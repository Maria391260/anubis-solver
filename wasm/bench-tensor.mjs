#!/usr/bin/env node
/**
 * bench-tensor.mjs — how fast is the wasm tensor PoW?
 *
 * One "solve" is the whole spec v1 pipeline: seed, stream, A/B fill, C = A@B,
 * digest. There is no nonce search, so the whole `anubis_work` call is the unit
 * of work and it is timed directly.
 *
 *   node wasm/bench-tensor.mjs [--runs 5] [--module path.wasm]
 *
 * Reports per-solve milliseconds (median of 5) and the effective rate as
 * 2*N^3 MACs / second, the same convention the task spec uses. The challenge is
 * fixed so the numbers are reproducible; the machine is printed with them.
 */
import { readFileSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import os from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const RUNS = Number(flag('runs', 5));
const MODULE = flag('module', join(HERE, 'tensor_pow.wasm'));

const bytes = readFileSync(MODULE);
const ex = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}).exports;

const nFor = (d) => 64 * Math.min(16, Math.max(2, d | 0));

/* fixed challenge: deterministic, and the same one every run */
const seed = createHash('sha256').update('anubis tensor bench').digest('hex');
const CHALLENGE = Buffer.from(seed, 'ascii');   // 64 ASCII bytes

function solve(difficulty) {
  const m = new Uint8Array(ex.memory.buffer);
  m.set(CHALLENGE, ex.data_ptr());
  ex.set_data_length(CHALLENGE.length);
  const rc = ex.anubis_work(difficulty, 0, 1) >>> 0;
  if (rc !== 0) throw new Error(`anubis_work returned ${rc}`);
  const rp = ex.result_hash_ptr();
  return Buffer.from(new Uint8Array(ex.memory.buffer, rp, ex.result_hash_size()))
    .toString('hex');
}

const median = (xs) => {
  const a = [...xs].sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length & 1 ? a[m] : (a[m - 1] + a[m]) / 2;
};

const cpu = os.cpus()[0]?.model ?? 'unknown CPU';
console.log(`machine: ${cpu} | node ${process.version} | ${os.platform()} ${os.arch()}`);
console.log(`module:  ${MODULE} (${statSync(MODULE).size} bytes), ` +
            `kernels: ${ex.has_simd?.() ? 'scalar + simd' : 'scalar'}\n`);

const results = [];
for (const difficulty of [8, 12, 16]) {
  const n = nFor(difficulty);
  solve(difficulty);                       // warm up / compile tier
  const times = [];
  for (let r = 0; r < RUNS; r++) {
    const t0 = process.hrtime.bigint();
    solve(difficulty);
    const t1 = process.hrtime.bigint();
    times.push(Number(t1 - t0) / 1e6);
  }
  const ms = median(times);
  const macs = 2 * n ** 3;
  const gmacs = macs / (ms / 1000) / 1e9;
  results.push({ difficulty, n, ms, gmacs });
  console.log(`d=${String(difficulty).padStart(2)}  N=${String(n).padStart(4)}  ` +
              `${ms.toFixed(1).padStart(8)} ms/solve  ` +
              `${gmacs.toFixed(3).padStart(7)} GMAC/s  ` +
              `(2*N^3 = ${(macs / 1e6).toFixed(0)} MMAC, runs: ` +
              times.map((t) => t.toFixed(1)).join(', ') + ')');
}

const r512 = results.find((r) => r.n === 512);
const r768 = results.find((r) => r.n === 768);
if (r512 && r768) {
  console.log(`\nsummary: N=512 ${r512.ms.toFixed(1)} ms (${r512.gmacs.toFixed(3)} GMAC/s), ` +
              `N=768 ${r768.ms.toFixed(1)} ms (${r768.gmacs.toFixed(3)} GMAC/s)`);
}
