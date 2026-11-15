#!/usr/bin/env node
/*
 * bench-js-hashx.mjs — rate of the page's pure-JavaScript HashX interpreter.
 *
 * The wasm path either compiles the challenge's program to wasm or interprets
 * it with hashx_pow.wasm. This is the third path: the page's own interpreter,
 * extracted from betterAnusbis.c so it benchmarks the shipped code. It reports
 * the one-off program derivation and the per-hash cost; SPEED.md uses both to
 * compute the tier times.
 *
 *   node bench-js-hashx.mjs              # 500 hashes, best of 3
 *   node bench-js-hashx.mjs --json       # JSON for results/
 */
import { readFileSync } from 'node:fs';

const argv = process.argv.slice(2);
const flag = (name, def) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? Number(argv[i + 1]) : def;
};
const N = flag('hashes', 500);
const ROUNDS = flag('rounds', 3);

const src = readFileSync(new URL('./betterAnusbis.c', import.meta.url), 'utf8');
const page = src.slice(src.indexOf('R"JS(') + 5, src.indexOf(')JS"'));

// The generator is the page's HASHX_JIT block; the interpreter is the hx*
// functions and hashxHash that follow it.
const jitStart = page.indexOf('/* HASHX-JIT-BEGIN */');
const jitEnd = page.indexOf('/* HASHX-JIT-END */') + '/* HASHX-JIT-END */'.length;
const HASHX_JIT = new Function(page.slice(jitStart, jitEnd) + '\nreturn HASHX_JIT;')();
const hxStart = page.indexOf('function hxU64');
const hxEnd = page.indexOf('async function solveHashxInJs');
const hx = page.slice(hxStart, hxEnd);
const { hashxInit, hashxHash } =
    new Function('HASHX_JIT', hx + '\nreturn {hashxInit,hashxHash};')(HASHX_JIT);

const raw = Uint8Array.from(
    '9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b'
        .match(/../g).map((x) => parseInt(x, 16)));

const now = () => Number(process.hrtime.bigint() / 1000n) / 1000;
let t0 = now();
const init = hashxInit(raw);
const deriveMs = now() - t0;

const out = new Uint8Array(32);
let best = Infinity;
for (let r = 0; r < ROUNDS; r++) {
    const t = now();
    for (let n = 0; n < N; n++) hashxHash(init, n, out);
    best = Math.min(best, now() - t);
}
const rate = N / (best / 1000);
const result = {
    challenge_hex: '9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b',
    derive_ms: deriveMs,
    hashes: N,
    best_ms: best,
    hashes_per_second: rate,
    us_per_hash: 1e6 / rate,
};

if (argv.includes('--json')) {
    console.log(JSON.stringify(result, null, 2));
} else {
    console.log(`pure-JavaScript HashX, ${N} hashes, best of ${ROUNDS}\n`);
    console.log(`  program derivation   ${deriveMs.toFixed(2)} ms (once per challenge)`);
    console.log(`  interpreter          ${(rate / 1000).toFixed(2)} kH/s   ` +
                `${(1e6 / rate).toFixed(2)} us/hash   ${best.toFixed(1)} ms for ${N}`);
}
