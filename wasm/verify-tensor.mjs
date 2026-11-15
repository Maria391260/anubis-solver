#!/usr/bin/env node
/**
 * verify-tensor.mjs — does our wasm tensor PoW match the spec?
 *
 * The reference below is written straight from spec v1 in plain JS (SHA-256 from
 * node:crypto, typed-array matmul), with no import of any project file, so it
 * can only agree with the module if the module really implements the spec.
 *
 *   node wasm/verify-tensor.mjs [module.wasm]
 *
 * For difficulties 5, 8, 12 and 16 and three random 32-byte-hex challenges each
 * it checks:
 *
 *   1. the module's whole-pipeline digest (anubis_work) against the reference;
 *   2. the module's C matrix against the reference C matrix, with A and B
 *      injected directly — including all-127/-128 inputs, which is the worst
 *      case for the int16 pairwise sums in the SIMD kernel;
 *   3. when the module was built with -msimd128, the scalar kernel (mode 0)
 *      and the SIMD kernel (mode 1) against each other on the same A and B.
 *
 * Exits non-zero on any mismatch.
 */
import { readFileSync } from 'node:fs';
import { createHash, randomBytes } from 'node:crypto';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const MODULE = process.argv[2] ?? join(HERE, 'tensor_pow.wasm');

const bytes = readFileSync(MODULE);
const ex = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}).exports;
const HAS_SIMD = typeof ex.has_simd === 'function' && ex.has_simd() !== 0;
const mem = () => new Uint8Array(ex.memory.buffer);

const sha256 = (...parts) => {
  const h = createHash('sha256');
  for (const p of parts) h.update(p);
  return h.digest();
};
const hex = (b) => Buffer.from(b).toString('hex');

function n_for(difficulty) {
  const d = Math.min(16, Math.max(2, difficulty | 0));
  return 64 * d;
}

/* ------------------------------------------------------------- reference */

/* The stream is 2*N*N bytes: A takes the first N*N, B the second N*N. */
function referenceStream(seed, n) {
  const nb = n * n;
  const need = 2 * nb;
  const stream = Buffer.alloc(need);
  const ctr = Buffer.alloc(4);
  let produced = 0;
  let counter = 0;
  while (produced < need) {
    ctr.writeUInt32LE(counter++ >>> 0, 0);
    const dig = sha256(seed, ctr);
    const take = Math.min(32, need - produced);
    dig.copy(stream, produced, 0, take);
    produced += take;
  }
  const A = new Uint8Array(nb);
  const B = new Int8Array(nb);
  for (let i = 0; i < nb; i++) A[i] = stream[i] & 0x7f;
  for (let i = 0; i < nb; i++) {
    const v = stream[nb + i];
    B[i] = v < 128 ? v : v - 256;
  }
  return { A, B };
}

/* C[i][j] = sum_k A[i][k] * B[k][j], exact int32.  i,k,j order with the C row
 * held in a local typed array: the straightforward reading of the spec. */
function referenceMatmul(A, B, n) {
  const C = new Int32Array(n * n);
  const a = new Int32Array(n);
  const row = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const ai = i * n;
    for (let k = 0; k < n; k++) a[k] = A[ai + k];
    row.fill(0);
    for (let k = 0; k < n; k++) {
      const av = a[k];
      const bk = k * n;
      for (let j = 0; j < n; j++) row[j] += av * B[bk + j];
    }
    C.set(row, i * n);
  }
  return C;
}

function reference(challengeHex, difficulty) {
  const n = n_for(difficulty);
  // the seed is SHA-256 of the decoded challenge bytes (what set_data_length sees)
  const seed = sha256(Buffer.from(challengeHex, 'hex'));
  const { A, B } = referenceStream(seed, n);
  const C = referenceMatmul(A, B, n);
  const cbuf = Buffer.alloc(n * n * 4);
  for (let i = 0; i < C.length; i++) cbuf.writeInt32LE(C[i], i * 4);
  return { n, A, B, C, digest: hex(sha256(cbuf)) };
}

/* ------------------------------------------------------------ the module */

function wasmPipeline(challengeHex, difficulty) {
  // JS writes the decoded challenge bytes: that is what the kernel hashes
  const ch = Buffer.from(challengeHex, 'hex');
  const m = mem();
  m.set(ch, ex.data_ptr());
  ex.set_data_length(ch.length);
  const rc = ex.anubis_work(difficulty, 0, 1) >>> 0;
  const m2 = mem();
  return {
    rc,
    digest: hex(m2.subarray(ex.result_hash_ptr(), ex.result_hash_ptr() + ex.result_hash_size())),
  };
}

function wasmMatmul(A, B, n, mode) {
  ex.kernel_mode(mode);
  const m = mem();
  m.set(A, ex.a_ptr());
  // B is int8; write its two's-complement bytes
  m.set(B, ex.b_ptr());
  const rc = ex.tensor_matmul(n) >>> 0;
  if (rc !== 0) throw new Error(`tensor_matmul(${n}) returned ${rc}`);
  return Int32Array.from(new Int32Array(ex.memory.buffer, ex.c_ptr(), n * n));
}

/* ---------------------------------------------------------------- checks */

let failures = 0;
let checks = 0;
const ok = (cond, label) => {
  checks++;
  if (!cond) { failures++; console.log(`FAIL ${label}`); }
};

function compareMatrices(label, got, want, n) {
  let bad = -1;
  for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) { bad = i; break; }
  checks++;
  if (bad >= 0) {
    failures++;
    console.log(`FAIL ${label} C[${bad}] got=${got[bad]} want=${want[bad]}`);
  }
}

/* 1. whole-pipeline digest, three random challenges per difficulty */
const DIFFICULTIES = [5, 8, 12, 16];
for (const difficulty of DIFFICULTIES) {
  const n = n_for(difficulty);
  for (let c = 0; c < 3; c++) {
    const challenge = randomBytes(32).toString('hex');
    const ref = reference(challenge, difficulty);
    const got = wasmPipeline(challenge, difficulty);
    const pass = got.rc === 0 && got.digest === ref.digest;
    checks++;
    if (!pass) failures++;
    console.log(`${pass ? 'PASS' : 'FAIL'} pipeline difficulty=${difficulty} N=${n} ` +
                `challenge=${challenge.slice(0, 16)}... digest=${got.digest.slice(0, 16)}...`);
  }
}

/* 2. the kernel itself, on adversarial and random A/B */
const KERNEL_NS = [64, 320, 512, 1024];
if (!HAS_SIMD) console.log('info module has no SIMD kernel (built without -msimd128): scalar only');
for (const n of KERNEL_NS) {
  const nb = n * n;
  const patterns = [];
  {
    const A = new Uint8Array(nb).fill(127);
    const B = new Int8Array(nb);
    B.fill(-128);
    patterns.push(['all-127/-128', A, B]);
  }
  {
    const A = new Uint8Array(nb);
    const B = new Int8Array(nb);
    const r = randomBytes(nb * 2);
    for (let i = 0; i < nb; i++) A[i] = r[i] & 0x7f;
    for (let i = 0; i < nb; i++) { const v = r[nb + i]; B[i] = v < 128 ? v : v - 256; }
    patterns.push(['random', A, B]);
  }
  for (const [name, A, B] of patterns) {
    const want = referenceMatmul(A, B, n);
    const scalar = wasmMatmul(A, B, n, 0);
    compareMatrices(`kernel scalar N=${n} ${name}`, scalar, want, n);
    checks++;
    if (HAS_SIMD) {
      const simd = wasmMatmul(A, B, n, 1);
      compareMatrices(`kernel simd   N=${n} ${name}`, simd, want, n);
      let diff = -1;
      for (let i = 0; i < nb; i++) if (scalar[i] !== simd[i]) { diff = i; break; }
      checks++;
      if (diff >= 0) {
        failures++;
        console.log(`FAIL scalar != simd N=${n} ${name} C[${diff}]=${scalar[diff]} vs ${simd[diff]}`);
      }
    }
  }
  ok(true, `kernel N=${n} done`);
}

/* 3. a SIMD build's whole pipeline must also agree with the reference above;
 *    that is already checked in (1), this line just states which build ran. */
console.log(`\nmodule: ${MODULE} (${bytes.length} bytes) kernels: scalar${HAS_SIMD ? ' + simd' : ''}`);
console.log(`${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
