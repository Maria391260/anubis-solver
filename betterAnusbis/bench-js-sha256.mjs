#!/usr/bin/env node
/*
 * bench-js-sha256.mjs — rate of the page's pure-JavaScript SHA-256 fallback.
 *
 * The browser benchmarks in SPEED.md measure the actual challenge page; this
 * is the headless reproduction of the same functions, extracted from
 * betterAnusbis.c so the numbers cannot drift from the code that ships. It
 * measures three things:
 *
 *   reused    the sha256 fallback: one padded block, only the 4 nonce bytes
 *             rewritten per attempt (what solveSha256InJs does)
 *   one-shot  sha256bytes(): a fresh padded block per candidate (fast/slow)
 *   search    the full search to a difficulty, hashes counted from the nonce
 *
 * With --wasm it also runs wasm/sha256_pow_batch.wasm on the same challenge,
 * which is the fast path the JS fallback replaces when WebAssembly is off.
 *
 *   node bench-js-sha256.mjs                 # 3 rounds, difficulty 20
 *   node bench-js-sha256.mjs --wasm --json   # plus the 4-wide kernel, JSON out
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const argv = process.argv.slice(2);
const flag = (name, def) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? Number(argv[i + 1]) : def;
};
const DIFF = flag('difficulty', 20);
const ROUNDS = flag('rounds', 3);
const RAW_ITERS = flag('iters', 2000000);
const WANT_WASM = argv.includes('--wasm');
const WANT_JSON = argv.includes('--json');

// The functions as they ship: K, rotr, sha256padded, sha256bytes, sha256length, lz.
const src = readFileSync(new URL('./betterAnusbis.c', import.meta.url), 'utf8');
const page = src.slice(src.indexOf('R"JS(') + 5, src.indexOf(')JS"'));
const core = page.slice(page.indexOf('const K=new Uint32Array'),
                        page.indexOf('const lz=(b)=>'));
const lzEnd = page.indexOf('\n', page.indexOf('const lz=(b)=>')) + 1;
const hash = new Function(core + '\n' + page.slice(page.indexOf('const lz=(b)=>'), lzEnd) +
                          '\nreturn {sha256padded, sha256bytes, lz};')();

// Fixed challenge used in SPEED.md; last byte < 0x80 -> big-endian nonce.
const CH = Uint8Array.from(
    '9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b'
        .match(/../g).map((x) => parseInt(x, 16)));
const little = (CH[CH.length - 1] & 0x80) !== 0;
const total = CH.length + 4;
const padded = new Uint8Array(((total + 9 + 63) / 64 | 0) * 64);
padded.set(CH);
padded[total] = 0x80;
{
    const bits = total * 8;
    const hi = Math.floor(bits / 4294967296) >>> 0, lo = bits >>> 0;
    for (let i = 0; i < 4; i++) {
        padded[padded.length - 1 - i] = (lo >>> (8 * i)) & 255;
        padded[padded.length - 5 - i] = (hi >>> (8 * i)) & 255;
    }
}
const P = CH.length;
const writeNonce = (n) => {
    if (little) {
        padded[P] = n & 255; padded[P + 1] = (n >>> 8) & 255;
        padded[P + 2] = (n >>> 16) & 255; padded[P + 3] = (n >>> 24) & 255;
    } else {
        padded[P] = (n >>> 24) & 255; padded[P + 1] = (n >>> 16) & 255;
        padded[P + 2] = (n >>> 8) & 255; padded[P + 3] = n & 255;
    }
};
const now = () => Number(process.hrtime.bigint() / 1000n) / 1000;
const best = (rounds, fn) => {
    let t = Infinity;
    for (let i = 0; i < rounds; i++) {
        const t0 = now();
        fn();
        t = Math.min(t, now() - t0);
    }
    return t;
};

function reusedRate(iters) {
    const ms = best(ROUNDS, () => {
        for (let n = 0; n < iters; n++) { writeNonce(n); hash.sha256padded(padded); }
    });
    return { iters, best_ms: ms, rate: iters / (ms / 1000) };
}
function oneShotRate(iters) {
    const enc = new TextEncoder();
    const ms = best(ROUNDS, () => {
        for (let n = 0; n < iters; n++)
            hash.sha256bytes(enc.encode('9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f' + n));
    });
    return { iters, best_ms: ms, rate: iters / (ms / 1000) };
}
function jsSearch(diff) {
    let nonce = -1, digest = null;
    const ms = best(ROUNDS, () => {
        for (let n = 0; ; n++) {
            writeNonce(n);
            const d = hash.sha256padded(padded);
            if (hash.lz(d) >= diff) { nonce = n; digest = d; break; }
        }
    });
    const hashes = nonce + 1;
    return { hashes, best_ms: ms, rate: hashes / (ms / 1000), nonce,
             digest: Buffer.from(digest).toString('hex') };
}
async function wasmSearch(diff) {
    const bytes = readFileSync(new URL('../wasm/sha256_pow_batch.wasm', import.meta.url));
    const { instance } = await WebAssembly.instantiate(bytes,
        { anubis: { anubis_update_nonce: () => {} } });
    const ex = instance.exports;
    new Uint8Array(ex.memory.buffer).set(CH, ex.data_ptr());
    ex.set_data_length(CH.length);
    ex.anubis_work(8, 0, 1); // warm the tier-up compiler
    let nonce = -1;
    const ms = best(ROUNDS, () => { nonce = ex.anubis_work(diff, 0, 1) >>> 0; });
    const hashes = nonce + 1;
    const digest = new Uint8Array(ex.memory.buffer, ex.result_hash_ptr(), ex.result_hash_size());
    return { hashes, best_ms: ms, rate: hashes / (ms / 1000), nonce,
             digest: Buffer.from(digest).toString('hex') };
}

// warm the JIT
reusedRate(100000); oneShotRate(20000); jsSearch(10);

const out = {
    challenge_hex: Buffer.from(CH).toString('hex'),
    difficulty: DIFF,
    js_search: jsSearch(DIFF),
    js_reused: reusedRate(RAW_ITERS),
    js_one_shot: oneShotRate(200000),
};
if (WANT_WASM) out.wasm_4wide = await wasmSearch(DIFF);

if (WANT_JSON) {
    console.log(JSON.stringify(out, null, 2));
} else {
    const M = (r) => `${(r.rate / 1e6).toFixed(3)} M/s`;
    console.log(`pure-JavaScript SHA-256, ${DIFF} bits, best of ${ROUNDS}\n`);
    console.log(`  reused padded block   ${M(out.js_reused)}   ${(1e6 / out.js_reused.rate).toFixed(3)} us/hash`);
    console.log(`  one-shot (fast/slow)  ${M(out.js_one_shot)}   ${(1e6 / out.js_one_shot.rate).toFixed(3)} us/hash`);
    console.log(`  search d${DIFF}           ${M(out.js_search)}   ${out.js_search.hashes} hashes`);
    if (out.wasm_4wide) {
        console.log(`  wasm 4-wide           ${M(out.wasm_4wide)}   ${out.wasm_4wide.hashes} hashes   ` +
                    `${(out.wasm_4wide.rate / out.js_reused.rate).toFixed(1)}x JS`);
        console.log(`  same answer: ${out.wasm_4wide.nonce === out.js_search.nonce &&
                    out.wasm_4wide.digest === out.js_search.digest ? 'yes' : 'NO'}`);
    }
}
