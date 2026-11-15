#!/usr/bin/env node
/*
 * bench-js-argon2id.mjs — rate of the page's pure-JavaScript Argon2id.
 *
 * Argon2id is the expensive challenge: 38,912 1 KiB block fills per hash at
 * Anubis's parameters (m=19456, t=2, p=1). This measures the page's fallback,
 * extracted from betterAnusbis.c, so the number is the shipped code. The wasm
 * module is the fast path; wasm/bench-a2i.mjs measures that one.
 *
 *   node bench-js-argon2id.mjs            # 2 hashes, best of 2
 *   node bench-js-argon2id.mjs --json     # JSON for results/
 */
import { readFileSync } from 'node:fs';

const argv = process.argv.slice(2);
const flag = (name, def) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? Number(argv[i + 1]) : def;
};
const HASHES = flag('hashes', 2);
const ROUNDS = flag('rounds', 2);

const src = readFileSync(new URL('./betterAnusbis.c', import.meta.url), 'utf8');
const page = src.slice(src.indexOf('R"JS(') + 5, src.indexOf(')JS"'));
const start = page.indexOf('// ---------------------------------------------------------------- Argon2id');
const end = page.indexOf('  const shaAlgo=algo==="sha256"');
if (start < 0 || end < 0) throw new Error('Argon2id block not found in the page');
const code = page.slice(start, end);
const { argon2idTag } = new Function(code + '\nreturn {argon2idTag};')();

const raw = Uint8Array.from(
    '9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b'
        .match(/../g).map((x) => parseInt(x, 16)));
const now = () => Number(process.hrtime.bigint() / 1000n) / 1000;

let best = Infinity, tag = null;
for (let r = 0; r < ROUNDS; r++) {
    const t = now();
    for (let n = 0; n < HASHES; n++) tag = argon2idTag(raw, n);
    best = Math.min(best, now() - t);
}
const msPerHash = best / HASHES;
const result = {
    challenge_hex: '9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b',
    hashes: HASHES,
    best_ms: best,
    ms_per_hash: msPerHash,
    hashes_per_second: 1000 / msPerHash,
    tag: Buffer.from(tag).toString('hex'),
};
if (argv.includes('--json')) console.log(JSON.stringify(result, null, 2));
else {
    console.log(`pure-JavaScript Argon2id, ${HASHES} hashes, best of ${ROUNDS}\n`);
    console.log(`  ${msPerHash.toFixed(0)} ms/hash   ${(1000 / msPerHash).toFixed(2)} hashes/s`);
    console.log(`  tag ${result.tag.slice(0, 16)}... (check against anubis_verify_argon2id)`);
}
