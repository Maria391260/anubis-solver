#!/usr/bin/env node
/*
 * bench-js-tensor.mjs — rate of the page's pure-JavaScript int8 GEMM.
 *
 * The `tensor` tier is the one challenge whose cost is pure matrix algebra, so
 * the interesting number is not hashes per second but multiply-adds per
 * second: that is what separates the JavaScript floor from the wasm kernel and
 * from WebNN. This measures the functions exactly as they ship, extracted from
 * betterAnusbis.c, at every matrix size the difficulty rule can produce.
 *
 *   node bench-js-tensor.mjs                 # N = 256..1024, median of 5
 *   node bench-js-tensor.mjs --wasm --json   # plus the wasm kernel, JSON out
 *
 * The digest is checked against node:crypto for every size, so a fast result
 * that is wrong cannot be reported as a fast result.
 */
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';

const HERE = new URL('.', import.meta.url);
const argv = process.argv.slice(2);
const ROUNDS = argv.includes('--rounds') ? Number(argv[argv.indexOf('--rounds') + 1]) : 5;
const WANT_WASM = argv.includes('--wasm');
const WANT_JSON = argv.includes('--json');

const src = readFileSync(new URL('./betterAnusbis.c', HERE), 'utf8');
const page = src.slice(src.indexOf('R"JS(') + 5, src.indexOf(')JS"'));

const hashCore = page.slice(page.indexOf('const K=new Uint32Array'),
                            page.indexOf('const lz=(b)=>'));
const lzLine = page.slice(page.indexOf('const lz=(b)=>'),
                          page.indexOf('\n', page.indexOf('const lz=(b)=>')) + 1);
const hexHelpers = page.slice(page.indexOf('const hex2bytes='),
                              page.indexOf('\n', page.indexOf('const tohex=')) + 1);
const tensorBlock = page.slice(page.indexOf('const tenN='),
                               page.indexOf('async function tensorInJs'));
const { tensorExpand, tensorRows, tensorDigest } = new Function(
    hashCore + lzLine + hexHelpers + tensorBlock +
    '\nreturn {tensorExpand, tensorRows, tensorDigest};')();

// Fixed challenge; the last byte is below 0x80, same vector family as SPEED.md.
const RD = '9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b';
const SIZES = [256, 512, 768, 1024];

function median(xs) {
    const s = [...xs].sort((a, b) => a - b);
    return s[(s.length - 1) >> 1];
}
const digestOf = (c) => createHash('sha256').update(Buffer.from(c.buffer, c.byteOffset, c.byteLength)).digest('hex');

async function benchJs(n) {
    const { a, b } = tensorExpand(RD, n);
    const times = [];
    let c = null;
    for (let r = 0; r < ROUNDS; r++) {
        // a fresh C per round: tensorRows accumulates, it does not overwrite
        c = new Int32Array(n * n);
        const t0 = performance.now();
        tensorRows(a, b, c, n, 0, n);
        times.push(performance.now() - t0);
    }
    const ms = median(times);
    const macs = 2 * n * n * n;
    return {
        n, macs, ms,
        gmacs: macs / (ms / 1000) / 1e9,
        digest: digestOf(c),
        check: tensorDigest(c, n) === digestOf(c),
    };
}

async function benchWasm(n) {
    const path = new URL('./web/tensor_pow.wasm', HERE);
    if (!existsSync(path)) return null;
    const bytes = readFileSync(path);
    const { instance } = await WebAssembly.instantiate(bytes, { anubis: { anubis_update_nonce: () => {} } });
    const ex = instance.exports;
    const raw = Buffer.from(RD, 'hex');
    new Uint8Array(ex.memory.buffer).set(raw, ex.data_ptr());
    ex.set_data_length(raw.length);
    const times = [];
    let digest = '';
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        ex.anubis_work(n / 64, 0, 1);
        times.push(performance.now() - t0);
        digest = Buffer.from(new Uint8Array(ex.memory.buffer, ex.result_hash_ptr(), ex.result_hash_size())).toString('hex');
    }
    const ms = median(times);
    const macs = 2 * n * n * n;
    return { n, macs, ms, gmacs: macs / (ms / 1000) / 1e9, digest, bytes: bytes.length };
}

const rows = [];
for (const n of SIZES) {
    const js = await benchJs(n);
    const wasm = WANT_WASM ? await benchWasm(n) : null;
    rows.push({ js, wasm });
    if (!WANT_JSON) {
        console.log(`N=${String(n).padEnd(4)} JS ${js.ms.toFixed(1).padStart(8)} ms ` +
                    `${js.gmacs.toFixed(3).padStart(7)} GMAC/s  ${js.check ? 'digest ok' : 'DIGEST WRONG'}`);
        if (wasm) {
            console.log(`      wasm ${wasm.ms.toFixed(1).padStart(6)} ms ` +
                        `${wasm.gmacs.toFixed(3).padStart(7)} GMAC/s  ` +
                        `${(js.ms / wasm.ms).toFixed(1)}x faster than JS, ` +
                        `${wasm.digest === js.digest ? 'same digest' : 'DIGEST MISMATCH'}`);
        }
    }
}

if (WANT_JSON) {
    console.log(JSON.stringify({
        date: new Date().toISOString().slice(0, 10),
        host: cpus()[0]?.model ?? 'unknown',
        node: process.version,
        rounds: ROUNDS,
        challenge: RD,
        rows,
    }, null, 2));
}
