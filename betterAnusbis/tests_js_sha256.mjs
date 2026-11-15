#!/usr/bin/env node
/*
 * tests_js_sha256.mjs — the challenge page's pure-JavaScript proof solvers.
 *
 * The page carries a compact SHA-256 and two solvers: nibble challenges
 * (fast/slow) and the wasm-style sha256 challenge. Both run without wasm, and
 * a browser that cannot use WebAssembly must still pass. The regression this
 * guards against is the padding length: `bits>>>(8*i)` for i>=4 repeats the low
 * word because JavaScript masks shift counts to five bits, and that silently
 * produced wrong digests for every message.
 *
 * No browser and no gate are needed. The page's IIFE is extracted from
 * betterAnusbis.c and run against a stub DOM/location; the answers are checked
 * with node:crypto, independently of the page's own SHA-256.
 *
 *   node tests_js_sha256.mjs
 */
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const source = readFileSync(new URL('./betterAnusbis.c', import.meta.url), 'utf8');
const start = source.indexOf('static const char *CHALLENGE_JS = R"JS(');
if (start < 0) throw new Error('CHALLENGE_JS not found in betterAnusbis.c');
const begin = source.indexOf('R"JS(', start) + 5;
const end = source.indexOf(')JS"', begin);
if (end < 0) throw new Error('CHALLENGE_JS terminator not found');
const pageJs = source.slice(begin, end);

const sha256 = (buf) => createHash('sha256').update(buf).digest();

function leadingZeroBits(buf) {
    let n = 0;
    for (const x of buf) {
        if (x === 0) { n += 8; continue; }
        n += Math.clz32(x) - 24;
        break;
    }
    return n;
}

function hexToBytes(hex) {
    return Uint8Array.from(hex.match(/../g).map((h) => parseInt(h, 16)));
}

// Run the page's solver for one synthetic challenge, with `wasm` present or
// absent and with/without the #pure-js hash. Returns {nonce, response}.
async function solve({ algorithm, difficulty, randomData, hash = '', wasm = true }) {
    const payload = JSON.stringify({
        rules: { algorithm, difficulty },
        challenge: {
            issuedAt: '1', metadata: {}, id: 'test.1', method: algorithm,
            randomData, policyRuleHash: '', difficulty, spent: false, extensions: null,
        },
    });
    let answer = null;
    const sandbox = {
        console,
        TextEncoder,
        URL,
        setTimeout,
        clearTimeout,
        performance: { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 },
        document: {
            getElementById: (id) => id === 'anubis_challenge' ? { textContent: payload } : null,
        },
        location: {
            hash, search: '', pathname: '/', origin: 'http://gate.test',
            replace: (u) => { answer = u; },
        },
    };
    if (wasm) sandbox.WebAssembly = WebAssembly;
    else delete sandbox.WebAssembly;
    const context = vm.createContext(sandbox);
    vm.runInContext(pageJs, context, { filename: 'challenge.js' });

    const deadline = Date.now() + 60000;
    while (answer === null && Date.now() < deadline)
        await new Promise((r) => setTimeout(r, 20));
    if (answer === null) throw new Error(`solver did not finish (${algorithm})`);
    const u = new URL(answer);
    return { nonce: Number(u.searchParams.get('nonce')), response: u.searchParams.get('response') };
}

let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  PASS  ${name}`); }
    else { failed++; console.log(`  FAIL  ${name}${detail ? ': ' + detail : ''}`); }
}

// 1. the page's SHA-256 against node:crypto, including the padding boundaries
{
    const jsStart = pageJs.indexOf('const K=new Uint32Array');
    const jsEnd = pageJs.indexOf('const lz=(b)=>');
    if (jsStart < 0 || jsEnd < 0) throw new Error('hash functions not found in the page');
    const hash = new Function(pageJs.slice(jsStart, jsEnd) + '\nreturn sha256bytes;')();
    const vectors = [
        ['empty', new Uint8Array(0)],
        ['abc', new TextEncoder().encode('abc')],
        ['55 bytes', new TextEncoder().encode('a'.repeat(55))],
        ['56 bytes', new TextEncoder().encode('a'.repeat(56))],
        ['64 bytes', new TextEncoder().encode('a'.repeat(64))],
        ['1000 bytes', Uint8Array.from({ length: 1000 }, (_, i) => (i * 37 + 11) & 255)],
    ];
    for (const [name, data] of vectors) {
        const got = Buffer.from(hash(data)).toString('hex');
        check(`sha256 ${name}`, got === sha256(Buffer.from(data)).toString('hex'), got);
    }
}

// 2. sha256 challenge, pure JS forced by #pure-js, little-endian nonce
{
    const randomData = '00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0fef';
    const difficulty = 4;
    const { nonce, response } = await solve({ algorithm: 'sha256', difficulty, randomData, hash: '#pure-js', wasm: true });
    const data = hexToBytes(randomData);
    const nonceBytes = Buffer.alloc(4);
    nonceBytes.writeUInt32LE(nonce >>> 0, 0);
    const expected = sha256(Buffer.concat([Buffer.from(data), nonceBytes])).toString('hex');
    check('sha256 LE nonce digest', response === expected, `${response} != ${expected}`);
    check('sha256 LE leading bits', leadingZeroBits(Buffer.from(response, 'hex')) >= difficulty);
}

// 3. sha256 challenge, WebAssembly missing -> automatic JS fallback, big-endian
{
    const randomData = 'ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221117';
    const difficulty = 4;
    const { nonce, response } = await solve({ algorithm: 'sha256', difficulty, randomData, wasm: false });
    const data = hexToBytes(randomData);
    const nonceBytes = Buffer.alloc(4);
    nonceBytes.writeUInt32BE(nonce >>> 0, 0);
    const expected = sha256(Buffer.concat([Buffer.from(data), nonceBytes])).toString('hex');
    check('sha256 BE nonce digest (no wasm)', response === expected, `${response} != ${expected}`);
    check('sha256 BE leading bits', leadingZeroBits(Buffer.from(response, 'hex')) >= difficulty);
}

// 4. fast/slow nibble solver: sha256("<randomData><nonce decimal>"), nibbles
{
    const randomData = '00112233445566778899aabbccddeeff';
    const difficulty = 3;
    const { nonce, response } = await solve({ algorithm: 'fast', difficulty, randomData });
    const expected = sha256(Buffer.from(randomData + nonce)).toString('hex');
    check('fast nonce digest', response === expected, `${response} != ${expected}`);
    check('fast leading nibbles', /^0+/.test(expected.slice(0, difficulty)) && expected.slice(0, difficulty) === '0'.repeat(difficulty));
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
