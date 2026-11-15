#!/usr/bin/env node
/*
 * tests_js_hashx.mjs — the challenge page's pure-JavaScript HashX fallback.
 *
 * HashX has no fixed hash: the challenge derives a 512-instruction program,
 * which the wasm path compiles or interprets. The JS fallback interprets it
 * too, with BigInt. The regression surface is therefore large — PRF register
 * fill, every opcode, the guarded backward branch, finalisation — so each
 * solved challenge is checked against the wasm interpreter on the same nonce,
 * and, when the C library is present, against anubis_verify_hashx().
 *
 * No browser and no gate are needed. The page's IIFE is extracted from
 * betterAnusbis.c and run with a stub DOM/location; the answer is then checked
 * independently.
 *
 *   node tests_js_hashx.mjs
 */
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const ROOT = new URL('..', import.meta.url);
const source = readFileSync(new URL('./betterAnusbis.c', import.meta.url), 'utf8');
const start = source.indexOf('static const char *CHALLENGE_JS = R"JS(');
if (start < 0) throw new Error('CHALLENGE_JS not found');
const begin = source.indexOf('R"JS(', start) + 5;
const end = source.indexOf(')JS"', begin);
const pageJs = source.slice(begin, end);

const wasmBytes = readFileSync(new URL('../wasm/hashx_pow.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasmBytes,
    { anubis: { anubis_update_nonce: () => {} } });
const ex = instance.exports;

function wasmDigest(raw, nonce) {
    new Uint8Array(ex.memory.buffer).set(raw, ex.data_ptr());
    ex.set_data_length(raw.length);
    const n = ex.anubis_work(0, nonce, 1) >>> 0;
    if (n !== nonce) throw new Error('wasm did not return the nonce');
    return Buffer.from(new Uint8Array(ex.memory.buffer, ex.result_hash_ptr(),
                                      ex.result_hash_size())).toString('hex');
}

const lz = (hex) => {
    let n = 0;
    for (const b of Buffer.from(hex, 'hex')) {
        if (b === 0) { n += 8; continue; }
        n += Math.clz32(b) - 24;
        break;
    }
    return n;
};

// Verify with the C engine when the library is built; ctypes directly, so the
// Python package's imports (requests and friends) do not matter.
function cDigest(rawHex, nonce, difficulty) {
    const lib = new URL('libanubis_solver.so', ROOT).pathname;
    const script = `
import ctypes, sys
lib = ctypes.CDLL(${JSON.stringify(lib)})
out = ctypes.create_string_buffer(65)
rc = lib.anubis_verify_hashx(${JSON.stringify(rawHex)}.encode(), ctypes.c_uint32(${nonce}),
                             ctypes.c_int(${difficulty}), out)
print(rc, out.value.decode())
`;
    const r = spawnSync('python3', ['-c', script], { encoding: 'utf8' });
    if (r.status !== 0) return null; // library or python missing: skip
    const [rc, digest] = r.stdout.trim().split(/\s+/);
    return rc === '0' ? digest : null;
}

async function jsSolve(randomData, difficulty, wasmAvailable) {
    const payload = JSON.stringify({
        rules: { algorithm: 'hashx', difficulty },
        challenge: {
            issuedAt: '1', metadata: {}, id: 'test.1', method: 'hashx',
            randomData, policyRuleHash: '', difficulty, spent: false, extensions: null,
        },
    });
    let answer = null;
    const sandbox = {
        console, TextEncoder, URL, setTimeout, clearTimeout,
        performance: { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 },
        document: { getElementById: (id) => id === 'anubis_challenge' ? { textContent: payload } : null },
        location: {
            hash: '#pure-js', search: '', pathname: '/', origin: 'http://gate.test',
            replace: (u) => { answer = u; },
        },
    };
    if (wasmAvailable) sandbox.WebAssembly = WebAssembly;
    else delete sandbox.WebAssembly;
    vm.createContext(sandbox);
    vm.runInContext(pageJs, sandbox, { filename: 'challenge.js' });
    const deadline = Date.now() + 120000;
    while (answer === null && Date.now() < deadline)
        await new Promise((r) => setTimeout(r, 20));
    if (answer === null) throw new Error('solver did not finish');
    const u = new URL(answer);
    return { nonce: Number(u.searchParams.get('nonce')), response: u.searchParams.get('response') };
}

let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  PASS  ${name}`); }
    else { failed++; console.log(`  FAIL  ${name}${detail ? ': ' + detail : ''}`); }
}

// Three challenges, including little- and big-endian seeds and both dispatch
// paths: #pure-js with WebAssembly present, and WebAssembly absent entirely.
const CASES = [
    ['00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f', 1, true],
    ['ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221117', 1, false],
    ['9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b', 2, false],
];

for (const [randomData, difficulty, wasmAvailable] of CASES) {
    const raw = Uint8Array.from(randomData.match(/../g).map((x) => parseInt(x, 16)));
    const { nonce, response } = await jsSolve(randomData, difficulty, wasmAvailable);
    const expected = wasmDigest(raw, nonce);
    const tag = wasmAvailable ? '#pure-js' : 'no WebAssembly';
    check(`hashx d${difficulty} ${tag}, nonce ${nonce}`, response === expected,
          `js ${response} wasm ${expected}`);
    check(`hashx d${difficulty} leading bits`, lz(response) >= difficulty);
    const fromC = cDigest(randomData, nonce, difficulty);
    if (fromC !== null) {
        check(`hashx d${difficulty} C engine`, fromC === response, `C ${fromC}`);
    }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
