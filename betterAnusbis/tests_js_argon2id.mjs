#!/usr/bin/env node
/*
 * tests_js_argon2id.mjs — the challenge page's pure-JavaScript Argon2id.
 *
 * The page's fallback is the reference fill (third_party/argon2/ref.c) ported
 * to 32-bit pairs, with the host side (BLAKE2b, H0, B0/B1, tag) from the
 * WebGPU demo. One hash is ~0.7-1.5 s, so the cases solve at difficulty 0
 * (the first nonce is accepted) and check the tag against two independent
 * implementations: argon2id_pow.wasm and the C engine's
 * anubis_verify_argon2id.
 *
 *   node tests_js_argon2id.mjs
 */
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = new URL('..', import.meta.url);
const source = readFileSync(new URL('./betterAnusbis.c', import.meta.url), 'utf8');
const start = source.indexOf('static const char *CHALLENGE_JS = R"JS(');
if (start < 0) throw new Error('CHALLENGE_JS not found');
const begin = source.indexOf('R"JS(', start) + 5;
const pageJs = source.slice(begin, source.indexOf(')JS"', begin));

const wasmBytes = readFileSync(new URL('../wasm/argon2id_pow.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasmBytes,
    { anubis: { anubis_update_nonce: () => {} } });
const ex = instance.exports;

function wasmTag(raw, nonce) {
    new Uint8Array(ex.memory.buffer).set(raw, ex.data_ptr());
    ex.set_data_length(raw.length);
    const n = ex.anubis_work(0, nonce, 1) >>> 0;
    if (n !== nonce) throw new Error('wasm did not return the nonce');
    return Buffer.from(new Uint8Array(ex.memory.buffer, ex.result_hash_ptr(),
                                      ex.result_hash_size())).toString('hex');
}
function cTag(rawHex, nonce) {
    const lib = new URL('libanubis_solver.so', ROOT).pathname;
    const script = `
import ctypes
lib = ctypes.CDLL(${JSON.stringify(lib)})
out = ctypes.create_string_buffer(65)
rc = lib.anubis_verify_argon2id(${JSON.stringify(rawHex)}.encode(), ctypes.c_uint32(${nonce}),
                                ctypes.c_int(0), out)
print(rc, out.value.decode())
`;
    const r = spawnSync('python3', ['-c', script], { encoding: 'utf8' });
    if (r.status !== 0) return null;
    const [rc, digest] = r.stdout.trim().split(/\s+/);
    return rc === '0' ? digest : null;
}
async function jsSolve(rd, wasmAvailable) {
    const difficulty = 0;
    const payload = JSON.stringify({
        rules: { algorithm: 'argon2id', difficulty },
        challenge: {
            issuedAt: '1', metadata: {}, id: 'test.1', method: 'argon2id',
            randomData: rd, policyRuleHash: '', difficulty, spent: false, extensions: null,
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
    const deadline = Date.now() + 180000;
    while (answer === null && Date.now() < deadline)
        await new Promise((r) => setTimeout(r, 50));
    if (answer === null) throw new Error('solver did not finish');
    const u = new URL(answer);
    return { nonce: Number(u.searchParams.get('nonce')), response: u.searchParams.get('response') };
}

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  PASS  ${name}`); }
    else { failed++; console.log(`  FAIL  ${name}${detail ? ': ' + detail : ''}`); }
}

const CASES = [
    ['9f2c1b4a7e5d3c0b8a1f6e2d4c7b0a9f5e3d1c8b7a6f4e2d1c0b9a8f7e6d5c4b', true],
    ['ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221117', false],
];
for (const [rd, wasmAvailable] of CASES) {
    const t0 = Date.now();
    const { nonce, response } = await jsSolve(rd, wasmAvailable);
    const raw = Uint8Array.from(rd.match(/../g).map((x) => parseInt(x, 16)));
    const fromWasm = wasmTag(raw, nonce);
    const fromC = cTag(rd, nonce);
    const tag = wasmAvailable ? '#pure-js' : 'no WebAssembly';
    check(`argon2id ${tag}, nonce ${nonce} (${Date.now() - t0} ms)`, response === fromWasm,
          `js ${response} wasm ${fromWasm}`);
    if (fromC !== null) check('argon2id C engine', fromC === response, `C ${fromC}`);
}
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
