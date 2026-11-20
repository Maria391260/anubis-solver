#!/usr/bin/env node
/*
 * tests_js_tensor.mjs — the `tensor` tier's three browser paths.
 *
 * The tier's proof is one integer GEMM whose digest is the whole answer, so
 * a browser that gets the derivation, the accumulation width or the digest
 * serialisation wrong produces a digest the server rejects, with no way to
 * notice locally. This test extracts the page's own solver and checks it
 * against an independent implementation (node:crypto + a plain Int32Array
 * GEMM), not against the page's helpers.
 *
 * Covered: the pure-JavaScript GEMM (forced with #pure-js), the automatic
 * fallback when WebAssembly is missing, the wasm kernel (when it is built),
 * WebNN used when it works, and WebNN rejected when it returns wrong numbers.
 * No browser and no gate are needed.
 *
 *   node tests_js_tensor.mjs
 */
import vm from 'node:vm';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';

const here = new URL('.', import.meta.url);
const source = readFileSync(new URL('./betterAnusbis.c', here), 'utf8');
const start = source.indexOf('static const char *CHALLENGE_JS = R"JS(');
if (start < 0) throw new Error('CHALLENGE_JS not found in betterAnusbis.c');
const begin = source.indexOf('R"JS(', start) + 5;
const end = source.indexOf(')JS"', begin);
if (end < 0) throw new Error('CHALLENGE_JS terminator not found');
const pageJs = source.slice(begin, end);

const sha256 = (buf) => createHash('sha256').update(buf).digest();
const dim = (difficulty) => 64 * Math.max(2, Math.min(16, difficulty));

/* The spec, written out again on purpose. */
function expand(rdHex, n) {
    const cells = n * n;
    const seed = sha256(Buffer.from(rdHex, 'hex'));
    const a = new Uint8Array(cells);
    const b = new Int8Array(cells);
    const ctr = Buffer.alloc(4);
    let pos = 0;
    for (let counter = 0; pos < 2 * cells; counter++) {
        ctr.writeUInt32LE(counter >>> 0);
        const block = sha256(Buffer.concat([seed, ctr]));
        for (let i = 0; i < 32 && pos < 2 * cells; i++, pos++) {
            if (pos < cells) a[pos] = block[i] & 0x7f;
            else b[pos - cells] = (block[i] << 24) >> 24;
        }
    }
    return { a, b };
}
function gemm(a, b, n) {
    const c = new Int32Array(n * n);
    for (let i = 0; i < n; i++) {
        const row = i * n;
        for (let k = 0; k < n; k++) {
            const av = a[row + k];
            if (av === 0) continue;
            const brow = k * n;
            for (let j = 0; j < n; j++) c[row + j] += av * b[brow + j];
        }
    }
    return c;
}
function refDigest(rdHex, difficulty) {
    const n = dim(difficulty);
    const c = gemm(...Object.values(expand(rdHex, n)), n);
    const bytes = Buffer.alloc(n * n * 4);
    for (let i = 0; i < c.length; i++) bytes.writeInt32LE(c[i], 4 * i);
    return sha256(bytes).toString('hex');
}

/* A stand-in WebNN that computes the gemm with the reference above, and a
 * variant that returns zeros so the page's row check has to reject it. */
function fakeWebNN({ garbage = false } = {}) {
    const compute = (a, b, n) => (garbage ? new Int32Array(n * n) : gemm(a, b, n));
    let pending = null;
    const context = {
        createTensor: (d) => ({ dataType: d.dataType, shape: d.shape }),
        writeTensor: (t, data) => { t.data = data; },
        dispatch: (graph, inputs, outputs) => { pending = { graph, inputs, outputs }; },
        readTensor: async (t) => t.data ?? (t.data = Float32Array.from(pending && pending.graph.run(pending.inputs, pending.outputs))),
    };
    const builder = {
        input: (name, desc) => ({ kind: 'input', name, desc }),
        gemm: (A, B) => ({ kind: 'gemm', A, B }),
        build: async () => ({
            run: (inputs, outputs) => {
                const n = inputs.A.shape[0];
                const c = compute(inputs.A.data, inputs.B.data, n);
                outputs.C.data = c;
                return c;
            },
        }),
    };
    return { createContext: async () => context, builder, MLGraphBuilder: function () { return builder; } };
}

const wasmPath = new URL('./web/tensor_pow.wasm', here);
const wasmBytes = existsSync(wasmPath) ? readFileSync(wasmPath) : null;

async function solve({ algorithm = 'tensor', difficulty, randomData, hash = '', wasm = false, webnn = null }) {
    const payload = JSON.stringify({
        rules: { algorithm, difficulty },
        challenge: {
            issuedAt: '1', metadata: {}, id: 'test.1', method: algorithm,
            randomData, policyRuleHash: '', difficulty, spent: false, extensions: null,
        },
    });
    let answer = null;
    const sandbox = {
        // failures are the point in two of these tests; do not spam CI
        console: { log: console.log, warn: () => {}, error: console.error },
        TextEncoder,
        URL,
        setTimeout,
        clearTimeout,
        performance: { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 },
        document: { getElementById: (id) => (id === 'anubis_challenge' ? { textContent: payload } : null) },
        location: {
            hash, search: '', pathname: '/', origin: 'http://gate.test',
            replace: (u) => { answer = u; },
        },
    };
    if (webnn) {
        sandbox.navigator = { ml: webnn };
        sandbox.MLGraphBuilder = webnn.MLGraphBuilder;
    }
    if (wasm) {
        sandbox.WebAssembly = WebAssembly;
        sandbox.fetch = async (url) => {
            if (String(url).includes('tensor_pow.wasm')) {
                if (!wasmBytes) throw new Error('tensor_pow.wasm is not built');
                return { arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) };
            }
            throw new Error('unexpected fetch ' + url);
        };
    }
    const context = vm.createContext(sandbox);
    vm.runInContext(pageJs, context, { filename: 'challenge.js' });

    const deadline = Date.now() + 120000;
    while (answer === null && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
    if (answer === null) throw new Error('tensor solver did not finish');
    const u = new URL(answer);
    return { nonce: Number(u.searchParams.get('nonce')), response: u.searchParams.get('response') };
}

let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  PASS  ${name}`); }
    else { failed++; console.log(`  FAIL  ${name}${detail ? ': ' + detail : ''}`); }
}

const VECTORS = [
    ['00', '00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f'],
    ['01', '9f3c1a7b24d5e8061c2b3a495867f0e1d2c3b4a5968778695a4b3c2d1e0f1a2b'],
];

// 1. forced pure JavaScript, with and without WebAssembly present
for (const [tag, randomData] of VECTORS) {
    for (const difficulty of [2, 5]) {
        const want = refDigest(randomData, difficulty);
        const forced = await solve({ difficulty, randomData, hash: '#pure-js', wasm: true });
        check(`pure-js ${tag} d=${difficulty} digest`, forced.response === want, forced.response);
        check(`pure-js ${tag} d=${difficulty} nonce`, forced.nonce === 0, String(forced.nonce));

        const noWasm = await solve({ difficulty, randomData, wasm: false });
        check(`no-wasm ${tag} d=${difficulty} digest`, noWasm.response === want, noWasm.response);
    }
}

// 2. the wasm kernel, when it has been built
if (wasmBytes) {
    const randomData = VECTORS[1][1];
    const difficulty = 5;
    const want = refDigest(randomData, difficulty);
    const got = await solve({ difficulty, randomData, wasm: true });
    check(`wasm kernel digest`, got.response === want, got.response);
    check(`wasm kernel nonce`, got.nonce === 0, String(got.nonce));
} else {
    console.log('  SKIP  wasm kernel (web/tensor_pow.wasm not built)');
}

// 3. WebNN: accepted when it agrees, rejected and fallen back when it does not
{
    const randomData = VECTORS[0][1];
    const difficulty = 2;
    const want = refDigest(randomData, difficulty);

    const good = await solve({ difficulty, randomData, webnn: fakeWebNN() });
    check('WebNN good digest', good.response === want, good.response);

    const bad = await solve({ difficulty, randomData, webnn: fakeWebNN({ garbage: true }), wasm: Boolean(wasmBytes) });
    check('WebNN garbage rejected, fell back to a correct digest', bad.response === want, bad.response);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
