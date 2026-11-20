#!/usr/bin/env node
/*
 * tests_js_tensor_v.mjs — the `tensor-v` browser client, WebNN included.
 *
 * The page waits for the server to pick sampled blocks, so this test stands in
 * for the gate: it hands out a fixed sample, captures the root from the first
 * request and the witness from the second, and checks both against an
 * independent implementation of the commitment scheme. WebNN is faked at the
 * builder/context boundary, which is enough to exercise the bTranspose path,
 * the explicit-transpose retry and the row check that rejects a broken
 * backend.
 *
 *   node tests_js_tensor_v.mjs
 */
import vm from 'node:vm';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';

const here = new URL('.', import.meta.url);
const source = readFileSync(new URL('./betterAnusbis.c', here), 'utf8');
const begin = source.indexOf('R"JS(', source.indexOf('static const char *CHALLENGE_JS = R"JS(')) + 5;
const end = source.indexOf(')JS"', begin);
if (begin < 5 || end < 0) throw new Error('CHALLENGE_JS not found in betterAnusbis.c');
const pageJs = source.slice(begin, end);

const sha256 = (...parts) => createHash('sha256').update(Buffer.concat(parts)).digest();
const cat = (...parts) => Buffer.concat(parts.map((p) => (Buffer.isBuffer(p) ? p : Buffer.from(p))));
const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v >>> 0); return b; };
const DOM_LEAF = Buffer.from('ANUBIS-TENSOR-VL');
const DOM_NODE = Buffer.from('ANUBIS-TENSOR-VN');
const DOM_CHAL = Buffer.from('ANUBIS-TENSOR-VC');
const MAGIC = 0x31565654;

const dim = (difficulty) => 64 * Math.max(2, Math.min(16, difficulty));

/* The spec, written out again on purpose. */
function expand(rdHex, n) {
    const cells = n * n;
    const seed = sha256(Buffer.from(rdHex, 'hex'));
    const a = new Uint8Array(cells);
    const b = new Int8Array(cells);
    let pos = 0;
    for (let counter = 0; pos < 2 * cells; counter++) {
        const block = sha256(seed, u32(counter));
        for (let i = 0; i < 32 && pos < 2 * cells; i++, pos++) {
            if (pos < cells) a[pos] = block[i] & 0x7f;
            else b[pos - cells] = (block[i] << 24) >> 24;
        }
    }
    return { a, b };
}
/* C[i][j] = sum_k A[i][k] * BT[j][k] */
function gemmv(a, bt, n) {
    const c = new Int32Array(n * n);
    for (let i = 0; i < n; i++) {
        const row = i * n;
        for (let j = 0; j < n; j++) {
            const brow = j * n;
            let sum = 0;
            for (let k = 0; k < n; k++) sum += a[row + k] * bt[brow + k];
            c[row + j] = sum;
        }
    }
    return c;
}
function blockBytes(c, b) {
    const out = Buffer.alloc(128);
    for (let j = 0; j < 32; j++) out.writeInt32LE(c[b * 32 + j], 4 * j);
    return out;
}
function refTree(rdHex, n) {
    const chal = sha256(DOM_CHAL, Buffer.from(rdHex, 'hex'));
    const c = gemmv(...Object.values(expand(rdHex, n)), n);
    const leaves = n * n / 32;
    let level = [];
    for (let b = 0; b < leaves; b++) level.push(sha256(DOM_LEAF, chal, u32(b), blockBytes(c, b)));
    const levels = [level];
    while (level.length > 1) {
        const next = [];
        for (let i = 0; i < level.length; i += 2) {
            next.push(sha256(DOM_NODE, level[i], level[i + 1] || level[i]));
        }
        levels.push(next);
        level = next;
    }
    return { c, leaves, levels, root: level[0].toString('hex') };
}
/* Parse and verify the witness blob the page would POST. */
function checkWitness(hex, tree, n, blocks, positions) {
    const buf = Buffer.from(hex, 'hex');
    if (buf.readUInt32LE(0) !== MAGIC) throw new Error('bad magic');
    if (buf.readUInt32LE(4) !== n) throw new Error('bad n');
    if (buf.readUInt32LE(8) !== blocks.length) throw new Error('bad count');
    if (buf.readUInt32LE(12) !== tree.leaves) throw new Error('bad leaves');
    const depth = tree.levels.length - 1;
    let at = 16;
    for (let i = 0; i < blocks.length; i++) {
        const block = buf.readUInt32LE(at);
        const position = buf.readUInt32LE(at + 4);
        const d = buf.readUInt32LE(at + 8);
        if (block !== blocks[i] || position !== positions[i]) throw new Error('sample order changed');
        if (d !== depth) throw new Error('bad depth at ' + i);
        at += 12;
    }
    const proofs = [];
    for (let i = 0; i < blocks.length; i++) {
        const proof = [];
        for (let l = 0; l < depth; l++) { proof.push(buf.subarray(at, at + 32)); at += 32; }
        proofs.push(proof);
    }
    for (let i = 0; i < blocks.length; i++) {
        const want = blockBytes(tree.c, blocks[i]);
        const got = buf.subarray(at, at + 128); at += 128;
        if (!got.equals(want)) throw new Error('block ' + blocks[i] + ' does not match C');
    }
    if (at !== buf.length) throw new Error('trailing bytes');
    return { buf, depth, proofs, at };
}
/* Verify proofs properly (leaf hash over the real challenge). */
function verifyProofs(tree, n, rdHex, blocks, proofs) {
    const chal = sha256(DOM_CHAL, Buffer.from(rdHex, 'hex'));
    const depth = tree.levels.length - 1;
    for (let i = 0; i < blocks.length; i++) {
        let idx = blocks[i];
        let node = sha256(DOM_LEAF, chal, u32(blocks[i]), blockBytes(tree.c, blocks[i]));
        for (let l = 0; l < depth; l++) {
            let sib = idx ^ 1;
            if (sib >= tree.levels[l].length) sib = idx;
            const other = proofs[i][l];
            if (!other.equals(tree.levels[l][sib])) throw new Error('bad sibling at level ' + l);
            node = (idx & 1) === 0 ? sha256(DOM_NODE, node, other) : sha256(DOM_NODE, other, node);
            idx >>>= 1;
        }
        if (node.toString('hex') !== tree.root) throw new Error('block ' + blocks[i] + ' does not reach the root');
    }
}

/* A WebNN stand-in. It computes the requested GEMM using the operand layout
 * the options ask for, records how the graph was built, and can be told to
 * reject bTranspose or to return zeros. */
function fakeWebNN({ transpose = true, garbage = false } = {}) {
    const calls = { bTranspose: false, transposedData: false, built: 0, float32: false };
    const context = {
        createTensor: (d) => {
            if (d.dataType === 'float32') calls.float32 = true;
            return { dataType: d.dataType, shape: d.shape };
        },
        writeTensor: (t, data) => { t.data = data; },
        dispatch: (graph, inputs, outputs) => {
            const n = inputs.A.shape[0];
            if (inputs.A.dataType !== 'float32' || inputs.B.dataType !== 'float32') {
                throw new Error('gemm operands must be float32 per the spec');
            }
            const c = garbage ? new Float32Array(n * n) : graph.compute(inputs.A.data, inputs.B.data, n, inputs.B.transposed);
            outputs.C.data = Float32Array.from(c);
        },
        readTensor: async (t) => t.data,
    };
    const makeBuilder = () => ({
        input: (name, desc) => ({ kind: 'input', name, desc }),
        gemm: (A, B, opts) => {
            if (opts && opts.bTranspose && !transpose) throw new Error('bTranspose unsupported');
            if (opts && opts.bTranspose) calls.bTranspose = true;
            return { kind: 'gemm', A, B, transposed: Boolean(opts && opts.bTranspose) };
        },
        build: async () => {
            calls.built++;
            return {
                compute: (a, b, n, t) => {
                    if (!t) calls.transposedData = true;
                    const c = new Int32Array(n * n);
                    for (let i = 0; i < n; i++) {
                        for (let j = 0; j < n; j++) {
                            let sum = 0;
                            for (let k = 0; k < n; k++) {
                                sum += t ? a[i * n + k] * b[j * n + k] : a[i * n + k] * b[k * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                    return c;
                },
            };
        },
    });
    return {
        calls,
        ml: { createContext: async () => context },
        MLGraphBuilder: function () { return makeBuilder(); },
    };
}

const wasmPath = new URL('./web/tensor_pow.wasm', here);
const wasmBytes = existsSync(wasmPath) ? readFileSync(wasmPath) : null;

async function solve({ difficulty, randomData, webnn = null, wasm = true, hash = '' }) {
    const payload = JSON.stringify({
        rules: { algorithm: 'tensor-v', difficulty },
        challenge: {
            issuedAt: '1', metadata: {}, id: 'test.1', method: 'tensor-v',
            randomData, policyRuleHash: '', difficulty, spent: false, extensions: null,
        },
    });
    const seen = { root: null, audit: null, done: false };
    // d=2 has 512 blocks; a sample of 3 keeps the independent proof check cheap
    const blocks = [0, 1, 511];
    const positions = [0, 31, 7];
    const sandbox = {
        console: { log: console.log, warn: () => {}, error: console.error },
        TextEncoder,
        URL,
        setTimeout,
        clearTimeout,
        performance: { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 },
        document: { getElementById: (id) => (id === 'anubis_challenge' ? { textContent: payload } : null) },
        location: {
            hash, search: '', pathname: '/', origin: 'http://gate.test',
            replace: () => { seen.done = true; },
        },
        fetch: async (url, opts = {}) => {
            const u = new URL(String(url), 'http://gate.test');
            if (u.pathname.includes('tensor_pow.wasm')) {
                if (!wasm) throw new Error('wasm disabled');
                if (!wasmBytes) throw new Error('web/tensor_pow.wasm is not built');
                return { ok: true, arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) };
            }
            if (u.pathname.endsWith('/pass-challenge')) {
                seen.root = u.searchParams.get('response');
                return { ok: true, status: 200, json: async () => ({ audit: { blocks, positions } }) };
            }
            if (u.pathname.endsWith('/audit-tensor')) {
                seen.audit = JSON.parse(opts.body);
                return { ok: true, status: 200, json: async () => ({ ok: true, redir: '/' }) };
            }
            throw new Error('unexpected fetch ' + url);
        },
    };
    if (webnn) {
        sandbox.navigator = { ml: webnn.ml };
        sandbox.MLGraphBuilder = webnn.MLGraphBuilder;
    }
    if (wasm) {
        sandbox.WebAssembly = WebAssembly;
    }
    vm.runInContext(pageJs, vm.createContext(sandbox), { filename: 'challenge.js' });
    const deadline = Date.now() + 120000;
    while (!seen.done && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
    if (!seen.done) throw new Error('tensor-v solver did not finish');
    if (!seen.root) throw new Error('no root was committed');
    if (!seen.audit) throw new Error('no audit was posted');
    return { ...seen, blocks, positions };
}

let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  PASS  ${name}`); }
    else { failed++; console.log(`  FAIL  ${name}${detail ? ': ' + detail : ''}`); }
}

const randomData = '00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f';

// 1. WebNN + wasm tree: bTranspose path, root and witness against the spec
{
    const difficulty = 2, n = dim(difficulty);
    const tree = refTree(randomData, n);
    const nn = fakeWebNN();
    const got = await solve({ difficulty, randomData, webnn: nn });
    check('WebNN used bTranspose', nn.calls.bTranspose === true);
    check('WebNN operands are float32', nn.calls.float32 === true);
    check('WebNN root matches the reference', got.root === tree.root, got.root);
    check('audit root matches the committed root', got.audit.root === tree.root);
    check('audit blocks match the server sample', typeof got.audit.audit === 'string');
    let parsed;
    try {
        parsed = checkWitness(got.audit.audit, tree, n, got.blocks, got.positions);
        verifyProofs(tree, n, randomData, got.blocks, parsed.proofs);
        check('witness blocks and proofs verify', true);
    } catch (e) {
        check('witness blocks and proofs verify', false, e.message);
    }
}

// 2. a backend without bTranspose: the page transposes by hand and retries
{
    const difficulty = 2, n = dim(difficulty);
    const tree = refTree(randomData, n);
    const nn = fakeWebNN({ transpose: false });
    const got = await solve({ difficulty, randomData, webnn: nn });
    check('transpose retry built the graph once', nn.calls.built === 1, String(nn.calls.built));
    check('transpose retry feeds a col-major B', nn.calls.transposedData === true);
    check('transpose retry root matches', got.root === tree.root, got.root);
}

// 3. a backend that returns zeros is rejected; wasm recomputes the real C
{
    const difficulty = 2, n = dim(difficulty);
    const tree = refTree(randomData, n);
    const nn = fakeWebNN({ garbage: true });
    const got = await solve({ difficulty, randomData, webnn: nn });
    check('garbage WebNN is rejected and wasm takes over', got.root === tree.root, got.root);
}

// 4. no WebNN, no wasm: the pure-JS GEMM and tree still answer the sample
{
    const difficulty = 2, n = dim(difficulty);
    const tree = refTree(randomData, n);
    const got = await solve({ difficulty, randomData, wasm: false });
    check('no-WebNN no-wasm root matches', got.root === tree.root, got.root);
    let parsed;
    try {
        parsed = checkWitness(got.audit.audit, tree, n, got.blocks, got.positions);
        verifyProofs(tree, n, randomData, got.blocks, parsed.proofs);
        check('pure-JS witness verifies', true);
    } catch (e) {
        check('pure-JS witness verifies', false, e.message);
    }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
