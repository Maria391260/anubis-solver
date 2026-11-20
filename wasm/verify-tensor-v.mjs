// verify-tensor-v.mjs — the wasm `tensor-v` client against the C core.
//
// The C test (tests/test_tensor_v.py) runs this, then verifies the printed
// root/witness with anubis_tensor_v_verify(). It pins the parts that must agree
// byte for byte between the browser kernel and the server: the challenge
// binding, the leaf and node hashing, the tree shape, and the witness layout.
//
//   node wasm/verify-tensor-v.mjs
//
// Prints {"root":"...","witness":"..."} on success.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const challenge = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
const difficulty = process.argv[2] ? Number(process.argv[2]) : 2;
const blocks = [0, 1, 2, 3];
const positions = [0, 5, 31, 7];

const { instance } = await WebAssembly.instantiate(
  readFileSync(join(here, "tensor_pow.wasm")), {});
const ex = instance.exports;

const raw = Uint8Array.from(Buffer.from(challenge, "hex"));
new Uint8Array(ex.memory.buffer).set(raw, ex.data_ptr());
ex.set_data_length(raw.length);

if ((ex.tv_solve(difficulty) >>> 0) !== 0) {
  throw new Error("tv_solve rejected the challenge");
}
const root = Buffer.from(
  new Uint8Array(ex.memory.buffer, ex.tv_root_ptr(), 32)).toString("hex");

const inp = new Uint8Array(ex.memory.buffer, ex.tv_in_ptr(), blocks.length * 8);
const dv = new DataView(inp.buffer, inp.byteOffset, inp.byteLength);
blocks.forEach((b, i) => {
  dv.setUint32(i * 8, b >>> 0, true);
  dv.setUint32(i * 8 + 4, positions[i] >>> 0, true);
});
const len = ex.tv_make_witness(difficulty, blocks.length) >>> 0;
if (!len) throw new Error("tv_make_witness failed");
const witness = Buffer.from(
  new Uint8Array(ex.memory.buffer, ex.tv_wit_ptr(), len)).toString("hex");

console.log(JSON.stringify({ root, witness }));
