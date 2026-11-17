#!/usr/bin/env python3
"""Tiny PTX interpreter covering the instruction subset used by the kernels in
this directory (sha256_pow.ptx, sha256_opt.ptx, hashx_interp.ptx), so they can
be checked against the CPU references without a GPU. Verification only -- this
is not a general PTX emulator and must not be used to run untrusted code."""
import re
import struct

LOCAL_BASE = 0x100000
CONST_BASE = 0x200000
GLOBAL_BASE = 0x300000

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]


def sha256_compress(state, block):
    Kk = K
    w = list(struct.unpack('>16I', block)) + [0] * 48
    for i in range(16, 64):
        x, y = w[i - 15], w[i - 2]
        s0 = ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3)
        s1 = ((y >> 17) | (y << 15)) ^ ((y >> 19) | (y << 13)) ^ (y >> 10)
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xffffffff
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7))
        ch = (e & f) ^ ((~e) & g)
        t1 = (h + S1 + ch + Kk[i] + w[i]) & 0xffffffff
        S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10))
        mj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + mj) & 0xffffffff
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xffffffff, c, b, a, (t1 + t2) & 0xffffffff
    return [(s + v) & 0xffffffff for s, v in zip(state, [a, b, c, d, e, f, g, h])]


IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def midstate(prefix):
    assert len(prefix) % 64 == 0
    st = list(IV)
    for off in range(0, len(prefix), 64):
        st = sha256_compress(st, prefix[off:off + 64])
    return st


def sha_from_midstate(st, tail):
    assert len(tail) == 64
    return b''.join(struct.pack('>I', x) for x in sha256_compress(st, tail))


class Kernel:
    def __init__(self, path):
        txt = open(path).read()
        txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.S)
        txt = re.sub(r'//[^\n]*', '', txt)
        self.const = bytearray()
        m = re.search(r'c_k\[64\]\s*=\s*\{([^}]*)\}', txt, re.S)
        if m:
            vals = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))]
            for v in vals:
                self.const += struct.pack('<I', v)
        i = txt.index('.entry')
        b0 = txt.index('{', i)
        depth, j = 0, b0
        while True:
            if txt[j] == '{':
                depth += 1
            elif txt[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = txt[b0 + 1:j]
        body = re.sub(r'([A-Za-z_][A-Za-z0-9_]*):', r'\n\1:\n', body)
        self.local_size = 256
        self.regs = {}
        self.preds = set()
        self.instrs = []
        self.labels = {}
        for raw in re.split(r'[;\n]+', body):
            s = raw.strip()
            if not s:
                continue
            if s.endswith(':'):
                self.labels[s[:-1]] = len(self.instrs)
                continue
            if s.startswith('.reg'):
                toks = s.replace(',', ' ').split()
                typ = toks[1]
                names = [t for t in toks[2:] if t.startswith('%')]
                if typ == '.pred':
                    self.preds.update(names)
                elif typ == '.u32':
                    self.regs.update({n: (32, 0) for n in names})
                elif typ == '.u64':
                    self.regs.update({n: (64, 0) for n in names})
                continue
            if s.startswith('.local'):
                m = re.search(r'\[(\d+)\]', s)
                if m:
                    self.local_size = int(m.group(1))
                continue
            self.instrs.append(s)
        # reg types
        self.rtype = dict(self.regs)

    # ---- operand helpers ----
    def val(self, op):
        op = op.strip()
        if op.startswith('%'):
            return self.r[op]
        if op.startswith('0x') or op.startswith('-'):
            return int(op, 0)
        if op[0].isdigit():
            return int(op, 0)
        if op == 'lmem':
            return LOCAL_BASE
        if op == 'c_k':
            return CONST_BASE
        raise ValueError('bad operand %r' % op)

    def addr(self, op):
        op = op.strip()
        assert op.startswith('[') and op.endswith(']'), op
        inner = op[1:-1].strip()
        if inner in ('lmem', 'c_k'):
            return self.val(inner)
        m = re.match(r'(%\w+)\s*([+-])\s*(0x[0-9a-fA-F]+|\d+)', inner)
        if m:
            base = self.r[m.group(1)]
            imm = int(m.group(3), 0)
            return base + imm if m.group(2) == '+' else base - imm
        if inner.startswith('%'):
            return self.r[inner]
        return self.val(inner)

    def local(self):
        return self._local

    def run_thread(self, params, global_mem, ntid, ctaid, tid, stop_pc=None):
        self.r = {n: 0 for n in self.regs}
        self.p = {n: False for n in self.preds}
        self.r['%ntid.x'] = ntid
        self.r['%ctaid.x'] = ctaid
        self.r['%tid.x'] = tid
        self._local = bytearray(self.local_size)
        self.global_mem = global_mem
        self.params = params
        pc = 0
        n = len(self.instrs)
        steps = 0
        while pc < n:
            steps += 1
            if steps > 200000:
                raise RuntimeError('runaway')
            if stop_pc is not None and pc == stop_pc:
                return steps
            ins = self.instrs[pc]
            pc += 1
            skip = False
            if ins.startswith('@'):
                g, ins = ins.split(None, 1)
                if g.startswith('@!'):
                    skip = self.p[g[2:]]
                else:
                    skip = not self.p[g[1:]]
            if skip:
                continue
            op, _, rest = ins.partition(' ')
            parts = [x.strip() for x in rest.split(',')] if rest else []
            d = self.do(op, parts)
            if d is not None:
                pc = d
        return steps

    def do(self, op, a):
        r = self.r
        if op == 'ret':
            return len(self.instrs)
        if op == 'bra':
            return self.labels[a[0].strip()]
        if op == 'ld.param.u32':
            r[a[0]] = self.params[a[1].strip()[1:-1]] & 0xffffffff
        elif op == 'ld.param.u64':
            r[a[0]] = self.params[a[1].strip()[1:-1]]
        elif op == 'ld.local.u8':
            off = self.addr(a[1]) - LOCAL_BASE
            r[a[0]] = self._local[off]
        elif op == 'ld.local.u32':
            off = self.addr(a[1]) - LOCAL_BASE
            r[a[0]] = int.from_bytes(self._local[off:off + 4], 'little')
        elif op == 'ld.const.u32':
            off = self.addr(a[1]) - CONST_BASE
            r[a[0]] = int.from_bytes(self.const[off:off + 4], 'little')
        elif op == 'st.local.u8':
            off = self.addr(a[0]) - LOCAL_BASE
            self._local[off] = self.val(a[1]) & 0xff
        elif op == 'st.local.u32':
            off = self.addr(a[0]) - LOCAL_BASE
            self._local[off:off + 4] = struct.pack('<I', self.val(a[1]) & 0xffffffff)
        elif op == 'ld.local.u64':
            off = self.addr(a[1]) - LOCAL_BASE
            r[a[0]] = int.from_bytes(self._local[off:off + 8], 'little')
        elif op == 'st.local.u64':
            off = self.addr(a[0]) - LOCAL_BASE
            self._local[off:off + 8] = struct.pack('<Q', self.val(a[1]) & 0xffffffffffffffff)
        elif op == 'ld.global.u32':
            off = self.addr(a[1]) - GLOBAL_BASE
            r[a[0]] = int.from_bytes(self.global_mem[off:off + 4], 'little')
        elif op == 'st.global.u32':
            off = self.addr(a[0]) - GLOBAL_BASE
            self.global_mem[off:off + 4] = struct.pack('<I', self.val(a[1]) & 0xffffffff)
        elif op == 'atom.global.add.u32':
            off = self.addr(a[1]) - GLOBAL_BASE
            old = int.from_bytes(self.global_mem[off:off + 4], 'little')
            self.global_mem[off:off + 4] = struct.pack('<I', (old + self.val(a[2])) & 0xffffffff)
            r[a[0]] = old
        elif op == 'mov.u32':
            r[a[0]] = self.val(a[1]) & 0xffffffff
        elif op == 'mov.u64':
            r[a[0]] = self.val(a[1]) & 0xffffffffffffffff
        elif op == 'add.u32':
            r[a[0]] = (self.val(a[1]) + self.val(a[2])) & 0xffffffff
        elif op == 'add.u64':
            r[a[0]] = (self.val(a[1]) + self.val(a[2])) & 0xffffffffffffffff
        elif op == 'sub.u32':
            r[a[0]] = (self.val(a[1]) - self.val(a[2])) & 0xffffffff
        elif op == 'sub.u64':
            r[a[0]] = (self.val(a[1]) - self.val(a[2])) & 0xffffffffffffffff
        elif op == 'mul.lo.u64':
            r[a[0]] = (self.val(a[1]) * self.val(a[2])) & 0xffffffffffffffff
        elif op == 'mul.hi.u64':
            r[a[0]] = ((self.val(a[1]) & 0xffffffffffffffff) *
                       (self.val(a[2]) & 0xffffffffffffffff)) >> 64
        elif op == 'mul.hi.s64':
            x, y = self.val(a[1]), self.val(a[2])
            x = x - (1 << 64) if x >> 63 else x
            y = y - (1 << 64) if y >> 63 else y
            r[a[0]] = ((x * y) >> 64) & 0xffffffffffffffff
        elif op == 'mad.lo.u32':
            r[a[0]] = (self.val(a[1]) * self.val(a[2]) + self.val(a[3])) & 0xffffffff
        elif op == 'mul.lo.u32':
            r[a[0]] = (self.val(a[1]) * self.val(a[2])) & 0xffffffff
        elif op == 'mul.wide.u32':
            r[a[0]] = (self.val(a[1]) * self.val(a[2])) & 0xffffffffffffffff
        elif op == 'div.u32':
            r[a[0]] = (self.val(a[1]) // self.val(a[2])) & 0xffffffff
        elif op == 'rem.u32':
            r[a[0]] = (self.val(a[1]) % self.val(a[2])) & 0xffffffff
        elif op == 'shl.b32':
            r[a[0]] = (self.val(a[1]) << (self.val(a[2]) & 31)) & 0xffffffff
        elif op in ('shr.b32', 'shr.u32'):
            r[a[0]] = (self.val(a[1]) & 0xffffffff) >> (self.val(a[2]) & 31)
        elif op == 'shl.b64':
            r[a[0]] = (self.val(a[1]) << (self.val(a[2]) & 63)) & 0xffffffffffffffff
        elif op == 'shr.u64':
            r[a[0]] = (self.val(a[1]) & 0xffffffffffffffff) >> (self.val(a[2]) & 63)
        elif op == 'shf.r.wrap.b32':
            x, y, c = self.val(a[1]), self.val(a[2]), self.val(a[3]) & 31
            v = ((y & 0xffffffff) << 32) | (x & 0xffffffff)
            r[a[0]] = (v >> c) & 0xffffffff
        elif op == 'and.b32':
            r[a[0]] = (self.val(a[1]) & self.val(a[2])) & 0xffffffff
        elif op == 'or.b32':
            r[a[0]] = (self.val(a[1]) | self.val(a[2])) & 0xffffffff
        elif op == 'xor.b32':
            r[a[0]] = (self.val(a[1]) ^ self.val(a[2])) & 0xffffffff
        elif op == 'and.b64':
            r[a[0]] = (self.val(a[1]) & self.val(a[2])) & 0xffffffffffffffff
        elif op == 'or.b64':
            r[a[0]] = (self.val(a[1]) | self.val(a[2])) & 0xffffffffffffffff
        elif op == 'xor.b64':
            r[a[0]] = (self.val(a[1]) ^ self.val(a[2])) & 0xffffffffffffffff
        elif op == 'not.b32':
            r[a[0]] = (~self.val(a[1])) & 0xffffffff
        elif op == 'cvt.u64.u32':
            r[a[0]] = self.val(a[1]) & 0xffffffff
        elif op == 'cvt.s64.s32':
            x = self.val(a[1]) & 0xffffffff
            r[a[0]] = (x - (1 << 32) if x >> 31 else x) & 0xffffffffffffffff
        elif op == 'cvt.u32.u64':
            r[a[0]] = self.val(a[1]) & 0xffffffff
        elif op == 'cvta.const.u64':
            r[a[0]] = CONST_BASE
        elif op == 'and.pred':
            self.p[a[0]] = self.p[a[1]] and self.p[a[2]]
        elif op.startswith('setp.'):
            m = op[5:]
            x, y = self.val(a[1]), self.val(a[2])
            if m == 'ge.u32': self.p[a[0]] = (x & 0xffffffff) >= (y & 0xffffffff)
            elif m == 'le.u32': self.p[a[0]] = (x & 0xffffffff) <= (y & 0xffffffff)
            elif m == 'lt.u32': self.p[a[0]] = (x & 0xffffffff) < (y & 0xffffffff)
            elif m == 'eq.u32': self.p[a[0]] = (x & 0xffffffff) == (y & 0xffffffff)
            elif m == 'ne.u32': self.p[a[0]] = (x & 0xffffffff) != (y & 0xffffffff)
            else: raise ValueError(op)
        else:
            raise ValueError('unhandled: ' + op)
        return None


def launch(k, params, count, blockdim=256, ilp=1, base=0):
    blocks = (count + blockdim * ilp - 1) // (blockdim * ilp)
    gm = bytearray(304)
    for b in range(blocks):
        for tx in range(blockdim):
            k.run_thread(params, gm, blockdim, b, tx)
    return gm


def results(gm):
    cnt = int.from_bytes(gm[0:4], 'little')
    nonces = [int.from_bytes(gm[16 + 4 * i:20 + 4 * i], 'little') for i in range(8)]
    hashes = []
    for i in range(8):
        off = 48 + 32 * i
        words = [int.from_bytes(gm[off + 4 * j:off + 4 * j + 4], 'little')
                 for j in range(8)]
        hashes.append(b''.join(struct.pack('>I', x) for x in words))
    return cnt, nonces, hashes


def tail_mode1(nonce, prefix_len, use_le):
    b = bytearray(64)
    if use_le:
        b[0:4] = struct.pack('<I', nonce)
    else:
        b[0:4] = struct.pack('>I', nonce)
    b[4] = 0x80
    bits = (prefix_len + 4) * 8
    b[56:64] = struct.pack('>Q', bits)
    return bytes(b)


def tail_mode0(nonce, prefix_len):
    s = str(nonce).encode()
    b = bytearray(64)
    b[0:len(s)] = s
    b[len(s)] = 0x80
    bits = (prefix_len + len(s)) * 8
    b[56:64] = struct.pack('>Q', bits)
    return bytes(b)


def params_for(st, prefix_len, mode, use_le, difficulty, check_bits, base, count):
    p = {}
    for i in range(8):
        p['mid%d' % i] = st[i]
    p['prefix_len'] = prefix_len
    p['mode'] = mode
    p['use_le'] = use_le
    p['difficulty'] = difficulty
    p['check_bits'] = check_bits
    p['base'] = base
    p['count'] = count
    p['results_ptr'] = GLOBAL_BASE
    return p
