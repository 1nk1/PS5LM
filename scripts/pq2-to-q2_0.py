#!/usr/bin/env python3
"""Repack a Prism PQ2_0 GGUF (Ternary Bonsai 2) into stock llama.cpp's Q2_0.

PQ2_0 (GGML type 142, PrismML's fork) and Q2_0 (type 42, upstream) share the
2-bit codec, value = (q - 1) * d with one fp16 scale per group; PQ2_0 groups
128 weights, Q2_0 groups 64. Each PQ2_0 block becomes two Q2_0 blocks with the
same scale, so the repack is lossless. Metadata (prism.hadamard.*) is copied
as is; upstream llama.cpp applies it since PR #29600 (v0.7.0).

usage: scripts/pq2-to-q2_0.py IN.gguf OUT.gguf [NAME]   (NAME replaces general.name)
       scripts/pq2-to-q2_0.py --self-test
"""
import struct, sys

import numpy as np

PQ2_0, Q2_0 = 142, 42
LLAMA_FTYPE_MOSTLY_Q2_0 = 41
# bytes per block and weights per block of the types a Bonsai file holds
SIZES = {0: (1, 4), 1: (1, 2), 30: (1, 2), Q2_0: (64, 18), PQ2_0: (128, 34)}
VAL = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def repack_blocks(src: bytes) -> bytes:
    """PQ2_0 blocks (2-byte scale, 32 bytes of codes) -> twice as many Q2_0 blocks."""
    b = np.frombuffer(src, np.uint8).reshape(-1, 34)
    out = np.empty((len(b), 2, 18), np.uint8)
    out[:, :, :2] = b[:, None, :2]
    out[:, 0, 2:] = b[:, 2:18]
    out[:, 1, 2:] = b[:, 18:34]
    return out.tobytes()


class Header:
    def __init__(self, f):
        self.f, self.o = f, 0
        assert self.read(4) == b'GGUF', 'not a GGUF file'
        self.version, n_tensors, n_kv = self.u('I'), self.u('Q'), self.u('Q')
        assert self.version == 3, f'GGUF version {self.version}'
        self.kv_start, self.alignment, self.ftype_at, self.name_at = self.o, 32, None, None
        for _ in range(n_kv):
            key, t = self.str(), self.u('I')
            if key == 'general.file_type':
                self.ftype_at = (self.o, t)
            at = self.o
            v = self.value(t)
            if key == 'general.name':
                self.name_at = (at, self.o)
            if key == 'general.alignment':
                self.alignment = v
        self.kv_end = self.o
        self.tensors = []
        for _ in range(n_tensors):
            name = self.str()
            dims = [self.u('Q') for _ in range(self.u('I'))]
            self.tensors.append([name, dims, self.u('I'), self.u('Q')])
        self.data_start = -(-self.o // self.alignment) * self.alignment

    def read(self, n):
        b = self.f.read(n)
        assert len(b) == n, 'truncated header'
        self.o += n
        return b

    def u(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def str(self):
        return self.read(self.u('Q')).decode()

    def value(self, t):
        if t == 8:
            return self.str()
        if t == 9:
            et, n = self.u('I'), self.u('Q')
            return [self.value(et) for _ in range(n)]
        return self.u(VAL[t])


def nbytes(dims, t):
    if t not in SIZES:
        raise SystemExit(f'pq2-to-q2_0: tensor type {t} is not handled')
    n = 1
    for x in dims:
        n *= x
    per, size = SIZES[t]
    assert n % per == 0
    return n // per * size


def convert(src, dst, name=None):
    with open(src, 'rb') as f:
        h = Header(f)
        f.seek(0)
        kv = bytearray(f.read(h.kv_end))
        if h.ftype_at and h.ftype_at[1] == 4:   # uint32
            struct.pack_into('<I', kv, h.ftype_at[0], LLAMA_FTYPE_MOSTLY_Q2_0)
        if name and h.name_at:   # spliced after the in-place file type patch, so its offset still held
            kv[h.name_at[0]:h.name_at[1]] = struct.pack('<Q', len(name.encode())) + name.encode()
        ti, plan, off = bytearray(), [], 0
        for tname, dims, t, src_off in h.tensors:
            out_t = Q2_0 if t == PQ2_0 else t
            ti += struct.pack("<Q", len(tname)) + tname.encode() + struct.pack("<I", len(dims))
            ti += struct.pack(f'<{len(dims)}Q', *dims) + struct.pack('<IQ', out_t, off)
            plan.append((h.data_start + src_off, nbytes(dims, t), t == PQ2_0, off))
            off = -(-(off + nbytes(dims, out_t)) // h.alignment) * h.alignment
        with open(dst, 'wb') as g:
            g.write(kv + ti)
            g.write(b'\0' * (-g.tell() % h.alignment))
            base = g.tell()
            chunk = 34 << 20   # whole PQ2_0 blocks, ~34 MiB at a time
            for i, (at, size, repack, out_off) in enumerate(plan):
                g.write(b'\0' * (base + out_off - g.tell()))
                f.seek(at)
                left = size
                while left:
                    b = f.read(min(chunk, left))
                    left -= len(b)
                    g.write(repack_blocks(b) if repack else b)
                if i % 40 == 0:
                    print(f'pq2-to-q2_0: {100 * i // len(plan)}%', flush=True)
            g.write(b'\0' * (base + off - g.tell()))
    print(f'pq2-to-q2_0: {sum(p[2] for p in plan)} tensors repacked -> {dst}')


def dequant(blocks: bytes, per: int, size: int):
    out = []
    for i in range(0, len(blocks), size):
        d = float(np.frombuffer(blocks[i:i + 2], np.float16)[0])
        qs = blocks[i + 2:i + size]
        out += [((qs[j // 4] >> (j % 4 * 2) & 3) - 1) * d for j in range(per)]
    return out


def self_test():
    rng = np.random.default_rng(0)
    pq = b''.join(np.float16(rng.uniform(0.01, 2)).tobytes() + rng.integers(0, 256, 32, np.uint8).tobytes()
                  for _ in range(3))
    q2 = repack_blocks(pq)
    assert len(q2) == 3 * 2 * 18
    assert dequant(q2, 64, 18) == dequant(pq, 128, 34)
    print('pq2-to-q2_0: self-test passed')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    elif len(sys.argv) in (3, 4):
        convert(*sys.argv[1:])
    else:
        sys.exit(__doc__)
