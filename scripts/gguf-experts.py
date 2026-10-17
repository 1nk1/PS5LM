"""Bytes per token for a GGUF read from its header: resident (non-expert) vs routed experts.
usage: python3 scripts/gguf-experts.py FILE  (a GGUF, or its first 24 MiB: curl -r 0-25165823)"""
import struct, sys

# ggml type -> (block size, bytes per block)
T = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),
     10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),
     16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),22:(256,82),
     23:(256,136),29:(256,56),30:(1,2),39:(32,17)}

d = open(sys.argv[1], 'rb').read(64 << 20)  # the header only, never the weights
o = 0
def rd(fmt):
    global o; v = struct.unpack_from('<' + fmt, d, o); o += struct.calcsize('<' + fmt); return v[0]
def rs():
    global o; n = rd('Q'); s = d[o:o+n].decode('utf-8', 'replace'); o += n; return s
SZ = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def val(t):
    if t == 8: return rs()
    if t == 9:
        et = rd('I'); n = rd('Q'); return [val(et) for _ in range(n)]
    return rd(SZ[t])
assert d[:4] == b'GGUF'; o = 4; rd('I'); nt = rd('Q'); nkv = rd('Q')
kv = {}
for _ in range(nkv):
    k = rs(); t = rd('I'); v = val(t); kv[k] = v if not isinstance(v, list) or len(v) < 64 else f'[{len(v)}]'
arch = kv['general.architecture']
tot = exp = 0; shexp = 0
for _ in range(nt):
    name = rs(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; ty = rd('I'); rd('Q')
    n = 1
    for x in dims: n *= x
    bs, bb = T[ty]; b = n // bs * bb
    tot += b
    if '_exps' in name: exp += b
for k in ('expert_count', 'expert_used_count', 'block_count', 'embedding_length', 'expert_feed_forward_length',
          'attention.head_count_kv', 'attention.key_length', 'full_attention_interval', 'nextn_predict_layers'):
    print(k, kv.get(f'{arch}.{k}'))
ne = kv.get(f'{arch}.expert_count') or 0; nu = kv.get(f'{arch}.expert_used_count') or 0
G = 2**30
print(f'arch {arch}  tensors {tot/G:.2f} GiB  routed experts {exp/G:.2f} GiB  resident rest {(tot-exp)/G:.2f} GiB')
if ne:
    print(f'one expert (all layers) {exp/ne/G*1024:.1f} MiB  active experts/token {exp*nu/ne/G:.3f} GiB  '
          f'bytes/token total {(tot-exp+exp*nu/ne)/G:.2f} GiB')
