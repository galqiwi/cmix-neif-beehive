"""gate.py <container.c4> <cpp digest file> : Python reference decode, same digest, compare."""
import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'export4')); sys.path.insert(0, os.environ.get('TFT_TREE', '.'))
import weights_compress_v3 as wc3
T = wc3.read_tensor_file_v3(sys.argv[1])
cpp = dict(l.split() for l in open(sys.argv[2]))
def fnv(dt, a):
    h = 0xcbf29ce484222325
    data = bytes([dt]) + b''.join(int(s).to_bytes(4, 'little') for s in a.shape) + np.ascontiguousarray(a).tobytes()
    for b in data:
        h ^= b; h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return '%016x' % h
bad = [n for n, (dt, a) in T.items() if cpp.get(n) != fnv(dt, a)]
missing = set(cpp) ^ set(T)
print('tensors py %d cpp %d, mismatches %d, name differences %d' % (len(T), len(cpp), len(bad), len(missing)))
for n in bad[:10]: print('  MISMATCH', n)
sys.exit(1 if (bad or missing) else 0)
