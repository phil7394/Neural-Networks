import gzip
import struct

import numpy as np


def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        print(shape)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

if __name__ == '__main__':
    read_idx('train-images-idx3-ubyte.gz')