import numpy as np


def show(mat):
    result = '%d %d' % (mat.shape[0], mat.shape[1])
    result += '\n'

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            result += str(mat[i,j]) + ' '

        result += '\n'

    print(result)


m = 128
k = 64
n = 256

sparsity = 0.01

A_mask = np.random.rand(m, k) < sparsity
B_mask = np.random.rand(k, n) < sparsity

A = np.random.randn(m, k) * A_mask
B = np.random.randn(k, n) * B_mask
C = A.dot(B)

show(A)
show(B)
show(C)
