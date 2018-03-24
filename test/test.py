import numpy as np


d, m = map(int, input().split())
A = np.float32([list(map(float, input().split())) for _ in range(d)])

d, n = map(int, input().split())
B = np.float32([list(map(float, input().split())) for _ in range(d)])

print(B)

m, n = map(int, input().split())
C_mine = np.float32([list(map(float, input().split())) for _ in range(m)])

C_true = -2.0 * A.T.dot(B)
# C_true = A.T.dot(B)

A_norms = [np.sum(A[:,i] * A[:,i]) for i in range(m)]
B_norms = [np.sum(B[:,i] * B[:,i]) for i in range(n)]

print(m, n)
print(len(A_norms))
print(len(B_norms))
print(C_true.shape)

for i in range(m):
    for j in range(n):
        C_true[i,j] += A_norms[i] + B_norms[j]


for i in range(m):
    for j in range(n):
        print(C_mine[i,j], C_true[i,j])
