import numpy as np


def dense_to_CSR(A):
    m,n = A.shape
    ia = np.zeros(m+1, dtype=np.int)
    aa = np.arange(0,dtype=np.float)
    ja = np.arange(0,dtype=np.int)

    for i in range(m):
        k = 0
        for j in range(n):
            if A[i,j] != 0:
                aa = np.append(aa,A[i,j])
                ja = np.append(ja,j)
                k = k + 1
                print(k)
        ia[i+1] = ia[i] + k
    return aa, ia, ja

A = np.random.permutation(12).reshape(4,3)
print('A',A)
aa, ia, ja = dense_to_CSR(A)
print('aa',aa)
print('ja', ja)
print('ia',ia)


