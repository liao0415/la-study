import numpy as np
from scipy.sparse import csr_matrix

ia = np.array([0,2,5,6,8])
ja = np.array([0,2,0,1,2,3,1,4])
a = np.array([3,2,1,2,8,4,1,9])
A = csr_matrix((a,ja,ia),shape=(4,5)).toarray()

ib = np.array([0,2,4,7,9,10])
jb = np.array([0,2,0,1,1,2,3,0,3,2])
b = np.array([1,2,3,4,5,2,1,2,6,4])
B = csr_matrix((b,gb,ib),shape=(5,4))

