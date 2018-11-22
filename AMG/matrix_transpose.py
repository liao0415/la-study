import numpy as np

def matrix_transpose(ia,ja,aa,A):
    m,n = A.shape

    ## Step 1:Initialize the array to 0
    jat = np.zeros(ja.shape[0], dtype=np.int)
    at = np.arange(aa.shape[0], dtype=np.float)
    iat = np.zeros(n+1,dtype=np.int)

    ## Step 2:Accounting non_zero element,placed in the iat
    for i in range(ja.shape[0]):
        k = ja[i]
        iat[k+1] = iat[k+1] + 1

    ## Step 3: add 
    
    iat = np.add.accumulate(iat)
    iat = np.r_[1,iat]

    ## Step 4:get jat and at
    for i in range(m):
        for j in range(ia[i],ia[i+1]):
            k = ja[j]
            jat[iat[k]] = i #The column number of the transposed matrix
            at[iat[k]] = aa[j]
            iat[k] = iat[k] + 1

    ## Step 5: Restore array
    for i in range(n,0,-1):
        iat[i] = iat[i-1]
    iat[0] = 1

    return iat,jat,at








