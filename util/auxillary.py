import numpy as np

normalise =  lambda arr: (arr - min(arr)) / (max(arr) - min(arr))

def truncate(arr: np.ndarray, tol:float=1e-12): 
    # Given an array and tolerance set all values within the tolerance range to 0
    arr.real[abs(arr.real) < tol] = 0.0
    arr.imag[abs(arr.imag) < tol] = 0.0
    return arr

def log_mat(mat: np.ndarray) -> np.ndarray: # Matrix logarithm via eigendecomposition
    evals, emat = np.linalg.eig(mat) # Get matrix V of eigenvectors of input matrix A
    emat_inv = np.linalg.inv(emat) # Get inverse of matrix V
    matp = emat @ mat @ emat_inv # Compute A' with a diagonal of eigenvalues tr(A') = evals
    tr = matp.diagonal() # Get the trace of the matrix
    np.fill_diagonal(matp, np.log2(tr, out=np.zeros_like(tr, dtype=np.complex128), where=(tr!=0))) # Element wise base 2 log of diagonal
    # Line above ignores log(0) error by replacing it with 0, which may lead to a wrong answer
    return emat_inv @ matp @ emat # Change basis back 

def prt_trace(mat: np.ndarray) -> (np.ndarray, np.ndarray):
    # Compute reduced density matrix for a bipartide quantum system
    dims_a = int(2**(np.floor(np.log2(mat.shape[0])/2)))
    dims_b = int(2**(np.ceil(np.log2(mat.shape[0])/2)))
    id_a = np.identity(dims_a)
    id_b = np.identity(dims_b)

    rho_a = np.zeros((dims_a, dims_a))
    rho_b = np.zeros((dims_b, dims_b))

    for base in id_b:
        bra = np.kron(id_a, base)
        ket = np.kron(id_a, base).T
        rho_a = rho_a + (bra @ mat) @ ket
    for base in id_a:
        bra = np.kron(id_b, base)
        ket = np.kron(id_b, base).T
        rho_b = rho_b + (bra @ mat) @ ket        
    return rho_a, rho_b

def prt_transpose(mat: np.ndarray, trb=True) -> np.ndarray:
    res = np.zeros((4,4))
    comp_basis = np.identity(2)
    for i in comp_basis:
        for j in comp_basis:
            for k in comp_basis:
                for l in comp_basis:
                    mat_a = np.outer(i,j)
                    mat_b = np.outer(k,l)
                    p = np.trace(mat @ np.kron(mat_a, mat_b))
                    res = res + p * np.kron(mat_a, mat_b.T)
    if trb:
        return res
    else:
        return res.T 

def trace_norm(mat: np.ndarray) -> np.ndarray:
    mat = mat.conj().T @ mat
    evals, emat = np.linalg.eig(mat) 
    emat_inv = np.linalg.inv(emat) 

    matp = emat_inv @ mat @ emat 
    sqrt_mat = emat @ np.sqrt(matp) @ emat_inv 

    return np.trace(sqrt_mat)

def lr_kron(array): 
    if len(array) == 1: 
        return array[0]
    return np.kron(array[0], lr_kron(array[1:]))

def sep_state(psi: [float]):
    psi_len = len(psi)
    psi_sep = []
    for n in range(psi_len):
        base = np.zeros(psi_len) 
        base[n] = psi[n]
        psi_sep.append(base)
    return psi_sep

def wedge_prod(u: [float], v: [float]): 
    dims = len(u)
    u_decomp = sep_state(u) 
    v_decomp = sep_state(v)  
    res = 0
    
    for x in range(dims): 
        for y in range(x+1,dims):
            res += abs(u_decomp[x][x]*v_decomp[y][y] - u_decomp[y][y]*v_decomp[x][x])**2
    return res


