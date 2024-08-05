import numpy as np
from lambeq.backend.quantum import Rx, Ry, Rz
from math import sin, cos

def calc_violation(state: np.array, contexts) -> float:
    expectations = [(np.conjugate(state) @ (contexts[ops] @ state)) for ops in list(contexts.keys())]
    return max([sum(expectations) - 2*exp for exp in expectations])

def state2dense(state, tol=1e-12):
    dense_mat = np.outer(state, np.conjugate(state))
    dense_mat.real[abs(dense_mat.real) < tol] = 0.0
    dense_mat.imag[abs(dense_mat.imag) < tol] = 0.0
    return dense_mat

def log_mat(mat): # Matrix logarithm via eigendecomposition
    evals, emat = np.linalg.eig(mat) # Get matrix V of eigenvectors of input matrix A
    emat_inv = np.linalg.inv(emat) # Get inverse of matrix V
    matp = emat @ mat @ emat_inv # Compute A' with a diagonal of eigenvalues tr(A') = evals
    tr = matp.diagonal() # Get the trace of the matrix
    np.fill_diagonal(matp, np.log2(tr, out=np.zeros_like(tr, dtype=np.complex128), where=(tr!=0))) # Element wise base 2 log of diagonal
    # Line above ignores log(0) error by replacing it with 0, which may lead to a wrong answer
    return emat_inv @ matp @ emat # Change basis back

def partial_trace(dense_mat):
    # Compute reduced density matrix for a bipartide quantum system
    dims_a = int(2**(np.floor(np.log2(dense_mat.shape[0])/2)))
    dims_b = int(2**(np.ceil(np.log2(dense_mat.shape[0])/2)))
    id_a = np.identity(dims_a)
    id_b = np.identity(dims_b)

    rho_a = np.zeros((dims_a, dims_a))
    rho_b = np.zeros((dims_b, dims_b))

    for base in id_b:
        bra = np.kron(id_a, base)
        ket = np.kron(id_a, base).T
        rho_a = rho_a + (bra @ dense_mat) @ ket
    for base in id_a:
        bra = np.kron(id_b, base)
        ket = np.kron(id_b, base).T
        rho_b = rho_b + (bra @ dense_mat) @ ket        
    return rho_a, rho_b



def calc_vne(dense_mat, direct=True):
    if direct:
        ent =  -np.trace(dense_mat @ log_mat(dense_mat))
    else:
        evals = np.linalg.eigvals(dense_mat)
        evals = evals[np.abs(evals) > 1e-12]
        ent = -np.sum(evals * np.log2(evals))
    ent = ent.round(12)
    return abs(ent)

def qrel_ent(mat1, mat2):
    return np.trace(mat1 @ (log_mat(mat1) - log_mat(mat2)))


def convert_dist(pr_dist, is_cyc=False):
    if is_cyc:
        # Converts a cyclic distribution to a standard one 
        new_dist = np.zeros_like(pr_dist)
        new_dist[0] = pr_dist[0]
        new_dist[1] = pr_dist[3][[0,2,1,3]]
        new_dist[2] = pr_dist[1][[0,2,1,3]]
        new_dist[3] = pr_dist[2]
    else:
        # Converts a standard distribution to a cyclic one
        new_dist = np.zeros_like(pr_dist)
        new_dist[0] = pr_dist[0]
        new_dist[1] = pr_dist[2][[0,2,1,3]]
        new_dist[2] = pr_dist[3]
        new_dist[3] = pr_dist[1][[0,2,1,3]]
    return new_dist

conv_theta = lambda theta: theta / (-2*np.pi)
conv_phi = lambda phi: phi / (2*np.pi)
normalise =  lambda arr: (arr - min(arr)) / (max(arr) - min(arr))

def convert_phase(theta, neg=False):
    return theta / (2 * np.pi * (1-neg*2))

def get_onb(theta=0, phi=0):
    ry = np.array([[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]])
    rz = np.array([[np.e**(-1j*phi/2),0], [0, np.e**(1j*phi/2)]])
    return ry @ rz

def gen_basis(theta=0, phi=0, tol=1e-9, gates=False):
    theta1 = convert_phase(theta, 1)
    phi1 = convert_phase(phi)

    if gates:
        return (Ry(theta1), Rz(phi1))

    theta2 = convert_phase(np.pi - theta, 1)
    phi2 = convert_phase(np.pi + phi)

    onb1 = Ry(theta1).array @ Rz(phi1).array
    onb2 = Ry(theta2).array @ Rz(phi2).array

    onb1.real[abs(onb1.real) < tol] = 0
    onb1.imag[abs(onb1.imag) < tol] = 0
    onb2.real[abs(onb2.real) < tol] = 0
    onb2.imag[abs(onb2.imag) < tol] = 0
    return (onb1, onb2)

def get_table(state, cnxt):
    state1 = cnxt['ab'] @ state
    state2 = cnxt['aB'] @ state
    state3 = cnxt['Ab'] @ state
    state4 = cnxt['AB'] @ state
    pr_dist = np.array([abs(state1)**2, 
                        abs(state2)**2,
                        abs(state3)**2, 
                        abs(state4)**2])
    return convert_dist(pr_dist)

def rand_state(n=2, ghz=False):
    state = np.zeros((n**2), dtype=np.complex128)
    state.real = np.random.uniform(0,1,(n**2))
    state.imag = np.random.uniform(0,1,(n**2))
    if ghz:
        state[1:n**2-1] = 0 + 0j
    state = np.sqrt(state / sum(abs(state)))
    return state 

def partial_transpose(mat, trb=True):
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

def trace_norm(mat):
    mat = mat.conj().T @ mat
    evals, emat = np.linalg.eig(mat) 
    emat_inv = np.linalg.inv(emat) 

    matp = emat_inv @ mat @ emat 
    sqrt_mat = emat @ np.sqrt(matp) @ emat_inv 

    return np.trace(sqrt_mat)

def calc_neg(dense_mat, direct=True):
    rho_tra = partial_transpose(dense_mat, False)
    if direct:
        evals = np.linalg.eigvalsh(rho_tra)
        evals[evals > 0] = 0
        return abs(sum(evals))
    else:
        return abs((trace_norm(rho_tra) - 1)/2)

def log_neg(dense_mat):
    return np.log2(abs(trace_norm(partial_transpose(dense_mat, False))))

def calc_eoe(dense_mat):
    rho_a, rho_b = partial_trace(dense_mat)
    return calc_vne(rho_a)
