import numpy as np
from lambeq.backend.quantum import Rx, Ry, Rz

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
    return ent.real

def qrel_ent(mat1, mat2):
    return np.trace(mat1 @ (log_mat(mat1) - log_mat(mat2)))

def std2cyc(pr_dist):
    new_dist = np.zeros_like(pr_dist)
    new_dist[0] = pr_dist[0]
    new_dist[1] = pr_dist[2][[0,2,1,3]]
    new_dist[2] = pr_dist[3]
    new_dist[3] = pr_dist[1][[0,2,1,3]]
    return new_dist

def cyc2std(pr_dist):
    new_dist = np.zeros_like(pr_dist)
    new_dist[0] = pr_dist[0]
    new_dist[1] = pr_dist[3][[0,2,1,3]]
    new_dist[2] = pr_dist[1][[0,2,1,3]]
    new_dist[3] = pr_dist[2]
    return new_dist

def convert_distribution(pr_dist, cyc=False):
    if cyc:
        new_dist = np.zeros_like(pr_dist)
        new_dist[0] = pr_dist[0]
        new_dist[1] = pr_dist[3][[0,2,1,3]]
        new_dist[2] = pr_dist[1][[0,2,1,3]]
        new_dist[3] = pr_dist[2]
        return new_dist
    else:
        new_dist = np.zeros_like(pr_dist)
        new_dist[0] = pr_dist[0]
        new_dist[1] = pr_dist[2][[0,2,1,3]]
        new_dist[2] = pr_dist[3]
        new_dist[3] = pr_dist[1][[0,2,1,3]]
        return new_dist

conv_theta = lambda theta: theta / (-2*np.pi)
conv_phi = lambda phi: phi / (2*np.pi)

def convert_phase(theta, neg=False):
    return theta / (2 * np.pi * (1-neg*2))

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


