import numpy as np
import scipy.linalg
from typing import Tuple, List, Optional
from ..core.state import QuantumState
from ..core.operations import kraus_cirac_global_unitary

def schmidt_decomposition_state(state_vector: np.ndarray, dim_A: int, dim_B: int, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    matrix = state_vector.reshape((dim_A, dim_B))
    U, s, Vh = np.linalg.svd(matrix)
    non_zero = s > tol
    rank = int(np.sum(non_zero))
    return s, U, Vh, rank

def operator_schmidt_decomposition(matrix_AB: np.ndarray, dim_A: int = 2, dim_B: int = 2, tol: float = 1e-10) -> Tuple[np.ndarray, int]:
    # Reshuffle matrix M_{(i_A, i_B), (j_A, j_B)} into R_{(i_A, j_A), (i_B, j_B)}
    tensor = matrix_AB.reshape((dim_A, dim_B, dim_A, dim_B))
    reshuffled = np.transpose(tensor, (0, 2, 1, 3)).reshape((dim_A * dim_A, dim_B * dim_B))
    U, s, Vh = np.linalg.svd(reshuffled)
    s_norm = s / np.sqrt(dim_A * dim_B)
    rank = int(np.sum(s > tol))
    return s_norm, rank

def operator_schmidt_rank(matrix_AB: np.ndarray, dim_A: int = 2, dim_B: int = 2, tol: float = 1e-10) -> int:
    _, rank = operator_schmidt_decomposition(matrix_AB, dim_A, dim_B, tol)
    return rank

MAGIC_BASIS_Q = (1.0 / np.sqrt(2.0)) * np.array([
    [1, 0, 0, 1j],
    [0, 1j, 1, 0],
    [0, 1j, -1, 0],
    [1, 0, 0, -1j]
], dtype=complex)

def makhlin_invariants(U: np.ndarray) -> Tuple[float, float, float]:
    assert U.shape == (4, 4)
    det_U = np.linalg.det(U)
    U_b = MAGIC_BASIS_Q.conj().T @ U @ MAGIC_BASIS_Q
    M = U_b.T @ U_b
    tr_M = np.trace(M)
    tr2_M = np.trace(M @ M)
    
    # Scale by det(U) to make phase invariant
    g1 = float(np.real(tr_M**2 / (16.0 * det_U)))
    g2 = float(np.imag(tr_M**2 / (16.0 * det_U)))
    g3 = float(np.real((tr_M**2 - tr2_M) / (4.0 * det_U)))
    return g1, g2, g3

def kraus_cirac_decomposition_2qubit(U: np.ndarray, tol: float = 1e-4) -> Tuple[float, float, float]:
    assert U.shape == (4, 4)
    g1_target, g2_target, g3_target = makhlin_invariants(U)
    
    from scipy.optimize import minimize
    def loss(p):
        x, y, z = p
        # Makhlin invariant formula for U_global(x, y, z)
        cos2x = np.cos(2.0 * x)
        cos2y = np.cos(2.0 * y)
        cos2z = np.cos(2.0 * z)
        sin2x = np.sin(2.0 * x)
        sin2y = np.sin(2.0 * y)
        sin2z = np.sin(2.0 * z)
        
        g1_model = (cos2x * cos2y * cos2z)**2 - (sin2x * sin2y * sin2z)**2
        g2_model = 0.25 * np.sin(4.0 * x) * np.sin(4.0 * y) * np.sin(4.0 * z)
        g3_model = np.cos(4.0 * x) + np.cos(4.0 * y) + np.cos(4.0 * z)
        
        return (g1_model - g1_target)**2 + (g2_model - g2_target)**2 + (g3_model - g3_target)**2

    bounds = [(0.0, np.pi / 4.0), (0.0, np.pi / 4.0), (0.0, np.pi / 4.0)]
    starts = [
        [0.0, 0.0, 0.0],
        [np.pi / 4.0, 0.0, 0.0],
        [np.pi / 4.0, np.pi / 4.0, 0.0],
        [np.pi / 4.0, np.pi / 4.0, np.pi / 4.0],
        [np.pi / 8.0, np.pi / 8.0, np.pi / 8.0],
        [0.3, 0.2, 0.1]
    ]
    best_dist = float('inf')
    best_xyz = (0.0, 0.0, 0.0)
    for st in starts:
        res = minimize(loss, st, bounds=bounds, method='L-BFGS-B', tol=1e-12)
        if res.fun < best_dist:
            best_dist = res.fun
            sorted_c = sorted(res.x, reverse=True)
            best_xyz = (float(sorted_c[0]), float(sorted_c[1]), float(sorted_c[2]))
            
    x_c = 0.0 if best_xyz[0] < tol else best_xyz[0]
    y_c = 0.0 if best_xyz[1] < tol else best_xyz[1]
    z_c = 0.0 if best_xyz[2] < tol else best_xyz[2]
    return (x_c, y_c, z_c)

def kraus_cirac_number(U: np.ndarray, tol: float = 1e-4) -> int:
    x, y, z = kraus_cirac_decomposition_2qubit(U, tol=tol)
    kc = 0
    if abs(x) > tol:
        kc += 1
    if abs(y) > tol:
        kc += 1
    if abs(z) > tol:
        kc += 1
    return kc

