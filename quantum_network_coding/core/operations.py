import numpy as np
import scipy.linalg
from typing import List, Tuple, Optional

PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
PHASE_S = np.array([[1, 0], [0, 1j]], dtype=complex)
PHASE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    res = matrices[0]
    for mat in matrices[1:]:
        res = np.kron(res, mat)
    return res

def kron_list(matrix_list: List[np.ndarray]) -> np.ndarray:
    return tensor_product(*matrix_list)

def u_x_gate(x: float) -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([
        [np.exp(1j * x), -1j * np.exp(-1j * x)],
        [np.exp(1j * x), 1j * np.exp(-1j * x)]
    ], dtype=complex)

def kraus_cirac_global_unitary(x: float, y: float, z: float) -> np.ndarray:
    XX = np.kron(PAULI_X, PAULI_X)
    YY = np.kron(PAULI_Y, PAULI_Y)
    ZZ = np.kron(PAULI_Z, PAULI_Z)
    H = x * XX + y * YY + z * ZZ
    return scipy.linalg.expm(1j * H)

def spectral_kraus_cirac_unitary(x: float, y: float, z: float) -> np.ndarray:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    psi0 = np.array([inv_sqrt2, 0, 0, inv_sqrt2], dtype=complex)
    psi1 = np.array([inv_sqrt2, 0, 0, -inv_sqrt2], dtype=complex)
    psi2 = np.array([0, inv_sqrt2, inv_sqrt2, 0], dtype=complex)
    psi3 = np.array([0, inv_sqrt2, -inv_sqrt2, 0], dtype=complex)
    lam0 = np.exp(1j * (x - y + z))
    lam1 = np.exp(1j * (-x + y + z))
    lam2 = np.exp(1j * (x + y - z))
    lam3 = np.exp(1j * (-x - y - z))
    U = (lam0 * np.outer(psi0, np.conj(psi0)) +
         lam1 * np.outer(psi1, np.conj(psi1)) +
         lam2 * np.outer(psi2, np.conj(psi2)) +
         lam3 * np.outer(psi3, np.conj(psi3)))
    return U

def weyl_chamber_unitary(x: float, y: float, z: float) -> np.ndarray:
    return kraus_cirac_global_unitary(x, y, z)

def two_qubit_controlled_unitary(u0: np.ndarray, u1: np.ndarray) -> np.ndarray:
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    return np.kron(P0, u0) + np.kron(P1, u1)

def three_qubit_fully_controlled_unitary(u00: np.ndarray, u01: np.ndarray, u10: np.ndarray, u11: np.ndarray) -> np.ndarray:
    P00 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    P01 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    P10 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=complex)
    P11 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex)
    return np.kron(P00, u00) + np.kron(P01, u01) + np.kron(P10, u10) + np.kron(P11, u11)

def random_unitary(dim: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    Z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    return Q @ np.diag(ph)
