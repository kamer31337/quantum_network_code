import numpy as np
from typing import List, Tuple, Union, Optional

class QuantumState:
    def __init__(self, vector: Union[np.ndarray, List[complex]], normalize: bool = True):
        self.vector = np.array(vector, dtype=complex).flatten()
        if normalize:
            norm = np.linalg.norm(self.vector)
            if norm > 1e-15:
                self.vector = self.vector / norm
        self.dim = len(self.vector)
        self.num_qubits = int(np.round(np.log2(self.dim)))
        assert 2**self.num_qubits == self.dim, f"Dimension {self.dim} is not a power of 2"

    def tensor(self, other: 'QuantumState') -> 'QuantumState':
        return QuantumState(np.kron(self.vector, other.vector), normalize=False)

    def __matmul__(self, other: 'QuantumState') -> 'QuantumState':
        return self.tensor(other)

    def to_density_matrix(self) -> 'DensityMatrix':
        dm = np.outer(self.vector, np.conj(self.vector))
        return DensityMatrix(dm, self.num_qubits)

    def apply_unitary(self, U: np.ndarray) -> 'QuantumState':
        assert U.shape == (self.dim, self.dim), f"Unitary shape {U.shape} mismatch with state dim {self.dim}"
        return QuantumState(U @ self.vector, normalize=False)

    def apply_gate(self, gate: np.ndarray, target_qubits: List[int]) -> 'QuantumState':
        k = len(target_qubits)
        assert gate.shape == (2**k, 2**k), f"Gate shape {gate.shape} mismatch with {k} target qubits"
        full_u = embed_gate(gate, target_qubits, self.num_qubits)
        return self.apply_unitary(full_u)

    def measure_qubit(self, qubit_idx: int, outcome: Optional[int] = None) -> Tuple[int, float, 'QuantumState']:
        assert 0 <= qubit_idx < self.num_qubits, f"Invalid qubit index {qubit_idx}"
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)
        full_P0 = embed_gate(P0, [qubit_idx], self.num_qubits)
        full_P1 = embed_gate(P1, [qubit_idx], self.num_qubits)
        v0 = full_P0 @ self.vector
        v1 = full_P1 @ self.vector
        prob0 = float(np.real(np.vdot(v0, v0)))
        prob1 = float(np.real(np.vdot(v1, v1)))
        if outcome is None:
            chosen = 0 if np.random.rand() < prob0 else 1
        else:
            chosen = outcome
        prob = prob0 if chosen == 0 else prob1
        post_vec = v0 if chosen == 0 else v1
        norm = np.linalg.norm(post_vec)
        if norm > 1e-15:
            post_vec = post_vec / norm
        return chosen, prob, QuantumState(post_vec, normalize=False)

    def inner_product(self, other: 'QuantumState') -> complex:
        return complex(np.vdot(other.vector, self.vector))

    def fidelity(self, other: 'QuantumState') -> float:
        return float(np.abs(self.inner_product(other))**2)

    def copy(self) -> 'QuantumState':
        return QuantumState(self.vector.copy(), normalize=False)

    def __repr__(self) -> str:
        return f"QuantumState(num_qubits={self.num_qubits}, dim={self.dim})"


class DensityMatrix:
    def __init__(self, matrix: np.ndarray, num_qubits: Optional[int] = None):
        self.matrix = np.array(matrix, dtype=complex)
        self.dim = self.matrix.shape[0]
        if num_qubits is not None:
            self.num_qubits = num_qubits
        else:
            self.num_qubits = int(np.round(np.log2(self.dim)))
        assert self.matrix.shape == (self.dim, self.dim), "Matrix must be square"

    def trace(self) -> complex:
        return complex(np.trace(self.matrix))

    def purity(self) -> float:
        return float(np.real(np.trace(self.matrix @ self.matrix)))

    def apply_unitary(self, U: np.ndarray) -> 'DensityMatrix':
        return DensityMatrix(U @ self.matrix @ U.conj().T, self.num_qubits)

    def partial_trace(self, keep_qubits: List[int]) -> 'DensityMatrix':
        keep_qubits = sorted(keep_qubits)
        trace_out = [q for q in range(self.num_qubits) if q not in keep_qubits]
        if not trace_out:
            return DensityMatrix(self.matrix.copy(), self.num_qubits)
        tensor_shape = [2] * (2 * self.num_qubits)
        reshaped = self.matrix.reshape(tensor_shape)
        num_trace = len(trace_out)
        for offset, q in enumerate(sorted(trace_out)):
            curr_q = q - offset
            curr_n = self.num_qubits - offset
            reshaped = np.trace(reshaped, axis1=curr_q, axis2=curr_q + curr_n)
        new_n = len(keep_qubits)
        new_dim = 2**new_n
        new_mat = reshaped.reshape((new_dim, new_dim))
        return DensityMatrix(new_mat, new_n)

    def fidelity(self, target: Union['DensityMatrix', QuantumState]) -> float:
        if isinstance(target, QuantumState):
            psi = target.vector
            val = np.vdot(psi, self.matrix @ psi)
            return float(np.real(val))
        import scipy.linalg
        sqrt_rho = scipy.linalg.sqrtm(self.matrix)
        product = sqrt_rho @ target.matrix @ sqrt_rho
        sqrt_product = scipy.linalg.sqrtm(product)
        return float(np.real(np.trace(sqrt_product))**2)

    def __repr__(self) -> str:
        return f"DensityMatrix(num_qubits={self.num_qubits}, dim={self.dim})"


def embed_gate(gate: np.ndarray, target_qubits: List[int], total_qubits: int) -> np.ndarray:
    k = len(target_qubits)
    assert gate.shape == (2**k, 2**k)
    ordered_qubits = target_qubits + [q for q in range(total_qubits) if q not in target_qubits]
    I_rest = np.eye(2**(total_qubits - k), dtype=complex)
    big_gate = np.kron(gate, I_rest)
    perm = [0] * total_qubits
    for new_idx, old_idx in enumerate(ordered_qubits):
        perm[old_idx] = new_idx
    tensor_shape = [2] * (2 * total_qubits)
    perm_in = perm
    perm_out = [p + total_qubits for p in perm]
    full_perm = perm_in + perm_out
    reshaped = big_gate.reshape(tensor_shape)
    transposed = np.transpose(reshaped, full_perm)
    return transposed.reshape((2**total_qubits, 2**total_qubits))

def computational_basis_state(bitstring: str) -> QuantumState:
    vec = np.zeros(2**len(bitstring), dtype=complex)
    idx = int(bitstring, 2)
    vec[idx] = 1.0
    return QuantumState(vec, normalize=False)

def bell_state(index: int = 0) -> QuantumState:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if index == 0:
        return QuantumState(np.array([inv_sqrt2, 0, 0, inv_sqrt2], dtype=complex))
    elif index == 1:
        return QuantumState(np.array([inv_sqrt2, 0, 0, -inv_sqrt2], dtype=complex))
    elif index == 2:
        return QuantumState(np.array([0, inv_sqrt2, inv_sqrt2, 0], dtype=complex))
    elif index == 3:
        return QuantumState(np.array([0, inv_sqrt2, -inv_sqrt2, 0], dtype=complex))
    else:
        raise ValueError(f"Invalid Bell index {index}. Must be 0, 1, 2, or 3.")

def random_quantum_state(num_qubits: int, seed: Optional[int] = None) -> QuantumState:
    if seed is not None:
        np.random.seed(seed)
    dim = 2**num_qubits
    real_part = np.random.randn(dim)
    imag_part = np.random.randn(dim)
    vec = real_part + 1j * imag_part
    return QuantumState(vec, normalize=True)
