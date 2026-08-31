import numpy as np
from typing import Tuple, Optional, List
from ..core.state import QuantumState, DensityMatrix, bell_state
from ..core.operations import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, CNOT

def teleport_single_qubit_locc(input_qubit: QuantumState, bell_index: int = 0) -> Tuple[QuantumState, int]:
    assert input_qubit.num_qubits == 1
    phi_plus = bell_state(bell_index)
    total_state = input_qubit @ phi_plus  # Qubits: 0 (input), 1 (sender ancilla), 2 (receiver ancilla)
    
    # 1. CNOT on qubits 0 and 1
    total_state = total_state.apply_gate(CNOT, [0, 1])
    # 2. Hadamard on qubit 0
    total_state = total_state.apply_gate(HADAMARD, [0])
    
    # 3. Measure qubits 0 and 1 in computational basis
    m0, p0, state_m0 = total_state.measure_qubit(0)
    m1, p1, state_m01 = state_m0.measure_qubit(1)
    
    # 4. Partial trace / extract target qubit 2
    dm = state_m01.to_density_matrix().partial_trace([2])
    
    # 5. Apply LOCC conditional Pauli correction: X^{m1} Z^{m0}
    correction = np.eye(2, dtype=complex)
    if m1 == 1:
        correction = PAULI_X @ correction
    if m0 == 1:
        correction = PAULI_Z @ correction
    
    corrected_dm = dm.apply_unitary(correction)
    # Recover pure statevector from rank-1 density matrix
    w, v = np.linalg.eigh(corrected_dm.matrix)
    rec_vec = v[:, np.argmax(w)]
    # Match global phase
    phase_factor = np.exp(1j * np.angle(np.vdot(rec_vec, input_qubit.vector)))
    rec_vec = rec_vec * phase_factor
    outcome_key = (m0 << 1) | m1
    return QuantumState(rec_vec, normalize=True), outcome_key

def teleport_qubit(state: QuantumState, source_qubit_idx: int, bell_pair_qubits: Tuple[int, int]) -> QuantumState:
    assert len(bell_pair_qubits) == 2
    b1, b2 = bell_pair_qubits
    
    # Apply Bell measurement on (source_qubit_idx, b1)
    st = state.apply_gate(CNOT, [source_qubit_idx, b1])
    st = st.apply_gate(HADAMARD, [source_qubit_idx])
    
    m_src, _, st1 = st.measure_qubit(source_qubit_idx)
    m_b1, _, st2 = st1.measure_qubit(b1)
    
    # Correction on b2: X^{m_b1} Z^{m_src}
    if m_b1 == 1:
        st2 = st2.apply_gate(PAULI_X, [b2])
    if m_src == 1:
        st2 = st2.apply_gate(PAULI_Z, [b2])
    
    return st2

def teleport_qubit_chain(input_qubit: QuantumState, num_hops: int) -> QuantumState:
    curr_qubit = input_qubit
    for _ in range(num_hops):
        curr_qubit, _ = teleport_single_qubit_locc(curr_qubit)
    return curr_qubit
