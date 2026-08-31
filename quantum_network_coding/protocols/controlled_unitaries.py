import numpy as np
from typing import Dict, Tuple, List, Optional
from ..core.state import QuantumState, DensityMatrix, bell_state
from ..core.operations import (
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, CNOT,
    three_qubit_fully_controlled_unitary, two_qubit_controlled_unitary
)

def implement_three_qubit_fully_controlled_unitary_locc(
    input_state: QuantumState,
    control1_idx: int,
    control2_idx: int,
    target_idx: int,
    u00: np.ndarray,
    u01: np.ndarray,
    u10: np.ndarray,
    u11: np.ndarray
) -> QuantumState:
    assert input_state.num_qubits == 3
    # Qubit mapping: control1=0, control2=1, target=2
    # 1. Ancillae at nodes: q3 (ancilla for control 1), q4 (ancilla for control 2)
    zero_ancilla = QuantumState(np.array([1, 0], dtype=complex))
    state = input_state @ zero_ancilla @ zero_ancilla  # 5 qubits: [0:c1, 1:c2, 2:tgt, 3:a1, 4:a2]
    
    # 2. Local CNOTs on (c1 -> a1) and (c2 -> a2)
    state = state.apply_gate(CNOT, [control1_idx, 3])
    state = state.apply_gate(CNOT, [control2_idx, 4])
    
    # 3. Simulate teleportation of a1 and a2 to target node:
    # Teleportation preserves the quantum state of a1 and a2 exactly (modulo LOCC Pauli corrections)
    # The teleported ancillae at node target are denoted a1', a2'
    
    # 4. At target node, apply local 3-qubit gate C_{a1', a2'; target}
    # Targets in the 5-qubit system: [3:a1', 4:a2', 2:target]
    gate_3q = three_qubit_fully_controlled_unitary(u00, u01, u10, u11)
    state = state.apply_gate(gate_3q, [3, 4, target_idx])
    
    # 5. At target node, apply Hadamard on a1' (qubit 3) and a2' (qubit 4)
    state = state.apply_gate(HADAMARD, [3])
    state = state.apply_gate(HADAMARD, [4])
    
    # 6. Measure a1' and a2' in computational basis
    m1, _, state_m1 = state.measure_qubit(3)
    m2, _, state_m12 = state_m1.measure_qubit(4)
    
    # 7. Apply LOCC phase corrections Z^{m1} on c1 and Z^{m2} on c2
    if m1 == 1:
        state_m12 = state_m12.apply_gate(PAULI_Z, [control1_idx])
    if m2 == 1:
        state_m12 = state_m12.apply_gate(PAULI_Z, [control2_idx])
        
    # 8. Extract the 3 remaining qubits [c1, c2, target]
    dm = state_m12.to_density_matrix().partial_trace([control1_idx, control2_idx, target_idx])
    w, v = np.linalg.eigh(dm.matrix)
    final_vec = v[:, np.argmax(w)]
    
    # Correct global phase alignment
    ideal_gate = three_qubit_fully_controlled_unitary(u00, u01, u10, u11)
    # Target index in the 3-qubit subspace
    ideal_state = input_state.apply_gate(ideal_gate, [control1_idx, control2_idx, target_idx])
    phase = np.exp(1j * np.angle(np.vdot(final_vec, ideal_state.vector)))
    final_vec = final_vec * phase
    
    return QuantumState(final_vec, normalize=True)

def implement_two_qubit_controlled_unitary_locc(
    input_state: QuantumState,
    control_idx: int,
    target_idx: int,
    u0: np.ndarray,
    u1: np.ndarray
) -> QuantumState:
    assert input_state.num_qubits == 2
    zero_ancilla = QuantumState(np.array([1, 0], dtype=complex))
    state = input_state @ zero_ancilla  # 3 qubits: [0:c, 1:tgt, 2:ancilla]
    
    # CNOT from control to ancilla
    state = state.apply_gate(CNOT, [control_idx, 2])
    
    # Apply local controlled unitary on ancilla and target
    gate_2q = two_qubit_controlled_unitary(u0, u1)
    state = state.apply_gate(gate_2q, [2, target_idx])
    
    # Hadamard on ancilla and measure
    state = state.apply_gate(HADAMARD, [2])
    m, _, state_m = state.measure_qubit(2)
    
    # Correction on control: Z^m
    if m == 1:
        state_m = state_m.apply_gate(PAULI_Z, [control_idx])
        
    dm = state_m.to_density_matrix().partial_trace([control_idx, target_idx])
    w, v = np.linalg.eigh(dm.matrix)
    final_vec = v[:, np.argmax(w)]
    
    ideal_gate = two_qubit_controlled_unitary(u0, u1)
    ideal_state = input_state.apply_gate(ideal_gate, [control_idx, target_idx])
    phase = np.exp(1j * np.angle(np.vdot(final_vec, ideal_state.vector)))
    final_vec = final_vec * phase
    
    return QuantumState(final_vec, normalize=True)
