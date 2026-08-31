import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..core.state import QuantumState, DensityMatrix, bell_state
from ..core.operations import (
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD,
    u_x_gate, kraus_cirac_global_unitary, three_qubit_fully_controlled_unitary
)

class ButterflyProtocolRunner:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.u_first_00 = PAULI_I
        self.u_first_11 = PAULI_I
        self.u_first_01 = PAULI_Z
        self.u_first_10 = PAULI_Z
        
        # Second controlled gate parameters w(ab)
        self.w_00 = np.array([
            [np.exp(1j * (z - y)), 0],
            [0, -1j * np.exp(1j * (z + y))]
        ], dtype=complex)
        self.w_11 = self.w_00
        self.w_01 = np.array([
            [np.exp(-1j * (z - y)), 0],
            [0, -1j * np.exp(-1j * (z + y))]
        ], dtype=complex)
        self.w_10 = self.w_01
        
        self.u_x = u_x_gate(x)
        self.C1 = three_qubit_fully_controlled_unitary(
            self.u_first_00, self.u_first_01, self.u_first_10, self.u_first_11
        )
        self.C2 = three_qubit_fully_controlled_unitary(
            self.w_00, self.w_01, self.w_10, self.w_11
        )

    def run_step_by_step(self, input_2qubit_state: QuantumState, forced_measurement: Optional[int] = None) -> Dict[str, Any]:
        assert input_2qubit_state.num_qubits == 2
        # Wire layout: wire 0 = i1 (v1,1), wire 1 = ancilla (v2,1), wire 2 = i2 (v3,1)
        # Note: input_2qubit_state is on (i1, i2). We construct 3-qubit state on (wire 0, wire 2, wire 1)
        zero_ancilla = QuantumState(np.array([1, 0], dtype=complex))
        
        # Stage (i): Construct 3-qubit state where wire 0 is input 1, wire 1 is ancilla |0>, wire 2 is input 2
        v_in = input_2qubit_state.vector  # basis |00>, |01>, |10>, |11> for (0, 2)
        v3 = np.zeros(8, dtype=complex)
        # |q0, q1, q2> where q1 = 0
        v3[0] = v_in[0]  # |000>
        v3[1] = v_in[1]  # |001>
        v3[4] = v_in[2]  # |100>
        v3[5] = v_in[3]  # |101>
        state_i = QuantumState(v3, normalize=False)
        
        # Stage (ii): Apply Hadamard H on all 3 wires
        state_ii = state_i.apply_gate(HADAMARD, [0]).apply_gate(HADAMARD, [1]).apply_gate(HADAMARD, [2])
        
        # Stage (iii): Apply first fully controlled gate C_{1,3;2} (controls 0 & 2, target 1)
        state_iii = state_ii.apply_gate(self.C1, [0, 2, 1])
        
        # State transmission (teleportation over horizontal edges K1 from column 1 to column 2)
        # In ideal LOCC network, teleportation transmits the exact state of wires 0, 1, 2
        
        # Stage (iv): Apply Hadamard on 0, 1, 2 and Pauli X on 0, 2
        state_iv = state_iii.apply_gate(HADAMARD, [0]).apply_gate(HADAMARD, [1]).apply_gate(HADAMARD, [2])
        state_iv = state_iv.apply_gate(PAULI_X, [0]).apply_gate(PAULI_X, [2])
        
        # Stage (v): Apply second fully controlled gate C'_{1,3;2} (controls 0 & 2, target 1)
        state_v = state_iv.apply_gate(self.C2, [0, 2, 1])
        
        # Stage (vi): Apply u(x) on wire 1 (node v2,2)
        state_vi = state_v.apply_gate(self.u_x, [1])
        
        # Stage (vii) & LOCC map Gamma: Measure wire 1 in computational basis
        meas_outcome, prob, state_post = state_vi.measure_qubit(1, outcome=forced_measurement)
        
        # If meas_outcome == 1, apply conditional Pauli X on wire 0 and wire 2
        state_corrected = state_post
        if meas_outcome == 1:
            state_corrected = state_corrected.apply_gate(PAULI_X, [0]).apply_gate(PAULI_X, [2])
            
        # Extract the 2 output qubits (wire 0 -> o1, wire 2 -> o2)
        dm_out = state_corrected.to_density_matrix().partial_trace([0, 2])
        w_eig, v_eig = np.linalg.eigh(dm_out.matrix)
        out_vec = v_eig[:, np.argmax(w_eig)]
        
        # Align global phase with ideal state
        ideal_U = kraus_cirac_global_unitary(self.x, self.y, self.z)
        ideal_out_vec = ideal_U @ input_2qubit_state.vector
        phase_align = np.exp(1j * np.angle(np.vdot(out_vec, ideal_out_vec)))
        out_vec = out_vec * phase_align
        out_state = QuantumState(out_vec, normalize=True)
        
        fidelity = float(np.abs(np.vdot(ideal_out_vec, out_vec))**2)
        
        return {
            'state_i': state_i,
            'state_ii': state_ii,
            'state_iii': state_iii,
            'state_iv': state_iv,
            'state_v': state_v,
            'state_vi': state_vi,
            'measurement_outcome': meas_outcome,
            'measurement_prob': prob,
            'output_state': out_state,
            'ideal_state': QuantumState(ideal_out_vec, normalize=True),
            'fidelity': fidelity
        }

def simulate_butterfly_quantum_computation(
    input_state: QuantumState,
    x: float,
    y: float,
    z: float,
    u_in: Optional[np.ndarray] = None,
    u_out: Optional[np.ndarray] = None,
    forced_measurement: Optional[int] = None
) -> Dict[str, Any]:
    # Local input pre-rotations
    curr_state = input_state
    if u_in is not None:
        curr_state = curr_state.apply_unitary(u_in)
        
    runner = ButterflyProtocolRunner(x, y, z)
    res = runner.run_step_by_step(curr_state, forced_measurement=forced_measurement)
    
    # Local output post-rotations
    out_state = res['output_state']
    if u_out is not None:
        out_state = out_state.apply_unitary(u_out)
        res['output_state'] = out_state
        
        ideal_vec = u_out @ res['ideal_state'].vector
        res['ideal_state'] = QuantumState(ideal_vec, normalize=True)
        res['fidelity'] = float(np.abs(np.vdot(ideal_vec, out_state.vector))**2)
        
    return res
