import numpy as np
import scipy.linalg
from typing import Dict, Any, Optional, Tuple
from ..core.state import QuantumState, DensityMatrix
from ..core.operations import (
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, CNOT,
    kraus_cirac_global_unitary, tensor_product
)

def r_x(theta: float) -> np.ndarray:
    return np.cos(theta / 2.0) * PAULI_I - 1j * np.sin(theta / 2.0) * PAULI_X

def r_y(theta: float) -> np.ndarray:
    return np.cos(theta / 2.0) * PAULI_I - 1j * np.sin(theta / 2.0) * PAULI_Y

def r_z(theta: float) -> np.ndarray:
    return np.cos(theta / 2.0) * PAULI_I - 1j * np.sin(theta / 2.0) * PAULI_Z

def cnot_21() -> np.ndarray:
    # CNOT with control = qubit 1, target = qubit 0
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)

def single_qubit_zyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    return r_z(alpha) @ r_y(beta) @ r_z(gamma)

class GrailProtocolRunner:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.cnot12 = CNOT  # control 0, target 1
        self.cnot21 = cnot_21()  # control 1, target 0
        self.circuit_unitary = self._synthesize_circuit()

    def _synthesize_circuit(self) -> np.ndarray:
        target_U = kraus_cirac_global_unitary(self.x, self.y, self.z)
        
        # If parameters are zero (Identity), return Identity
        if abs(self.x) < 1e-7 and abs(self.y) < 1e-7 and abs(self.z) < 1e-7:
            return np.eye(4, dtype=complex)
            
        from scipy.optimize import minimize
        
        # We parameterize 8 single-qubit unitaries across the 3 CNOT slices of (2,3)-cluster
        # (u4 (x) v4) C12 (u3 (x) v3) C21 (u2 (x) v2) C12 (u1 (x) v1)
        def make_circuit(params):
            # 8 unitaries * 3 angles = 24 params, or simplified 12 params (Ry and Rz only)
            # params: [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6]
            u1 = r_z(params[0]) @ r_y(params[1])
            v1 = r_z(params[2]) @ r_y(params[3])
            u2 = r_y(params[4]) @ r_z(params[5])
            v2 = r_y(params[6]) @ r_z(params[7])
            u3 = r_z(params[8]) @ r_y(params[9])
            v3 = r_z(params[10]) @ r_y(params[11])
            u4 = r_y(params[12]) @ r_z(params[13])
            v4 = r_y(params[14]) @ r_z(params[15])
            
            U = tensor_product(u1, v1)
            U = self.cnot12 @ U
            U = tensor_product(u2, v2) @ U
            U = self.cnot21 @ U
            U = tensor_product(u3, v3) @ U
            U = self.cnot12 @ U
            U = tensor_product(u4, v4) @ U
            return U

        def cost(params):
            U = make_circuit(params)
            inner = np.trace(target_U.conj().T @ U)
            return 1.0 - float(np.abs(inner) / 4.0)

        # Initial seed based on Vatan-Williams canonical form
        p0 = [
            -np.pi/2, -np.pi/2, -np.pi/2, 0.0,
            2.0*self.y, 2.0*self.z, 2.0*self.y, 0.0,
            0.0, 0.0, -2.0*self.x, 0.0,
            np.pi/2, np.pi/2, np.pi/2, 0.0
        ]
        
        res = minimize(cost, p0, method='BFGS', options={'gtol': 1e-8, 'maxiter': 300})
        if res.fun > 1e-4:
            # Fallback multiple restarts
            for _ in range(5):
                prand = np.random.uniform(-np.pi, np.pi, size=16)
                res2 = minimize(cost, prand, method='BFGS', options={'gtol': 1e-8, 'maxiter': 300})
                if res2.fun < res.fun:
                    res = res2
                if res.fun < 1e-6:
                    break
                    
        return make_circuit(res.x)

    def get_circuit_unitary(self) -> np.ndarray:
        return self.circuit_unitary

    def run(self, input_state: QuantumState) -> Dict[str, Any]:
        assert input_state.num_qubits == 2
        circuit_U = self.get_circuit_unitary()
        out_state = input_state.apply_unitary(circuit_U)
        
        ideal_U = kraus_cirac_global_unitary(self.x, self.y, self.z)
        ideal_vec = ideal_U @ input_state.vector
        
        # Check equivalence up to global phase
        inner = np.vdot(ideal_vec, out_state.vector)
        fidelity = float(np.abs(inner)**2)
        
        return {
            'output_state': out_state,
            'ideal_state': QuantumState(ideal_vec, normalize=True),
            'fidelity': fidelity,
            'circuit_unitary': circuit_U
        }

def simulate_grail_quantum_computation(input_state: QuantumState, x: float, y: float, z: float) -> Dict[str, Any]:
    runner = GrailProtocolRunner(x, y, z)
    return runner.run(input_state)

