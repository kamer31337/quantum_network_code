import unittest
import numpy as np
from quantum_network_coding.core.state import QuantumState, random_quantum_state
from quantum_network_coding.core.operations import (
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, random_unitary,
    three_qubit_fully_controlled_unitary, two_qubit_controlled_unitary
)
from quantum_network_coding.protocols.controlled_unitaries import (
    implement_two_qubit_controlled_unitary_locc,
    implement_three_qubit_fully_controlled_unitary_locc
)

class TestControlledUnitariesLOCC(unittest.TestCase):
    def test_two_qubit_cnot_locc(self):
        np.random.seed(10)
        for _ in range(5):
            psi_in = random_quantum_state(num_qubits=2)
            u0 = PAULI_I
            u1 = PAULI_X
            res_state = implement_two_qubit_controlled_unitary_locc(psi_in, 0, 1, u0, u1)
            ideal_gate = two_qubit_controlled_unitary(u0, u1)
            ideal_state = psi_in.apply_gate(ideal_gate, [0, 1])
            fid = ideal_state.fidelity(res_state)
            self.assertAlmostEqual(fid, 1.0, places=6)

    def test_three_qubit_fully_controlled_locc(self):
        np.random.seed(20)
        for _ in range(5):
            psi_in = random_quantum_state(num_qubits=3)
            u00 = random_unitary(2)
            u01 = random_unitary(2)
            u10 = random_unitary(2)
            u11 = random_unitary(2)
            res_state = implement_three_qubit_fully_controlled_unitary_locc(
                psi_in, control1_idx=0, control2_idx=1, target_idx=2,
                u00=u00, u01=u01, u10=u10, u11=u11
            )
            ideal_gate = three_qubit_fully_controlled_unitary(u00, u01, u10, u11)
            ideal_state = psi_in.apply_gate(ideal_gate, [0, 1, 2])
            fid = ideal_state.fidelity(res_state)
            self.assertAlmostEqual(fid, 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
