import unittest
import numpy as np
from quantum_network_coding.core.state import QuantumState, random_quantum_state, computational_basis_state
from quantum_network_coding.protocols.teleportation import teleport_single_qubit_locc, teleport_qubit_chain

class TestLOCCTeleportation(unittest.TestCase):
    def test_single_qubit_basis_teleportation(self):
        for bit in ["0", "1"]:
            psi = computational_basis_state(bit)
            rec_psi, _ = teleport_single_qubit_locc(psi)
            fid = psi.fidelity(rec_psi)
            self.assertAlmostEqual(fid, 1.0, places=7)

    def test_single_qubit_random_state_teleportation(self):
        np.random.seed(42)
        for _ in range(10):
            psi = random_quantum_state(num_qubits=1)
            rec_psi, _ = teleport_single_qubit_locc(psi)
            fid = psi.fidelity(rec_psi)
            self.assertAlmostEqual(fid, 1.0, places=7)

    def test_multi_hop_teleportation(self):
        np.random.seed(99)
        psi = random_quantum_state(num_qubits=1)
        rec_psi = teleport_qubit_chain(psi, num_hops=4)
        fid = psi.fidelity(rec_psi)
        self.assertAlmostEqual(fid, 1.0, places=7)

if __name__ == '__main__':
    unittest.main()
