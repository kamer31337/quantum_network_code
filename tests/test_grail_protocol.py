import unittest
import numpy as np
from quantum_network_coding.core.state import QuantumState, random_quantum_state
from quantum_network_coding.protocols.grail_protocol import simulate_grail_quantum_computation

class TestGrailProtocol(unittest.TestCase):
    def test_canonical_gates_grail(self):
        cases = [
            (0.0, 0.0, 0.0),                     # Identity
            (np.pi / 4.0, 0.0, 0.0),             # CNOT
            (np.pi / 4.0, np.pi / 4.0, 0.0),     # DCNOT / iSWAP
            (np.pi / 4.0, np.pi / 4.0, np.pi / 4.0), # SWAP
        ]
        for x, y, z in cases:
            psi = random_quantum_state(num_qubits=2, seed=15)
            res = simulate_grail_quantum_computation(psi, x, y, z)
            self.assertAlmostEqual(res['fidelity'], 1.0, places=6)

    def test_random_weyl_points_grail(self):
        np.random.seed(202)
        for _ in range(10):
            raw = np.sort(np.random.uniform(0, np.pi / 4.0, size=3))
            z, y, x = float(raw[0]), float(raw[1]), float(raw[2])
            psi = random_quantum_state(num_qubits=2)
            res = simulate_grail_quantum_computation(psi, x, y, z)
            self.assertAlmostEqual(res['fidelity'], 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
