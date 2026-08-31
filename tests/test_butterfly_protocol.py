import unittest
import numpy as np
from quantum_network_coding.core.state import QuantumState, bell_state, random_quantum_state
from quantum_network_coding.core.operations import kraus_cirac_global_unitary
from quantum_network_coding.protocols.butterfly_protocol import (
    simulate_butterfly_quantum_computation, ButterflyProtocolRunner
)

class TestButterflyProtocol(unittest.TestCase):
    def test_bell_eigenstates_butterfly(self):
        x, y, z = np.pi / 6.0, np.pi / 8.0, np.pi / 12.0
        for bell_idx in range(4):
            psi_bell = bell_state(bell_idx)
            for forced_k in [0, 1]:
                res = simulate_butterfly_quantum_computation(
                    psi_bell, x, y, z, forced_measurement=forced_k
                )
                self.assertAlmostEqual(res['fidelity'], 1.0, places=6)
                
    def test_canonical_gates_butterfly(self):
        cases = [
            (0.0, 0.0, 0.0),                     # Identity
            (np.pi / 4.0, 0.0, 0.0),             # CNOT
            (np.pi / 4.0, np.pi / 4.0, 0.0),     # DCNOT / iSWAP
            (np.pi / 4.0, np.pi / 4.0, np.pi / 4.0), # SWAP
        ]
        for x, y, z in cases:
            psi = random_quantum_state(num_qubits=2, seed=7)
            res = simulate_butterfly_quantum_computation(psi, x, y, z)
            self.assertAlmostEqual(res['fidelity'], 1.0, places=6)

    def test_random_weyl_points_butterfly(self):
        np.random.seed(101)
        for _ in range(10):
            raw = np.sort(np.random.uniform(0, np.pi / 4.0, size=3))
            z, y, x = float(raw[0]), float(raw[1]), float(raw[2])
            psi = random_quantum_state(num_qubits=2)
            res = simulate_butterfly_quantum_computation(psi, x, y, z)
            self.assertAlmostEqual(res['fidelity'], 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
