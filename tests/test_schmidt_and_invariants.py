import unittest
import numpy as np
from quantum_network_coding.core.state import bell_state, computational_basis_state
from quantum_network_coding.core.operations import (
    PAULI_I, PAULI_X, HADAMARD, CNOT, CZ, SWAP, ISWAP, tensor_product
)
from quantum_network_coding.analysis.schmidt import (
    schmidt_decomposition_state, operator_schmidt_rank, kraus_cirac_number
)
from quantum_network_coding.analysis.implementability import is_implementable_on_ladder_network
from quantum_network_coding.analysis.slocc_invariants import verify_no_4qubit_state_4_2_2

class TestSchmidtAndInvariants(unittest.TestCase):
    def test_state_schmidt_rank(self):
        phi_plus = bell_state(0)
        _, _, _, r_bell = schmidt_decomposition_state(phi_plus.vector, 2, 2)
        self.assertEqual(r_bell, 2)
        
        prod_state = computational_basis_state("01")
        _, _, _, r_prod = schmidt_decomposition_state(prod_state.vector, 2, 2)
        self.assertEqual(r_prod, 1)

    def test_operator_schmidt_ranks(self):
        self.assertEqual(operator_schmidt_rank(tensor_product(PAULI_I, PAULI_I)), 1)
        self.assertEqual(operator_schmidt_rank(CNOT), 2)
        self.assertEqual(operator_schmidt_rank(CZ), 2)
        self.assertEqual(operator_schmidt_rank(ISWAP), 4)
        self.assertEqual(operator_schmidt_rank(SWAP), 4)

    def test_kraus_cirac_numbers(self):
        self.assertEqual(kraus_cirac_number(tensor_product(HADAMARD, HADAMARD)), 0)
        self.assertEqual(kraus_cirac_number(CNOT), 1)
        self.assertEqual(kraus_cirac_number(CZ), 1)
        self.assertEqual(kraus_cirac_number(ISWAP), 2)
        self.assertEqual(kraus_cirac_number(SWAP), 3)

    def test_ladder_network_implementability_theorem_3(self):
        # 1-bridge ladder
        self.assertTrue(is_implementable_on_ladder_network(CNOT, num_bridges_N=1)['deterministic_implementable'])
        self.assertFalse(is_implementable_on_ladder_network(ISWAP, num_bridges_N=1)['deterministic_implementable'])
        self.assertFalse(is_implementable_on_ladder_network(SWAP, num_bridges_N=1)['deterministic_implementable'])
        
        # 2-bridge ladder (square network)
        self.assertTrue(is_implementable_on_ladder_network(ISWAP, num_bridges_N=2)['deterministic_implementable'])
        self.assertFalse(is_implementable_on_ladder_network(SWAP, num_bridges_N=2)['deterministic_implementable'])
        self.assertFalse(is_implementable_on_ladder_network(SWAP, num_bridges_N=2)['probabilistic_implementable'])
        
        # 3-bridge ladder
        self.assertTrue(is_implementable_on_ladder_network(SWAP, num_bridges_N=3)['deterministic_implementable'])

    def test_theorem_5_slocc_invariants(self):
        res = verify_no_4qubit_state_4_2_2(num_samples=100)
        self.assertTrue(res['theorem_5_holds'])

if __name__ == '__main__':
    unittest.main()
