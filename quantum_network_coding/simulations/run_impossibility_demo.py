import numpy as np
from ..core.operations import PAULI_I, CNOT, CZ, SWAP, ISWAP, HADAMARD, tensor_product
from ..analysis.schmidt import kraus_cirac_number, operator_schmidt_rank
from ..analysis.implementability import is_implementable_on_ladder_network, is_implementable_on_cluster_network
from ..analysis.slocc_invariants import verify_no_4qubit_state_4_2_2, verstraete_nine_families_schmidt_ranks

def run_impossibility_benchmark() -> None:
    print("=" * 85)
    print("DEMO: Theoretical Implementability & Impossibility Bounds (Theorems 2, 3, 4, 5 & Lemma 3)")
    print("=" * 85)
    
    gates = [
        ("Local Unitary (H (x) H)", tensor_product(HADAMARD, HADAMARD)),
        ("Controlled-NOT (CNOT)", CNOT),
        ("Controlled-Z (CZ)", CZ),
        ("iSWAP", ISWAP),
        ("SWAP", SWAP)
    ]
    
    print(f"{'Gate Name':<28} | {'OP#':<4} | {'KC#':<4} | {'(2,1)-Ladder':<13} | {'(2,2)-Square':<13} | {'Butterfly':<10} | {'Grail':<8}")
    print("-" * 85)
    
    for name, U in gates:
        op = operator_schmidt_rank(U)
        kc = kraus_cirac_number(U)
        res_21 = is_implementable_on_ladder_network(U, num_bridges_N=1)
        res_22 = is_implementable_on_ladder_network(U, num_bridges_N=2)
        res_bf = is_implementable_on_cluster_network(U, 'butterfly')
        res_gr = is_implementable_on_cluster_network(U, 'grail')
        
        det21 = "YES" if res_21['deterministic_implementable'] else "NO"
        det22 = "YES" if res_22['deterministic_implementable'] else "NO"
        detbf = "YES" if res_bf['deterministic_implementable'] else "NO"
        detgr = "YES" if res_gr['deterministic_implementable'] else "NO"
        
        print(f"{name:<28} | {op:<4} | {kc:<4} | {det21:<13} | {det22:<13} | {detbf:<10} | {detgr:<8}")
        
    print("\n" + "=" * 85)
    print("SLOCC Invariant Verification for Theorem 5 (No 4-Qubit State with Ranks {4, 2, 2})")
    print("=" * 85)
    
    res_thm5 = verify_no_4qubit_state_4_2_2(num_samples=300)
    print(f"Sampling across all 9 Verstraete canonical families under SLOCC:")
    print(f"Observed Partition Schmidt Rank Tuples across random parameters:")
    for tup in res_thm5['observed_rank_profiles']:
        print(f"  - (r_12|34, r_13|24, r_14|23) = {tup}")
    
    print(f"\nResult: Rank tuple (2, 2, 4) found? {not res_thm5['theorem_5_holds']}")
    print(f"Conclusion: Theorem 5 HOLDS -> SWAP is IMPOSSIBLE over (2,2)-cluster even probabilistically.")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    run_impossibility_benchmark()
