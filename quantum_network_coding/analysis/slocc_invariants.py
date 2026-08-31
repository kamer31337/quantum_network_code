import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from .schmidt import schmidt_decomposition_state

def four_qubit_bipartite_schmidt_ranks(state_vec: np.ndarray, tol: float = 1e-8) -> Tuple[int, int, int]:
    assert len(state_vec) == 16
    tensor_4q = state_vec.reshape((2, 2, 2, 2))  # indices (0, 1, 2, 3) for qubits (1, 2, 3, 4)
    
    # 1. Partition (1,2) vs (3,4): indices (0,1) vs (2,3)
    mat_12_34 = np.transpose(tensor_4q, (0, 1, 2, 3)).reshape((4, 4))
    _, _, _, r12_34 = schmidt_decomposition_state(mat_12_34.flatten(), 4, 4, tol=tol)
    
    # 2. Partition (1,3) vs (2,4): indices (0,2) vs (1,3)
    mat_13_24 = np.transpose(tensor_4q, (0, 2, 1, 3)).reshape((4, 4))
    _, _, _, r13_24 = schmidt_decomposition_state(mat_13_24.flatten(), 4, 4, tol=tol)
    
    # 3. Partition (1,4) vs (2,3): indices (0,3) vs (1,2)
    mat_14_23 = np.transpose(tensor_4q, (0, 3, 1, 2)).reshape((4, 4))
    _, _, _, r14_23 = schmidt_decomposition_state(mat_14_23.flatten(), 4, 4, tol=tol)
    
    return (r12_34, r13_24, r14_23)

def generate_verstraete_family_state(
    family_idx: int,
    a: complex = 1.0,
    b: complex = 0.5,
    c: complex = 0.3,
    d: complex = 0.2
) -> np.ndarray:
    basis = {f"{i:04b}": idx for idx, i in enumerate(range(16))}
    vec = np.zeros(16, dtype=complex)
    
    def add_term(coeff: complex, bitstr: str):
        vec[basis[bitstr]] += coeff

    if family_idx == 1:
        add_term((a + d) / 2.0, "0000")
        add_term((a + d) / 2.0, "1111")
        add_term((a - d) / 2.0, "0011")
        add_term((a - d) / 2.0, "1100")
        add_term((b + c) / 2.0, "0101")
        add_term((b + c) / 2.0, "1010")
        add_term((b - c) / 2.0, "0110")
        add_term((b - c) / 2.0, "1001")
    elif family_idx == 2:
        add_term((a + b) / 2.0, "0000")
        add_term((a + b) / 2.0, "1111")
        add_term((a - b) / 2.0, "0011")
        add_term((a - b) / 2.0, "1100")
        add_term(c, "0101")
        add_term(c, "1010")
        add_term(1.0, "0110")
    elif family_idx == 3:
        add_term(a, "0000")
        add_term(a, "1111")
        add_term(b, "0101")
        add_term(b, "1010")
        add_term(1.0, "0110")
        add_term(1.0, "0011")
    elif family_idx == 4:
        add_term(a, "0000")
        add_term(a, "1111")
        add_term((a + b) / 2.0, "0101")
        add_term((a + b) / 2.0, "1010")
        add_term((a - b) / 2.0, "0110")
        add_term((a - b) / 2.0, "1001")
        add_term(1j / np.sqrt(2.0), "0001")
        add_term(1j / np.sqrt(2.0), "0010")
        add_term(1j / np.sqrt(2.0), "0111")
        add_term(1j / np.sqrt(2.0), "1011")
    elif family_idx == 5:
        add_term(a, "0000")
        add_term(a, "0101")
        add_term(a, "1010")
        add_term(a, "1111")
        add_term(1j, "0001")
        add_term(1.0, "0110")
        add_term(-1j, "1011")
    elif family_idx == 6:
        add_term(a, "0000")
        add_term(a, "1111")
        add_term(1.0, "0011")
        add_term(1.0, "0101")
        add_term(1.0, "0110")
    elif family_idx == 7:
        add_term(1.0, "0000")
        add_term(1.0, "0101")
        add_term(1.0, "1000")
        add_term(1.0, "1110")
    elif family_idx == 8:
        add_term(1.0, "0000")
        add_term(1.0, "1011")
        add_term(1.0, "1101")
        add_term(1.0, "1110")
    elif family_idx == 9:
        add_term(1.0, "0000")
        add_term(1.0, "0111")
    else:
        raise ValueError(f"Family index {family_idx} must be between 1 and 9.")
        
    norm = np.linalg.norm(vec)
    if norm > 1e-15:
        vec = vec / norm
    return vec

def verstraete_nine_families_schmidt_ranks(samples_per_family: int = 50) -> Dict[int, List[Tuple[int, int, int]]]:
    results: Dict[int, List[Tuple[int, int, int]]] = {}
    for fam in range(1, 10):
        ranks_set = set()
        for _ in range(samples_per_family):
            a = np.random.randn() + 1j * np.random.randn()
            b = np.random.randn() + 1j * np.random.randn()
            c = np.random.randn() + 1j * np.random.randn()
            d = np.random.randn() + 1j * np.random.randn()
            vec = generate_verstraete_family_state(fam, a, b, c, d)
            ranks = four_qubit_bipartite_schmidt_ranks(vec)
            ranks_set.add(tuple(sorted(ranks)))
        results[fam] = sorted(list(ranks_set))
    return results

def verify_no_4qubit_state_4_2_2(num_samples: int = 500) -> Dict[str, Any]:
    found_422 = False
    violating_sample = None
    all_observed_tuples = set()
    
    for fam in range(1, 10):
        for _ in range(num_samples):
            a = (np.random.randn() + 1j * np.random.randn())
            b = (np.random.randn() + 1j * np.random.randn())
            c = (np.random.randn() + 1j * np.random.randn())
            d = (np.random.randn() + 1j * np.random.randn())
            vec = generate_verstraete_family_state(fam, a, b, c, d)
            ranks = four_qubit_bipartite_schmidt_ranks(vec)
            sorted_ranks = tuple(sorted(ranks))
            all_observed_tuples.add(sorted_ranks)
            if sorted_ranks == (2, 2, 4):
                found_422 = True
                violating_sample = {'family': fam, 'params': (a, b, c, d), 'ranks': ranks}
                break
        if found_422:
            break
            
    return {
        'theorem_5_holds': not found_422,
        'observed_rank_profiles': sorted(list(all_observed_tuples)),
        'violating_sample': violating_sample
    }
