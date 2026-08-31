import numpy as np
from typing import Dict, List, Tuple, Optional
from .state import QuantumState, bell_state

class Node:
    def __init__(self, i: int, j: int, name: Optional[str] = None, is_input: bool = False, is_output: bool = False):
        self.i = i
        self.j = j
        self.name = name if name is not None else f"v_{i},{j}"
        self.is_input = is_input
        self.is_output = is_output

    def __repr__(self) -> str:
        return f"Node({self.name}, coord=({self.i},{self.j}))"


class Edge:
    def __init__(self, u: Node, v: Node, edge_type: str, name: Optional[str] = None):
        self.u = u
        self.v = v
        self.edge_type = edge_type  # 'vertical' (S) or 'horizontal' (K)
        self.name = name if name is not None else f"({u.name}->{v.name})"

    def __repr__(self) -> str:
        return f"Edge({self.name}, type={self.edge_type})"


class ClusterNetwork:
    def __init__(self, k: int, N: int):
        self.k = k
        self.N = N
        self.nodes: Dict[Tuple[int, int], Node] = {}
        for i in range(1, k + 1):
            for j in range(1, N + 1):
                is_inp = (j == 1)
                is_out = (j == N)
                self.nodes[(i, j)] = Node(i, j, is_input=is_inp, is_output=is_out)
        self.vertical_edges: List[Edge] = []
        for j in range(1, N + 1):
            for i in range(1, k):
                u = self.nodes[(i, j)]
                v = self.nodes[(i + 1, j)]
                self.vertical_edges.append(Edge(u, v, 'vertical', name=f"S_{i},{j}"))
        self.horizontal_edges: List[Edge] = []
        for i in range(1, k + 1):
            for j in range(1, N):
                u = self.nodes[(i, j)]
                v = self.nodes[(i, j + 1)]
                self.horizontal_edges.append(Edge(u, v, 'horizontal', name=f"K_{i},{j}"))

    @property
    def num_vertical_bell_pairs(self) -> int:
        return len(self.vertical_edges)

    @property
    def num_horizontal_bell_pairs(self) -> int:
        return len(self.horizontal_edges)

    @property
    def total_bell_pairs(self) -> int:
        return self.num_vertical_bell_pairs + self.num_horizontal_bell_pairs

    def generate_resource_state(self) -> QuantumState:
        total = self.total_bell_pairs
        phi_plus = bell_state(0)
        state = phi_plus
        for _ in range(1, total):
            state = state @ phi_plus
        return state

    def __repr__(self) -> str:
        return f"ClusterNetwork(k={self.k}, N={self.N}, nodes={len(self.nodes)}, vert_edges={len(self.vertical_edges)}, horiz_edges={len(self.horizontal_edges)})"


class ButterflyNetwork:
    def __init__(self):
        self.cluster = ClusterNetwork(k=3, N=2)
        self.node_mapping = {
            'i1': (1, 1),
            'n1': (2, 1),
            'i2': (3, 1),
            'o1': (1, 2),
            'n2': (2, 2),
            'o2': (3, 2)
        }
        self.edge_mapping = {
            'E1': 'K_1,1',
            'E2': 'S_1,1',
            'E3': 'K_3,1',
            'E4': 'S_2,1',
            'E5': 'K_2,1',
            'E6': 'S_1,2',
            'E7': 'S_2,2'
        }

    def __repr__(self) -> str:
        return "ButterflyNetwork(isomorphic to (3,2)-cluster network)"


class GrailNetwork:
    def __init__(self):
        self.cluster = ClusterNetwork(k=2, N=3)
        self.node_mapping = {
            'n1': (1, 1),
            'n2': (1, 2),
            'o1': (1, 3),
            'i2': (2, 1),
            'n3': (2, 2),
            'n4': (2, 3)
        }

    def __repr__(self) -> str:
        return "GrailNetwork(isomorphic to (2,3)-cluster network)"
