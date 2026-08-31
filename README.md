# Quantum Network Coding for Distributed Quantum Computation over Cluster and Butterfly Networks

Python implementation and simulation framework based on the paper:
> **"Network coding for distributed quantum computation over cluster and butterfly networks"**  
> *Seiseki Akibue and Mio Murao* (arXiv:1503.07740v2)

---

## 🌟 Overview

In distributed quantum computation, spatial node separation creates communication bottlenecks when multiple quantum data flows intersect. This repository implements quantum network coding schemes that simultaneously compute and route quantum information over bottleneck networks using pre-shared entanglement (Bell pairs) and LOCC.

### Key Results Implemented

| Network Topology | Cluster Model | Implementable Unitary Class | Key Protocol / Reference |
|---|---|---|---|
| **Butterfly Network** | $(3, 2)$-cluster | **Any 2-qubit unitary** ($KC\# \le 3$) | 7-stage LOCC protocol with $C_{1,3;2}$ and feedforward $\Gamma$ (Theorem 1) |
| **Grail Network** | $(2, 3)$-cluster | **Any 2-qubit unitary** ($KC\# \le 3$) | 3-CNOT + single-qubit Euler rotations (Section V) |
| **$N$-Bridge Ladder** | $(2, N)$-cluster | **$KC\#(U) \le N$** | CNOT chain simulation (Theorem 3) |
| **$(2, 2)$-Cluster (Square)** | $(2, 2)$-cluster | **No SWAP** ($KC\#(U) \le 2$ only) | SWAP impossible even probabilistically (Theorem 4, Theorem 5, Lemma 3) |

---

## 📁 Repository Structure

```
├── quantum_network_coding/
│   ├── core/
│   │   ├── state.py                 # Pure state & density matrix engine, Bell states, partial trace
│   │   ├── operations.py            # Pauli, CNOT, CZ, SWAP, Kraus-Cirac canonical decompositions
│   │   └── network.py               # (k, N)-cluster network graph model & resource state generator
│   ├── protocols/
│   │   ├── teleportation.py         # Single-hop & multi-hop LOCC quantum teleportation
│   │   ├── controlled_unitaries.py  # C_{l;n} and 3-qubit fully controlled C_{l,m;n} via LOCC
│   │   ├── butterfly_protocol.py    # 7-stage deterministic Butterfly network protocol (Appendix D)
│   │   └── grail_protocol.py        # 3-CNOT Kraus-Cirac protocol on Grail network (Section V)
│   ├── analysis/
│   │   ├── schmidt.py               # State Schmidt rank, Operator Schmidt rank (OP#), Kraus-Cirac (KC#)
│   │   ├── slocc_invariants.py      # Verstraete 9-family analysis & Theorem 5 rank {4,2,2} verification
│   │   └── implementability.py      # Deterministic / SLOCC implementability tests for cluster networks
│   └── simulations/
│       ├── run_butterfly_demo.py    # Butterfly network benchmark across Weyl chamber
│       ├── run_grail_demo.py        # Grail network benchmark across canonical gates
│       └── run_impossibility_demo.py# Theoretical implementability & SWAP impossibility demo
├── tests/
│   ├── test_locc_teleportation.py
│   ├── test_controlled_unitaries.py
│   ├── test_butterfly_protocol.py
│   ├── test_grail_protocol.py
│   └── test_schmidt_and_invariants.py
├── main.py                          # Unified CLI runner and test orchestrator
└── README.md
```

---

## 🚀 Quick Start

### Requirements
- Python 3.8+
- `numpy`, `scipy`

```bash
pip install numpy scipy
```

### Running the Entire Suite
```bash
python main.py
```

### Running Individual Demos
```bash
# Run Butterfly Network Simulation
python -m quantum_network_coding.simulations.run_butterfly_demo

# Run Grail Network Simulation
python -m quantum_network_coding.simulations.run_grail_demo

# Run Impossibility Analysis
python -m quantum_network_coding.simulations.run_impossibility_demo
```

### Running Unit Tests
```bash
python -m unittest discover tests
```

---

## 🔬 Mathematical Formulas & Protocol Mapping

### 1. Kraus-Cirac Decomposition & Weyl Chamber
Any 2-qubit unitary $U \in U(4)$ can be expressed as:
$$U = (u \otimes u') e^{i(x X\otimes X + y Y\otimes Y + z Z\otimes Z)} (w \otimes w')$$
where $0 \le z \le y \le x \le \pi/4$. The Kraus-Cirac number $KC\#(U)$ is the number of non-zero parameters among $(x, y, z)$.

### 2. Butterfly Protocol (Theorem 1 & Appendix D)
1. Input state: $|\psi\rangle_{1,3} |0\rangle_2$.
2. Stage (ii): $H_1, H_2, H_3$.
3. Stage (iii): $C_{1,3;2}$ with $u^{(00)}=u^{(11)}=I, u^{(01)}=u^{(10)}=Z$.
4. Stage (iv): $H_1, H_2, H_3$ and $X_1, X_3$.
5. Stage (v): $C'_{1,3;2}$ with $w^{(ab)}$ parameterized by $y, z$.
6. Stage (vi): Single-qubit rotation $u(x)$ on node 2.
7. Stage (vii): Measurement on wire 2 with classical feedforward $X$ correction on wires 1 and 3 if $k=1$.

Deterministic output: $U_{global}(x, y, z)|\psi\rangle_{1,3}$.
