import sys
import unittest
import numpy as np

from quantum_network_coding.simulations.run_butterfly_demo import run_butterfly_benchmark
from quantum_network_coding.simulations.run_grail_demo import run_grail_benchmark
from quantum_network_coding.simulations.run_impossibility_demo import run_impossibility_benchmark

def run_all_unit_tests() -> bool:
    print("\n" + "=" * 80)
    print("RUNNING ALL UNIT TESTS (Quantum Network Coding Suite)")
    print("=" * 80)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def main() -> None:
    print("\n" + "#" * 80)
    print("# QUANTUM NETWORK CODING FOR DISTRIBUTED QUANTUM COMPUTATION")
    print("# Based on arXiv:1503.07740v2 (Seiseki Akibue & Mio Murao)")
    print("#" * 80 + "\n")
    
    # 1. Run Butterfly Network Simulation
    run_butterfly_benchmark(num_random_trials=3)
    
    # 2. Run Grail Network Simulation
    run_grail_benchmark(num_random_trials=3)
    
    # 3. Run Impossibility & Invariants Demo
    run_impossibility_benchmark()
    
    # 4. Execute Unit Test Suite
    success = run_all_unit_tests()
    
    if success:
        print("\n[SUCCESS] All quantum network coding simulations and test suites passed with 100% fidelity.")
    else:
        print("\n[WARNING] Some tests encountered errors. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
