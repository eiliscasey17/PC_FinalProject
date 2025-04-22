
import argparse

parser = argparse.ArgumentParser(description="MPI Clifford Circuit Simulation")
parser.add_argument("--qubits", type=int, default=20, help="Number of qubits")
parser.add_argument("--depth", type = int, default=15, help="Depth of circuit")
parser.add_argument("--scaling", type=int, default=1, help="Scaler")

args = parser.parse_args()

# Import necessary libraries
from mpi4py import MPI  # Import MPI
from qiskit.circuit.library import QuantumVolume
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.random import random_circuit
from qiskit.circuit.random import random_clifford_circuit
from qiskit.quantum_info import Clifford
import numpy as np
import time
import sys # To flush output



# --- MPI Initialization ---
# Get the MPI communicator, rank (process ID), and size (total number of processes)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

weak = args.scaling

# --- Simulation Parameters ---
qubits = args.qubits # Number of qubits
depth = args.depth * 3 # Depth of the circuit
seed = 42   # Seed for reproducibility
shots =1024 # Number of shots (number of times the circuit is executed)


experiment_name = "clifford_circuit_" + str(qubits) + "q_" + str(depth) + "d"


if weak:
    experiment_name += "_weak"
    
else:
    experiment_name += "_strong"
    shots = (shots * 8) //size

# --- Qiskit Setup (Executed by all processes, but AerSimulator handles distribution) ---
# Initialize the simulator
# Note: device='GPU' requires a system with MPI-aware CUDA setup across nodes.

sim = AerSimulator(
    method= "statevector",
    device='CPU', # Change to 'CPU' if running on a CPU cluster without MPI-aware GPU setup
    blocking_enable=True, # May help with GPU memory management
    blocking_qubits=27,
    max_parallel_shots = 1
)

# Create a quantum circuit with the specified number of qubits
circ = random_clifford_circuit(qubits, depth, seed=seed)
circ.measure_all()

# Transpile the circuit for the simulator
if rank == 0:
    print("\n","-" * 30, flush=True)
    print(experiment_name, flush=True)
    print(f"Starting transpilation for {qubits} qubits on {size} MPI processes...", flush=True)
    print(f"Number of shots per process: {shots}", flush=True)
    print(f"Total number of shots per process: {shots//size}", flush=True)
#######
comm.Barrier() # Synchronize before timing the run
start_time = time.time()
circ_transpiled = transpile(circ, sim)
comm.Barrier() # Synchronize after transpilation
end_time = time.time()
########

transpile_time = end_time - start_time



if rank == 0:
    print(f"Transpilation finished in {transpile_time:.2f} seconds.", flush=True)
    print("-" * 30, flush=True)
    print(f"Running simulation...", flush=True)

# --- Run Simulation (Parallel execution handled by AerSimulator) ---
comm.Barrier()
start_time = time.time()
result = sim.run(circ_transpiled, shots=shots).result() # shots added for clarity, though statevector doesn't strictly need them for$
comm.Barrier() # Ensure all processes finish before stopping timer
end_time = time.time()

simulation_time = end_time - start_time



# --- Process Results (Rank 0 handles output/plotting) ---
if rank == 0:
    print(f"Simulation finished in {simulation_time:.2f} seconds.", flush=True)
    print("-" * 30, flush=True)
    try:
        counts = result.get_counts(circ_transpiled)
        print("\n Counts obtained successfully.")
        #print(f"Counts: {counts}", flush=True)

        # Plot histogram of counts
        try:
            '''
            print("Histogram", flush=True)
            # plot_histogram function requires matplotlib
            import matplotlib.pyplot as plt

            fig = plot_histogram(
                #counts,
                #title=f'Circuit ({qubits} qubits) Counts - MPI ({size} processes)',
                #figsize= (15,10)
            )
            ax = fig.axes[0]

            ax.set_xlabel("Measurement outcome", fontsize=12)
            ax.set_ylabel("Counts", fontsize=12)

            plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)
            for label in ax.get_xticklabels():
                #label.set_verticalalignment('top')   # 'center' or 'bottom' also possible
                #label.set_y(-0.05)
            # Save the plot instead of showing interactively in MPI runs
            plot_filename = f"{experiment_name}_mpi_{size}p_hist.png"

            fig.savefig(plot_filename)
            print(f"Histogram saved to {plot_filename}", flush=True)

            '''
        except ImportError:
            print("Matplotlib not found. Cannot generate plot. Please install it: pip install matplotlib", flush=True)

        except Exception as e:
            print(f"An error occurred during plotting: {e}", flush=True)
    except Exception as e:
        print(f"Error occured getting counts: {e}", flush=True)
    print("-" * 30, flush=True)
    print("MPI Qiskit Run Complete.", flush=True)

comm.Barrier()


