"""
hhl_main.py
-----------
Full HHL circuit assembled from modular steps in hhl_steps.py.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from numpy import pi

from qiskit_funcs import (
    make_unitary,
    encode_b,
    apply_qpe_unitaries,
    apply_inverse_qft,
    apply_ancilla_rotation,
    apply_forward_qft,
    apply_inverse_qpe_unitaries,
    print_sv,
)

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
A     = np.array([[3, 1], [1, 3]])
b_vec = np.array([1, 2], dtype=float)
n_l   = 4   # clock qubits

eigvals_true = np.linalg.eigvalsh(A)
t            = 1 * (2 * pi) / (min(abs(eigvals_true)) * 2**n_l)
x_classical  = np.linalg.solve(A, b_vec)
kappa        = max(abs(eigvals_true)) / min(abs(eigvals_true))

print("=" * 50)
print("SYSTEM BEING SOLVED: Ax = b")
print("=" * 50)
print(f"\nA =\n  {A[0]}\n  {A[1]}")
print(f"b = {b_vec}")
print(f"\nEigenvalues: {eigvals_true}")
print(f"Condition number κ = {kappa:.4f}")
print(f"t = {t:.6f}")
print(f"\nClassical solution:              x = {x_classical}")
print(f"Classical solution (normalized): x = {x_classical / np.linalg.norm(x_classical)}")
print(f"Classical ratio x[0]/x[1] = {x_classical[0]/x_classical[1]:.4f}")

# ---------------------------------------------------------------------------
# Build circuit
# ---------------------------------------------------------------------------
n_a, n_b    = 1, 1
b_reg       = QuantumRegister(n_b, name="b")
clock       = QuantumRegister(n_l, name="clock")
ancilla     = QuantumRegister(n_a, name="ancilla")
measurement = ClassicalRegister(2,  name="c")
hhl         = QuantumCircuit(ancilla, clock, b_reg, measurement)

hhl.barrier(label="ψ0")
encode_b(hhl, b_reg, b_vec)
hhl.barrier(label="ψ1")

apply_qpe_unitaries(hhl, clock, b_reg, A, t)
hhl.barrier()

apply_inverse_qft(hhl, clock)
hhl.barrier(label="ψ2")

apply_ancilla_rotation(hhl, clock, ancilla, A, t)
hhl.barrier(label="ψ3")

hhl.measure(ancilla, measurement[0])
hhl.barrier(label="ψ4")

apply_forward_qft(hhl, clock)
hhl.barrier()

apply_inverse_qpe_unitaries(hhl, clock, b_reg, A, t)
hhl.barrier(label="ψ5")

hhl.measure(b_reg, measurement[1])

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
sim    = AerSimulator()
job    = sim.run(transpile(hhl, sim), shots=8192)
counts = job.result().get_counts()

print("\n" + "=" * 50)
print("SIMULATION RESULTS (8192 shots)")
print("=" * 50)
print("\nRaw counts:")
for k, v in sorted(counts.items()):
    print(f"  |{k}⟩ : {v:5d}  ({100*v/8192:.1f}%)")

# ---------------------------------------------------------------------------
# Extract solution
# ---------------------------------------------------------------------------
post_1_0        = counts.get('01', 0)
post_1_1        = counts.get('11', 0)
total_ancilla_1 = post_1_0 + post_1_1

print(f"\nPost-selected on ancilla=1:")
print(f"  |b=0, ancilla=1⟩ : {post_1_0:5d}  ({100*post_1_0/8192:.1f}%)")
print(f"  |b=1, ancilla=1⟩ : {post_1_1:5d}  ({100*post_1_1/8192:.1f}%)")
print(f"  P(ancilla=1) = {100*total_ancilla_1/8192:.1f}%")

print("\n" + "=" * 50)
print("SOLUTION COMPARISON")
print("=" * 50)

if total_ancilla_1 > 0:
    amp0     = np.sqrt(post_1_0 / total_ancilla_1)
    amp1     = np.sqrt(post_1_1 / total_ancilla_1)
    x_q      = np.array([amp0, amp1])
    x_q      = x_q / np.linalg.norm(x_q)
    x_c_norm = x_classical / np.linalg.norm(x_classical)
    print("Qiskit solution:")
    print(f"\nQuantum  |x⟩ (normalized): [{x_q[0]:.4f}, {x_q[1]:.4f}]")
    print(f"Classical x  (normalized): [{x_c_norm[0]:.4f}, {x_c_norm[1]:.4f}]")
    print(f"\nQuantum  ratio x[0]/x[1] = {x_q[0]/x_q[1]:.4f}")
    print(f"Classical ratio x[0]/x[1] = {x_c_norm[0]/x_c_norm[1]:.4f}")
    fidelity = np.dot(x_q, np.abs(x_c_norm))**2
    print(f"\nFidelity |⟨x_q|x_c⟩|² = {fidelity:.4f}")
    print("\nNote: sign of amplitudes is lost — measurement gives |amplitude|² only.")
    print("Full sign recovery requires quantum state tomography.")