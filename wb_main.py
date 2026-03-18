"""
hhl_main_workbench.py
---------------------
Full HHL circuit assembled from modular steps in hhl_steps_workbench.py.
"""

import numpy as np
from numpy import pi
from psiqworkbench import QPU, Qubits

from wb_funcs import (
    make_unitary,
    encode_b,
    apply_qpe,
    apply_ancilla_rotation,
    apply_inverse_qpe,
    extract_solution,
)

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
A     = np.array([[3, 1], [1, 6]])
b_vec = np.array([1, 2], dtype=float)
n_l   = 2   # clock qubits

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

def hhl():
    n_a, n_b = 1, 1
    qpu      = QPU(num_qubits=n_a + n_b + n_l)
    b_reg    = Qubits(n_b, name="b",       qpu=qpu)
    clock    = Qubits(n_l, name="clock",   qpu=qpu)
    ancilla  = Qubits(n_a, name="ancilla", qpu=qpu)


    encode_b(b_reg, b_vec)
    # print("\nafter encoding b:")
    # qpu.print_state_vector()

    iqft = apply_qpe(b_reg, clock, A, t)
    # print("\nafter QPE:")
    # qpu.print_state_vector()

    apply_ancilla_rotation(clock, ancilla, A, t)
    # print("\nafter ancilla rotation:")
    # qpu.print_state_vector()
    ancilla_readout = ancilla.read()
    print("ancilla: ", ancilla_readout)
    apply_inverse_qpe(iqft, b_reg, clock, A, t)
    # print("\nafter inverse QPE:")
    # qpu.print_state_vector()
    extract_solution(qpu, n_b, n_l, n_a, A, b_vec, verbose=True)
    b_readout = b_reg.read()
    print("ancilla: ", b_readout)
    return ancilla_readout, b_readout
    
    
b_0 = 0
b_1 = 0
num_shots = 8192
post_select_shots = 0
for _ in range(num_shots):
    anc, b = hhl()
    if anc == 1:
        post_select_shots +=1
        if b == 1:
            b_1+=1
        elif b == 0:
            b_0 +=1
amp0 = np.sqrt(b_0/post_select_shots)
amp1 = np.sqrt(b_1/post_select_shots)
x_q      = np.array([amp0, amp1])
x_q      = x_q / np.linalg.norm(x_q)
x_c_norm = x_classical / np.linalg.norm(x_classical)
print("Workbench solution:")

print(f"\nQuantum  |x⟩ (normalized): [{x_q[0]:.4f}, {x_q[1]:.4f}]")
print(f"Classical x  (normalized): [{x_c_norm[0]:.4f}, {x_c_norm[1]:.4f}]")
print(f"\nQuantum  ratio x[0]/x[1] = {x_q[0]/x_q[1]:.4f}")
print(f"Classical ratio x[0]/x[1] = {x_c_norm[0]/x_c_norm[1]:.4f}")
fidelity = np.dot(x_q, np.abs(x_c_norm))**2
print(f"\nFidelity |⟨x_q|x_c⟩|² = {fidelity:.4f}")
print("\nNote: sign of amplitudes is lost — measurement gives |amplitude|² only.")
print("Full sign recovery requires quantum state tomography.")
