#################### importing ####################################
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.library import QFT, UnitaryGate, RYGate
from scipy.linalg import expm
import numpy as np
from numpy import pi

#################### define problem ###############################
A = np.array([[3, 1],
              [1, 2]])
b_vec = np.array([0, 1], dtype=float)

#################### compute unitaries ############################
def make_unitary(A, t, power=1):
    """Compute e^(i * A * t * power) via direct matrix exponential."""
    return expm(1j * A * t * power)

t      = pi / 2
U1     = make_unitary(A, t,  power= 1)
U2     = make_unitary(A, t,  power= 2)
U1_inv = make_unitary(A, t,  power=-1)   # BUG FIX 2: was reusing U1/U2
U2_inv = make_unitary(A, t,  power=-2)

#################### problem info ##################################
eigvals     = np.linalg.eigvalsh(A)
kappa       = max(abs(eigvals)) / min(abs(eigvals))
x_classical = np.linalg.solve(A, b_vec)

print("=" * 50)
print("SYSTEM BEING SOLVED: Ax = b")
print("=" * 50)
print(f"\nA =\n  {A[0]}\n  {A[1]}")
print(f"b = {b_vec}")
print(f"\nEigenvalues: {eigvals}")
print(f"Condition number κ = {kappa:.4f}")
print(f"\nClassical solution:              x = {x_classical}")
print(f"Classical solution (normalized): x = {x_classical / np.linalg.norm(x_classical)}")
print(f"Classical ratio x[0]/x[1] = {x_classical[0]/x_classical[1]:.4f}")

#################### initializing #################################
n_a, n_l, n_b = 1, 2, 1
b_reg       = QuantumRegister(n_b, name="b")
clock       = QuantumRegister(n_l, name="clock")
ancilla     = QuantumRegister(n_a, name="ancilla")
measurement = ClassicalRegister(2,  name="c")
hhl         = QuantumCircuit(ancilla, clock, b_reg, measurement)

#################### encoding b ###################################
# BUG FIX 1: atan2(b[0], b[1]) gave wrong angle for b=[0,1]
# Correct: RY(theta) where theta = 2*arccos(b_norm[0])
b_norm  = b_vec / np.linalg.norm(b_vec)
theta_b = 2 * np.arccos(np.clip(b_norm[0], -1, 1))
print(f"\n|b⟩ encoding: RY({theta_b:.4f}) → [{np.cos(theta_b/2):.4f}, {np.sin(theta_b/2):.4f}]")

hhl.barrier(label="$\\psi_0$")
hhl.ry(theta_b, b_reg)
hhl.barrier(label="$\\psi_1$")

#################### quantum phase estimation #####################
hhl.h(clock)

ctrl_U1 = UnitaryGate(U1, label="U").control(1)
ctrl_U2 = UnitaryGate(U2, label="U²").control(1)
hhl.append(ctrl_U1, [clock[0], b_reg[0]])
hhl.append(ctrl_U2, [clock[1], b_reg[0]])
hhl.barrier()

#################### inverse QFT ##################################
for j in range(n_l - 1, -1, -1):
    hhl.h(clock[j])
    for k in range(j - 1, -1, -1):
        angle = -pi / (2 ** (j - k))
        hhl.cp(angle, clock[j], clock[k])
hhl.swap(clock[0], clock[1])
hhl.barrier(label="$\\psi_2$")

#################### ancilla rotation #############################
# BUG FIX 3: angles must be computed from eigenvalues, not hardcoded.
# For clock bin m: lambda_m = m * 2π / (t * 2^n_clock)
# Ancilla angle: 2 * arcsin(C / lambda_m)
C = 0.9 * min(abs(eigvals))   # scaling constant < smallest eigenvalue

print("\nAncilla rotation angles:")
for m in range(1, 2**n_l):
    lam_m = m * (2 * pi) / (t * 2**n_l)
    if abs(C / lam_m) <= 1.0:
        angle = 2 * np.arcsin(C / lam_m)
        bits  = format(m, f'0{n_l}b')
        print(f"  m={m} (bits={bits}): λ={lam_m:.4f}, angle={angle:.4f} rad")

        # Build CCRy: flip qubits where bit='0' so gate fires on correct pattern
        ccry = RYGate(angle).control(n_l)
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                hhl.x(clock[i])
        hhl.append(ccry, [*clock, ancilla[0]])
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                hhl.x(clock[i])

hhl.barrier(label="$\\psi_3$")

#################### ancilla measurement ##########################
hhl.measure(ancilla, measurement[0])
hhl.barrier(label="$\\psi_4$")

#################### inverse QPE ##################################
hhl.swap(clock[0], clock[1])
for j in range(n_l):
    hhl.h(clock[j])
    for k in range(j + 1, n_l):
        angle = pi / (2 ** (k - j))
        hhl.cp(angle, clock[j], clock[k])
hhl.barrier()

# BUG FIX 2: use U_inv gates, not U gates
ctrl_U2_inv = UnitaryGate(U2_inv, label="U²†").control(1)
ctrl_U1_inv = UnitaryGate(U1_inv, label="U†").control(1)
hhl.append(ctrl_U2_inv, [clock[1], b_reg[0]])
hhl.append(ctrl_U1_inv, [clock[0], b_reg[0]])
hhl.h(clock)
hhl.barrier(label="$\\psi_5$")

#################### measure x ####################################
hhl.measure(b_reg, measurement[1])

#################### print circuit ################################
print("\n" + "=" * 50)
print("CIRCUIT")
print("=" * 50)
print(hhl.draw('text'))

#################### simulate #####################################
sim    = AerSimulator()
job    = sim.run(transpile(hhl, sim), shots=8192)
counts = job.result().get_counts()

print("\n" + "=" * 50)
print("SIMULATION RESULTS (8192 shots)")
print("=" * 50)
print("\nRaw counts (bit string: 'b_bit ancilla_bit'):")
for k, v in sorted(counts.items()):
    print(f"  |{k}⟩ : {v:5d}  ({100*v/8192:.1f}%)")

#################### post-select on ancilla=1 #####################
post_1_0       = counts.get('01', 0)
post_1_1       = counts.get('11', 0)
total_ancilla_1 = post_1_0 + post_1_1

print(f"\nPost-selected on ancilla=1:")
print(f"  |b=0, ancilla=1⟩ : {post_1_0:5d}  ({100*post_1_0/8192:.1f}%)")
print(f"  |b=1, ancilla=1⟩ : {post_1_1:5d}  ({100*post_1_1/8192:.1f}%)")
print(f"  P(ancilla=1) = {100*total_ancilla_1/8192:.1f}%")

#################### extract solution #############################
print("\n" + "=" * 50)
print("SOLUTION COMPARISON")
print("=" * 50)

if total_ancilla_1 > 0:
    amp0     = np.sqrt(post_1_0 / total_ancilla_1)
    amp1     = np.sqrt(post_1_1 / total_ancilla_1)
    x_q      = np.array([amp0, amp1])
    x_q      = x_q / np.linalg.norm(x_q)
    x_c_norm = x_classical / np.linalg.norm(x_classical)

    print(f"\nQuantum  |x⟩ (normalized): [{x_q[0]:.4f}, {x_q[1]:.4f}]")
    print(f"Classical x  (normalized): [{x_c_norm[0]:.4f}, {x_c_norm[1]:.4f}]")
    print(f"\nQuantum  ratio x[0]/x[1] = {x_q[0]/x_q[1]:.4f}")
    print(f"Classical ratio x[0]/x[1] = {x_c_norm[0]/x_c_norm[1]:.4f}")
    fidelity = np.dot(x_q, np.abs(x_c_norm))**2
    print(f"\nFidelity |⟨x_q|x_c⟩|² = {fidelity:.4f}")
    print("\nNote: sign of amplitudes is lost — measurement gives |amplitude|² only.")
    print("Full sign recovery requires quantum state tomography.")