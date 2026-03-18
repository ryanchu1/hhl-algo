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
              [1, 3]])
b_vec = np.array([1, 2], dtype=float)

#################### compute unitaries ############################
def make_unitary(A, t, power=1):
 
    """Compute e^(i * A * t * power) via direct matrix exponential."""
    return expm(1j * A * t * power)

n_l = 4  # clock qubits — change this to scale QPE precision

# t chosen so eigenvalues map to integer bins:
#   bin width = 2pi / (t * 2^n_l)
#   eigenvalues of A are 1 and 2, so t = 2pi / 2^n_l maps them to bins 1 and 2.
#   with n_l=3 we have 8 bins, bin width = 2pi/8 = pi/4.
#   lambda=1 -> m=1*(pi/4)^-1 ... re-deriving: lam_m = m*2pi/(t*8)
#   set t so lam_1 = eigval_min: t = 1*2pi / (eigval_min * 2^n_l)
eigvals_true = np.linalg.eigvalsh(A)
# map smallest eigenvalue to bin 1, largest to bin 2 (same relative mapping as before)
t = 1 * (2 * pi) / (min(abs(eigvals_true)) * 2**n_l)

#################### problem info ##################################
kappa       = max(abs(eigvals_true)) / min(abs(eigvals_true))
x_classical = np.linalg.solve(A, b_vec)

print("=" * 50)
print("SYSTEM BEING SOLVED: Ax = b")
print("=" * 50)
print(f"\nA =\n  {A[0]}\n  {A[1]}")
print(f"b = {b_vec}")
print(f"\nEigenvalues: {eigvals_true}")
print(f"Condition number κ = {kappa:.4f}")
print(f"t = {t:.6f}  (maps eigvals to integer bins with {n_l} clock qubits)")
print(f"\nClassical solution:              x = {x_classical}")
print(f"Classical solution (normalized): x = {x_classical / np.linalg.norm(x_classical)}")
print(f"Classical ratio x[0]/x[1] = {x_classical[0]/x_classical[1]:.4f}")

#################### verify bin mapping ###########################
print(f"\nBin width = 2π / (t · 2^n_l) = {2*pi / (t * 2**n_l):.4f}")
print("Eigenvalue → bin mapping:")
for lam in eigvals_true:
    m = lam * t * 2**n_l / (2 * pi)
    print(f"  λ={lam:.4f} → m={m:.4f} {'✓ integer' if abs(m - round(m)) < 1e-9 else '✗ NOT integer — QPE will have error'}")

#################### initializing #################################
n_a, n_b = 1, 1
b_reg       = QuantumRegister(n_b,  name="b")
clock       = QuantumRegister(n_l,  name="clock")
ancilla     = QuantumRegister(n_a,  name="ancilla")
measurement = ClassicalRegister(2,  name="c")
hhl         = QuantumCircuit(ancilla, clock, b_reg, measurement)

#################### encoding b ###################################
b_norm  = b_vec / np.linalg.norm(b_vec)
theta_b = 2 * np.arccos(np.clip(b_norm[0], -1, 1))
print(f"\n|b⟩ encoding: RY({theta_b:.4f}) → [{np.cos(theta_b/2):.4f}, {np.sin(theta_b/2):.4f}]")

hhl.barrier(label="$\\psi_0$")
hhl.ry(theta_b, b_reg)
hhl.barrier(label="$\\psi_1$")
print("after encoding")
#################### quantum phase estimation #####################
hhl.h(clock)

# Apply controlled-U^(2^j) for j = 0 .. n_l-1
for j in range(n_l):
    power = 2**j
    U_pow = make_unitary(A, t, power=power)
    ctrl_U = UnitaryGate(U_pow, label=f"U^{power}").control(1)
    hhl.append(ctrl_U, [clock[j], b_reg[0]])

hhl.barrier()

#################### inverse QFT ##################################
for j in range(n_l - 1, -1, -1):
    hhl.h(clock[j])
    for k in range(j - 1, -1, -1):
        angle = -pi / (2 ** (j - k))
        hhl.cp(angle, clock[j], clock[k])

# Bit-reversal swaps for n_l=3: swap clock[0] and clock[2]
for i in range(n_l // 2):
    hhl.swap(clock[i], clock[n_l - 1 - i])

hhl.barrier(label="$\\psi_2$")

#################### ancilla rotation #############################
C = 0.9 * min(abs(eigvals_true))

print("\nAncilla rotation angles:")
for m in range(1, 2**n_l):
    lam_m = m * (2 * pi) / (t * 2**n_l)
    if abs(C / lam_m) <= 1.0:
        angle = 2 * np.arcsin(C / lam_m)
        bits  = format(m, f'0{n_l}b')
        print(f"  m={m} (bits={bits}): λ={lam_m:.4f}, angle={angle:.4f} rad")

        # CCRy controlled on all n_l clock qubits
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
# Undo bit-reversal swaps
for i in range(n_l // 2):
    hhl.swap(clock[i], clock[n_l - 1 - i])

# Inverse QFT (forward direction)
for j in range(n_l):
    hhl.h(clock[j])
    for k in range(j + 1, n_l):
        angle = pi / (2 ** (k - j))
        hhl.cp(angle, clock[j], clock[k])

hhl.barrier()

# Apply controlled-U^(-2^j) in reverse order: j = n_l-1 .. 0
for j in range(n_l - 1, -1, -1):
    power = 2**j
    U_pow_inv = make_unitary(A, t, power=-power)
    ctrl_U_inv = UnitaryGate(U_pow_inv, label=f"U^{power}†").control(1)
    hhl.append(ctrl_U_inv, [clock[j], b_reg[0]])

hhl.h(clock)
hhl.barrier(label="$\\psi_5$")


#################### measure x ####################################
hhl.measure(b_reg, measurement[1])

#################### print circuit ################################
print("\n" + "=" * 50)
print("CIRCUIT")
print("=" * 50)
# print(hhl.draw('text'))

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
post_1_0        = counts.get('01', 0)
post_1_1        = counts.get('11', 0)
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