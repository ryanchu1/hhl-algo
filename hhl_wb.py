#################### importing ####################################


from scipy.linalg import expm
import numpy as np
from numpy import pi
from psiqworkbench import QPU, Qubits, Units, Qubrick

from psiqworkbench.qubricks import QFT
from workbench_algorithms import QPE
from psiqworkbench.qubricks import Matrix
#################### define problem ###############################
A = np.array([[3, 1],
              [1, 3]])
b_vec = np.array([1, 0], dtype=float)

#################### compute unitaries ############################
def make_unitary(A, t, power=1):
 
    """Compute e^(i * A * t * power) via direct matrix exponential."""
    return expm(1j * A * t * power)

n_l = 5  # clock qubits — change this to scale QPE precision

# t chosen so eigenvalues map to integer bins:
#   bin width = 2pi / (t * 2^n_l)
#   eigenvalues of A are 1 and 2, so t = 2pi / 2^n_l maps them to bins 1 and 2.
#   with n_l=3 we have 8 bins, bin width = 2pi/8 = pi/4.
#   lambda=1 -> m=1*(pi/4)^-1 ... re-deriving: lam_m = m*2pi/(t*8)
#   set t so lam_1 = eigval_min: t = 1*2pi / (eigval_min * 2^n_l)
eigvals_true = np.linalg.eigvalsh(A)

# Find best integer bin assignment
ratio = max(eigvals_true) / min(eigvals_true)  # ≈ 3.125

# With n_l=3 we have bins 1..7. Find m1,m2 with m2/m1 closest to ratio
best_err = np.inf
best_m1, best_m2 = 1, 3
for m1 in range(1, 2**n_l):
    for m2 in range(m1+1, 2**n_l):
        err = abs(m2/m1 - ratio)
        if err < best_err:
            best_err = err
            best_m1, best_m2 = m1, m2

print(f"Best bin assignment: λ_min→m={best_m1}, λ_max→m={best_m2}, ratio error={best_err:.6f}")

# Set t so λ_min maps to best_m1
t = best_m1 * (2 * pi) / (min(abs(eigvals_true)) * 2**n_l)
# map smallest eigenvalue to bin 1, largest to bin 2 (same relative mapping as before)
# t = 1 * (2 * pi) / (min(abs(eigvals_true)) * 2**n_l)

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
qpu = QPU(num_qubits=n_a + n_b + n_l)
b_reg  = Qubits(n_b, name="b", qpu=qpu)
clock   = Qubits(n_l, name="clock", qpu=qpu)
ancilla  = Qubits(n_a, name="ancilla", qpu=qpu)


#################### encoding b ###################################
b_norm  = b_vec / np.linalg.norm(b_vec)
theta_b = 2 * np.arccos(np.clip(b_norm[0], -1, 1))
print(f"\n|b⟩ encoding: RY({theta_b:.4f}) → [{np.cos(theta_b/2):.4f}, {np.sin(theta_b/2):.4f}]")

b_reg.ry(np.rad2deg(theta_b))
qpu.print_state_vector()
#################### quantum phase estimation #####################
U_pow = make_unitary(A, t, power=1)

class SimplePhaseUnitary(Qubrick):
    """Returns a Qubrickified ``Rz`` gate for debugging QPE routines.

    Allows for exact eigenphases to be implemented in QPE.

    Args:
        params (dict): Parameters that the Qubrick can access. May contain ``phase``.
    """

    def __init__(self, A, **kwargs):
        self.A = A
        super().__init__(**kwargs)

    def _compute(self, psi, ctrl=0):
        """Compute the dummy block encoding.

        Args:
            psi (Qubits): State register for the computation.
            ctrl (int, Qubits): Register to control the unitary on. Defaults to ``0``.
        """
        U_pow = make_unitary(self.A, t, power=1)
        mat = Matrix()
        mat.compute(U_pow,psi,ctrl)
unitary = SimplePhaseUnitary(A=A)
qpe = QPE(bits_of_precision=n_l, unitary=unitary)
qpe.compute(b_reg, clock)
# clock.had()

# # Apply controlled-U^(2^j) for j = 0 .. n_l-1
# for j in range(n_l):
#     power = 2**j
#     U_pow = make_unitary(A, t, power=power)
#     apply_matrix = Matrix()
#     apply_matrix.compute(U_pow, b_reg[0], condition_qubits=clock[j] )
#     # ctrl_U = UnitaryGate(U_pow, label=f"U^{power}").control(1)
#     # hhl.append(ctrl_U, [clock[j], b_reg[0]])
# print("applied U gates")
# qpu.print_state_vector()

# #################### inverse QFT ##################################
# from psiqworkbench import QFT
# qft = QFT()
# # for j in range(n_l - 1, -1, -1):
# #     clock[j].had()
# #     for k in range(j - 1, -1, -1):
# #         angle = -pi / (2 ** (j - k))
# #         clock[j].phase(np.rad2deg(angle), cond=clock[k])
# qft.compute(clock)
# print("applied QFT")
# qpu.print_state_vector()
# # Bit-reversal swaps for n_l=3: swap clock[0] and clock[2]
# for i in range(n_l // 2):
#     clock[i].swap(clock[n_l - 1 - i])


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
        # ccry = RYGate(angle).control(n_l)
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                clock[i].x()
        ancilla[0].ry(np.rad2deg(angle), cond=clock)
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                clock[i].x()
print("applied ancilla rotations")
qpu.print_state_vector()
# hhl.barrier(label="$\\psi_3$")

#################### ancilla measurement ##########################

# hhl.barrier(label="$\\psi_4$")

#################### inverse QPE ##################################
# Undo bit-reversal swaps
# for i in range(n_l // 2):
#     clock[i].swap(clock[n_l - 1 - i])

# Inverse QFT (forward direction)
# for j in range(n_l):
#     clock[j].had()
#     for k in range(j + 1, n_l):
#         angle = pi / (2 ** (k - j))
#         clock[j].phase(np.rad2deg(angle), cond=clock[k])
qpe.uncompute()


# Don't call read() — extract directly from state vector
sv = qpu.pull_state()  # VERIFY: exact method name
print("sv:", sv)
# post-select on clock=0, ancilla=1
# b is LSB, then clock, then ancilla is MSB
# ancilla=1, clock=00, b=0  →  1000 in binary = index 8
# ancilla=1, clock=00, b=1  →  1001 in binary = index 9
n_total = n_a + n_b + n_l
ancilla_shift = n_b + n_l   # ancilla is the highest bits

amp0 = sv[(1 << ancilla_shift) | 0]   # ancilla=1, b=0  →  index 8
amp1 = sv[(1 << ancilla_shift) | 1]   # ancilla=1, b=1  →  index 9

x_q = np.array([abs(amp0), abs(amp1)])
x_q = x_q / np.linalg.norm(x_q)

x_c_norm = x_classical / np.linalg.norm(x_classical)

print(f"\nQuantum  |x⟩ (normalized): [{x_q[0]:.4f}, {x_q[1]:.4f}]")
print(f"Classical x  (normalized): [{x_c_norm[0]:.4f}, {x_c_norm[1]:.4f}]")
print(f"Ratio x[0]/x[1] quantum:   {x_q[0]/x_q[1]:.4f}")
print(f"Ratio x[0]/x[1] classical: {x_c_norm[0]/x_c_norm[1]:.4f}")
fidelity = np.dot(x_q, np.abs(x_c_norm))**2
print(f"Fidelity |⟨x_q|x_c⟩|²:    {fidelity:.4f}")