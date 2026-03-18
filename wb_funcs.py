"""
functions to implement hhl in workbench
"""

from scipy.linalg import expm
import numpy as np
from numpy import pi
from psiqworkbench import QPU, Qubits, Qubrick
from psiqworkbench.qubricks import QFT, Matrix
from workbench_algorithms import QPE


def make_unitary(A: np.ndarray, t: float, power: int = 1) -> np.ndarray:
    """Return e^{i A t power} via matrix exponential."""
    return expm(1j * A * t * power)



class HHLUnitary(Qubrick):
    """Here we build the unitary gate using Matrix() qubrick, similar to 
    building in qiskit using UnitaryGate.
    """

    def __init__(self, A: np.ndarray, t: float, **kwargs):
        self.A = A
        self.t = t
        super().__init__(**kwargs)

    def _compute(self, psi, ctrl=0, compute_iterations: int = 1):
        U_pow = make_unitary(self.A, self.t, power=compute_iterations)
        mat   = Matrix()
        mat.compute(U_pow, psi, ctrl)


def encode_b(
    b_reg: Qubits,
    b_vec: np.ndarray,
) -> None:
    """encodes b vector into b register using ry gate
    """
    b_norm  = b_vec / np.linalg.norm(b_vec)
    theta_b = 2 * np.arccos(np.clip(b_norm[0], -1, 1))
    b_reg.ry(np.rad2deg(theta_b))


def apply_qpe(
    b_reg: Qubits,
    clock: Qubits,
    A: np.ndarray,
    t: float,
) -> QFT:
    """Multiplies b register by even powers of the unitary for each clock register as part of QPE algorithm
    Uses make_unitary() to obtain e^{i A t power} and creates gate using UnitaryGate()
    Applies QFT using built-in qft function from workbench.
    
    Note: we would have used built-in QPE() from workbench, but we could not get it to work properly.
    """
    n_l = len(clock)
    clock.had()
    for j in range(n_l):
        U_pow = make_unitary(A, t, 2**j)
        mat   = Matrix()
        mat.compute(U_pow, b_reg, condition_qubits=clock[j])
    iqft = QFT(dagger=True)
    iqft.compute(clock)
    return iqft


def apply_ancilla_rotation(
    clock: Qubits,
    ancilla: Qubits,
    A: np.ndarray,
    t: float,
    C: float | None = None,
) -> None:
    """
    Applies rotation to encode eigenvalues into the ancilla qubit. 
    We compute the C scaling factor using eigenvalues. In actual implementation, a different method would have to be used.
    For example, C could be set if we somehow computed the lower bound of the lowest eigenvalue. This might reduce efficiency of 
    HHL algo and would require more shots to obtain the |1> state in the ancilla qubit after reading. 
    """
    n_l     = len(clock)
    eigvals = np.linalg.eigvalsh(A)
    if C is None:
        C = 0.9 * min(abs(eigvals))

    for m in range(1, 2**n_l):
        lam_m = m * (2 * pi) / (t * 2**n_l)
        if abs(C / lam_m) > 1.0:
            continue
        angle = 2 * np.arcsin(C / lam_m)
        bits  = format(m, f'0{n_l}b')

        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                clock[i].x()
        ancilla[0].ry(np.rad2deg(angle), cond=clock)
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                clock[i].x()

def apply_inverse_qpe(
    qpe_or_iqft,
    b_reg: Qubits,
    clock: Qubits,
    A: np.ndarray,
    t: float,
) -> None:
    """Call uncompute for the QFT and then manually uncomputes unitaries
    """
    
    iqft = qpe_or_iqft
    iqft.uncompute()
    n_l = len(clock)
    for j in range(n_l):
        U_pow_inv = make_unitary(A, t, -2**j)
        mat       = Matrix()
        mat.compute(U_pow_inv, b_reg, condition_qubits=clock[j])
    clock.had()



def extract_solution(
    qpu: QPU,
    n_b: int,
    n_l: int,
    n_a: int,
    A: np.ndarray,
    b_vec: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Pull the state vector and extract the HHL solution amplitudes.

    Post-selects on ancilla=1 and clock=0 to recover the b-register
    amplitudes proportional to A^{-1}|b>.

    Args:
        qpu:     QPU instance.
        n_b:     Number of b-register qubits (1 for 2x2 systems).
        n_l:     Number of clock qubits.
        n_a:     Number of ancilla qubits (1).
        A:       Hermitian matrix.
        b_vec:   Right-hand side vector.
        verbose: Whether to print comparison with classical solution.

    Returns:
        Normalised quantum solution vector x_q as a numpy array.
    """
    sv            = qpu.pull_state()
    ancilla_shift = n_b + n_l   # ancilla occupies the highest bits

    amp0 = sv[(1 << ancilla_shift) | 0]   # ancilla=1, clock=0, b=0
    amp1 = sv[(1 << ancilla_shift) | 1]   # ancilla=1, clock=0, b=1

    x_q      = np.array([abs(amp0), abs(amp1)])
    x_q      = x_q / np.linalg.norm(x_q)

    if verbose:
        x_classical  = np.linalg.solve(A, b_vec)
        x_c_norm     = x_classical / np.linalg.norm(x_classical)
        fidelity     = max(
            np.dot(x_q,  x_c_norm)**2,
            np.dot(x_q, -x_c_norm)**2,
        )
        print(f"\nQuantum  |x⟩ (normalized): [{x_q[0]:.4f}, {x_q[1]:.4f}]")
        print(f"Classical x  (normalized): [{x_c_norm[0]:.4f}, {x_c_norm[1]:.4f}]")
        print(f"Ratio x[0]/x[1] quantum:   {x_q[0]/x_q[1]:.4f}")
        print(f"Ratio x[0]/x[1] classical: {x_c_norm[0]/x_c_norm[1]:.4f}")
        print(f"Fidelity |⟨x_q|x_c⟩|²:    {fidelity:.4f}")

    return x_q