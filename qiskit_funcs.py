"""
functions to implement hhl in qiskit
"""

from qiskit.circuit.library import UnitaryGate, RYGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
import numpy as np
from numpy import pi



def make_unitary(A: np.ndarray, t: float, power: int = 1) -> np.ndarray:
    """Return e^{i A t power} via matrix exponential."""
    return expm(1j * A * t * power)



def encode_b(
    circuit: QuantumCircuit,
    b_reg: QuantumRegister,
    b_vec: np.ndarray,
) -> QuantumCircuit:
    """
    encodes b vector into b register using ry gate
    """
    b_norm  = b_vec / np.linalg.norm(b_vec)
    theta_b = 2 * np.arccos(np.clip(b_norm[0], -1, 1))
    circuit.ry(theta_b, b_reg)
    return circuit


def apply_qpe_unitaries(
    circuit: QuantumCircuit,
    clock: QuantumRegister,
    b_reg: QuantumRegister,
    A: np.ndarray,
    t: float,
) -> QuantumCircuit:
    """
    Multiplies b register by even powers of the unitary for each clock register as part of QPE algorithm
    Uses make_unitary() to obtain e^{i A t power} and creates gate using UnitaryGate()
    """
    circuit.h(clock)
    for j in range(len(clock)):
        power  = 2**j
        U_pow  = make_unitary(A, t, power=power)
        ctrl_U = UnitaryGate(U_pow, label=f"U^{power}").control(1)
        circuit.append(ctrl_U, [clock[j], b_reg[0]])
    return circuit


def apply_inverse_qft(
    circuit: QuantumCircuit,
    clock: QuantumRegister,
) -> QuantumCircuit:
    """
    inverse qft for QPE algorithm
    """
    n_l = len(clock)
    for j in range(n_l - 1, -1, -1):
        circuit.h(clock[j])
        for k in range(j - 1, -1, -1):
            angle = -pi / (2 ** (j - k))
            circuit.cp(angle, clock[j], clock[k])
    for i in range(n_l // 2):
        circuit.swap(clock[i], clock[n_l - 1 - i])
    return circuit

def apply_ancilla_rotation(
    circuit: QuantumCircuit,
    clock: QuantumRegister,
    ancilla: QuantumRegister,
    A: np.ndarray,
    t: float,
    C: float | None = None,
) -> QuantumCircuit:
    """
    Applies rotation to encode eigenvalues into the ancilla qubit. 
    We compute the C scaling factor using eigenvalues. In actual implementation, a different method would have to be used.
    For example, C could be set if we somehow computed the lower bound of the lowest eigenvalue. This might reduce efficiency of 
    HHL algo and would require more shots to obtain the |1> state in the ancilla qubit after reading. 
    """
    n_l       = len(clock)
    eigvals   = np.linalg.eigvalsh(A)
    if C is None:
        C = 0.9 * min(abs(eigvals))

    for m in range(1, 2**n_l):
        lam_m = m * (2 * pi) / (t * 2**n_l)
        if abs(C / lam_m) > 1.0:
            continue
        angle = 2 * np.arcsin(C / lam_m)
        bits  = format(m, f'0{n_l}b')

        # X-flip qubits where the bit pattern has '0' so cond becomes |1..1>
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                circuit.x(clock[i])
        ccry = RYGate(angle).control(n_l)
        circuit.append(ccry, [*clock, ancilla[0]])
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                circuit.x(clock[i])

    return circuit

def apply_forward_qft(
    circuit: QuantumCircuit,
    clock: QuantumRegister,
) -> QuantumCircuit:
    """
    uncompute QPE step
    """
    n_l = len(clock)
    # Undo bit-reversal swaps first
    for i in range(n_l // 2):
        circuit.swap(clock[i], clock[n_l - 1 - i])
    for j in range(n_l):
        circuit.h(clock[j])
        for k in range(j + 1, n_l):
            angle = pi / (2 ** (k - j))
            circuit.cp(angle, clock[j], clock[k])
    return circuit


def apply_inverse_qpe_unitaries(
    circuit: QuantumCircuit,
    clock: QuantumRegister,
    b_reg: QuantumRegister,
    A: np.ndarray,
    t: float,
) -> QuantumCircuit:
    """
    uncompute QPE step
    """
    n_l = len(clock)
    for j in range(n_l - 1, -1, -1):
        power       = 2**j
        U_pow_inv   = make_unitary(A, t, power=-power)
        ctrl_U_inv  = UnitaryGate(U_pow_inv, label=f"U^{power}†").control(1)
        circuit.append(ctrl_U_inv, [clock[j], b_reg[0]])
    circuit.h(clock)
    return circuit


def print_sv(circuit: QuantumCircuit, label: str = "") -> None:
    """Print non-negligible amplitudes of the current state vector."""
    sv = Statevector.from_instruction(circuit)
    print(f"\n--- {label} ---")
    for i, amp in enumerate(sv):
        if abs(amp) > 1e-6:
            bits = format(i, f'0{circuit.num_qubits}b')
            print(f"  |{bits}⟩  {amp:.4f}  (|amp|={abs(amp):.4f})")