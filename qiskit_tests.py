"""
test_hhl_steps.py
-----------------
Unit tests for each HHL step function in hhl_steps.py.

Run with:  pytest test_hhl_steps.py -v
"""

import numpy as np
import pytest
from numpy import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

from qiskit_funcs import (
    make_unitary,
    encode_b,
    apply_qpe_unitaries,
    apply_inverse_qft,
    apply_ancilla_rotation,
    apply_forward_qft,
    apply_inverse_qpe_unitaries,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

A_DIAG  = np.array([[3, 1], [1, 3]], dtype=float)   # eigenvalues 2, 4
N_L     = 2
EIGVALS = np.linalg.eigvalsh(A_DIAG)
T       = 1 * (2 * pi) / (min(abs(EIGVALS)) * 2**N_L)


def fresh_circuit(n_l=N_L):
    """Return a blank (ancilla, clock, b) circuit with no gates."""
    b_reg   = QuantumRegister(1,   name="b")
    clock   = QuantumRegister(n_l, name="clock")
    ancilla = QuantumRegister(1,   name="ancilla")
    circ    = QuantumCircuit(ancilla, clock, b_reg)
    return circ, ancilla, clock, b_reg


# ---------------------------------------------------------------------------
# make_unitary
# ---------------------------------------------------------------------------

class TestMakeUnitary:
    def test_returns_unitary_matrix(self):
        U = make_unitary(A_DIAG, T)
        assert U.shape == (2, 2)
        # U†U should be identity
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-10)

    def test_power_scaling(self):
        # e^{iAt*2} should equal (e^{iAt})^2
        U1 = make_unitary(A_DIAG, T, power=1)
        U2 = make_unitary(A_DIAG, T, power=2)
        np.testing.assert_allclose(U2, U1 @ U1, atol=1e-10)

    def test_inverse(self):
        U     = make_unitary(A_DIAG, T, power=1)
        U_inv = make_unitary(A_DIAG, T, power=-1)
        np.testing.assert_allclose(U @ U_inv, np.eye(2), atol=1e-10)

    def test_hermitian_input_gives_unitary(self):
        # Any Hermitian A should produce a unitary e^{iAt}
        A = np.array([[2, 1+1j], [1-1j, 3]])
        U = make_unitary(A, 1.0)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-10)


# ---------------------------------------------------------------------------
# encode_b
# ---------------------------------------------------------------------------

class TestEncodeB:
    def test_encodes_zero_vector(self):
        """b = [1, 0] should leave b_reg in |0>."""
        circ, _, _, b_reg = fresh_circuit()
        encode_b(circ, b_reg, np.array([1.0, 0.0]))
        sv = Statevector.from_instruction(circ)
        # |ancilla=0, clock=00, b=0> should have amplitude 1
        assert abs(sv[0]) == pytest.approx(1.0, abs=1e-6)

    def test_encodes_one_vector(self):
        """b = [0, 1] should put b_reg in |1>."""
        circ, _, _, b_reg = fresh_circuit()
        encode_b(circ, b_reg, np.array([0.0, 1.0]))
        sv = Statevector.from_instruction(circ)
        # b=1 is index 1 in the state vector (b is LSB)
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-6)

    def test_normalisation_preserved(self):
        """Total probability should remain 1 after encoding."""
        circ, _, _, b_reg = fresh_circuit()
        encode_b(circ, b_reg, np.array([1.0, 2.0]))
        sv = Statevector.from_instruction(circ)
        assert sum(abs(a)**2 for a in sv) == pytest.approx(1.0, abs=1e-6)

    def test_correct_amplitudes(self):
        """Amplitudes should match cos(theta/2) and sin(theta/2)."""
        b_vec  = np.array([1.0, 2.0])
        b_norm = b_vec / np.linalg.norm(b_vec)
        theta  = 2 * np.arccos(b_norm[0])
        circ, _, _, b_reg = fresh_circuit()
        encode_b(circ, b_reg, b_vec)
        sv = Statevector.from_instruction(circ)
        # b=0 amplitude (index 0)
        assert abs(sv[0]) == pytest.approx(np.cos(theta / 2), abs=1e-6)
        # b=1 amplitude (index 1)
        assert abs(sv[1]) == pytest.approx(np.sin(theta / 2), abs=1e-6)

    def test_returns_circuit(self):
        circ, _, _, b_reg = fresh_circuit()
        result = encode_b(circ, b_reg, np.array([1.0, 1.0]))
        assert result is circ


# ---------------------------------------------------------------------------
# apply_qpe_unitaries
# ---------------------------------------------------------------------------

class TestApplyQPEUnitaries:
    def test_clock_not_all_zero_after_qpe(self):
        """After QPE, amplitude should spread across clock bins."""
        circ, _, clock, b_reg = fresh_circuit()
        encode_b(circ, b_reg, np.array([1.0, 0.0]))
        apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        sv    = Statevector.from_instruction(circ)
        probs = np.abs(sv.data)**2
        # At least 2 clock bins should have non-negligible probability
        nonzero_bins = sum(1 for p in probs if p > 1e-6)
        assert nonzero_bins > 1

    def test_returns_circuit(self):
        circ, _, clock, b_reg = fresh_circuit()
        result = apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        assert result is circ

    def test_state_is_normalised(self):
        circ, _, clock, b_reg = fresh_circuit()
        encode_b(circ, b_reg, np.array([1.0, 2.0]))
        apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        sv = Statevector.from_instruction(circ)
        assert sum(abs(a)**2 for a in sv) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_inverse_qft
# ---------------------------------------------------------------------------

class TestApplyInverseQFT:
    def test_qft_then_inverse_is_identity(self):
        """Forward QFT followed by inverse QFT should return to original state."""
        from qiskit.circuit.library import QFT as QiskitQFT

        circ, _, clock, b_reg = fresh_circuit()
        encode_b(circ, b_reg, np.array([1.0, 2.0]))
        apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)

        sv_before = Statevector.from_instruction(circ)

        # Apply inverse QFT then forward QFT — should get back to sv_before
        apply_inverse_qft(circ, clock)
        apply_forward_qft(circ, clock)

        sv_after = Statevector.from_instruction(circ)
        np.testing.assert_allclose(
            np.abs(sv_before.data), np.abs(sv_after.data), atol=1e-6
        )

    def test_returns_circuit(self):
        circ, _, clock, _ = fresh_circuit()
        result = apply_inverse_qft(circ, clock)
        assert result is circ


# ---------------------------------------------------------------------------
# apply_ancilla_rotation
# ---------------------------------------------------------------------------

class TestApplyAncillaRotation:
    def _run_full_qpe(self, b_vec):
        """Helper: run encode + QPE unitaries + inverse QFT."""
        circ, ancilla, clock, b_reg = fresh_circuit()
        encode_b(circ, b_reg, b_vec)
        apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        apply_inverse_qft(circ, clock)
        return circ, ancilla, clock, b_reg

    def test_ancilla_gets_nonzero_amplitude(self):
        """After ancilla rotation, some amplitude should be in ancilla=1."""
        circ, ancilla, clock, b_reg = self._run_full_qpe(np.array([1.0, 0.0]))
        apply_ancilla_rotation(circ, clock, ancilla, A_DIAG, T)
        sv = Statevector.from_instruction(circ)
        # ancilla=1 is MSB in workbench ordering; in Qiskit (ancilla, clock, b)
        # ancilla qubit is qubit 0 (highest index in state vector)
        # sum prob where ancilla bit = 1
        n_qubits   = circ.num_qubits
        ancilla_on = sum(
            abs(sv.data[i])**2
            for i in range(2**n_qubits)
            if (i >> (n_qubits - 1)) & 1
        )
        assert ancilla_on > 1e-4

    def test_custom_C_accepted(self):
        """Should not raise with a custom C value."""
        circ, ancilla, clock, b_reg = self._run_full_qpe(np.array([1.0, 1.0]))
        C = 0.5 * min(abs(EIGVALS))
        apply_ancilla_rotation(circ, clock, ancilla, A_DIAG, T, C=C)
        sv = Statevector.from_instruction(circ)
        assert sum(abs(a)**2 for a in sv) == pytest.approx(1.0, abs=1e-6)

    def test_C_too_large_raises_no_rotations(self):
        """If C >= all lam_m, no rotations fire and ancilla stays in |0>."""
        circ, ancilla, clock, b_reg = self._run_full_qpe(np.array([1.0, 0.0]))
        # C larger than all possible lam_m — no arcsin fires
        apply_ancilla_rotation(circ, clock, ancilla, A_DIAG, T, C=1e10)
        sv = Statevector.from_instruction(circ)
        n_qubits   = circ.num_qubits
        ancilla_on = sum(
            abs(sv.data[i])**2
            for i in range(2**n_qubits)
            if (i >> (n_qubits - 1)) & 1
        )
        assert ancilla_on == pytest.approx(0.0, abs=1e-6)

    def test_returns_circuit(self):
        circ, ancilla, clock, _ = fresh_circuit()
        result = apply_ancilla_rotation(circ, clock, ancilla, A_DIAG, T)
        assert result is circ


# ---------------------------------------------------------------------------
# apply_forward_qft / apply_inverse_qpe_unitaries (inverse QPE)
# ---------------------------------------------------------------------------

class TestInverseQPE:
    def test_full_qpe_then_inverse_restores_b(self):
        """QPE followed by inverse QPE should return clock to |0> and leave b intact."""
        circ, ancilla, clock, b_reg = fresh_circuit()
        b_vec = np.array([1.0, 2.0])
        encode_b(circ, b_reg, b_vec)

        sv_after_encode = Statevector.from_instruction(circ)

        # Forward QPE
        apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        apply_inverse_qft(circ, clock)

        # Inverse QPE
        apply_forward_qft(circ, clock)
        apply_inverse_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)

        sv_final = Statevector.from_instruction(circ)

        # Amplitudes should match the post-encode state (up to global phase)
        np.testing.assert_allclose(
            np.abs(sv_after_encode.data),
            np.abs(sv_final.data),
            atol=1e-5,
        )

    def test_returns_circuit_forward_qft(self):
        circ, _, clock, _ = fresh_circuit()
        result = apply_forward_qft(circ, clock)
        assert result is circ

    def test_returns_circuit_inverse_unitaries(self):
        circ, _, clock, b_reg = fresh_circuit()
        result = apply_inverse_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        assert result is circ


# ---------------------------------------------------------------------------
# Integration: full HHL pipeline recovers correct ratio
# ---------------------------------------------------------------------------

class TestFullHHLIntegration:
    @pytest.mark.parametrize("b_vec,expected_ratio", [
        (np.array([1.0, 1.0]), 1.0),   # symmetric b -> x[0]/x[1] = 1
        (np.array([1.0, 0.0]), 0.5),   # b=[1,0] -> x = A^{-1}[1,0] = [3/8,-1/8] normalized ratio
    ])
    def test_ratio_matches_classical(self, b_vec, expected_ratio):
        """Post-selected amplitudes should give the classical x[0]/x[1] ratio."""
        circ, ancilla, clock, b_reg = fresh_circuit()

        encode_b(circ, b_reg, b_vec)
        apply_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)
        apply_inverse_qft(circ, clock)
        apply_ancilla_rotation(circ, clock, ancilla, A_DIAG, T)
        apply_forward_qft(circ, clock)
        apply_inverse_qpe_unitaries(circ, clock, b_reg, A_DIAG, T)

        sv = Statevector.from_instruction(circ)

        # Extract ancilla=1, clock=0 amplitudes
        # Circuit order: (ancilla, clock, b) — ancilla is qubit 0 in circuit = MSB in index
        n_qubits      = circ.num_qubits  # 1 + N_L + 1 = 4
        ancilla_shift = n_qubits - 1     # ancilla is the top qubit

        amp0 = sv.data[(1 << ancilla_shift) | 0]  # b=0, clock=0, ancilla=1
        amp1 = sv.data[(1 << ancilla_shift) | 1]  # b=1, clock=0, ancilla=1

        if abs(amp0) < 1e-8 and abs(amp1) < 1e-8:
            pytest.skip("No amplitude in ancilla=1 subspace — C may be too small")

        ratio = abs(amp0) / abs(amp1)
        x_classical  = np.linalg.solve(A_DIAG, b_vec)
        x_c_norm     = x_classical / np.linalg.norm(x_classical)
        classical_ratio = abs(x_c_norm[0]) / abs(x_c_norm[1])

        assert ratio == pytest.approx(classical_ratio, rel=0.05)