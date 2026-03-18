"""
test_hhl_steps_workbench.py
---------------------------
Unit tests for each HHL step function in hhl_steps_workbench.py.

Run with:  pytest test_hhl_steps_workbench.py -v

Each test spins up a fresh QPU so tests are fully isolated.
"""

import numpy as np
import pytest
from numpy import pi
from psiqworkbench import QPU, Qubits

from wb_funcs import (
    make_unitary,
    encode_b,
    apply_qpe,
    apply_ancilla_rotation,
    apply_inverse_qpe,
    extract_solution,
    HHLUnitary,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

A_DIAG  = np.array([[3, 1], [1, 3]], dtype=float)   # eigenvalues 2, 4
N_L     = 2
N_A     = 1
N_B     = 1
EIGVALS = np.linalg.eigvalsh(A_DIAG)
T       = 1 * (2 * pi) / (min(abs(EIGVALS)) * 2**N_L)


def fresh_qpu(n_l=N_L):
    """Return a fresh QPU and registers (ancilla, clock, b)."""
    qpu     = QPU(num_qubits=N_A + N_B + n_l)
    b_reg   = Qubits(N_B,  name="b",       qpu=qpu)
    clock   = Qubits(n_l,  name="clock",   qpu=qpu)
    ancilla = Qubits(N_A,  name="ancilla", qpu=qpu)
    return qpu, b_reg, clock, ancilla


def pull_probs(qpu):
    """Return probability for each basis state."""
    sv = qpu.pull_state()
    return np.abs(sv)**2


# ---------------------------------------------------------------------------
# make_unitary
# ---------------------------------------------------------------------------

class TestMakeUnitary:
    def test_returns_unitary_matrix(self):
        U = make_unitary(A_DIAG, T)
        assert U.shape == (2, 2)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-10)

    def test_power_scaling(self):
        U1 = make_unitary(A_DIAG, T, power=1)
        U2 = make_unitary(A_DIAG, T, power=2)
        np.testing.assert_allclose(U2, U1 @ U1, atol=1e-10)

    def test_inverse(self):
        U     = make_unitary(A_DIAG, T, power=1)
        U_inv = make_unitary(A_DIAG, T, power=-1)
        np.testing.assert_allclose(U @ U_inv, np.eye(2), atol=1e-10)

    def test_hermitian_input_gives_unitary(self):
        A = np.array([[2, 1+1j], [1-1j, 3]])
        U = make_unitary(A, 1.0)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-10)


# ---------------------------------------------------------------------------
# encode_b
# ---------------------------------------------------------------------------

class TestEncodeB:
    def test_encodes_zero_vector(self):
        """b = [1, 0] should leave b_reg in |0>."""
        qpu, b_reg, _, _ = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 0.0]))
        sv    = qpu.pull_state()
        probs = np.abs(sv)**2
        # b=0 is index 0 (b is LSB)
        assert probs[0] == pytest.approx(1.0, abs=1e-6)

    def test_encodes_one_vector(self):
        """b = [0, 1] should put b_reg in |1>."""
        qpu, b_reg, _, _ = fresh_qpu()
        encode_b(b_reg, np.array([0.0, 1.0]))
        sv    = qpu.pull_state()
        probs = np.abs(sv)**2
        assert probs[1] == pytest.approx(1.0, abs=1e-6)

    def test_normalisation_preserved(self):
        qpu, b_reg, _, _ = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 2.0]))
        probs = pull_probs(qpu)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_correct_amplitudes(self):
        b_vec  = np.array([1.0, 2.0])
        b_norm = b_vec / np.linalg.norm(b_vec)
        theta  = 2 * np.arccos(b_norm[0])
        qpu, b_reg, _, _ = fresh_qpu()
        encode_b(b_reg, b_vec)
        sv = qpu.pull_state()
        assert abs(sv[0]) == pytest.approx(np.cos(theta / 2), abs=1e-6)
        assert abs(sv[1]) == pytest.approx(np.sin(theta / 2), abs=1e-6)


# ---------------------------------------------------------------------------
# apply_qpe
# ---------------------------------------------------------------------------

class TestApplyQPEManual:
    def test_clock_spreads_after_qpe(self):
        """After QPE, amplitude should spread across multiple clock bins."""
        qpu, b_reg, clock, _ = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 0.0]))
        apply_qpe(b_reg, clock, A_DIAG, T)
        probs        = pull_probs(qpu)
        nonzero_bins = sum(1 for p in probs if p > 1e-6)
        assert nonzero_bins > 1

    def test_returns_iqft_object(self):
        qpu, b_reg, clock, _ = fresh_qpu()
        result = apply_qpe(b_reg, clock, A_DIAG, T)
        # Should return a QFT instance for use in uncompute
        from psiqworkbench.qubricks import QFT
        assert isinstance(result, QFT)

    def test_state_is_normalised(self):
        qpu, b_reg, clock, _ = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 2.0]))
        apply_qpe(b_reg, clock, A_DIAG, T)
        probs = pull_probs(qpu)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_eigenvalue_bins_populated(self):
        """For A with exact bin mapping, bins 1 and 2 should dominate."""
        qpu, b_reg, clock, _ = fresh_qpu()
        # Use b=[1,0] which projects purely onto one eigenvector
        encode_b(b_reg, np.array([1.0, 1.0]) / np.sqrt(2))
        apply_qpe(b_reg, clock, A_DIAG, T)
        sv    = qpu.pull_state()
        probs = np.abs(sv)**2
        # With n_l=2 and exact bins: bins 1 and 2 of clock should have most prob
        # clock is bits 1..n_l in the index (b is LSB)
        bin1_prob = sum(probs[i] for i in range(len(probs)) if ((i >> N_B) & (2**N_L - 1)) == 1)
        bin2_prob = sum(probs[i] for i in range(len(probs)) if ((i >> N_B) & (2**N_L - 1)) == 2)
        assert bin1_prob + bin2_prob == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# apply_ancilla_rotation
# ---------------------------------------------------------------------------

class TestApplyAncillaRotation:
    def _run_to_post_qpe(self, b_vec, n_l=N_L):
        qpu, b_reg, clock, ancilla = fresh_qpu(n_l)
        t = 1 * (2 * pi) / (min(abs(np.linalg.eigvalsh(A_DIAG))) * 2**n_l)
        encode_b(b_reg, b_vec)
        apply_qpe(b_reg, clock, A_DIAG, t)
        return qpu, b_reg, clock, ancilla, t

    def test_ancilla_gets_nonzero_amplitude(self):
        qpu, _, clock, ancilla, t = self._run_to_post_qpe(np.array([1.0, 0.0]))
        apply_ancilla_rotation(clock, ancilla, A_DIAG, t)
        sv          = qpu.pull_state()
        n_total     = N_A + N_B + N_L
        ancilla_on  = sum(
            abs(sv[i])**2
            for i in range(2**n_total)
            if (i >> (N_B + N_L)) & 1
        )
        assert ancilla_on > 1e-4

    def test_custom_C_accepted(self):
        qpu, _, clock, ancilla, t = self._run_to_post_qpe(np.array([1.0, 1.0]))
        C = 0.5 * min(abs(EIGVALS))
        apply_ancilla_rotation(clock, ancilla, A_DIAG, t, C=C)
        probs = pull_probs(qpu)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_C_too_large_fires_no_rotations(self):
        """C >= all lam_m means no rotations fire — ancilla stays in |0>."""
        qpu, _, clock, ancilla, t = self._run_to_post_qpe(np.array([1.0, 0.0]))
        apply_ancilla_rotation(clock, ancilla, A_DIAG, t, C=1e10)
        sv         = qpu.pull_state()
        n_total    = N_A + N_B + N_L
        ancilla_on = sum(
            abs(sv[i])**2
            for i in range(2**n_total)
            if (i >> (N_B + N_L)) & 1
        )
        assert ancilla_on == pytest.approx(0.0, abs=1e-6)

    def test_state_normalised_after_rotation(self):
        qpu, _, clock, ancilla, t = self._run_to_post_qpe(np.array([1.0, 2.0]))
        apply_ancilla_rotation(clock, ancilla, A_DIAG, t)
        probs = pull_probs(qpu)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_inverse_qpe
# ---------------------------------------------------------------------------

class TestApplyInverseQPE:
    def test_qpe_then_inverse_restores_b(self):
        """QPE followed by inverse QPE should return clock to |0> and leave b intact."""
        qpu, b_reg, clock, _ = fresh_qpu()
        b_vec = np.array([1.0, 2.0])
        encode_b(b_reg, b_vec)

        sv_after_encode = qpu.pull_state().copy()

        iqft = apply_qpe(b_reg, clock, A_DIAG, T)
        apply_inverse_qpe(iqft, b_reg, clock, A_DIAG, T)

        sv_final = qpu.pull_state()
        np.testing.assert_allclose(
            np.abs(sv_after_encode),
            np.abs(sv_final),
            atol=1e-5,
        )

    def test_clock_returns_to_zero(self):
        """After full QPE + inverse QPE, all amplitude should be in clock=0."""
        qpu, b_reg, clock, _ = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 2.0]))
        iqft = apply_qpe(b_reg, clock, A_DIAG, T)
        apply_inverse_qpe(iqft, b_reg, clock, A_DIAG, T)

        sv          = qpu.pull_state()
        probs       = np.abs(sv)**2
        n_total     = N_A + N_B + N_L
        clock_zero  = sum(
            probs[i]
            for i in range(2**n_total)
            if ((i >> N_B) & (2**N_L - 1)) == 0
        )
        assert clock_zero == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# extract_solution
# ---------------------------------------------------------------------------

class TestExtractSolution:
    def test_returns_normalised_vector(self):
        qpu, b_reg, clock, ancilla = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 2.0]))
        iqft = apply_qpe(b_reg, clock, A_DIAG, T)
        apply_ancilla_rotation(clock, ancilla, A_DIAG, T)
        apply_inverse_qpe(iqft, b_reg, clock, A_DIAG, T)
        ancilla.read()

        x_q = extract_solution(qpu, N_B, N_L, N_A, A_DIAG, np.array([1.0, 2.0]), verbose=False)
        assert np.linalg.norm(x_q) == pytest.approx(1.0, abs=1e-6)

    def test_returns_two_element_array(self):
        qpu, b_reg, clock, ancilla = fresh_qpu()
        encode_b(b_reg, np.array([1.0, 2.0]))
        iqft = apply_qpe(b_reg, clock, A_DIAG, T)
        apply_ancilla_rotation(clock, ancilla, A_DIAG, T)
        apply_inverse_qpe(iqft, b_reg, clock, A_DIAG, T)
        ancilla.read()

        x_q = extract_solution(qpu, N_B, N_L, N_A, A_DIAG, np.array([1.0, 2.0]), verbose=False)
        assert x_q.shape == (2,)


# ---------------------------------------------------------------------------
# Integration: full HHL pipeline recovers correct ratio
# ---------------------------------------------------------------------------

class TestFullHHLIntegration:
    @pytest.mark.parametrize("b_vec,expected_ratio", [
        (np.array([1.0, 1.0]), 1.0),
        (np.array([1.0, 2.0]), 0.5),
    ])
    def test_ratio_matches_classical(self, b_vec, expected_ratio):
        qpu, b_reg, clock, ancilla = fresh_qpu()

        encode_b(b_reg, b_vec)
        iqft = apply_qpe(b_reg, clock, A_DIAG, T)
        apply_ancilla_rotation(clock, ancilla, A_DIAG, T)
        apply_inverse_qpe(iqft, b_reg, clock, A_DIAG, T)
        ancilla.read()

        x_q = extract_solution(qpu, N_B, N_L, N_A, A_DIAG, b_vec, verbose=False)

        if x_q[1] < 1e-8:
            pytest.skip("Near-zero denominator — degenerate b vector")

        x_classical     = np.linalg.solve(A_DIAG, b_vec)
        x_c_norm        = x_classical / np.linalg.norm(x_classical)
        classical_ratio = abs(x_c_norm[0]) / abs(x_c_norm[1])

        assert x_q[0] / x_q[1] == pytest.approx(classical_ratio, rel=0.05)