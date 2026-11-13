"""Check restart selection, early stopping, and car-control layout."""

from types import SimpleNamespace

import numpy as np
import pytest

from rd3g.core.base_solver import Solution
from rd3g.solvers.interior_point_game import InteriorPointGame


def make_solver(outcomes, racing=False):
    solver = object.__new__(InteriorPointGame)
    solver.cpp_solver = object()
    solver.m, solver.N, solver.T = 2, 2, 3
    solver.game = SimpleNamespace(__module__=(
        'rd3g.games.car_racing_casadi' if racing
        else 'rd3g.games.car_merge_kinematic_bicycle_casadi'))
    guesses = []

    def solve(guess):
        assert guess.shape == (solver.m * solver.N, solver.T) and guess.flags.f_contiguous
        converged, optimal, residual = outcomes[min(len(guesses), len(outcomes) - 1)]
        guesses.append(guess.copy())
        solver.residual_vec = [residual + 1, residual]
        return Solution(u=guess.copy(), residual=residual,
                        has_converged=converged, is_optimal=optimal)

    solver.solve_cpp_backend = solve
    return solver, guesses


def test_successful_initial_guess_needs_no_retries():
    solver, guesses = make_solver([(True, True, 1e-5)])
    base = np.ones((4, 3))
    result = solver.solve_cpp_backend_rand_restart(restarts=10, u_ref=base, control_std=[0.1, 0.2])
    assert len(guesses) == 1
    assert solver.restart_selected_attempt == 0
    np.testing.assert_array_equal(result.u, base)
    assert result.elapsed_time > 0


def test_exhausted_retries_preserve_an_earlier_converged_solution():
    solver, guesses = make_solver([(True, False, 1e-4), (False, True, 2.0)])
    result = solver.solve_cpp_backend_rand_restart(restarts=10, control_std=[0.1, 0.2])
    assert len(guesses) == 11
    assert len(solver.restart_attempts) == 11
    assert result.has_converged and not result.is_optimal
    assert solver.restart_selected_attempt == 0
    assert solver.residual_vec == solver.restart_attempts[0]['residual_history']
    np.testing.assert_array_equal(result.u, np.zeros((4, 3)))


def test_racing_retries_preserve_stanley_guess_and_control_order(monkeypatch):
    solver, guesses = make_solver(
        [(False, False, 2.0), (False, True, 1.0), (True, True, 1e-5)], racing=True)
    samples = np.arange(12, dtype=float).reshape(2, 3, 2) / 100

    def sample(loc, scale, size):
        assert loc == 0
        np.testing.assert_allclose(scale, [0.1, 0.2])
        assert size == (2, 3, 2)
        return samples

    monkeypatch.setattr(np.random, 'normal', sample)
    base = np.full((4, 3), 0.25)
    result = solver.solve_cpp_backend_rand_restart(restarts=10, u_ref=base, control_std=[0.1, 0.2])
    expected_noise = np.vstack([samples[0, :, 0], samples[0, :, 1],
                                samples[1, :, 0], samples[1, :, 1]])
    assert len(guesses) == 3 and solver.restart_selected_attempt == 2
    np.testing.assert_allclose(guesses[1], base + expected_noise)
    np.testing.assert_allclose(result.u, base + expected_noise)
    np.testing.assert_array_equal(base, np.full((4, 3), 0.25))
    assert result.has_converged and result.is_optimal


def test_restart_noise_scales_are_required():
    solver, guesses = make_solver([(True, True, 1e-5)])
    with pytest.raises(ValueError, match='control_std is required'):
        solver.solve_cpp_backend_rand_restart(restarts=1)
    assert not guesses


def test_restart_sampling_supports_arbitrary_control_dimensions():
    solver, guesses = make_solver([(False, False, 1.0), (True, True, 1e-5)])
    solver.m = 3
    solver.solve_cpp_backend_rand_restart(restarts=1, control_std=[0, 0, 0])
    assert len(guesses) == 2
    np.testing.assert_array_equal(guesses[1], np.zeros((solver.m * solver.N, solver.T)))
