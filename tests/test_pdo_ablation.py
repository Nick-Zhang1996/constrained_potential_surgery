"""Regression coverage for the three PDO linear-solver ablations.

Each level must solve more than half of 100 fixed three-car merge games.
"""

import numpy as np
import pytest

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.interior_point_game import (
    InteriorPointGame,
    InteriorPointGameConfig,
)


SEEDS = range(100)


def test_pdo_ablation_requires_the_pdo_split() -> None:
    """Nonzero levels cannot silently alter a non-PDO solver configuration."""
    InteriorPointGameConfig(ablation_level=0,
                            precondition_with_potential=False)
    with pytest.raises(AssertionError, match="PDO ablations require"):
        InteriorPointGameConfig(ablation_level=1,
                                precondition_with_potential=False)
    with pytest.raises(AssertionError, match="retain the vGNE"):
        InteriorPointGameConfig(ablation_level=2,
                                precondition_with_potential=True,
                                variational_gne=False)
    with pytest.raises(AssertionError, match="ablation_level must"):
        InteriorPointGameConfig(ablation_level=3,
                                precondition_with_potential=True)


def make_config(ablation_level: int) -> InteriorPointGameConfig:
    """Return the explicitly selected configuration for one ablation level."""
    return InteriorPointGameConfig(
        ablation_level=ablation_level,
        inertia_correction=True,
        precondition_with_potential=True,
        abs_split=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("ablation_level", range(3))
@pytest.mark.parametrize("use_cpp", [False, True], ids=["python", "cpp"])
def test_pdo_ablation_three_car_success_rate(ablation_level: int,
                                              use_cpp: bool) -> None:
    """Every Python and C++ ablation solves more than 50% of fixed cases."""
    successes = 0
    for seed in SEEDS:
        np.random.seed(seed)
        config = make_config(ablation_level)
        game = create_random_game(car_count=3,
                                  horizon=20,
                                  variational_gne=config.variational_gne)
        solver = InteriorPointGame(config, game, cpp_only=use_cpp)
        if use_cpp:
            solver.init_cpp_backend()
            solution = solver.solve_cpp_backend()
        else:
            solution = solver.solve()
        successes += solution.has_converged

    print(f"level {ablation_level} ({'C++' if use_cpp else 'Python'}): "
          f"{successes}/{len(SEEDS)} converged ({successes / len(SEEDS):.1%})")
    assert successes > len(SEEDS) / 2, (
        f"ablation level {ablation_level} ({'C++' if use_cpp else 'Python'}) "
        f"only converged on {successes}/{len(SEEDS)} three-car cases"
    )
