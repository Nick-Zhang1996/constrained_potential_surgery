"""Exercise the supported commands against real small game instances."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts._benchmark import constraint_stats, main as benchmark
from scripts._common import create_game
from scripts._solve import main as single_run


@pytest.mark.parametrize('game', ['merge', 'intersection', 'racing'])
def test_seeded_games_match_across_gne_formulations(game):
    cps = create_game(game, 2, 3, 17, 'cps').config
    rd3g = create_game(game, 2, 3, 17, 'rd3g').config
    for name in ('x0', 'target_x_ref', 'J_R'):
        np.testing.assert_array_equal(getattr(cps, name), getattr(rd3g, name))


@pytest.mark.parametrize('game', ['merge', 'intersection', 'racing'])
def test_headless_single_run_saves_separate_initial_and_solution(game, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', [f'solve_{game}', '--backend', 'python',
                                   '--car-count', '2', '--horizon', '3', '--iterations', '2',
                                   '--no-display', '--animate', '--initial',
                                   '--output-dir', str(tmp_path)])
    single_run(game)
    saved = np.load(tmp_path / 'solution.npz')
    assert saved['u'].size == 2 * 2 * 3
    assert np.isfinite(saved['u']).all()
    # Diagnostics must accept Julia's x0..xT layout as well as x1..xT.
    model = create_game(game, 2, 3, 0)
    n = model.config.n
    full_x = np.dstack([saved['x0'][:, :, None], saved['x'].reshape((n, 2, 3), order='F')])
    assert constraint_stats(model, SimpleNamespace(u=saved['u'], x=full_x)) == \
        constraint_stats(model, SimpleNamespace(u=saved['u'], x=saved['x']))
    for name in ('initial.png', 'solution.png', 'residuals.png', 'initial.gif', 'solution.gif'):
        with Image.open(tmp_path / name) as img:
            assert min(img.size) > 0


@pytest.mark.parametrize('game', ['merge', 'intersection', 'racing'])
def test_small_comparison_records_each_method_and_generates_report(game, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', [f'benchmark_{game}', '--backend', 'python',
                                   '--solvers', 'cps', 'rd3g', '--runs', '1',
                                   '--min-cars', '2', '--max-cars', '2',
                                   '--horizon', '3', '--iterations', '2',
                                   '--output-dir', str(tmp_path)])
    benchmark(game)
    with (tmp_path / 'runs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['method'] for row in rows] == ['cps', 'rd3g']
    assert all(row['seed'] == '0' and not row['error'] for row in rows)
    assert all(float(row['runtime_s']) > 0 for row in rows)
    records = json.loads((tmp_path / 'runs.json').read_text())
    assert len(records) == len(rows)
    assert (tmp_path / 'summary.txt').stat().st_size > 0
    with Image.open(tmp_path / 'comparison.png') as img:
        assert min(img.size) > 0


def test_comparison_rejects_pdo(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['benchmark_merge', '--solvers', 'pdo',
                                   '--output-dir', str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        benchmark('merge')
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('override_cli', [False, True])
def test_toml_settings_control_the_experiment_and_cli_takes_precedence(tmp_path, monkeypatch,
                                                                     override_cli):
    config = tmp_path / 'experiment.toml'
    config.write_text('''[run]
backend = "python"
horizon = 3
iterations = 2
tolerance = 0.01
seed = 19

[benchmark]
runs = 2
min_cars = 2
max_cars = 2
solvers = ["cps"]

[solvers.cps]
line_search_max_iter = 4
''')
    destination = tmp_path / 'results'
    args = ['benchmark_merge', '--config', str(config), '--output-dir', str(destination)]
    if override_cli:
        args.extend(['--runs', '3', '--seed', '23', '--tolerance', '0.02'])
    monkeypatch.setattr('sys.argv', args)
    benchmark('merge')
    records = json.loads((destination / 'runs.json').read_text())
    settings = json.loads((destination / 'config.json').read_text())
    expected_runs, first_seed, tolerance = (3, 23, 0.02) if override_cli else (2, 19, 0.01)
    assert [row['seed'] for row in records] == list(range(first_seed, first_seed + expected_runs))
    assert settings['solver_configs']['cps']['tolerance'] == tolerance
    assert settings['solver_configs']['cps']['iterations'] == 2
    assert settings['solver_configs']['cps']['line_search_max_iter'] == 4
    assert settings['solver_configs']['cps']['abs_split'] is True


def test_unreported_optimality_is_not_counted_as_failure():
    from scripts._benchmark import aggregates

    stats, = aggregates([dict(method='algames', car_count=2, runtime_s=0.1,
                             is_converged=True, is_optimal=None, error='')])
    assert stats['converged'] == 1
    assert stats['optimal'] is None
