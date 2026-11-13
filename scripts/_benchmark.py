"""Shared implementation for the merge, intersection, and racing benchmarks."""

import argparse
import csv
import json
import logging
from dataclasses import asdict
from time import perf_counter
from pathlib import Path

import numpy as np

from scripts._common import (ROOT, BENCHMARK_SOLVERS, create_game, create_solver,
                             initial_guess, reset_solver, solve, validate_args)
from scripts._settings import configure_parser

LOGGER = logging.getLogger(__name__)
FIELDS = ('game', 'method', 'car_count', 'seed', 'runtime_s', 'residual',
          'is_converged', 'is_optimal', 'optimality_check', 'iterations',
          'constraint_violation_count', 'max_constraint_violation', 'error')


def constraint_stats(game, solution):
    """Evaluate canonical h <= 0 on the returned state/control trajectory."""
    import casadi as cas

    gc = game.config
    states = np.asarray(solution.x)
    # Julia returns x0..xT; the native/Python wrappers return x1..xT.
    if states.size == gc.n * gc.N * (gc.T + 1):
        states = states.reshape((gc.n, gc.N, gc.T + 1), order='F')[:, :, 1:]
    x = cas.DM(states.reshape((gc.n * gc.N, gc.T), order='F'))
    u = cas.DM(np.asarray(solution.u).reshape((gc.m * gc.N, gc.T), order='F'))
    context = cas.horzcat(*[
        cas.vertcat(*[game.get_context(x[i * gc.n:(i + 1) * gc.n, k])
                      for i in range(gc.N)]) for k in range(gc.T)])
    fun = cas.Function('benchmark_h', [gc.get_int_param_sx(), gc.get_double_param_sx()],
                       [game.h(x, u, context)])
    h = np.asarray(fun(gc.get_int_param_np(), gc.get_double_param_np()), dtype=float)
    if not np.isfinite(h).all():
        raise ValueError('Returned trajectory contains nonfinite constraints')
    return int(np.count_nonzero(h > 0)), float(np.maximum(h, 0).max(initial=0))


def json_safe(value):
    """Keep failed residuals readable by strict JSON tools."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_records(path, records):
    path.write_text(json.dumps(json_safe(records), indent=2, allow_nan=False) + '\n')


def aggregates(records):
    groups = {}
    for row in records:
        groups.setdefault((row['method'], row['car_count']), []).append(row)
    result = []
    for (method, cars), rows in groups.items():
        times = np.array([r['runtime_s'] * 1000 for r in rows])
        optimal = [r['is_converged'] and r['is_optimal'] for r in rows
                   if r['is_optimal'] is not None]
        result.append(dict(
            method=method, cars=cars, runs=len(rows),
            mean_ms=float(times.mean()), std_ms=float(times.std()),
            converged=np.mean([r['is_converged'] for r in rows]),
            optimal=float(np.mean(optimal)) if optimal else None,
            errors=sum(bool(r['error']) for r in rows)))
    return result


def save_report(game_name, records, output_dir, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    stats = aggregates(records)
    lines = [f'{game_name.title()} benchmark',
             'Runtime includes all requested restarts; setup/codegen is excluded.',
             'Optimal rate counts converged AND solver-reported optimal; n/a means unavailable.',
             'See config.json for experiment and solver settings.', '']
    for row in stats:
        optimal_label = 'n/a' if row['optimal'] is None else f"{row['optimal']:.1%}"
        lines.append(f"{row['method']:8} {row['cars']} cars | {row['runs']} runs | "
                     f"{row['mean_ms']:.3f} +/- {row['std_ms']:.3f} ms | "
                     f"converged {row['converged']:.1%} | optimal {optimal_label} | "
                     f"errors {row['errors']}")
    (output_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for method in dict.fromkeys(r['method'] for r in stats):
        rows = [r for r in stats if r['method'] == method]
        cars = [r['cars'] for r in rows]
        axes[0].errorbar(cars, [r['mean_ms'] for r in rows],
                         yerr=[r['std_ms'] for r in rows], marker='o', capsize=3,
                         label=method.upper())
        line, = axes[1].plot(cars, [r['converged'] for r in rows], 'o-',
                            label=f'{method.upper()}: converged')
        optimal_rows = [r for r in rows if r['optimal'] is not None]
        if optimal_rows:
            axes[1].plot([r['cars'] for r in optimal_rows],
                         [r['optimal'] for r in optimal_rows], 's--', color=line.get_color(),
                         label=f'{method.upper()}: converged + optimal')
    axes[0].set(ylabel='Runtime (ms), mean ± std', title='Solve time')
    axes[1].set(ylabel='Fraction of all runs', ylim=(0, 1.05), title='Success rate')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel('Number of cars')
        ax.set_xticks(sorted({r['car_count'] for r in records}))
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7)
    fig.suptitle(f'{game_name.title()} solver comparison')
    fig.savefig(output_dir / 'comparison.png', dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main(game_name):
    parser = argparse.ArgumentParser(description=f'Benchmark the {game_name} game.')
    settings = configure_parser(parser)
    run, sweep = settings['run'], settings['benchmark']
    parser.add_argument('--solvers', nargs='+', choices=BENCHMARK_SOLVERS,
                        default=sweep['solvers'])
    parser.add_argument('--backend', choices=('cpp', 'python'), default=run['backend'],
                        help='CPS and RD3G backend; other methods use native backends')
    parser.add_argument('--runs', '--repeat-count', type=int, default=sweep['runs'])
    parser.add_argument('--min-cars', type=int, default=sweep['min_cars'])
    parser.add_argument('--max-cars', type=int, default=sweep['max_cars'])
    parser.add_argument('--horizon', type=int, default=run['horizon'])
    parser.add_argument('--iterations', type=int, default=run['iterations'])
    parser.add_argument('--tolerance', type=float, default=run['tolerance'])
    parser.add_argument('--seed', type=int, default=run['seed'], help='first seed in the sweep')
    parser.add_argument('--restarts', type=int, default=run['restarts'],
                        help='extra CPS control guesses; runtime includes every attempt')
    parser.add_argument('--restart-std', type=float, nargs='+', default=run['restart_std'],
                        help='restart standard deviation per control, in game control order')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs' / 'benchmarks' / game_name)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    validate_args(parser, args)
    if args.runs < 1 or not 2 <= args.min_cars <= args.max_cars:
        parser.error('require --runs > 0 and 2 <= --min-cars <= --max-cars')
    if any(method not in BENCHMARK_SOLVERS for method in args.solvers):
        parser.error('comparison solvers must be cps, rd3g, or algames')
    args.solvers = list(dict.fromkeys(args.solvers))
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'config': str(args.config),
              'output_dir': str(args.output_dir), 'game': game_name,
              'experiment_settings': settings,
              'timing': 'solve only; includes all restarts; excludes setup and diagnostics',
              'solver_configs': {}}
    records = []
    with (args.output_dir / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for cars in range(args.min_cars, args.max_cars + 1):
            for method in args.solvers:
                game = create_game(game_name, cars, args.horizon, args.seed, method)
                # Setup errors fail immediately, before producing a misleading full sweep.
                solver = create_solver(method, game, args.backend, args.iterations,
                                       tolerance=args.tolerance, settings=settings)
                config['solver_configs'][method] = asdict(solver.config)
                write_records(args.output_dir / 'config.json', config)
                try:
                    for seed in range(args.seed, args.seed + args.runs):
                        game = create_game(game_name, cars, args.horizon, seed, method)
                        reset_solver(solver, game)
                        row = dict(game=game_name, method=method, car_count=cars, seed=seed,
                                   is_converged=False,
                                   is_optimal=None if method == 'algames' else False,
                                   optimality_check='not_reported' if method == 'algames' else 'solver',
                                   error='', iterations=0, residual=None,
                                   constraint_violation_count=None, max_constraint_violation=None)
                        started = perf_counter()
                        try:
                            guess = initial_guess(game, game_name) if method != 'algames' else None
                            started = perf_counter()
                            solution = solve(solver, method, args.backend, guess, args.restarts,
                                             args.restart_std)
                            row.update(runtime_s=float(solution.elapsed_time),
                                       residual=float(solution.residual),
                                       is_converged=bool(solution.has_converged),
                                       is_optimal=None if method == 'algames' else bool(solution.is_optimal),
                                       iterations=int(solution.iterations))
                            row['residual_history'] = list(getattr(solver, 'residual_vec', []))
                            if args.restarts:
                                row['attempts'] = solver.restart_attempts
                            count, violation = constraint_stats(game, solution)
                            row.update(constraint_violation_count=count,
                                       max_constraint_violation=violation)
                        except Exception as exc:
                            row.setdefault('runtime_s', perf_counter() - started)
                            row['error'] = f'{type(exc).__name__}: {exc}'
                            LOGGER.exception('%s / %d cars / seed %d failed', method, cars, seed)
                        records.append(row)
                        writer.writerow({key: row.get(key) for key in FIELDS})
                        stream.flush()
                        LOGGER.info('%s / %s / %d cars / seed %d: converged=%s, %.3f ms',
                                    game_name, method, cars, seed, row['is_converged'],
                                    row['runtime_s'] * 1000)
                finally:
                    if hasattr(solver, 'close'):
                        solver.close()
                    write_records(args.output_dir / 'runs.json', records)
    save_report(game_name, records, args.output_dir, args.show)
    print(f'Results: {args.output_dir}')
    if any(row['error'] for row in records):
        raise SystemExit('Some runs failed; inspect the error column in runs.csv.')
