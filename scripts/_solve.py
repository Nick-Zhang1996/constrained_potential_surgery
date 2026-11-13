"""Single-game solves with trajectory, residual, and optional GIF outputs."""

import argparse
import logging
from pathlib import Path

import numpy as np

from scripts._common import (ROOT, SOLVERS, create_game, create_solver,
                             initial_guess, solve, validate_args)
from scripts._settings import configure_parser


def main(game_name):
    parser = argparse.ArgumentParser(description=f'Solve and visualize the {game_name} game.')
    settings = configure_parser(parser)
    run = settings['run']
    parser.add_argument('--solver', choices=SOLVERS, default='cps')
    parser.add_argument('--backend', choices=('cpp', 'python'), default=run['backend'])
    parser.add_argument('--car-count', type=int, default=settings['games'][game_name]['car_count'])
    parser.add_argument('--horizon', type=int, default=run['horizon'])
    parser.add_argument('--iterations', type=int, default=run['iterations'])
    parser.add_argument('--tolerance', type=float, default=run['tolerance'])
    parser.add_argument('--seed', type=int, default=run['seed'])
    parser.add_argument('--restarts', type=int, default=run['restarts'])
    parser.add_argument('--restart-std', type=float, nargs='+', default=run['restart_std'],
                        help='restart standard deviation per control, in game control order')
    parser.add_argument('--ablation-level', type=int, choices=(0, 1, 2), default=run['ablation_level'])
    parser.add_argument('--animate', action='store_true', help='also save a solution GIF')
    parser.add_argument('--initial', action='store_true', help='also render the initial control guess')
    parser.add_argument('--no-display', action='store_true', help='save files without opening windows')
    parser.add_argument('--output-dir', type=Path,
                        help='destination (default: outputs/single_runs/GAME_Ncar_seedSEED)')
    args = parser.parse_args()
    validate_args(parser, args)
    if args.car_count < 2:
        parser.error('--car-count must be at least 2')
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    logging.basicConfig(level=logging.INFO)
    destination = args.output_dir or (ROOT / 'outputs' / 'single_runs' /
                                      f'{game_name}_{args.car_count}car_seed{args.seed}')
    destination.mkdir(parents=True, exist_ok=True)
    game = create_game(game_name, args.car_count, args.horizon, args.seed, args.solver)
    solver = create_solver(args.solver, game, args.backend, args.iterations, args.ablation_level,
                           tolerance=args.tolerance, settings=settings)
    guess = initial_guess(game, game_name)
    try:
        solution = solve(solver, args.solver, args.backend, guess, args.restarts, args.restart_std)
        print(f'converged={solution.has_converged}, optimal={solution.is_optimal}, '
              f'residual={solution.residual:.6g}, runtime={solution.elapsed_time:.6f} s')
        np.savez_compressed(destination / 'solution.npz', u=solution.u, x=solution.x,
                            x0=game.config.x0, initial_u=guess, seed=args.seed,
                            solver=args.solver, residual=solution.residual,
                            elapsed_time=solution.elapsed_time,
                            has_converged=solution.has_converged, is_optimal=solution.is_optimal)
        controls = [('solution', solution.u, solution.x)]
        if args.initial:
            controls.insert(0, ('initial', guess, None))
        for label, u, x in controls:
            u = np.asarray(u).reshape((solver.m, solver.N, solver.T), order='F')
            if x is None:
                # Roll out initial controls independently of the solved trajectory.
                import casadi as cas
                gc = game.config
                rollout = cas.Function('initial_rollout',
                                       [gc.get_int_param_sx(), gc.get_double_param_sx()],
                                       [game.rollout(cas.DM(gc.x0), cas.DM(u.reshape((gc.m * gc.N, gc.T), order='F')))])
                future_x = np.asarray(rollout(gc.get_int_param_np(), gc.get_double_param_np()))
                full_x = np.dstack([gc.x0[:, :, None],
                                    future_x.reshape((gc.n, gc.N, gc.T), order='F')])
            else:
                full_x = solver._rollout_full_x(u, x)
            game.visualize(u, full_x, show=False, save_path=destination / f'{label}.png')
            if args.animate:
                game.animate(u, full_x, show=not args.no_display, save_gif=True,
                             save_path=destination / f'{label}.gif')
        history = np.asarray(getattr(solver, 'residual_vec', []), dtype=float)
        if not history.size:
            history = np.asarray([solution.residual])
        fig, ax = plt.subplots(layout='constrained')
        ax.plot(np.arange(1, len(history) + 1), history, 'o-')
        ax.axhline(solver.config.tolerance, color='tab:red', linestyle='--', label='Tolerance')
        if np.all(history > 0):
            ax.set_yscale('log')
        ax.set(xlabel='Iteration', ylabel='Residual', title=f'{game_name.title()} convergence')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(destination / 'residuals.png', dpi=180)
        if not args.no_display:
            plt.show()
        plt.close('all')
        print(f'Visualizations and solution: {destination}')
    finally:
        if hasattr(solver, 'close'):
            solver.close()
