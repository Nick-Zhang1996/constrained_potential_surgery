"""Game creation and solver settings shared by the supported CLI commands."""

from pathlib import Path

import numpy as np

from scripts._settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ('cps', 'pdo', 'ipm', 'rd3g', 'algames', 'ilqgame')
BENCHMARK_SOLVERS = ('cps', 'rd3g', 'algames')


def create_game(name, car_count, horizon, seed, solver_name='cps'):
    """Use identical seeded physical games across methods; RD3G uses GNE duals."""
    if name == 'merge':
        from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
    elif name == 'intersection':
        from rd3g.games.intersection_casadi import create_random_game
    elif name == 'racing':
        from rd3g.games.car_racing_casadi import create_random_game
    else:
        raise ValueError(f'Unknown game {name!r}')
    np.random.seed(seed)
    return create_random_game(car_count=car_count, horizon=horizon,
                              variational_gne=solver_name != 'rd3g')


def create_solver(name, game, backend=None, iterations=None, ablation_level=None,
                  tolerance=None, settings=None):
    """Build a solver; Python and Julia baselines use their native backends."""
    settings = load_settings() if settings is None else settings
    run = settings['run']
    backend = run['backend'] if backend is None else backend
    options = dict(settings['solvers'][name])
    options['iterations'] = run['iterations'] if iterations is None else iterations
    options['tolerance'] = run['tolerance'] if tolerance is None else tolerance
    if name in {'cps', 'pdo', 'ipm'}:
        from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
        options['ablation_level'] = run['ablation_level'] if ablation_level is None else ablation_level
        config = InteriorPointGameConfig(**options)
        solver = InteriorPointGame(config, game, cpp_only=backend == 'cpp')
    elif name == 'rd3g':
        from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig
        solver = RD3GCasadi(RD3GCasadiConfig(**options), game, cpp_only=backend == 'cpp')
    elif name == 'ilqgame':
        from rd3g.solvers.ilqgame import ILQGame, ILQGameConfig
        return ILQGame(ILQGameConfig(**options), game)
    elif name == 'algames':
        from rd3g.solvers.algames_julia import AlgamesJulia, AlgamesJuliaConfig
        return AlgamesJulia(AlgamesJuliaConfig(**options), game)
    else:
        raise ValueError(f'Unknown solver {name!r}')
    if backend == 'cpp':
        solver.init_cpp_backend()
    return solver


def reset_solver(solver, game):
    """Reuse symbolic functions and native libraries across same-sized games."""
    solver.game = game
    for name, value in (
        ('x0', game.config.x0), ('reg', getattr(solver.config, 'reg', None)),
        ('rho', getattr(solver.config, 'rho_0', None)),
        ('line_search_fail_count', 0), ('residual_vec', []), ('violation_vec', []),
        ('barrier_weight', getattr(solver.config, 'barrier_weight', None)),
        ('last_policy', None),
    ):
        if hasattr(solver, name):
            setattr(solver, name, value)


def initial_guess(game, game_name):
    if game_name == 'racing':
        return np.asfortranarray(game.initial_control_guess())
    gc = game.config
    return np.zeros((gc.m * gc.N, gc.T), order='F')


def solve(solver, name, backend, guess, restarts=0, restart_std=None):
    if name == 'algames':
        return solver.solve()
    if backend == 'cpp' and name in {'cps', 'pdo', 'ipm', 'rd3g'}:
        if name in {'cps', 'pdo', 'ipm'} and restarts:
            return solver.solve_cpp_backend_rand_restart(restarts=restarts, u_ref=guess,
                                                         control_std=restart_std)
        return solver.solve_cpp_backend(guess)
    return solver.solve(guess)


def validate_args(parser, args):
    if args.horizon < 1 or args.iterations < 1 or args.restarts < 0:
        parser.error('require positive --horizon/--iterations and nonnegative --restarts')
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('--tolerance must be finite and positive')
    selected = [args.solver] if hasattr(args, 'solver') else args.solvers
    if getattr(args, 'ablation_level', 0) and any(s != 'pdo' for s in selected):
        parser.error('--ablation-level 1/2 requires selecting only pdo')
    if args.restarts and any(s not in {'cps', 'pdo', 'ipm'} for s in selected):
        parser.error('--restarts requires selecting only cps/pdo/ipm')
    if args.restarts and args.backend != 'cpp':
        parser.error('--restarts requires --backend cpp')
    if args.restarts and not args.restart_std:
        parser.error('--restarts requires --restart-std in the game control order')
    if any(not np.isfinite(v) or v < 0 for v in args.restart_std):
        parser.error('--restart-std values must be finite and nonnegative')
