""" Generate CasADi cpp source code for all games """
import argparse
import logging
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game as create_merge_game
from rd3g.games.intersection_casadi import create_random_game as create_intersection_game
from rd3g.utilities.casadi_util import generate_code, normalize_solver_codegen_key
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig
from scripts._settings import configure_parser

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_solver(solver_name, game, variational_gne):
    """Construct the requested solver for code generation."""
    solver_name = normalize_solver_codegen_key(solver_name)
    if solver_name == 'ipm':
        solver_config = InteriorPointGameConfig(variational_gne=variational_gne)
        return InteriorPointGame(solver_config, game)
    if solver_name == 'rd3g':
        if variational_gne:
            raise ValueError('RD3G CasADi codegen does not support variational_gne=True.')
        solver_config = RD3GCasadiConfig()
        return RD3GCasadi(solver_config, game)
    raise ValueError(f'Unsupported solver {solver_name!r}.')


def solver_variants(solver_name):
    """Return supported variational_gne settings for a solver."""
    solver_name = normalize_solver_codegen_key(solver_name)
    if solver_name == 'ipm':
        # The three linear-solver ablations share the same vGNE code.
        return [True]
    if solver_name == 'rd3g':
        return [False]
    raise ValueError(f'Unsupported solver {solver_name!r}.')


def generate_solver_variants(create_game, car_count, horizon, game_name, solvers):
    """Generate CasADi code for the selected solver keys."""
    for solver_name in solvers:
        solver_name = normalize_solver_codegen_key(solver_name)
        for variational_gne in solver_variants(solver_name):
            logger.info(
                f'Codegen for {game_name}, solver={solver_name}, {car_count=}, '
                f'{horizon=}, {variational_gne=}'
            )
            game = create_game(car_count=car_count, horizon=horizon,
                               variational_gne=variational_gne)
            solver = create_solver(solver_name, game, variational_gne)
            generate_code(solver, solver_key=solver_name)
            logger.info('Done!')


def codegen_merge(car_count, horizon, solvers=('ipm',)):
    generate_solver_variants(create_merge_game, car_count, horizon, 'merge', solvers)


def codegen_intersection(car_count, horizon, solvers=('ipm',)):
    generate_solver_variants(create_intersection_game, car_count, horizon, 'intersection', solvers)


def codegen_racing(car_count, horizon, solvers=('ipm',)):
    from rd3g.games.car_racing_casadi import create_random_game as create_racing_game
    generate_solver_variants(create_racing_game, car_count, horizon, 'racing', solvers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    settings = configure_parser(parser)
    parser.add_argument('--games', nargs='+', choices=('merge', 'intersection', 'racing'),
                        default=['merge', 'intersection', 'racing'])
    parser.add_argument('--solvers', nargs='+', choices=('ipm', 'rd3g'),
                        default=['ipm', 'rd3g'])
    parser.add_argument('--min-cars', type=int, default=settings['benchmark']['min_cars'])
    parser.add_argument('--max-cars', type=int, default=settings['benchmark']['max_cars'])
    parser.add_argument('--horizon', type=int, default=settings['run']['horizon'])
    args = parser.parse_args()
    if not 2 <= args.min_cars <= args.max_cars or args.horizon < 1:
        parser.error('require 2 <= --min-cars <= --max-cars and --horizon > 0')
    generators = {'merge': codegen_merge, 'intersection': codegen_intersection,
                  'racing': codegen_racing}
    for game in dict.fromkeys(args.games):
        for count in range(args.min_cars, args.max_cars + 1):
            generators[game](count, args.horizon, solvers=tuple(dict.fromkeys(args.solvers)))


if __name__ == '__main__':
    main()
