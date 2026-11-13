# Native solver backends

The C++ extensions accelerate the CasADi interior-point and RD3G solvers.
Game residuals, Jacobians, rollouts, and per-agent KKT matrices are generated
from the Python game implementations; separate hand-written C++ games are unnecessary.

From the repository root, after installing the dependencies in the
[main README](../README.md#installation):

```bash
uv run python -m scripts.generate_casadi_code
uv run cmake -S cpp -B build -DPython_EXECUTABLE="$PWD/.venv/bin/python"
uv run cmake --build build --parallel 2
```

CMake builds current generated sources for merge, intersection, and racing,
including both IPM/vGNE and RD3G/GNE variants. Native modules and generated shared
libraries are placed in `build/lib/`, where the Python wrappers load them.

Each game, solver, car count, and horizon requires a matching generated library.
Regenerate and rebuild after changing dynamics, costs, constraints, or symbolic
parameter layouts. Run commands from the repository root. For a small build,
select games, solvers, and sizes with `generate_casadi_code --help`.
