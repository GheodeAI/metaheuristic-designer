# Examples

This directory contains ready‑to‑run scripts that demonstrate how to apply
the metaheuristic‑designer library to different optimisation tasks.
Each script accepts command‑line arguments so you can experiment without
changing the code.

## Contents

| Script | What it shows |
|--------|---------------|
| `exec_basic.py` | Continuous problems (Sphere, Rastrigin, etc.) with the full API, including CMA‑ES, CEM, Bayesian Opt. |
| `exec_basic_simple.py` | Same problems, but using the `simple` module – GA, DE, PSO, SA, ES. |
| `image_evolution.py` | Evolving an image to match a reference (MSE, MAE, SSIM) or maximise entropy/contrast. Uses `pygame` for live display. |
| `image_evolution_simple.py` | Image evolution via the `simple` wrappers with live display. |
| `lambda_test.py` | Constructing a whole algorithm from scratch using `*FromLambda` components. |
| `np_problems.py` | NP‑hard problems: 0‑1 Knapsack, 3‑SAT, Maximum Clique, TSP. Full API. |
| `np_problems_simple.py` | Same NP‑hard problems with the `simple` wrappers. |

## Running an example

```bash
python examples/exec_basic.py -a ga -o sphere -d 5 --seed 42
```

Most scripts support `-h` / `--help` to list all options.  Common arguments:

- `-a` / `--algorithm` – which optimizer to use
- `-o` / `--objective` – the problem to solve
- `-r` / `--seed` – random seed for reproducibility
- `--log` – set logging level (`DEBUG`, `INFO`, `WARNING`)
- `-v` / `--reporter` – choose `tqdm`, `silent`, or `verbose`

## Required data files

Some examples read external data that is **not** included in the repository:

- `image_evolution.py` and `image_evolution_simple.py` expect an image at
  `data/images/cat.png`.  Place any RGB image there (a 32×32 PNG works
  out‑of‑the‑box) or change the `--image` argument.
- `np_problems.py` and `np_problems_simple.py` need:
  - `data/sat_examples/uf50-03.cnf` for 3‑SAT
  - `data/tsp_examples/r50_01.csv` for TSP (the script actually uses `r20_02.csv`, adjust as needed)

You can download the standard TSPLIB format files and convert them to CSV
(see the `benchmarks.TSP.from_csv` docstring for the expected columns).

## Dependencies

The examples require the library’s optional dependencies:

```bash
pip install metaheuristic-designer[examples]
```

This installs `seaborn`, `matplotlib`, `networkx`, `pygame`, and
`opencv‑python` – everything needed for the live display and plotting.

## Notes

- The `MemeticAlgorithm` class is used in some examples to wrap a local
  search around the main loop.  It demonstrates Baldwinian vs. Lamarckian
  strategies.
- All scripts use a fixed random seed by default – change it to see how
  different runs behave.
- While the scripts print the best solution found, they do **not** produce
  plots.  For plotting, see the tutorials or the `HistoryTracker` API.