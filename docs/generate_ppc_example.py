"""
Generate the posterior predictive check (PPC) example figure used in the README.

This is a self-contained demonstration: it draws synthetic peaks-over-threshold
exceedances from a known Generalised Pareto distribution, fits the model with the
project's own Bayesian MCMC routine, runs the posterior predictive check, and
writes the resulting figure to ``docs/ppc_example.png``.

The two analysis functions are loaded directly from their source files so the
figure can be regenerated without starting the Dash dashboard.

Run from the project root:

    python docs/generate_ppc_example.py
"""

from pathlib import Path
import importlib.util
import logging
import sys
import types

import numpy as np
from scipy.stats import genpareto

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POT = PROJECT_ROOT / "src" / "plotting" / "extreme_value_analysis_pot"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Build a minimal package skeleton so ppc_plots' relative import (``..constants``)
# resolves, without importing the package __init__ (which loads the dashboard).
_pkg = types.ModuleType("evapot")
_pkg.__path__ = []  # mark as package
sys.modules["evapot"] = _pkg
_plotting = types.ModuleType("evapot.plotting")
_plotting.__path__ = []
sys.modules["evapot.plotting"] = _plotting

_load("evapot.constants", POT / "constants.py")
bayes = _load("evapot.bayesian_analysis", POT / "bayesian_analysis.py")
ppc = _load("evapot.plotting.ppc_plots", POT / "plotting" / "ppc_plots.py")


def main() -> None:
    # Synthetic exceedances from a known GPD (shape xi=0.1, scale sigma=1.0).
    rng = np.random.default_rng(42)
    exceedances = genpareto.rvs(0.1, loc=0, scale=1.0, size=180, random_state=rng)

    stats = bayes.perform_bayesian_gpd_stationary(exceedances)
    if stats is None:
        raise SystemExit("Bayesian GPD fit failed.")

    fig, summary = ppc.generate_posterior_predictive_checks_gpd(
        stats, exceedances, analysis_type="bayesian_stationary", n_sim=200
    )

    print("\nPosterior predictive summary (observed vs 90% posterior interval):")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    out_path = PROJECT_ROOT / "docs" / "ppc_example.png"
    fig.write_image(str(out_path), width=1000, height=600, scale=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
