from math import floor, log10
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


def format_value(value):
    if value is None:
        return "default"
    if isinstance(value, (int, float)):
        abs_value = abs(value)
        if abs_value >= 1000 or (abs_value < 0.01 and abs_value != 0):
            exponent = floor(log10(abs_value))
            mantissa = value / (10**exponent)
            return f"${mantissa:.1f} \\cdot 10^{{{exponent}}}$"
        elif isinstance(value, int):
            return f"${value}$"
        else:  # float
            return f"${value:.2f}$".rstrip("0").rstrip(".")
    return value  # Return non-numeric values as-is


def generate_latex_table(run_details_df):
    """
    Generate a LaTeX table from a DataFrame containing run details.

    Parameters:
    -----------
    run_details_df : pd.DataFrame
        DataFrame containing one row of parameters and statistics

    Returns:
    --------
    str
        LaTeX formatted table
    """
    # Lookup dictionary for LaTeX representations
    latex_lookup = {
        "N": r"$N$",
        "K": r"$K$",
        "beta": r"$\beta$",
        "k": r"$k$",
        "u": r"$u$",
        "tau_s": r"$\tau_s$",
        "tau_J": r"$\tau_J$",
        "tau_h": r"$\tau_h$",
        "T": r"$T$",
        "dt": r"$\Delta t$",
        "rho": r"$\rho$",  # Added rho
        "pattern_distribution": "Pattern initialization",
        "prob_distribution": "Stimuli distribution",
        "offset_variance": r"$\mathrm{Var}(\sigma_{\Delta})$",
        "offset_std": r"$\sigma_{\Delta}$",
    }

    # Get parameters from DataFrame
    if len(run_details_df) > 1:
        raise ValueError("Expected DataFrame with single row of parameters")

    params = run_details_df.iloc[0].to_dict()

    # Count the number of parameters
    num_params = len(params)

    # Generate table header and rows
    header = []
    values = []
    for key, value in params.items():
        latex_key = latex_lookup.get(key, key)
        header.append(latex_key)
        values.append(format_value(value))

    # Generate LaTeX table
    table = f"""
\\begin{{tabular}}{{l{' c' * (num_params - 1)}}}
\\toprule
{' & '.join(header)} \\\\
\\midrule
{' & '.join(values)} \\\\
\\bottomrule
\\end{{tabular}}
"""

    return table


def create_run_details_statistics(simulation_dict: Dict[str, Any], cfg: Any) -> None:
    """
    Create statistics from simulation data and save to CSV.

    Parameters:
    -----------
    simulation_dict : Dict[str, Any]
        Dictionary containing simulation data including offset_from_theoretical
    cfg : Any
        Configuration object containing model parameters and save settings

    The function saves a CSV file with:
    - All model parameters
    - Variance and standard deviation of the residuals
    """
    # Extract residuals data diff_normF2_k

    residual = simulation_dict.get("diff_normF2_k") - simulation_dict.get(
        "theoretical_y"
    )

    # Ensure offset_data is 1D
    residual = np.array(residual).flatten()

    # Calculate statistics
    stats = {"residuals_variance": np.var(residual), "residuals_std": np.std(residual)}
    std = {"theoretical_std_last_step": (np.array(simulation_dict.get("theoretical_std")).flatten()[-1])}
    # Collect all parameters from model config
    params = {
        "N": cfg.model.N,
        "K": cfg.model.K,
        "beta": cfg.model.beta,
        "k": cfg.model.k if cfg.model.k is not None else np.tanh(cfg.model.beta),
        "u": cfg.model.u,
        "tau_s": cfg.model.tau_s,
        "tau_J": cfg.model.tau_J,
        "rho": cfg.model.rho,
        "pattern_distribution": cfg.model.pattern_distribution,
        "prob_distribution": cfg.model.prob_distribution,
        "tau_h": cfg.model.tau_h,
        "T": cfg.simulation.T,
        "dt": cfg.simulation.dt,
    }

    # Combine parameters and statistics
    all_stats = {**params, **stats, **std}

    # Create DataFrame
    df = pd.DataFrame([all_stats])

    # Save to CSV
    if cfg.save.enabled:
        os.makedirs("data", exist_ok=True)
        filename = f"run_stats.csv"
        # home_path = cfg.hydra.run.dir
        # dir_path = os.path.join(home_path, cfg.save.folders.static_vis)
        # filepath = os.path.join( cfg.save.folders.data, filename)
        filepath = os.path.join("data", filename)
        df.to_csv(filepath, index=False)
        log.info(f"Statistics saved to {filepath}")

    return df
