import os
from typing import Dict
import logging

from run_stats_and_latex_tables_merger import collect_and_merge_run_stats
from fluctuation_analysis import compute_theoretical_std

import pandas as pd

log = logging.getLogger(__name__)


def create_label(key, value):
    """Create a label based on the key and value."""
    # Extract the last part of the key if it contains dots
    label_key = key.split(".")[-1]

    if label_key.lower() == "beta":
        return rf"$\beta={value}$"
    elif label_key.lower() == "tau_j":
        return rf"$\tau_J={value}$"
    elif label_key.lower() == "tau_s":
        return rf"$\tau_\sigma={value}$"
    else:
        return f"${label_key}={value}$"


def save_latex_table(table: str, filename: str, cfg) -> None:
    """Save LaTeX table to file."""
    if cfg.save.enabled:
        os.makedirs(cfg.save.folders.data, exist_ok=True)
        file_path = os.path.join(cfg.save.folders.data, filename)
        with open(file_path, "w") as f:
            f.write(table)
        log.info(f"LaTeX table saved to {file_path}")


def get_combined_stats(multirun_dir, cfg):
    if isinstance(cfg.get('combined_stats_path', None), str):
        try:
            combined_df = pd.read_csv(cfg.combined_stats_path)
        except FileNotFoundError:
            log.error(f"Could not find combined stats file at {cfg.combined_stats_path}")
            log.warning("Collecting and merging run stats from the multirun directory instead")
            combined_df = collect_and_merge_run_stats(multirun_dir, cfg)
    else:
        combined_df = collect_and_merge_run_stats(multirun_dir, cfg)

    if cfg.get('append_combined_stats_path', None) is not None:
        df_to_append = pd.read_csv(cfg.get('append_combined_stats_path'))
        combined_df = pd.concat([combined_df, df_to_append])
        combined_df = combined_df.sort_values(by=['K', 'tau_J'], ascending=[True, True])
        
    if ("theoretical_std" not in combined_df.columns) or cfg.get("recompute_theoretical_std", False):
        log.warning("Computing theoretical standard deviation from a function defined in fluctuation_analysis.py")
        combined_df['theoretical_std'] = compute_theoretical_std(
            combined_df['N'], 
            combined_df['K'],
            combined_df['tau_J'], 
            combined_df['rho'],
        )
    
    return combined_df
