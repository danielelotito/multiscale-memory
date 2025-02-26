import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from collection_functions import get_combined_stats

from fluctuation_analysis import create_fluctuation_plots

from file_operations import (
    collect_and_process_data,
    get_multirun_path,
    save_combined_data,
)


from run_stats_and_latex_tables_merger import (
    collect_and_merge_latex_tables,
)
from utils import generate_identifier
from visualization import create_comparison_plots
from config_utils import get_first_run_config

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@hydra.main(
    config_path="config_multirun_collection",
    config_name="config_multirun_collection",
    version_base="1.1",
)
def main(cfg: DictConfig):
    os.chdir(get_original_cwd())
    multirun_dir = get_multirun_path(cfg)

    # Collect and merge run statistics
    combined_stats = get_combined_stats(multirun_dir, cfg)
    create_fluctuation_plots(combined_stats, cfg)


    if len(combined_stats) > cfg.max_runs_to_process:
        log.info(f"Too many runs to process: {len(combined_stats)}, plotting only combined stats")
        return

    combined_df, labels, varying_param = collect_and_process_data(
        multirun_dir, cfg.csv_filename, cfg=cfg
    )


    if combined_df is not None:
        # Get first run config for identifier generation
        first_cfg = get_first_run_config(multirun_dir)
        if first_cfg:
            identifier = generate_identifier(first_cfg, varying_param)
            save_combined_data(combined_df, f"cdata_{identifier}.csv", cfg)

        if hasattr(cfg.visualizations, "static") and cfg.visualizations.static:
            create_comparison_plots(combined_df, cfg, varying_param, labels)

    # Maintain existing latex table collection
    collect_and_merge_latex_tables(multirun_dir, cfg)

if __name__ == "__main__":
    main()
