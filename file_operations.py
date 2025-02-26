import os
import logging
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import pandas as pd
from utils import (
    create_label,
    find_varying_parameter,
    get_last_line,
    get_multirun_path_from_output,
    get_relative_multirun_path,
)
import shutil
import yaml
import logging


logging.basicConfig(level=logging.WARNING)

log = logging.getLogger(__name__)

log.setLevel(logging.WARNING)


def parse_override_string(override):
    """Parse override string into a dictionary."""
    key, value = override.split("=")
    return {key.strip(): value.strip()}


def append_multirun_path(hydra_output_dir):
    log.info(f"Hydra output directory: {hydra_output_dir}")

    potential_multirun_path = get_multirun_path_from_output(hydra_output_dir)

    if os.path.exists(potential_multirun_path):
        relative_path = get_relative_multirun_path(hydra_output_dir)
        path_file = os.path.join(get_original_cwd(), "multirun_paths.txt")

        last_line = get_last_line(path_file)
        if last_line != relative_path:
            with open(path_file, "a") as f:
                f.write(f"{relative_path}\n")
            log.info(f"Multirun path appended to {path_file}")
        else:
            log.info("Multirun path already exists in the file. Not appending.")
    else:
        log.info("This is not a multirun execution. No path appended.")


def collect_overrides(override_file):
    """Collect overrides from a file into a dictionary."""
    with open(override_file, "r") as f:
        overrides = yaml.safe_load(f)
    return {
        k: v
        for override in overrides
        for k, v in parse_override_string(override).items()
    }


def find_csv_files(directory, csv_filename):
    csv_files = []
    log.info(f"Searching for CSV files in directory: {directory}")

    for root, dirs, files in os.walk(directory):
        log.info(f"Examining directory: {root}")
        log.info(f"Subdirectories: {dirs}")
        log.info(f"Files: {files}")
        for file in files:
            if file == csv_filename:
                csv_files.append(os.path.join(root, file))

    log.info(f"Found CSV files: {csv_files}")
    return csv_files


def collect_and_process_data(multirun_dir, csv_filename, cfg=None):
    csv_files = find_csv_files(multirun_dir, csv_filename)
    if not csv_files:
        log.error("No CSV files found.")
        return None, None, None

    dfs, all_overrides = [], []
    for file in csv_files:
        log.info(f"Processing file: {file}")
        df = pd.read_csv(file)
        df["run"] = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
        dfs.append(df)
        override_file = os.path.join(
            os.path.dirname(file), "..", ".hydra", "overrides.yaml"
        )
        all_overrides.append(collect_overrides(override_file))

    combined_df = pd.concat(dfs, ignore_index=True)
    varying_param = find_varying_parameter(all_overrides)
    labels = [
        f"Run {i}"
        if not varying_param
        else create_label(varying_param, ovr[varying_param])
        for i, ovr in enumerate(all_overrides)
    ]
    return combined_df, labels, varying_param


def get_multirun_path_by_index(index: int) -> str:
    project_root = get_original_cwd()
    path_file = os.path.join(project_root, "multirun_paths.txt")
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            paths = f.readlines()
        if paths:
            if index < len(paths):
                return paths[-1 - index].strip()
            else:
                log.warning(
                    f"Requested multirun {index} is out of range. Using the oldest available."
                )
                return paths[0].strip()
    else:
        log.warning("No multirun_paths.txt file found. Using current directory.")
        return os.getcwd()


def get_multirun_path(cfg: DictConfig) -> str:
    log.info("Getting multirun path from configuration.")
    if cfg.get('dir', None):
        log.info(f"Using directory from config: {cfg.dir}")
        return cfg.dir
    if isinstance(cfg.path, int):
        log.info(f"Using path index from config: {cfg.path}")
        return get_multirun_path_by_index(cfg.path)
    else:
        resolved_path = os.path.abspath(os.path.join(get_original_cwd(), cfg.path))
        log.info(f"Using resolved path from config: {resolved_path}")
        return resolved_path


def save_combined_data(combined_df: pd.DataFrame, filename: str, cfg) -> None:
    """Save the combined data to a CSV file"""
    if cfg.save.enabled:
        data_dir = os.path.join(cfg.save.folders.data)
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, filename)
        combined_df.to_csv(file_path, index=False)
        log.info(f"Combined data saved to {file_path}")
