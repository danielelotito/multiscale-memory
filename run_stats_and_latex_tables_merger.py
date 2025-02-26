import os
import pandas as pd
import logging
from typing import Optional
from fluctuation_analysis import compute_theoretical_std

log = logging.getLogger(__name__)


def collect_and_merge_run_stats(multirun_dir: str, cfg) -> Optional[pd.DataFrame]:
    """
    Collect and merge run_stats.csv files from multiple simulation runs.

    Parameters
    ----------
    multirun_dir : str
        Directory containing multiple run directories
    cfg : DictConfig
        Configuration object containing save settings

    Returns
    -------
    Optional[pd.DataFrame]
        Combined DataFrame containing all run statistics, or None if no files found
    """
    # Find all run_stats.csv files
    csv_files = find_files(multirun_dir, ends_with="run_stats.csv")
    # csv_files = find_csv_files(multirun_dir, "run_stats.csv")
    if not csv_files:
        log.error("No run_stats.csv files found.")
        return None

    log.info(f"Found {len(csv_files)} run_stats.csv files")

    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add run number based on directory name
            run_dir = os.path.dirname(os.path.dirname(file))
            run_number = int(os.path.basename(run_dir))
            df["run"] = run_number
            dfs.append(df)
        except Exception as e:
            log.error(f"Error reading {file}: {str(e)}")
            continue

    if not dfs:
        log.error("No valid data found in CSV files.")
        return None

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by run number
    combined_df = combined_df.sort_values("run")


    # Save combined data if enabled
    if cfg.save.enabled:
        # output_dir = os.path.dirname(cfg.save.folders.data)
        os.makedirs(cfg.save.folders.data, exist_ok=True)
        output_file = os.path.join(cfg.save.folders.data, "combined_run_stats.csv")
        combined_df.to_csv(output_file, index=False)
        log.info(f"Combined run statistics saved to {output_file}")

    return combined_df


def merge_latex_tables(directory, output_file):
    latex_files = find_files(directory, ends_with="latex_table.tex")
    if not latex_files:
        log.error("No LaTeX table files found.")
        return

    tables = []
    for file in latex_files:
        with open(file, "r") as f:
            content = f.read()
            tables.append(parse_latex_table(content))

    if not all(len(table["header"]) == len(tables[0]["header"]) for table in tables):
        log.error("LaTeX tables have different numbers of columns.")
        return

    if not all(table["header"] == tables[0]["header"] for table in tables):
        log.error("LaTeX tables have different headers.")
        return

    merged_table = merge_tables(tables)
    write_merged_table(merged_table, output_file)


def find_files(directory, ends_with=".tex"):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(ends_with)
    ]


def parse_latex_table(content):
    lines = content.strip().split("\n")
    header = None
    data = None
    for line in lines:
        if "\\\\" in line:
            if header is None:
                header = line.strip().rstrip("\\\\").split("&")
            else:
                data = line.strip().rstrip("\\\\").split("&")
    return {"header": header, "data": data}


def merge_tables(tables):
    header = tables[0]["header"]
    data = [table["data"] for table in tables]
    return {"header": header, "data": data}


def write_merged_table(merged_table, output_file):
    with open(output_file, "w") as f:
        f.write(
            "\\begin{tabular}{" + "l" + " c" * (len(merged_table["header"]) - 1) + "}\n"
        )
        f.write("\\toprule\n")
        f.write(" & ".join(merged_table["header"]) + " \\\\\n")
        f.write("\\midrule\n")
        for row in merged_table["data"]:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def collect_and_merge_latex_tables(multirun_dir, cfg):
    if cfg.merge_tex:
        output_dir = os.path.dirname(cfg.save.folders.data)
        output_file = os.path.join(output_dir, "combined_table.tex")
        merge_latex_tables(multirun_dir, output_file)
        log.info(f"Merged LaTeX table saved to {output_file}")
