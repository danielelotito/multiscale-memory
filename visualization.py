import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from detect_stationarity import plot_stationarity
from utils import compute_overlap, autocorrelation
import os
from typing import Dict,  List, Optional
import pandas as pd
import logging

from datetime import datetime
from setup_matplotlib_style import setup_matplotlib_style


logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



def create_comparison_plots(
    combined_df: pd.DataFrame, cfg, varying_param: str, labels: List[str]
) -> None:
    """Create multiple comparison plots based on visualization config."""
    setup_matplotlib_style(big_fonts=True)
    if not combined_df.empty and hasattr(cfg.visualizations, "static"):
        plot_types = cfg.visualizations.static
        n_plots = len(plot_types)

        if n_plots == 0:
            return

        # Setup subplot grid
        n_rows = (n_plots + 1) // 2
        n_cols = min(2, n_plots)

        fig = plt.figure(figsize=(15 * n_cols, 8 * n_rows))

        # Store colors for each run to ensure consistency
        run_colors = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, run in enumerate(combined_df["run"].unique()):
            run_colors[run] = color_cycle[i % len(color_cycle)]
            
            
        for i, plot_name in enumerate(plot_types):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax._expected_series = len(labels)
            # Get the plot type from the map
            plot_type = PLOT_TYPE_MAP.get(plot_name)
            if not plot_type:
                log.warning(f"Unknown plot type: {plot_name}")
                continue

            # Get the plot configuration
            plot_config = PLOT_CONFIGS.get(plot_type)
            if not plot_config:
                log.warning(f"No configuration found for plot type: {plot_type}")
                continue

            for run, label in zip(combined_df["run"].unique(), labels):
                run_data = combined_df[combined_df["run"] == run]
                run_data = combined_df[combined_df["run"] == run]
                color = run_colors[run]  # Use consistent color for this runl  
                print(color)
                log.info(f"Plotting {plot_name} for run {run} with label {label}, color {color}") 
                # Get data according to configuration
                y_data_col = plot_config["y_data"]
                if y_data_col not in run_data:
                    log.warning(f"Missing data column: {y_data_col}")
                    continue

                x_data = run_data["time"].values
                y_data = run_data[y_data_col].values

                # Get additional data for plots
                theoretical_y = (
                    run_data["theoretical_y"].values
                    if "theoretical_y" in run_data
                    else None
                )
                asymptote_y = (
                    run_data["horizontal_line_y"].iloc[0]
                    if "horizontal_line_y" in run_data
                    else None
                )
                theoretical_std = (
                    run_data["theoretical_std"].values
                    if "theoretical_std" in run_data
                    else None
                )

                if plot_config.get("residuals", False):
                    log.debug(
                        "Plotting residuals, subtracting theoretical data and setting theoretical data to None"
                    )
                    y_data = y_data - theoretical_y
                    theoretical_y = None
                    asymptote_y = None
                    
                if plot_config.get("no_theoretical", False):
                    theoretical_y = None
                    asymptote_y = None
                    
                # Use the existing plot_metric function
                plot_metric(
                    ax=ax,
                    x=x_data,
                    y=y_data,
                    plot_type=plot_type,
                    label=label,
                    theoretical_y=theoretical_y,
                    asymptote_y=asymptote_y,
                    log_X=cfg.visualizations.get("log_X", False),
                    log_y=cfg.visualizations.get("log_y", False),
                    theoretical_std=theoretical_std,
                    color=color
                )

        if varying_param and cfg.get("global_title", False):
            param_name = varying_param.split(".")[-1]
            fig.suptitle(f"Results varying {param_name}")

        plt.tight_layout()

        if cfg.save.enabled:
            save_plot(
                fig,
                f"comparison_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                cfg,
            )
        plt.close()


# Plot configurations
PLOT_TYPE_MAP = {
    "plot_energy_mr": "energy",
    "plot_synaptic_hebb_normF_difference_mr": "fnorm",
    "plot_synaptic_hebb_normF_difference_mr_bnorm": "fnorm_normalized",
    "plot_synaptic_hebb_normF_difference_mr_th": "fnorm_theoretical",
    "fnorm_theoretical": "fnorm_theoretical",
    "fnorm_residuals": "fnorm_residuals",
    "residuals": "fnorm_residuals",
    "plot_delta_residuals": "fnorm_residuals",
    "deltaFnorm2_kt": "fnorm_theoretical",
    "deltaFnorm2_kt_no_theoretical": "fnorm_no_theoretical",
    "energy": "energy",
    "fnorm_residuals_hist": "fnorm_residuals_hist",
    "fluctuation_scaling": "fluctuation_scaling",
    "fluctuation_scaling_std": "fluctuation_scaling_std",
    "fluctuation_scaling_variance": "fluctuation_scaling_variance",
    "scaling": "fluctuation_scaling_std",
}
PLOT_CONFIGS = {
    "energy": {"title": "System energy", "ylabel": "Energy", "y_data": "energies"},
    "fnorm": {
        "title": r"Difference between $J$ and $k J^{(Hebb)}$",
        "ylabel": r"$\|\Delta^{(n)}\|^2_{\mathcal{F}}$",
        "y_data": "diff_normF2",
    },
    "fnorm_normalized": {
        "title": r"Difference between $J$ and $k J^{(Hebb)}$",
        "ylabel": r"$\|\Delta^{(n)}\|^2_{\mathcal{F}}/k^2$",
        "y_data": "diff_normF2_k",
    },
    "fnorm_theoretical": {
        "title": r"Difference between $J$ and $k J^{(Hebb)}$",
        "ylabel": r"$\|\Delta^{(n)}\|^2_{\mathcal{F}}/k^2$",  
        "y_data": "diff_normF2_k",
    },
    "fnorm_no_theoretical": {
        "title": r"Difference between $J$ and $k J^{(Hebb)}$",
        "ylabel": r"$\|\Delta^{(n)}\|^2_{\mathcal{F}}/k^2$",  
        "y_data": "diff_normF2_k",
        "no_theoretical": True,
    },
    "fnorm_residuals": {
        "title": r"Residuals from theoretical prediction",
        "ylabel": r"$\|\Delta^{(n)}\|^2_{\mathcal{F}}/k^2 - E(\|\Delta^{(n)}\|^2_{\mathcal{F}})/k^2$",
        "y_data": "diff_normF2_k",
        "residuals": True,
    },
    "fnorm_residuals_hist": {
        "title": r"$\|\Delta^{(n)}\|^2_{\mathcal{F}}/k^2 - E(\|\Delta^{(n)}\|^2_{\mathcal{F}})/k^2$",
        "ylabel": "Density",
        "y_data": "diff_normF2_k",
    },
    "fluctuation_scaling_variance": {
        "title": "Fluctuation Scaling Analysis",
        "ylabel": "Var(Residuals)",
        "y_data": "diff_normF2_k",
    },
    "fluctuation_scaling_std": {
        "title": "Fluctuation Scaling Analysis",
        "ylabel": "Std(Residuals)",
        "y_data": "diff_normF2_k",
    },
}

def save_plot(fig, filename, cfg):
    filename = filename.replace(" ", "_")
    if cfg.save.enabled:
        # Get the Hydra working directory
        save_dir = os.path.join(os.getcwd(), cfg.save.folders.static_vis)
        os.makedirs(save_dir, exist_ok=True)
        fig_path_and_name = os.path.join(save_dir, filename)
        fig.savefig(fig_path_and_name)
        log.info(f"Figure saved, you can find it at {fig_path_and_name}")

def add_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def save_plot_data(
    times, data, theoretical_y=None, horizontal_line_y=None, filename=None
):
    """Save plot data to CSV file."""
    if not filename:
        return

    plot_data = {
        "times": times,
        "diff": data.ravel() if isinstance(data, np.ndarray) else data,
    }
    if theoretical_y is not None:
        plot_data["theoretical_y"] = (
            theoretical_y.ravel()
            if isinstance(theoretical_y, np.ndarray)
            else theoretical_y
        )
    if horizontal_line_y is not None:
        plot_data["horizontal_line_y"] = [horizontal_line_y] * len(times)

    df = pd.DataFrame(plot_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv(os.path.join("data", filename), index=False)


def create_side_histogram(
    hist_ax, y, theoretical_std=None, log_y=False, color=None, label=None
):
    """Create a side histogram with statistics and theoretical predictions."""
    # Calculate statistics
    mean_y = np.mean(y)
    std_y = np.std(y)

    # Use same color as main plot
    hist_color = color if color is not None else "skyblue"

    # Create histogram
    n, bins, patches = hist_ax.hist(
        y,
        bins="auto",
        orientation="horizontal",
        density=True,
        alpha=0.3,
        color=hist_color,
    )

    # Add normal distribution fit with matching color
    y_range = np.linspace(np.min(y), np.max(y), 100)
    from scipy.stats import norm

    hist_ax.plot(norm.pdf(y_range, mean_y, std_y), y_range, "--", color=hist_color)

    # Add zero line (only once)
    if not hasattr(hist_ax, "_zero_line_added"):
        hist_ax.axhline(0, color="black", linestyle="-", alpha=0.5)
        hist_ax._zero_line_added = True

    # Add empirical mean and std lines with matching color
    hist_ax.axhline(mean_y, color=hist_color, linestyle="-", alpha=0.8)
    hist_ax.axhline(mean_y + std_y, color=hist_color, linestyle="--", alpha=0.8)
    hist_ax.axhline(mean_y - std_y, color=hist_color, linestyle="--", alpha=0.8)

    # Add theoretical offset if provided (with dotted lines)
    if theoretical_std is not None:
        mean_offset = np.mean(theoretical_std)
        if not hasattr(hist_ax, "_theo_lines_added"):
            hist_ax.axhline(mean_offset, color="black", linestyle=":", alpha=0.8)
            hist_ax.axhline(-mean_offset, color="black", linestyle=":", alpha=0.8)
            hist_ax._theo_lines_added = True

    # Formatting
    max_density = max(n)
    hist_ax.set_xlim(0, max_density * 1.2)
    hist_ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    hist_ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    hist_ax.set_xlabel("Density", labelpad=10)
    if log_y:
        hist_ax.set_yscale("log")
    hist_ax.grid(True, linestyle="--", alpha=0.3)

    # Remove redundant y-axis labels
    hist_ax.set_yticklabels([])

    return n


def plot_metric(
    ax,
    x,
    y,
    plot_type,
    label=None,
    theoretical_y=None,
    asymptote_y=None,
    log_X=False,
    log_y=False,
    theoretical_std=None,
    color = None
):
    """Updated plot_metric function to handle fluctuation scaling."""
    config = PLOT_CONFIGS.get(plot_type, PLOT_CONFIGS["fnorm"])

    if plot_type == "fluctuation_scaling_std":
        # For fluctuation scaling, we need all data series at once
        if not hasattr(ax, "_scaling_data"):
            ax._scaling_data = {"data": [], "theo": [], "labels": []}

        # Collect data
        ax._scaling_data["data"].append(y)
        ax._scaling_data["theo"].append(theoretical_y)
        ax._scaling_data["labels"].append(label)

        # Plot when we have all data
        if len(ax._scaling_data["labels"]) == ax._expected_series:
            plot_fluctuation_scaling_std(
                ax,
                ax._scaling_data["data"],
                ax._scaling_data["theo"],
                ax._scaling_data["labels"],
                log_X=log_X,
                log_y=log_y,
            )
        return

    if plot_type == "fluctuation_scaling_variance":
        # For fluctuation scaling, we need all data series at once
        if not hasattr(ax, "_scaling_data"):
            ax._scaling_data = {"data": [], "theo": [], "labels": []}

        # Collect data
        ax._scaling_data["data"].append(y)
        ax._scaling_data["theo"].append(theoretical_y)
        ax._scaling_data["labels"].append(label)

        # Plot when we have all data
        if len(ax._scaling_data["labels"]) == ax._expected_series:
            plot_fluctuation_scaling_variance(
                ax,
                ax._scaling_data["data"],
                ax._scaling_data["theo"],
                ax._scaling_data["labels"],
                log_X=log_X,
                log_y=log_y,
            )
        return

    if plot_type == "fnorm_residuals_hist":
        # For histogram plot, use the dedicated function
        plot_residuals_histogram(
            ax,
            y,
            theoretical_y,
            theoretical_std=theoretical_std,
            log_y=log_y,
            label=label,
        )
        return

    # Get color from current color cycle properly
    # if label:  # Only get new color if this is a labeled line
    #     try:
    #         color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
    #             len(ax.lines) % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    #         ]
    #     except:
    #         color = f"C{len(ax.lines)}"  # Fallback to default color cycle
    # else:
    #     color = None

    # if (not asymptote_y) and (theoretical_y is None) and plot_type != "energy":
    #     if not hasattr(ax, "_hist_ax"):
    #         divider = make_axes_locatable(ax)
    #         ax._hist_ax = divider.append_axes("right", size="25%", pad=0.1)

    
    # Use provided color or get from cycle
    plot_color = color if color is not None else plt.gca()._get_lines.get_next_color()
    
    y = np.ravel(y) if isinstance(y, np.ndarray) else y
    if theoretical_y is not None:
        theoretical_y = (
            np.ravel(theoretical_y)
            if isinstance(theoretical_y, np.ndarray)
            else theoretical_y
        )
    if theoretical_std is not None:
        theoretical_std = np.ravel(theoretical_std)

    # Main plot
    line = ax.plot(x, y, label=label, color=plot_color)[0]
    
    color_used = line.get_color()  # Get the actual color used

    # Theoretical prediction if provided
    if theoretical_y is not None:
        ax.plot(
            x,
            theoretical_y,
            color="black",
            linestyle="--",
            label=f"{label} (Theoretical)" if label else "Theoretical",
        )

        # Prediction of fluctuation if provided
        if theoretical_std is not None:
            ax.fill_between(
                x,
                theoretical_y - theoretical_std,
                theoretical_y + theoretical_std,
                color="gray",
                alpha=0.2,
            )
    else:
        # e.g. for residuals
        if theoretical_std is not None:
            ax.fill_between(
                x,
                0 - theoretical_std,
                0 + theoretical_std,
                color="gray",
                alpha=0.2,
            )

    # Asymptote if provided
    if asymptote_y is not None:
        ax.axhline(y=asymptote_y, color="red", linestyle=":", linewidth=1)

    # Add histogram if no theoretical or asymptotic predictions
    # if (not asymptote_y) and (theoretical_y is None) and plot_type == "fnorm_residuals":
    #     # Create side histogram with matching color
    #     create_side_histogram(
    #         ax._hist_ax,
    #         y,
    #         theoretical_std,
    #         log_y,
    #         color=color_used,
    #         label=label,
    #     )
    #     ax._hist_ax.set_ylim(ax.get_ylim())

    # Main plot formatting
    ax.set_xlabel("Time")
    ax.set_ylabel(config["ylabel"])
    ax.set_title(config["title"])
    ax.grid(True, linestyle="--", alpha=0.7)

    if log_X:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if label or theoretical_y is not None:
        ax.legend()


def plot_neural_activities(ax, times, sigmas):
    im = ax.imshow(
        sigmas.T,
        aspect="auto",
        cmap="coolwarm",
        extent=[times[0], times[-1], 0, sigmas.shape[1]],
    )
    ax.set_ylabel("Neuron index")
    ax.set_title("Neural activities")
    add_colorbar(ax, im)


def plot_external_fields(ax, times, fields):
    im = ax.imshow(
        fields.T,
        aspect="auto",
        cmap="viridis",
        extent=[times[0], times[-1], 0, fields.shape[1]],
    )
    ax.set_ylabel("Neuron index")
    ax.set_title("External fields")
    add_colorbar(ax, im)


def plot_overlaps(ax, times, sigmas, patterns):
    overlaps = np.array([compute_overlap(sigma, patterns) for sigma in sigmas])
    for i in range(overlaps.shape[1]):
        ax.plot(times, overlaps[:, i], label=f"Pattern {i+1}")
    ax.set_ylabel("Overlap")
    ax.set_xlabel("Time")
    ax.set_title("Overlap with patterns")
    ax.legend()


def plot_energy(ax, times, energies, label=None):
    plot_metric(ax, times, energies, "energy", label=label)


def plot_autocorrelation(ax, sigmas):
    max_lag = min(100, sigmas.shape[0] // 2)
    ac = autocorrelation(np.mean(sigmas, axis=1), max_lag)
    ax.plot(range(max_lag), ac)
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of mean neural activity")


def plot_deltaFnorm2_k(
    ax,
    times,
    deltaFnorm2_k,
    cfg,
    theoretical_y=None,
    horizontal_line_y=None,
    theoretical_std=None,
):
    """Plot normalized Frobenius norm difference."""
    plot_metric(
        ax,
        times,
        deltaFnorm2_k,
        "fnorm_normalized",
        theoretical_y=theoretical_y,
        asymptote_y=horizontal_line_y,
        log_X=cfg.visualizations.get("log_X", False),
        theoretical_std=theoretical_std,
    )

    if cfg.save.enabled and (
        theoretical_y is not None or horizontal_line_y is not None
    ):
        save_plot_data(
            times,
            deltaFnorm2_k,
            theoretical_y,
            horizontal_line_y,
            "synaptic_hebb_normF_difference_th_k.csv",
        )


def plot_synaptic_hebb_normF_difference(
    ax,
    times,
    deltaFnorm2,
    cfg,
    theoretical_y=None,
    horizontal_line_y=None,
    title=None,
    save_name_th="synaptic_hebb_normF_difference_th.csv",
):
    """Plot the Frobenius norm difference between synaptic weights and Hebbian weights."""
    plot_metric(
        ax,
        times,
        deltaFnorm2,
        "fnorm",
        label="Frobenius norm",
        theoretical_y=theoretical_y,
        asymptote_y=horizontal_line_y,
        log_X=cfg.visualizations.get("log_X", False),
    )

    if cfg.save.enabled and (
        theoretical_y is not None or horizontal_line_y is not None
    ):
        save_plot_data(
            times, deltaFnorm2, theoretical_y, horizontal_line_y, save_name_th
        )


def set_title(ax, varying_param):
    if varying_param:
        v_parameter_splitted = varying_param.split(".")[-1]
        if v_parameter_splitted.lower() == "beta":
            ax.set_title(rf"{ax.get_title()} (Varying $\beta$)")
        else:
            ax.set_title(f"{ax.get_title()} (Varying {v_parameter_splitted})")


def plot_results(simulation_dict, patterns, Hebb, cfg):
    """Plot the results of a single simulation with dynamically selected visualizations."""
    times = simulation_dict["times"]
    sigmas = np.array(simulation_dict["sigmas"])
    fields = simulation_dict["fields"]
    deltaFnorm2 = np.array(simulation_dict.get("diff_normF2", None))
    diff_normF2_k = np.array(simulation_dict.get("diff_normF2_k", None))
    horizontal_line_y = simulation_dict.get("horizontal_line_y", None)
    theoretical_y = simulation_dict.get("theoretical_y", None)
    theoretical_std = np.array(
        simulation_dict.get("theoretical_std", None)
    ).flatten()

    setup_matplotlib_style(big_fonts=True)
    energies = np.array(simulation_dict.get("energies", None))

    plots_to_show = cfg.visualizations.static
    n_plots = len(plots_to_show)
    n_rows = (n_plots + 1) // 2
    n_cols = 2 if n_plots > 1 else 1

    fig = plt.figure(figsize=(20, 6 * n_rows))
    fig.suptitle(cfg.name,   y=0.995)
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1] * n_rows)

    def create_axis(row, col, colspan=1):
        return fig.add_subplot(gs[row, col : col + colspan])

    plot_functions = {
        "synaptic_hebb_F_norm_difference": lambda ax: plot_synaptic_hebb_normF_difference(
            ax, times, deltaFnorm2, cfg
        ),
        "deltaFnorm2_k": lambda ax: plot_deltaFnorm2_k(ax, times, diff_normF2_k, cfg),
        "deltaFnorm2_kt": lambda ax: plot_deltaFnorm2_k(
            ax,
            times,
            diff_normF2_k,
            cfg,
            theoretical_y=theoretical_y,
            horizontal_line_y=horizontal_line_y,
            theoretical_std=theoretical_std,
        ),
        "deltaFnorm2_kt_no_theoretical": lambda ax: plot_deltaFnorm2_k(
            ax, times, diff_normF2_k, cfg, theoretical_y=None, horizontal_line_y=None, theoretical_std=None),
        "deltaFnorm2_kt_residuals": lambda ax: plot_delta_residuals(
            ax, times, diff_normF2_k, theoretical_y, cfg, offset=theoretical_std
        ),
        "deltaFnorm2_kt_residuals_hist": lambda ax: plot_delta_residuals_histogram(
            ax, diff_normF2_k, theoretical_y, cfg
        ),
        "neural_activities": lambda ax: plot_neural_activities(ax, times, sigmas),
        "external_fields": lambda ax: plot_external_fields(ax, times, fields),
        "overlaps": lambda ax: plot_overlaps(ax, times, sigmas, patterns),
        "energy": lambda ax: plot_energy(ax, times, energies),
        "autocorrelation": lambda ax: plot_autocorrelation(ax, sigmas),
        "stationarity": lambda ax: plot_stationarity(
            ax,
            times,
            deltaFnorm2,
            theoretical_y=theoretical_y,
            theoretical_std=simulation_dict.get(
                "theoretical_std", None
            ),
            cfg=cfg,
        ),
    }

    for i, plot_name in enumerate(plots_to_show):
        row = i // 2
        col = i % 2
        colspan = 2 if i == n_plots - 1 and n_plots % 2 != 0 else 1
        ax = create_axis(row, col, colspan)
        try:
            plot_functions[plot_name](ax)
        except KeyError:
            log.error(f"Unknown plot type: {plot_name}")
        except Exception as e:
            log.error(f"Error creating plot {plot_name}: {str(e)}")
            continue

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if cfg.save.enabled:
        save_plot(fig, f"{cfg.name}_selected_results.pdf", cfg)


def plot_delta_residuals(ax, times, data, theoretical_y, cfg, label=None, offset=None):
    """Plot the residuals between data and theoretical prediction."""
    residuals = data - theoretical_y
    residuals = residuals.flatten()

    plot_metric(
        ax,
        times,
        residuals,
        "fnorm_residuals",
        log_X=cfg.visualizations.get("log_X", False),
        theoretical_std=offset,
        label=label,
    )

    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)


def get_next_color(ax):
    """Get the next color from matplotlib's color cycle."""
    # Get the current color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Count how many lines/artists already exist that would have consumed colors
    num_existing = len(
        [
            child
            for child in ax.get_children()
            if isinstance(child, (plt.Line2D, plt.Polygon))
            and child.get_label() != "_nolegend_"
        ]
    )
    # Return the next color in the cycle
    return color_cycle[num_existing % len(color_cycle)]


def plot_residuals_histogram(
    ax, data, theoretical_y, theoretical_std=None, log_y=False, label=None
):
    """Create a residuals histogram with colors matching other plot panels.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        data (array-like): The data to plot
        theoretical_y (array-like): Theoretical predictions
        theoretical_std (array-like): Theoretical standard deviation bounds
        log_y (bool): Whether to use log scale for y-axis
        label (str): Label for the legend
    """
    from scipy.stats import norm
    import numpy as np

    # Get next color in the cycle
    hist_color = get_next_color(ax)

    # Calculate residuals
    residuals = data - theoretical_y

    # Calculate statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Create histogram
    n, bins, patches = ax.hist(
        residuals,
        bins="auto",
        density=True,
        alpha=0.3,
        color=hist_color,
        label=f"{label} (hist)" if label else "Histogram",
    )

    # Add normal distribution fit
    x_range = np.linspace(np.min(residuals), np.max(residuals), 100)
    ax.plot(
        x_range,
        norm.pdf(x_range, mean_residual, std_residual),
        "--",
        color=hist_color,
        label=f"{label} (fit)" if label else "Normal fit",
    )

    # Add zero line (only once)
    if not hasattr(ax, "_zero_line_added"):
        ax.axvline(0, color="black", linestyle="-", alpha=0.5, label="Zero")
        ax._zero_line_added = True

    # Add empirical mean and std lines
    ax.axvline(
        mean_residual,
        color=hist_color,
        linestyle="-",
        alpha=0.8,
        label=f"Mean: {mean_residual:.2e}",
    )
    ax.axvline(
        mean_residual + std_residual,
        color=hist_color,
        linestyle="--",
        alpha=0.8,
        label=f"Std: {std_residual:.2e}",
    )
    ax.axvline(
        mean_residual - std_residual, color=hist_color, linestyle="--", alpha=0.8
    )

    # Add theoretical bounds if provided (only once)
    if theoretical_std is not None and not hasattr(ax, "_theo_lines_added"):
        mean_offset = np.mean(theoretical_std)
        ax.axvline(
            mean_offset,
            color="black",
            linestyle=":",
            alpha=0.8,
            label=f"Theo. bounds: $\pm{mean_offset:.2e}$",
        )
        ax.axvline(-mean_offset, color="black", linestyle=":", alpha=0.8)
        ax._theo_lines_added = True

    # Formatting
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Residuals")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    return n, bins


def plot_delta_residuals_histogram(ax, data, theoretical_y, cfg):
    """Plot histogram of difference of squared normal variables with analysis."""
    from scipy import stats

    # Get data
    diff_squares = data - theoretical_y
    diff_squares = diff_squares.flatten()

    # Basic statistics
    mean_diff = float(np.mean(diff_squares))
    std_diff = float(np.std(diff_squares))
    skewness = float(stats.skew(diff_squares))
    kurtosis = float(stats.kurtosis(diff_squares))

    # Create histogram
    n, bins, patches = ax.hist(
        diff_squares,
        bins="auto",
        density=True,
        alpha=0.7,
        color="skyblue",
        label="Experimental",
    )

    # Generate points for smooth curve plotting
    x = np.linspace(min(diff_squares), max(diff_squares), 200)

    # Try to fit a normal distribution (as a reference)
    normal_pdf = stats.norm.pdf(x, mean_diff, std_diff)
    ax.plot(x, normal_pdf, "r--", lw=2, label="Normal fit (reference)")

    # Perform normality tests
    shapiro_stat, shapiro_p = stats.shapiro(diff_squares)
    ks_stat, ks_p = stats.kstest(diff_squares, "norm")

    # Calculate percentiles for checking tails
    percentiles = np.percentile(diff_squares, [1, 5, 25, 50, 75, 95, 99])

    # Add text box with statistics
    stats_text = (
        f"Distribution Statistics:\n"
        f"Mean: {mean_diff:.3e}\n"
        f"Std: {std_diff:.3e}\n"
        f"Skewness: {skewness:.3f}\n"
        f"Kurtosis: {kurtosis:.3f}\n\n"
        f"Tests:\n"
        f"Shapiro-Wilk p: {shapiro_p:.2e}\n"
        f"KS-test p: {ks_p:.2e}\n\n"
        f"Percentiles:\n"
        f"1%: {percentiles[0]:.2e}\n"
        f"25%: {percentiles[2]:.2e}\n"
        f"50%: {percentiles[3]:.2e}\n"
        f"75%: {percentiles[4]:.2e}\n"
        f"99%: {percentiles[6]:.2e}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Set labels and title
    ax.set_title(
        r"Distribution of $X^2 - Y^2$ where $X,Y \sim \mathcal{N}(\mu,\sigma^2)$"
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    # Apply styling
    ax.grid(True, linestyle="--", alpha=0.7)

    # Optional log scale
    if cfg.get("log_y", False):
        ax.set_yscale("log")

    # Add vertical lines for key statistics
    ax.axvline(mean_diff, color="red", linestyle="-", alpha=0.5)
    ax.axvline(mean_diff + std_diff, color="red", linestyle="--", alpha=0.5)
    ax.axvline(mean_diff - std_diff, color="red", linestyle="--", alpha=0.5)

    # Add QQ plot as an inset
    # Create an inset axes for the QQ plot
    inset_ax = ax.inset_axes([0.05, 0.5, 0.3, 0.4])
    stats.probplot(diff_squares, dist="norm", plot=inset_ax)
    inset_ax.set_title("Q-Q Plot", fontsize=18)

    # Clean up inset axis
    inset_ax.tick_params(labelsize=6)
    inset_ax.grid(True, linestyle="--", alpha=0.3)


def extract_param_value(label):
    """Extract parameter name and value from a label of form '$param=value$'."""
    if not label:
        return None, None

    # Remove $ signs and split by =
    label = label.strip("$")
    try:
        param, value = label.split("=")
        # Convert value to float, handling scientific notation
        value = float(value)
        return param.strip(), value
    except (ValueError, AttributeError):
        return None, None


def plot_fluctuation_scaling_std(
    ax, data, theoretical_y, labels, log_X=False, log_y=True
):
    """Plot scaling of fluctuations (residuals variance) against varying parameter.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        data (list): List of data arrays for each parameter value
        theoretical_y (list): List of theoretical prediction arrays
        labels (list): List of labels containing parameter information
        log_X (bool): Whether to use log scale for x-axis
        log_y (bool): Whether to use log scale for y-axis
    """
    import numpy as np
    from scipy import stats
    from scipy.optimize import curve_fit

    # Extract parameter values and compute variances
    param_name = None
    param_values = []
    stds = []

    for d, t, label in zip(data, theoretical_y, labels):
        # Extract parameter info
        name, value = extract_param_value(label)
        if name is not None:
            if param_name is None:
                param_name = name
            param_values.append(value)

            # Calculate residuals and their std
            residuals = d - t
            stds.append(np.std(residuals))

    param_values = np.array(param_values)
    stds = np.array(stds)

    # Sort by parameter value
    sort_idx = np.argsort(param_values)
    param_values = param_values[sort_idx]
    stds = stds[sort_idx]

    # Define power law fit function
    def power_law(x, a, b):
        return a * np.power(x, b)

    # Fit power law to the data
    try:
        if log_X or log_y:
            # Use linear fit in log space
            mask = (param_values > 0) & (stds > 0)
            log_x = np.log(param_values[mask])
            log_y = np.log(stds[mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            a = np.exp(intercept)
            b = slope
        else:
            # Direct power law fit
            popt, _ = curve_fit(power_law, param_values, stds)
            a, b = popt

        # Create fit line
        x_fit = np.logspace(
            np.log10(param_values.min()), np.log10(param_values.max()), 100
        )
        y_fit = power_law(x_fit, a, b)

        # Plot data points and fit
        ax.scatter(param_values, stds, color="blue", marker="o", label="Empirical stds")
        ax.plot(
            x_fit, y_fit, "r--", label=f"Power law fit: ${{a:.2e}}\times{param_name}^{{b:.2f}}$"
        )

        # Add fit quality metrics
        if log_X or log_y:
            r_squared = r_value**2
            ax.text(
                0.02,
                0.98,
                f"R² = {r_squared:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    except (RuntimeError, ValueError) as e:
        print(f"Fitting failed: {e}")
        # Just plot the points if fitting fails
        ax.scatter(param_values, stds, color="blue", marker="o", label="Empirical stds")

    # Set scales
    if log_X:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Labels and formatting
    ax.set_xlabel(f"{param_name}")
    ax.set_ylabel("Stds(Residuals)")
    ax.set_title(f"Fluctuation Scaling with {param_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()

    return param_values, stds


def plot_fluctuation_scaling_variance(
    ax, data, theoretical_y, labels, log_X=False, log_y=True
):
    """Plot scaling of fluctuations (residuals variance) against varying parameter.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        data (list): List of data arrays for each parameter value
        theoretical_y (list): List of theoretical prediction arrays
        labels (list): List of labels containing parameter information
        log_X (bool): Whether to use log scale for x-axis
        log_y (bool): Whether to use log scale for y-axis
    """
    import numpy as np
    from scipy import stats
    from scipy.optimize import curve_fit

    # Extract parameter values and compute variances
    param_name = None
    param_values = []
    variances = []

    for d, t, label in zip(data, theoretical_y, labels):
        # Extract parameter info
        name, value = extract_param_value(label)
        if name is not None:
            if param_name is None:
                param_name = name
            param_values.append(value)

            # Calculate residuals and their variance
            residuals = d - t
            variances.append(np.var(residuals))

    param_values = np.array(param_values)
    variances = np.array(variances)

    # Sort by parameter value
    sort_idx = np.argsort(param_values)
    param_values = param_values[sort_idx]
    variances = variances[sort_idx]

    # Define power law fit function
    def power_law(x, a, b):
        return a * np.power(x, b)

    # Fit power law to the data
    try:
        if log_X or log_y:
            # Use linear fit in log space
            mask = (param_values > 0) & (variances > 0)
            log_x = np.log(param_values[mask])
            log_y = np.log(variances[mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            a = np.exp(intercept)
            b = slope
        else:
            # Direct power law fit
            popt, _ = curve_fit(power_law, param_values, variances)
            a, b = popt

        # Create fit line
        x_fit = np.logspace(
            np.log10(param_values.min()), np.log10(param_values.max()), 100
        )
        y_fit = power_law(x_fit, a, b)

        # Plot data points and fit
        ax.scatter(
            param_values,
            variances,
            color="blue",
            marker="o",
            label="Empirical variances",
        )
        ax.plot(
            x_fit, y_fit, "r--", label=f"Power law fit: ${{a:.2e}}\times{param_name}^{{b:.2f}}$"
        )

        # Add fit quality metrics
        if log_X or log_y:
            r_squared = r_value**2
            ax.text(
                0.02,
                0.98,
                f"R² = {r_squared:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    except (RuntimeError, ValueError) as e:
        print(f"Fitting failed: {e}")
        # Just plot the points if fitting fails
        ax.scatter(
            param_values,
            variances,
            color="blue",
            marker="o",
            label="Empirical variances",
        )

    # Set scales
    if log_X:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Labels and formatting
    ax.set_xlabel(f"{param_name}")
    ax.set_ylabel("Var(Residuals)")
    ax.set_title(f"Fluctuation Scaling with {param_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()

    return param_values, variances
