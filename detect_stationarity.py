from typing import Any, Tuple


import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Optional
import logging

log = logging.getLogger(__name__)


def detect_stationarity(
    series: np.ndarray,
    significance: float = 0.05,
    window_size: int = 100,
    min_periods: int = 20,
    autolag: str = "AIC",
):
    """
    Detect the index at which a time series becomes stationary using a rolling
    Augmented Dickey-Fuller test.
    """
    series = np.asarray(series)
    if len(series) < min_periods:
        log.warning("Series too short for stationarity analysis")
        return None

    # Ensure the series is 1D
    if len(series.shape) > 1:
        series = series.ravel()

    convergence_candidates = []
    step_size = window_size // 4

    for i in range(min_periods, len(series), step_size):
        window = series[max(min_periods, i - window_size) : i]

        try:
            adf_stat = adfuller(window, autolag=autolag)
            p_value = adf_stat[1]

            if p_value < significance:
                return i

        except ValueError as e:
            log.debug(f"ADF test failed at index {i}: {str(e)}")
            continue

    # If we found candidates but couldn't find the exact point
    if convergence_candidates:
        point = convergence_candidates[0]
        log.info(f"Using first convergence candidate at {point}")
        return point

    return None


def analyze_convergence(series: np.ndarray, cfg: Any) -> Tuple[Optional[int], dict]:
    """Analyze the convergence of a time series using multiple methods."""
    stationarity_cfg = cfg.get("stationarity_analysis", {})

    # Get configuration values with defaults
    significance = stationarity_cfg.get("significance", 0.05)
    window_size = stationarity_cfg.get("window_size", 100)
    min_periods = stationarity_cfg.get("min_periods", 20)

    log.info(
        f"Running stationarity analysis with: significance={significance}, "
        f"window_size={window_size}, min_periods={min_periods}"
    )

    convergence_point = detect_stationarity(
        series,
        significance=significance,
        window_size=window_size,
        min_periods=min_periods,
        autolag="AIC",
    )

    results = {
        "convergence_point": convergence_point,
        "series_length": len(series),
        "converged": convergence_point is not None,
    }

    if convergence_point is not None:
        converged_series = series[convergence_point:]
        results.update(
            {
                "mean_after_convergence": np.mean(converged_series),
                "std_after_convergence": np.std(converged_series),
                "convergence_percentage": convergence_point / len(series) * 100,
            }
        )

        log.info(
            f"Convergence detected at index {convergence_point} "
            f"({results['convergence_percentage']:.1f}% of series)"
        )
    else:
        log.warning("No convergence detected")

    return convergence_point, results


def plot_stationarity(
    ax, times, data, theoretical_y=None, offset_from_theoretical=None, cfg=None
):
    """
    Analyze stationarity and plot histogram of stationary values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    times : array-like
        Time points
    data : array-like
        Data to analyze
    theoretical_y : array-like, optional
        Theoretical predictions (for mean)
    offset_from_theoretical : array-like, optional
        Theoretical predictions for standard deviation
    cfg : object, optional
        Configuration object
    """
    # Ensure data is 1-dimensional
    data = np.ravel(data) if isinstance(data, np.ndarray) else data

    # Get theoretical values if available
    theo_mean = theoretical_y[-1][0] if theoretical_y is not None else None
    theo_std = (
        offset_from_theoretical[-1][0] if offset_from_theoretical is not None else None
    )

    # Perform stationarity analysis
    convergence_point, stats = analyze_convergence(data, cfg)

    if convergence_point is None:
        log.warning("No stationarity point found")
        return

    # Get stationary portion of the data
    stationary_data = data[convergence_point:]

    # Calculate empirical statistics
    emp_mean = np.mean(stationary_data)
    emp_std = np.std(stationary_data)

    # Plot histogram
    ax.hist(
        stationary_data,
        bins=cfg.stationarity_analysis.bins,
        density=True,
        color="skyblue",
        alpha=0.6,
        label="Stationary values",
    )

    # Plot empirical statistics
    ax.axvline(emp_mean, color="red", linestyle="-", label=f"Mean: {emp_mean:.4f}")
    ax.axvline(
        emp_mean + emp_std, color="red", linestyle="--", label=f"Std: {emp_std:.4f}"
    )
    ax.axvline(emp_mean - emp_std, color="red", linestyle="--")

    # Plot theoretical values if both are available
    if theo_mean is not None and np.isscalar(theo_mean) and np.isscalar(theo_std):
        ax.axvline(
            theo_mean,
            color="black",
            linestyle="-",
            label=f"Theo. mean: {theo_mean:.4f}",
        )
        ax.axvline(
            theo_mean + theo_std,
            color="black",
            linestyle="--",
            label=f"Theo. std: {theo_std:.4f}",
        )
        ax.axvline(theo_mean - theo_std, color="black", linestyle="--")

    # Add stationarity point to title
    convergence_time = times[convergence_point]
    ax.set_title(
        f"Distribution of Stationary Values\n(Stationarity achieved at t={convergence_time:.2f})"
    )

    # Set labels and legend
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    # Add text box with statistics
    stats_text = (
        f"Statistics:\n"
        f"Empirical mean: {emp_mean:.4f}\n"
        f"Empirical std: {emp_std:.4f}"
    )
    if theo_mean is not None and np.isscalar(theo_mean) and np.isscalar(theo_std):
        rel_error_mean = abs(theo_mean - emp_mean) / abs(theo_mean)
        rel_error_std = abs(theo_std - emp_std) / abs(theo_std)
        stats_text += (
            f"\nTheoretical mean: {theo_mean:.4f}\n"
            f"Theoretical std: {theo_std:.4f}\n"
            f"Relative error (mean): {rel_error_mean:.2%}\n"
            f"Relative error (std): {rel_error_std:.2%}"
        )

    # Position text box in upper left corner
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
