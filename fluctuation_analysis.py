from calendar import c
from matplotlib import use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.optimize import curve_fit
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from setup_matplotlib_style import setup_matplotlib_style

log = logging.getLogger(__name__)


# Plot configurations for different visualization types
PLOT_CONFIGS = {
    "N_std": {
        "title": "Standard Deviation vs N (averaged over K)",
        "xlabel": "N",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "group_by": "N",
    },
    "N_var": {
        "title": "Variance vs N ",
        "xlabel": "N",
        "ylabel": "Variance",
        "data_col": "residuals_variance",
        "group_by": "N",
    },
    "K_std_N": {
        "title": "Standard Deviation vs K",
        "xlabel": "K",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "group_by": "K",
    },
    "K_std_tau_J": {
        "title": r"Standard Deviation vs K ",
        "xlabel": "K",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "group_by": "K",
    },
    "tau_J_std": {
        "title": r"Standard Deviation vs $\tau_J$ ",
        "xlabel": r"$\tau_J$",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "group_by": "tau_J",
    },
    "rho_std": {
        "title": r"Standard Deviation vs $\rho$",
        "xlabel": r"$\rho$",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "group_by": "rho",
    },
    "tau_J_var": {
        "title": r"Variance Deviation vs $\tau_J$ ",
        "xlabel": r"$\tau_J$",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_variance",
        "group_by": "tau_J",
    },
    "K_var": {
        "title": "Variance vs K",
        "xlabel": "K",
        "ylabel": "Variance",
        "data_col": "residuals_variance",
        "group_by": "K",
    },
    "alpha_std_fit": {
        "title": "Standard Deviation vs α with Polynomial Fit",
        "xlabel": "α = K/N",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "plot_type": "alpha",
    },
    "alpha_var_fit": {
        "title": "Variance vs α with Polynomial Fit",
        "xlabel": "α = K/N",
        "ylabel": "Variance",
        "data_col": "residuals_variance",
        "plot_type": "alpha",
    },
    "3d_var": {
        "title": "Variance vs N and K",
        "zlabel": "Variance",
        "data_col": "residuals_variance",
        "fit": False,
    },
    "3d_std": {
        "title": r"Standard Deviation vs N and K",
        "zlabel": "Standard Deviation",
        "data_col": "residuals_std",
        "fit": False,
    },
    "3d_var_fit": {
        "title": "Variance vs N and K with Surface Fit",
        "zlabel": "Variance",
        "data_col": "residuals_variance",
        "fit": True,
        "fit_type": "NK",
    },
    "3d_var_fit_alpha": {
        "title": "Variance vs N and K with α-based Surface Fit",
        "zlabel": "Variance",
        "data_col": "residuals_variance",
        "fit": True,
        "fit_type": "alpha",
    },
    "3d_std_fit": {
        "title": r"Standard Deviation vs N and K",
        "zlabel": "Standard Deviation",
        "data_col": "residuals_std",
        "fit": True,
        "fit_type": "NK",
    },
    "3d_std_fit_alpha": {
        "title": r"Standard Deviation vs N and K with α-based Surface Fit",
        "zlabel": "Standard Deviation",
        "data_col": "residuals_std",
        "fit": True,
        "fit_type": "alpha",
    },
    "slice_std_N": {
        "title": "Standard Deviation vs N for different K",
        "xlabel": "N",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "slice_var": "K",
    },
    "slice_std_tau_J_K": {
        "title": r"Standard Deviation vs $\tau_J$ for different K",
        "xlabel": r"$\tau_J$",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "slice_var": "K",
    },
    "slice_std_K_N": {
        "title": "Standard Deviation vs K for different N",
        "xlabel": "K",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "slice_var": "N",
    },
    "slice_std_K_tau_J": {
        "title": r"Standard Deviation vs K for different $\tau_J$",
        "xlabel": "K",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "slice_var": "tau_J",
    },
    "slice_var_N": {
        "title": "Variance vs N for different K",
        "xlabel": "N",
        "ylabel": "Variance",
        "data_col": "residuals_variance",
        "slice_var": "K",
    },
    "slice_var_K": {
        "title": "Variance vs K for different N",
        "xlabel": "K",
        "ylabel": "Variance",
        "data_col": "residuals_variance",
        "slice_var": "N",
    },
}


@dataclass
class FitResult:
    """Class to store fit results and metadata"""

    param_name: str
    function_type: str
    parameters: dict
    r_squared: float
    fit_equation: str


def plot_alpha_fit(ax: plt.Axes, df: pd.DataFrame, config: dict):
    """Create plot of std/var vs alpha with polynomial fit."""
    try:
        # Calculate alpha
        df = df.copy()
        df["alpha"] = df["K"] / df["N"]

        # Sort by alpha for smooth line plotting
        df = df.sort_values("alpha")

        X = df["alpha"].values
        Y = df[config["data_col"]].values

        log.info(f"Alpha range: [{X.min():.3f}, {X.max():.3f}]")
        log.info(f"{config['ylabel']} range: [{Y.min():.3e}, {Y.max():.3e}]")

        # Plot scatter points
        ax.scatter(X, Y, c="blue", marker="o", label="Data points")

        # Try polynomial fits of different degrees
        degrees = [1, 2, 3]  # linear, quadratic, cubic
        best_fit = {"degree": 1, "r2": -np.inf, "coeffs": None}

        x_smooth = np.linspace(X.min(), X.max(), 100)

        for degree in degrees:
            try:
                coeffs = np.polyfit(X, Y, degree)
                y_fit = np.polyval(coeffs, X)

                # Calculate R-squared
                r2 = 1 - (np.sum((Y - y_fit) ** 2) / np.sum((Y - np.mean(Y)) ** 2))

                if r2 > best_fit["r2"]:
                    best_fit = {"degree": degree, "r2": r2, "coeffs": coeffs}

                log.info(f"Degree {degree} polynomial R² = {r2:.3f}")

            except Exception as e:
                log.warning(f"Fit failed for degree {degree}: {str(e)}")

        # Plot the best fit
        if best_fit["coeffs"] is not None:
            y_smooth = np.polyval(best_fit["coeffs"], x_smooth)
            ax.plot(
                x_smooth,
                y_smooth,
                "r-",
                label=f'Degree {best_fit["degree"]} fit (R²={best_fit["r2"]:.3f})',
            )

            # Add equation to plot
            equation = "y = "
            for i, coeff in enumerate(best_fit["coeffs"]):
                power = len(best_fit["coeffs"]) - i - 1
                if power > 0:
                    equation += f"{coeff:.2e}α^{power} + "
                else:
                    equation += f"{coeff:.2e}"

            ax.text(
                0.05,
                0.95,
                equation,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Add power law fit for comparison
        try:

            def power_law(x, a, b):
                return a * np.power(x, b)

            popt, _ = curve_fit(power_law, X, Y)
            y_power = power_law(x_smooth, *popt)

            # Calculate R-squared for power law
            y_power_data = power_law(X, *popt)
            r2_power = 1 - (
                np.sum((Y - y_power_data) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
            )


            ax.plot(
                x_smooth,
                y_power,
                "g--",
                label=f"Power law: {popt[0]:.2e}α^{popt[1]:.2f} (R²={r2_power:.3f})",
            )

            log.info(
                f"Power law fit: a={popt[0]:.2e}, b={popt[1]:.2f}, R²={r2_power:.3f}"
            )

        except Exception as e:
            log.warning(f"Power law fit failed: {str(e)}")

        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Consider log scale if data spans multiple orders of magnitude
        if np.log10(Y.max()) - np.log10(Y.min()) > 2:
            ax.set_yscale("log")
            log.info("Set y-axis to log scale due to data range")

    except Exception as e:
        log.error(f"Error in alpha plot creation: {str(e)}")
        log.error("Stack trace:", exc_info=True)
        raise


def create_fluctuation_plots(combined_stats: pd.DataFrame, cfg) -> None:
    """Create fluctuation analysis plots based on configuration"""
    
    font_size = cfg.get("font_size", None)
    setup_matplotlib_style(tex_fonts=cfg.get("use_tex", False), big_fonts=True, font_size=font_size)
    
    if not hasattr(cfg.visualizations, "fluct"):
        log.warning("No fluctuation plots specified in configuration")
        return

    plot_types = cfg.visualizations.fluct
    if not plot_types:
        log.warning("No fluctuation plots specified in configuration")
        return
    n_plots = len(plot_types)

    if n_plots == 0:
        return

    # Setup subplot grid
    n_rows = (n_plots + 1) // 2
    n_cols = 2  # Always use 2 columns for consistent layout

    fig = plt.figure(figsize=(15 * n_cols, 10 * n_rows))

    # Create GridSpec for more control over subplot layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, n_cols, figure=fig)

    for i, plot_name in enumerate(plot_types):
        plot_config = PLOT_CONFIGS.get(plot_name)
        if not plot_config:
            log.warning(f"Unknown plot type: {plot_name}")
            continue

        try:
            # For the last plot in an odd-numbered set, span both columns
            if i == n_plots - 1 and n_plots % 2 == 1:
                subplot_spec = gs[i // 2, :]  # Span both columns
            else:
                subplot_spec = gs[i // 2, i % 2]  # Normal single-column plot

            # Create appropriate subplot based on plot type
            if "3d" in plot_name:
                ax = fig.add_subplot(subplot_spec, projection='3d')
                # ax.set_box_aspect([2, 2, 1.5])  # Increased aspect ratio for 3D plots
                plot_3d(ax, combined_stats, plot_config, cfg=cfg)
            elif "alpha" in plot_name:
                ax = fig.add_subplot(subplot_spec)
                plot_alpha_fit(ax, combined_stats, plot_config)
            elif plot_name.startswith(("N_", "K_", "tau_J_", "rho")): # see if rho should have appropriate plot
                ax = fig.add_subplot(subplot_spec)
                plot_with_errorbars(ax, combined_stats, plot_config, cfg)

            else:
                ax = fig.add_subplot(subplot_spec)
                plot_slice(ax, combined_stats, plot_config, cfg)

        except Exception as e:
            log.error(f"Error creating plot {plot_name}: {str(e)}")
            log.error("Stack trace:", exc_info=True)
            continue

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    if cfg.save.enabled:
        try:
            os.makedirs(cfg.save.folders.static_vis, exist_ok=True)
            plot_file = os.path.join(cfg.save.folders.static_vis, "fluctuation_analysis.pdf")
            plt.savefig(plot_file, bbox_inches='tight')
            log.info(f"Plots saved to {plot_file}")
        except Exception as e:
            log.error(f"Error saving plots: {str(e)}")
            log.error("Stack trace:", exc_info=True)

    plt.close()

def plot_slice(ax: plt.Axes, df: pd.DataFrame, config: dict, cfg):
    """Create slice plot with multiple lines and smooth theoretical predictions"""
    try:
        slice_var = config["slice_var"]
        x_var = config["xlabel"].replace(r"$\tau_J$", "tau_J")
        
        # Get fixed parameters first
        N = df['N'].iloc[0]  # Fixed N value
        tau_J = df['tau_J'].iloc[0] if x_var != "tau_J" else None
        K = df['K'].iloc[0] if x_var != "K" else None
        
        # Get unique values for the slice variable
        unique_vals = sorted(df[slice_var].unique())
        if len(unique_vals) <= 5:
            slice_values = unique_vals
        else:
            n_slices = 4
            indices = np.linspace(0, len(unique_vals) - 1, n_slices, dtype=int)
            slice_values = [unique_vals[i] for i in indices]

        colors = plt.cm.rainbow(np.linspace(0, 1, len(slice_values)))

        for val, color in zip(slice_values, colors):
            mask = df[slice_var] == val
            data = df[mask]
            
            # Plot empirical data
            slice_var_label = r"$\tau_J$" if slice_var=='tau_J' else slice_var
            ax.plot(
                data[x_var],
                data[config["data_col"]],
                marker="o",
                color=color,
                label=f"Emp {slice_var_label}={val}",
            )
            
            # Add theoretical curves if enabled
            if 'theoretical_std' in df.columns and cfg.get('show_theoretical', False):
                x_smooth = np.linspace(
                    data[x_var].min(),
                    data[x_var].max(),
                    100
                )
                
                # Compute theoretical values based on variable type
                if 'theoretical_std' in df.columns:
                    y_theo = data['theoretical_std'].values
                    log.info(f"Theoretical data found for {slice_var}={val}")
                else:
                    if x_var == "tau_J":
                        y_theo = [compute_theoretical_std(N, val, tj) 
                                for tj in x_smooth]  # val is K here
                    elif x_var == "K":
                        tau_J_val = val if slice_var == 'tau_J' else tau_J
                        y_theo = [compute_theoretical_std(N, k, tau_J_val) 
                                for k in x_smooth]
                    elif x_var == "N":
                        K_val = val if slice_var == 'K' else K
                    y_theo = [compute_theoretical_std(n, K_val, tau_J) 
                            for n in x_smooth]
                    
                
                ax.plot(
                    x_smooth, 
                    y_theo,
                    linestyle='--',
                    color=color,
                    label=f"Theo {slice_var_label}={val}"
                )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"])
        ax.legend()
        ax.grid(True)

    except Exception as e:
        log.error(f"Error in slice plot creation: {str(e)}")
        log.error("Stack trace:", exc_info=True)
        raise
    
def plot_3d(ax: plt.Axes, df: pd.DataFrame, config: dict, cfg=None):
    """Create 3D surface plot with improved visualization and K/tau_J-dependent fit"""
    try:
        X = df["N"].values
        tau_J_mode = False
        if len(set(X)) == 1:
            log.warning("Only one unique N value, plotting tau_J instead")
            X = df["tau_J"].values
            tau_J_mode = True
            
        Y = df["K"].values
        Z = df[config["data_col"]].values

        log.info(
            f"Data ranges - {'tau_J' if tau_J_mode else 'N'}: [{X.min()}, {X.max()}], "
            f"K: [{Y.min()}, {Y.max()}], "
            f"{config['data_col']}: [{Z.min():.2e}, {Z.max():.2e}]"
        )

        # Plot scattered data points with larger size and better visibility
        scatter = ax.scatter(X, Y, Z, c='red', marker='o', s=50, alpha=0.6)

        # Add theoretical points if enabled and available
        if cfg.get('show_theoretical', False) and 'theoretical_std' in df.columns:
            Z_theo = df['theoretical_std'].values
            scatter_theo = ax.scatter(X, Y, Z_theo, c='blue', marker='x', s=50, alpha=0.6, label='Theoretical')
            ax.legend()

        # Only attempt surface fitting if we have enough points and fit is requested
        elif len(Y) >= 4 and config.get("fit", False):
            try:
                # Create grid for surface
                xi = np.linspace(X.min(), X.max(), 50)
                yi = np.linspace(Y.min(), Y.max(), 50)
                Xi, Yi = np.meshgrid(xi, yi)
                
                # Define fitting function based on mode
                if tau_J_mode:
                    # For tau_J mode, we fit: Z = a * K^γ * tau_J^δ
                    def fit_func(XY, a, gamma, delta, b):
                        tau_J, K = XY
                        return a * np.power(K, gamma) * np.power(tau_J, delta) + b
                    
                    # Prepare data for fitting
                    XY = (df["tau_J"].values, df["K"].values)
                    Z_fit = df[config["data_col"]].values
                    
                    # Initial parameter guesses
                    p0 = [1, 0.5, -1, 0]  # Initial guess for a, gamma, delta, b
                    
                    # Perform the fit
                    try:
                        popt, pcov = curve_fit(fit_func, XY, Z_fit, p0=p0)
                        a, gamma, delta, b = popt
                        
                        # Generate surface using the fitted parameters
                        Zi = fit_func((Xi, Yi), a, gamma, delta, b)
                        
                        fit_label = f"Fit: {a:.2e}×K^{gamma:.3f}×"+r"$\tau_J$"+f"^{delta:.3f} + {b:.2e}"
                        
                        # Calculate R-squared
                        residuals = Z_fit - fit_func(XY, a, gamma, delta, b)
                
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((Z_fit - np.mean(Z_fit))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        fit_label += f"R² = {r_squared:.3f}"
                        
                    except RuntimeError as e:
                        log.warning(f"Curve fitting failed: {str(e)}")
                        return
                        
                else:
                    # Original N-dependent fitting logic
                    if cfg.visualizations.fit3d.power_coeff:
                        # Fit y = ax^γ
                        log_x = np.log(Y)
                        log_y = np.log(Z)
                        gamma, log_a = np.polyfit(log_x, log_y, 1)
                        a = np.exp(log_a)
                        Zi = a * np.power(Yi, gamma)
                        fit_label = f"Fit: {a:.2e}×K^{gamma:.3f}"
                    else:
                        # Pure power law y = x^γ
                        log_x = np.log(Y)
                        log_y = np.log(Z)
                        gamma = np.sum(log_x * log_y) / np.sum(log_x * log_x)
                        Zi = np.power(Yi, gamma)
                        fit_label = f"Fit: K^{gamma:.3f}"

                # Plot surface with improved appearance
                surf = ax.plot_surface(Xi, Yi, Zi, alpha=0.3, cmap='viridis')
                
                # Add text annotation with fit parameters
                ax.text2D(0.05, 0.95, fit_label,
                         transform=ax.transAxes,
                         bbox=dict(facecolor='white', alpha=0.8))

                log.info(f"Surface fit: {fit_label}")

            except Exception as e:
                log.warning(f"Surface fitting failed: {str(e)}")
                log.error("Stack trace:", exc_info=True)

        # Improve axis labels and viewing angle
        if tau_J_mode:
            ax.set_xlabel(r"$\tau_J$", labelpad=10)
            if cfg.get('display_title', False):
                ax.set_title(config["title"].replace("N", r"$\tau_J$"))
        else:
            ax.set_xlabel("N", labelpad=10)
            if cfg.get('display_title', False):
                ax.set_title(config["title"])
            
        ax.set_ylabel("K", labelpad=10)
        ax.set_zlabel(config["zlabel"], labelpad=10)

        # Set better viewing angle
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 0.5])  # Adjust the box aspect ratio

    except Exception as e:
        log.error(f"Error in 3D plot creation: {str(e)}")
        log.error("Stack trace:", exc_info=True)
        raise

def plot_with_errorbars(ax: plt.Axes, df: pd.DataFrame, config: dict, cfg):
    """Create plot with error bars and theoretical predictions for two varying parameters."""
    lwidth = 2.5
    try:
        # 1. Get primary variable (group_by) for x-axis
        group_var = config["group_by"]
        data_col = config["data_col"]
        
        # 2. Find second varying parameter
        potential_params = [j for j in ['tau_J', 'K', 'N', 'rho', 'tau_h'] if j != group_var]
        second_param = None
        max_unique_values = 1
        # df = df[df['rho']>0.5]
        
        for param in potential_params:
            if param in df.columns and param != group_var:
                n_unique = len(df[param].unique())
                if n_unique > max_unique_values:
                    max_unique_values = n_unique
                    second_param = param
        
        if second_param is None or max_unique_values <= 1:
            log.warning("No valid second parameter found, setting it to be K")
            second_param = 'K'
            # return
            
        # 3. Get unique values for second parameter and create color scheme
        unique_values = sorted(df[second_param].unique())
        # colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_values)))
        colors = COLORS[:len(unique_values)]
        # 4. Process each subset
        for value, color in zip(unique_values, colors):
            # Create subset for this value of second parameter
            subset_df = df[df[second_param] == value]
            
            # Calculate statistics for this subset
            stats = subset_df.groupby(group_var)[data_col].agg(["mean", "std", "count"]).reset_index()
            stats["sem"] = stats["std"] / np.sqrt(stats["count"])
            
            # Plot individual points (all with same color but transparent)
            ax.plot(
                subset_df[group_var],
                subset_df[data_col],
                "o",
                color=color,
                alpha=0.2,
                markersize=8,
            )
            
            # Plot error bars
            ax.errorbar(
                stats[group_var],
                stats["mean"],
                yerr=stats["sem"],
                fmt="o",
                color=color,
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                markersize=12,
            )
            
            # Get fixed parameters for theoretical prediction
            params = {param: subset_df[param].iloc[0] for param in potential_params 
                     if param in subset_df.columns and param != group_var}
            
            # Generate smooth x values for theoretical line
            x_smooth = np.logspace(
                np.log10(max(stats[group_var].min(), 0.01)),
                np.log10(stats[group_var].max()),
                400
            )
            
            # Calculate theoretical predictions
            y_theo = []
            for x in x_smooth:
                # Update the varying parameter in params
                if group_var == 'N':
                    pred = compute_theoretical_std(x, params['K'], params['tau_J'], params.get('rho', 0))
                elif group_var == 'K':
                    pred = compute_theoretical_std(params['N'], x, params['tau_J'], params.get('rho', 0))
                elif group_var == 'tau_J':
                    pred = compute_theoretical_std(params['N'], params['K'], x, params.get('rho', 0))
                elif group_var == 'rho':
                    pred = compute_theoretical_std(params['N'], params['K'], params['tau_J'], x)
                y_theo.append(pred)
            
            # Plot theoretical prediction
            ax.plot(
                x_smooth,
                y_theo,
                '--',
                color=color,
                linewidth=2,
            )
            
        # 5. Add legend entries
        # Create custom legend handles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', marker='o', markersize=8, alpha=0.2, 
               label='Individual points', linestyle='None'),
            Line2D([0], [0], color='darkgray', linestyle='--', linewidth=lwidth, 
               label='Theoretical predictions')
        ]
        
        # Add second parameter values to legend
        for value, color in zip(unique_values, colors):
            # Format parameter name for latex if needed
            if second_param in ['tau_J', 'tau_h']:
                param_name = r'$\tau_' + ('J' if second_param == 'tau_J' else 'h') + '$'
            else:
                param_name = second_param
                
            legend_elements.append(
                Line2D([0], [0], color=color, marker='o', markersize=16, linestyle='-', linewidth=lwidth,
                       label=f'{param_name}={value}')
            )
            
        ax.legend(handles=legend_elements)
        
        # 6. Set scales, labels and grid
        if cfg.visualizations.get("log_scale", False):
            ax.set_xscale("log")
            ax.set_yscale("log")
        
        # Remove the tick corresponding to 0 from y-axis
        yticks = ax.get_yticks()
        yticks = yticks[yticks != 0]
        ax.set_yticks(yticks)
        
        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        if cfg.get('display_title', False):
            ax.set_title(config["title"])
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # 7. Set axis limits
        x_min, x_max = df[group_var].min(), df[group_var].max()
        y_min, y_max = df[data_col].min(), df[data_col].max()
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
        ax.set_ylim(y_min * 0.9,  y_max * 1.1)
        
        if cfg.get('reduce_ylim', False):
            ax.set_ylim(y_min * 0.9, 0.0110)
            
        if cfg.get('ylim', None):
            ax.set_ylim(y_min * 0.9, cfg.get('ylim', None))            

    except Exception as e:
        log.error(f"Error in error bar plot creation: {str(e)}")
        log.error("Stack trace:", exc_info=True)
        raise


def plot_simple(ax: plt.Axes, df: pd.DataFrame, config: dict, cfg):
    """Create simple scatter plot with mean and standard deviation error bars.
    
    Particularly suitable for parameters like rho that vary in a fixed range [0,1].
    Uses linear scale and handles multiple measurements per x-value by showing mean and standard deviation.
    """
    try:
        x_var = config["group_by"]
        y_var = config["data_col"]
        
        # Group data by x_var and calculate statistics
        stats = df.groupby(x_var)[y_var].agg(['mean', 'std', 'count']).reset_index()
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        
        # Plot individual points with low opacity
        if cfg.visualizations.errorbars.get("show_points", True):
            ax.plot(df[x_var], df[y_var], 'o', 
                   color='lightgray', alpha=0.3, markersize=4,
                   label='Individual measurements')

        # Plot error bars for mean ± standard error
        ax.errorbar(stats[x_var], stats['mean'], yerr=stats['sem'],
                   fmt='o', color='blue', capsize=5, capthick=1.5,
                   elinewidth=1.5, markersize=8,
                   label='Mean ± SEM'  if cfg.get("show_stats", False) else "")

        # Try polynomial fit on means
        x_smooth = np.linspace(stats[x_var].min(), stats[x_var].max(), 100)
        try:
            # Use quadratic fit
            coeffs = np.polyfit(stats[x_var], stats['mean'], 2)
            y_smooth = np.polyval(coeffs, x_smooth)
            
            # Calculate R-squared using means
            y_fit = np.polyval(coeffs, stats[x_var])
            r2 = 1 - (np.sum((stats['mean'] - y_fit) ** 2) / 
                     np.sum((stats['mean'] - stats['mean'].mean()) ** 2))
            
            # Plot fit
            ax.plot(x_smooth, y_smooth, 'r-', 
                   label=f'Quadratic fit (R²={r2:.3f})')

            # Add equation
            equation = f"y = {coeffs[0]:.2e}ρ² + {coeffs[1]:.2e}ρ + {coeffs[2]:.2e}"
            ax.text(0.05, 0.95, equation, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))

        except Exception as e:
            log.warning(f"Polynomial fit failed: {str(e)}")

        # Add statistics box
        if cfg.get('show_stats', False):
            stats_text = (f"Statistics:\n"
                        f"Points per {x_var}: "
                        f"min={stats['count'].min()}, max={stats['count'].max()}\n"
                        f"Total points: {len(df)}")
            
            ax.text(0.95, 0.05, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7),
                    fontsize=30)

        # Set labels and title
        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"])
        
        # Set axis limits with padding for linear scale
        x_range = stats[x_var].max() - stats[x_var].min()
        y_all = np.concatenate([
            df[y_var], 
            stats['mean'] + stats['sem'],
            stats['mean'] - stats['sem'],
            y_smooth if 'y_smooth' in locals() else []
        ])
        y_min, y_max = np.min(y_all), np.max(y_all)
        y_padding = 0.1 * (y_max - y_min)
        
        # For x-axis, enforce limits between 0 and 1 if dealing with correlation
        if x_var.lower() == 'rho':
            ax.set_xlim(-0.05, 1.05)
        else:
            ax.set_xlim(stats[x_var].min() - 0.1 * x_range,
                       stats[x_var].max() + 0.1 * x_range)
            
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    except Exception as e:
        log.error(f"Error in simple plot creation: {str(e)}")
        log.error("Stack trace:", exc_info=True)
        raise
    
    
def compute_theoretical_std(N: int, K: int, tau_J: float, rho : float = 0) -> float:
        eps = 1/tau_J
        
        eps2 = eps**2
        eps3 = eps**3
        inv_K = 1/K
        inv_K2 = inv_K**2
        inv_K3 = inv_K**3
    
        #var 
        term1_bracket = eps2 - eps3 + 2 * inv_K*(-eps2 + 2*eps3) + ( 3*inv_K2 - 2 * inv_K3) *(eps2 - 3 * eps3)
        term1 = term1_bracket / (N*(N-1))
        
        #covar
        term2 = 1/2 *inv_K*(eps2 - eps3)
        term3 = 1/2 * inv_K2*(eps2 - 3 * eps3)
        term4 = inv_K2*(- eps2 + 2 * eps3)
        
        result = np.sqrt(term1 + (term2 + term3 + term4)) * (1 - rho ** 2)
        
        return result

COLORS = [
    (0.00784313725490196, 0.4470588235294118, 0.6352941176470588),
    (0.6235294117647059, 0.7647058823529411, 0.4666666666666667),
    (0.792156862745098, 0.043137254901960784, 0.011764705882352941),
    (0.8431372549019608, 0.7803921568627451, 0.011764705882352941),
    (0.5333333333333333, 0.792156862745098, 0.8549019607843137),
    (0.6470588235294118, 0.00784313725490196, 0.34509803921568627),
    "#F7930B",
    "#389855",
    "#D169B8",
    "#f5ea76",
    "#51bdb7",
]