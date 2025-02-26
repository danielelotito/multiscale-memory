"""
Available Plot Types for Fluctuation Analysis
-----------------------------------------

3D Surface Plots:
---------------
1. "3d_var"
   - 3D scatter plot of variance vs N and K
   - No surface fitting
   - Shows raw data points in 3D space

2. "3d_std"
   - 3D scatter plot of standard deviation vs N and K
   - No surface fitting
   - Shows raw data points in 3D space

3. "3d_var_fit"
   - 3D plot of variance vs N and K with fitted surface
   - Fits surface using power law: a * N^b * K^c + d
   - Includes colorbar for surface values

4. "3d_var_fit_alpha"
   - 3D plot of variance vs N and K with α-based surface fit
   - Fits surface using α = K/N relationship
   - Surface model: a * (K/N)^b + c

5. "3d_std_fit"
   - 3D plot of standard deviation vs N and K with fitted surface
   - Similar to 3d_var_fit but for standard deviation
   - Includes colorbar for surface values

6. "3d_std_fit_alpha"
   - 3D plot of standard deviation vs N and K with α-based surface fit
   - Similar to 3d_var_fit_alpha but for standard deviation
   - Surface model based on α = K/N

Slice Plots:
-----------
7. "slice_std_N"
   - Standard deviation vs N for different K values
   - Shows multiple lines, one for each K value
   - Uses equispaced K values if more than 5 available

8. "slice_std_K"
   - Standard deviation vs K for different N values
   - Shows multiple lines, one for each N value
   - Uses equispaced N values if more than 5 available

9. "slice_var_N"
   - Variance vs N for different K values
   - Similar to slice_std_N but for variance
   - Includes grid and legend

10. "slice_var_K"
    - Variance vs K for different N values
    - Similar to slice_std_K but for variance
    - Includes grid and legend

Alpha Analysis Plots:
------------------
11. "alpha_std_fit"
    - Standard deviation vs α (K/N) with polynomial fits
    - Tests polynomial fits up to degree 3
    - Includes power law fit for comparison
    - Shows best fit equation and R² values

12. "alpha_var_fit"
    - Variance vs α (K/N) with polynomial fits
    - Tests polynomial fits up to degree 3
    - Includes power law fit for comparison
    - Shows best fit equation and R² values

Example Configuration:
-------------------
visualizations:
  fluct:
    - 3d_var
    - 3d_std_fit
    - slice_std_N
    - slice_var_K
    - alpha_var_fit
    - alpha_std_fit

Notes:
-----
- All plots include proper axis labels and titles
- 3D plots use consistent colormaps for better comparison
- Slice plots automatically handle different numbers of unique values
- Alpha plots include both polynomial and power law fits for comparison
- All plots include error handling and logging
- Last plot in odd-numbered lists will span two columns
"""

# Dictionary of plot configurations
PLOT_CONFIGS = {
    "3d_var": {
        "title": "Variance vs N and K",
        "zlabel": "Variance",
        "data_col": "residuals_variance",
        "fit": False,
    },
    "3d_std": {
        "title": "Standard Deviation vs N and K",
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
        "title": "Standard Deviation vs N and K with Surface Fit",
        "zlabel": "Standard Deviation",
        "data_col": "residuals_std",
        "fit": True,
        "fit_type": "NK",
    },
    "3d_std_fit_alpha": {
        "title": "Standard Deviation vs N and K with α-based Surface Fit",
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
    "slice_std_K": {
        "title": "Standard Deviation vs K for different N",
        "xlabel": "K",
        "ylabel": "Standard Deviation",
        "data_col": "residuals_std",
        "slice_var": "N",
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
}
