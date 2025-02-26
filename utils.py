from datetime import datetime
import os
import numpy as np

import logging
from typing import Optional
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_patterns(N: int, K: int, distribution: str = "rademacher", rho: float = 0.0) -> np.ndarray:
    """Generate patterns based on the specified distribution with correlation parameter rho.
    
    Args:
        N: Number of neurons
        K: Number of patterns
        distribution: Pattern distribution type ("rademacher" or "gaussian")
        rho: Correlation parameter between patterns (0 = uncorrelated, 1 = fully correlated)
        
    Returns:
        np.ndarray: Array of shape (K, N) containing the patterns
    """
    if not 0 <= rho <= 1:
        raise ValueError(f"Correlation parameter rho must be between 0 and 1, got {rho}")
    
    # Generate independent random patterns
    if distribution == "rademacher":
        Z = np.random.choice([-1, 1], size=(K, N))
    elif distribution == "gaussian":
        Z = np.random.normal(0, 1, size=(K, N))
    else:
        raise ValueError(f"Unsupported pattern distribution: {distribution}")
    
    # Regulate correlation
    # Z = np.random.choice([-1, 1], size=(K, N))
    W = np.random.choice([0, 1],p=[np.sqrt(rho),1-np.sqrt(rho)], size=(K,N))
    patterns = (1-W) + np.multiply(W,Z)
    
    # Normalize patterns if using gaussian distribution
    if distribution == "gaussian":
        patterns /= np.sqrt(N)
        if rho != 0:
            log.error("Error: rho different from zero not implemented for gaussian distribution")
        
    return patterns

def compute_pattern_correlations(patterns: np.ndarray) -> tuple[float, float]:
    """Compute mean and standard deviation of pattern correlations.
    
    Args:
        patterns: Array of shape (K, N) containing the patterns
        
    Returns:
        tuple[float, float]: Mean and standard deviation of correlations
    """
    K, N = patterns.shape
    correlations = []
    
    # Compute correlations between all pairs of patterns
    for i in range(K):
        for j in range(i + 1, K):
            C_ij = np.dot(patterns[i], patterns[j]) / N
            correlations.append(C_ij)
            
    correlations = np.array(correlations)
    return np.mean(correlations), np.std(correlations)

def validate_pattern_correlations(patterns: np.ndarray, rho: float, rtol: float = 0.1) -> bool:
    """Validate that generated patterns have the expected correlation statistics.
    
    Args:
        patterns: Array of shape (K, N) containing the patterns
        rho: Target correlation parameter
        rtol: Relative tolerance for validation
        
    Returns:
        bool: True if correlations match theoretical predictions
    """
    K, N = patterns.shape
    mean_corr, std_corr = compute_pattern_correlations(patterns)
    
    # Theoretical predictions from tex notes
    expected_mean = rho
    expected_std = np.sqrt((1 - rho**2) / N)
    
    # Check if statistics match predictions within tolerance
    mean_valid = np.abs(mean_corr - expected_mean) <= rtol * abs(expected_mean) if rho != 0 else abs(mean_corr) <= rtol
    std_valid = np.abs(std_corr - expected_std) <= rtol * expected_std
    
    return mean_valid and std_valid

def generate_probabilities(K, distribution="uniform"):
    if distribution == "uniform":
        return [1 / K for i in range(K)]
    # other
    else:
        raise ValueError(f"Unsupported probabilities distribution: {distribution}")


def generate_hebb(patterns, probabilities = None):
    K, N = patterns.shape
    if probabilities is None:
        probabilities = [1 / K for i in range(K)]
    Hebb = 0
    for i in range(K):
        Hebb += np.outer(patterns[i], patterns[i]) * probabilities[i]
    np.fill_diagonal(Hebb, 0)
    return Hebb


def generate_persistence(tau_h):
    if isinstance(tau_h, int):
        return tau_h
    else:
        if "distribution" in tau_h.keys():
            raise NotImplementedError
        else:
            return np.random.randint(tau_h.min, tau_h.max)


def compute_overlap(sigma, patterns):
    """Compute the overlap between the current state and the patterns."""
    return np.dot(patterns, sigma) / len(sigma)


def energy(sigma, J, h):
    """Compute the energy of the current state."""
    return -0.5 * np.dot(sigma, np.dot(J, sigma)) - np.dot(h, sigma)


def diff_Fnorm2(J, Hebb, k, N):
    return (np.linalg.norm(J - Hebb * k, ord="fro")) ** 2 / (N * (N - 1))


def autocorrelation(x, max_lag):
    """Compute the autocorrelation of a time series."""
    result = np.correlate(x - np.mean(x), x - np.mean(x), mode="full")
    return (
        result[result.size // 2 : result.size // 2 + max_lag] / result[result.size // 2]
    )


def create_folders(cfg):
    if cfg.save.enabled:
        for folder in cfg.save.folders.values():
            os.makedirs(folder, exist_ok=True)


def update_simulation_dict(simulation_dict: dict, current_state: dict, isLarge):
    if isLarge:
        if "sigmas" in simulation_dict.keys():
            simulation_dict["sigmas"].append(current_state["sigmas"])
        else:
            simulation_dict["sigmas"] = [current_state["sigmas"]]
    else:
        s_keys = simulation_dict.keys()
        c_keys = current_state.keys()
        for key in c_keys:
            if key in s_keys:
                simulation_dict[key].append(current_state[key])
            else:
                simulation_dict[key] = [current_state[key]]
        return simulation_dict


def create_label(key, value):
    """Create a label based on the key and value."""
    # Extract the last part of the key if it contains dots
    log.info(f"Creating label for {key} with value {value}")
    
    label_key = key.split(".")[-1].strip()
    log.info(f"After split: {label_key}")

    if label_key.lower() == "beta":
        return rf"$\beta={value}$"
    elif label_key.lower() == "tau_j":
        return rf"$\tau_J={value}$"
    elif label_key.lower() == "tau_s":
        return rf"$\tau_\sigma={value}$"
    else:
        return f"${label_key}={value}$"


def find_varying_parameter(all_overrides):
    """Find the parameter that varies across runs."""
    if not all_overrides:
        return None

    reference = all_overrides[0]
    for key in reference:
        if any(override[key] != reference[key] for override in all_overrides):
            return key

    return None


import os
from hydra.utils import get_original_cwd


def get_last_line(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else None
    except FileNotFoundError:
        return None


def get_multirun_path_from_output(hydra_output_dir):
    path_parts = os.path.normpath(hydra_output_dir).split(os.path.sep)
    date_time = os.path.sep.join(path_parts[-2:])
    return os.path.join(get_original_cwd(), "multirun", *path_parts[-2:])


def get_relative_multirun_path(hydra_output_dir):
    path_parts = os.path.normpath(hydra_output_dir).split(os.path.sep)
    return os.path.join("multirun", *path_parts[-2:])


def get_horizonal_line_y(cfg):
    temp = np.zeros(cfg.simulation.steps)
    log.error(f"Not implemented yet")
    return temp


def generate_identifier(first_cfg: DictConfig, varying_param: Optional[str]) -> str:
    """Generate a standardized filename identifier based on configuration."""
    essential_params = {
        "N": first_cfg.model.N,
        "K": first_cfg.model.K,
        "tau_J": first_cfg.model.tau_J,
        "tau_s": first_cfg.model.tau_s,
        "T": first_cfg.simulation.T,
    }

    param_strs = [f"{param}{value}" for param, value in essential_params.items()]

    if varying_param:
        var_name = varying_param.split(".")[-1]
        param_strs.append(f"var{var_name}")

    timestamp = datetime.now().strftime("%m-%d-%H")
    return "_".join(param_strs + [timestamp])
