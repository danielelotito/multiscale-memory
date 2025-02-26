import logging
import numpy as np
from typing import List, Dict, Any
import os
import pandas as pd
from omegaconf import DictConfig
from dynamic_visualization import dynamic_visualizations
from run_manager import (
    check_dimensions_consistency,
    model_from_config,
    run_logger_and_verifier,
)
from utils import (
    generate_persistence,
    generate_patterns,
    generate_probabilities,
    generate_hebb,
    update_simulation_dict,
    energy,
    diff_Fnorm2,
)
from visualization import plot_results
from run_stat_and_latex_table_generator import (
    generate_latex_table,
    create_run_details_statistics,
)
from large_deviations_callback import large_deviations_callback, LargeDeviationsTracker
import logging 


log = logging.getLogger(__name__)


def simulate(
    model: Any,
    patterns: np.ndarray,
    probabilities: np.ndarray,
    Hebb: np.ndarray,
    cfg: DictConfig,
) -> Dict[str, Any]:
    
    T, dt = cfg.simulation.T, cfg.simulation.dt
    tau_h = cfg.model.tau_h
    steps = int(T / dt)
    times = np.arange(0, T, dt)

    fields = np.zeros((steps, model.N))
    energies = np.zeros((steps, 1))
    diff_normF2 = np.zeros((steps, 1))
    estimates_Fnorm2 = np.zeros((steps, 1))
    theoretical_std_step = np.zeros((steps, 1))
    
    tracker = None
    track_deviations = False
    if cfg.get("large_deviations_callback", {}).get("activate", False):
        track_deviations = initialize_large_deviations_tracker(model, patterns, cfg)
    
    simulation_dict = {}
    theoretical_std = model.compute_theoretical_std(Jhebb=Hebb)
    
    pattern_persistence = 0
    for i in range(steps):
        if pattern_persistence == 0:
            pattern_persistence = generate_persistence(tau_h)
            pattern_idx = np.random.choice(model.K, p=probabilities)
            h = patterns[pattern_idx]
        pattern_persistence -= 1

        fields[i] = h
        current_state = model.get_state()
        energies[i] = energy(current_state["sigmas"], current_state["Js"], h)
        diff_normF2[i] = diff_Fnorm2(current_state["Js"], Hebb, model.k, model.N)
        estimates_Fnorm2[i] = model.get_estimates_Fnorm2(i, Jhebb=Hebb)
        
                
        if cfg.get('transient_factor', True):
            theoretical_std_step[i] = theoretical_std * (1 - (1-model.eps)**(2*i))
        else:
            theoretical_std_step[i] = theoretical_std 
        
        if track_deviations:  
                    tracker = large_deviations_callback(
                        diff_normF2[i],
                        estimates_Fnorm2[i],
                        theoretical_std_step[i],
                        current_state,
                        patterns,
                        i,
                        cfg,
                        tracker  # Pass the existing tracker
                    )
                    
        model.step(h)

        update_simulation_dict(
            simulation_dict, current_state, bool(cfg.max_size < cfg.Js_size)
        )

        delay_transition = cfg.model.get("delay_transition", 1)

        if hasattr(cfg.model, "delay_transition") and isinstance(delay_transition, int):
            if i >= delay_transition:
                model.update_transition_matrix(
                    simulation_dict["sigmas"][i],
                    simulation_dict["sigmas"][i - delay_transition],
                )

    if cfg.get("dev_run", False):
        check_dimensions_consistency(simulation_dict, cfg)

    if cfg.get("large_deviations_callback", {}).get("process_now", False):
        tracker.process_events()

    simulation_dict.update(
        {
            "times": times,
            "fields": fields,
            "diff_normF2": diff_normF2,
            "diff_normF2_k": diff_normF2 / model.k**2,
            "energies": energies,
            "horizontal_line_y": model.get_horizonal_line_y(Jhebb=Hebb),
            "theoretical_y": estimates_Fnorm2,
            "theoretical_std": theoretical_std_step,
        }
    )
    
    
    
    # After simulation, process deviations if requested
    if cfg.get("large_deviations_callback", {}).get("process", False):
        from process_deviations import main as process_deviations
        from omegaconf import OmegaConf
        import yaml
        # Load the deviations config
        config_path = cfg.large_deviations_callback.config_path
        with open(config_path, 'r') as f:
            cfg_dev = OmegaConf.create(yaml.safe_load(f))
        process_deviations(cfg_dev)

    
    run_details = create_run_details_statistics(simulation_dict, cfg)
    latex_table = generate_latex_table(run_details)
    save_latex_table(latex_table, cfg)
    save_simulation_data(simulation_dict, cfg)

    return simulation_dict

def initialize_large_deviations_tracker(model, patterns, cfg):
    max_N = cfg.large_deviations_callback.get("max_N", 100)
    if model.N <= max_N:
        track_deviations = True
        tracker = LargeDeviationsTracker(cfg, patterns)  # Initialize once here
        log.info("Large deviations tracking enabled")
    else:
        log.warning(
                f"Network size {model.N} exceeds maximum allowed size {max_N}. "
                "Large deviations tracking disabled."
            )
        
    return track_deviations


def save_latex_table(latex_table: str, cfg: DictConfig) -> None:
    if cfg.save.enabled:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", "latex_table.tex")
        with open(file_path, "w") as f:
            f.write(latex_table)


def save_simulation_data(simulation_dict: Dict[str, Any], cfg: DictConfig) -> None:
    if cfg.save.enabled:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", "simulation_dict.csv")
        df = pd.DataFrame(
            {
                "time": simulation_dict["times"].astype(int),
                "diff_normF2": simulation_dict["diff_normF2"]
                .ravel()
                .astype(np.float16),
                "diff_normF2_k": simulation_dict["diff_normF2_k"]
                .ravel()
                .astype(np.float16),
                "energies": simulation_dict["energies"].ravel().astype(np.float16),
                "horizontal_line_y": np.float16(simulation_dict["horizontal_line_y"]),
                "theoretical_y": simulation_dict["theoretical_y"]
                .ravel()
                .astype(np.float16),
                "theoretical_std": simulation_dict["theoretical_std"]
                .ravel()
                .astype(np.float16),
                "residuals": (
                    simulation_dict["diff_normF2_k"].ravel()
                    - simulation_dict["theoretical_y"].ravel()
                ).astype(np.float16),
            }
        )
        df.to_csv(file_path, index=False)


def run_multiple_simulations(cfg_list: List[DictConfig]) -> List[Dict[str, Any]]:
    results = []

    run_logger_and_verifier(cfg_list)

    for cfg in cfg_list:
        model_class = model_from_config(cfg)
        model = model_class(cfg.model)

        patterns = generate_patterns(model.N, model.K, cfg.model.pattern_distribution, rho =  cfg.model.get('rho',0))
        probabilities = generate_probabilities(model.K, cfg.model.prob_distribution)
        Hebb = generate_hebb(patterns, probabilities)

        simulation_dict = simulate(model, patterns, probabilities, Hebb, cfg)

        if "static" in cfg.visualizations:
            plot_results(simulation_dict, patterns, Hebb, cfg)

        if "dynamic" in cfg.visualizations:
            dynamic_visualizations(simulation_dict, patterns, Hebb, cfg)

        results.append(simulation_dict)

    return results
