from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import logging

import os


def run_scheduler(cfg):
    log = logging.getLogger(__name__)
    original_wd = get_original_cwd()
    log.info(f"Original working directory: {original_wd}")

    if cfg.run_schedule:
        variant_cfgs = [
            OmegaConf.merge(
                cfg,
                OmegaConf.load(
                    os.path.join(original_wd, "config", "variants", name + ".yaml")
                ),
            )
            for name in cfg.run_schedule
        ]
        log.info(f"Loading cfgs: {cfg.run_schedule}")

        for modification in variant_cfgs:
            if "name" not in modification:
                modification.name = (
                    f"N={modification.model.N}, K={modification.model.K}, beta={modification.model.beta}, "
                    f"k={modification.model.k}, u={modification.model.u}, tau_s={modification.tau_s}, "
                    f"tau_J={modification.tau_J}"
                )
        cfgs = variant_cfgs if cfg.skip_default else [cfg] + variant_cfgs
    else:
        log.warning("Running single configuration, no run_schedule specified")
        cfgs = [cfg]

    return cfgs


def run_logger_and_verifier(cfgs):
    log = logging.getLogger(__name__)

    for i, cfg in enumerate(cfgs):
        log.info(
            f"Experiment {i+1}/{len(cfgs)}:\n name: {cfg.name}\n model: {cfg.model}, "
            f"model_class: {cfg.model_class}\n simulation steps: {cfg.simulation.T}"
        )

        if cfg.model_class in ["AssociativeMemory", "default"]:
            continue

        if cfg.model_class == "MemoryMatrixAssociativeMemory":
            if "delay_transition" in cfg.model and not isinstance(
                cfg.model.delay_transition, int
            ):
                raise ValueError(
                    f"delay_transition should be an integer, got {cfg.model.delay_transition}"
                )
            elif "delay_transition" not in cfg.model:
                log.warning(
                    "delay_transition not specified, setting to default value of 1 in simultation.py"
                )

            if "mc_J_update" in cfg.model:
                raise NotImplementedError(
                    "Markov Chain based update of J not implemented yet"
                )

        if cfg.max_size < cfg.Js_size:
            plot_compatibility(cfg)


def plot_compatibility(cfg):
    log = logging.getLogger(__name__)

    Js_plot_functions = [
        "synaptic_weights",
        "synaptic_hebb_difference",
        "synaptic_hebb_abs_difference",
        "synaptic_hebb_difference",
        "synaptic_hebb_norm_difference",
    ]

    if any(plot in cfg.visualizations.static for plot in Js_plot_functions):
        log.error("You want plots that require storing a too large matrix")

    if cfg.visualizations.dynamic:
        log.error("GIFs should be disabled if the memory requirements are high")


def model_from_config(cfg):
    if cfg.model_class == "AssociativeMemory" or cfg.model_class == "default":
        from model import AssociativeMemory

        return AssociativeMemory
    elif cfg.model_class == "MemoryMatrixAssociativeMemory":
        from model import MemoryMatrixAssociativeMemory

        return MemoryMatrixAssociativeMemory
    else:
        raise ValueError(f"Unsupported model class: {cfg.model}")


def check_dimensions_consistency(simulation_dict, cfg):
    log = logging.getLogger(__name__)

    time_size = int(cfg.simulation.T / cfg.simulation.dt)
    dimensions = {
        "times": (len(simulation_dict["times"]), time_size),
        "fields": (simulation_dict["fields"].shape, (time_size, cfg.model.N)),
        "sigmas": (simulation_dict["sigmas"].shape, (time_size, cfg.model.N)),
        "Js": (simulation_dict["Js"].shape, (time_size, cfg.model.N, cfg.model.N)),
    }

    for key, (actual, expected) in dimensions.items():
        if actual != expected:
            log.error(f"{key} has shape {actual}, expected {expected}")

    if cfg.model_class == "MemoryMatrixAssociativeMemory":
        optional_dims = {
            "memory_matrices": (
                simulation_dict["memory_matrices"].shape,
                (time_size, cfg.model.K, cfg.model.N),
            ),
            "transition_matrices": (
                simulation_dict["transition_matrices"].shape,
                (time_size, cfg.model.K, cfg.model.K),
            ),
        }

        for key, (actual, expected) in optional_dims.items():
            if actual != expected:
                log.error(f"{key} has shape {actual}, expected {expected}")
