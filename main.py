import hydra
from omegaconf import DictConfig
import os
import logging
from run_manager import run_scheduler
from simulation import run_multiple_simulations
from utils import create_folders
from hydra.core.hydra_config import HydraConfig
from file_operations import append_multirun_path

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    setup_environment(cfg)
    cfgs = run_scheduler(cfg)
    results = run_multiple_simulations(cfgs)
    append_multirun_path(HydraConfig.get().run.dir)


def setup_environment(cfg: DictConfig):
    logging.basicConfig(level=cfg.get("log_level", "INFO"))
    log.info(f"Working directory: {os.getcwd()}")
    create_folders(cfg)


if __name__ == "__main__":
    main()
