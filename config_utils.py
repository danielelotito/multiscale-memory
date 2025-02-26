import os
import yaml
from omegaconf import OmegaConf
from typing import Optional
import logging

log = logging.getLogger(__name__)


def get_first_run_config(multirun_dir: str) -> Optional[dict]:
    """Get configuration from the first run in a multirun directory."""
    try:
        # Find first run directory (usually '0')
        first_run_dir = os.path.join(multirun_dir, "0")

        if not os.path.exists(first_run_dir):
            log.error(f"Could not find first run directory in {multirun_dir}")
            return None

        # Load the configuration
        config_path = os.path.join(first_run_dir, ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            log.error(f"Could not find config.yaml in {config_path}")
            return None

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return OmegaConf.create(config)

    except Exception as e:
        log.error(f"Error loading first run config: {str(e)}")
        return None
