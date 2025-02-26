import numpy as np
from omegaconf import DictConfig
from scipy.special import softmax
from typing import Optional, Dict
from fluctuation_analysis import compute_theoretical_std

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class BaseAssociativeMemory:
    """Base class for associative memory models."""

    def __init__(self, cfg: DictConfig, initial_state: Optional[np.ndarray] = None, initial_synapses: Optional[np.ndarray] = None):
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object
            initial_state: Optional initial neural state array (shape: N)
            initial_synapses: Optional initial synaptic matrix (shape: N x N)
        """
        self._initialize_parameters(cfg)
        if initial_state is not None and initial_synapses is not None:
            # Use provided initial states
            self._initialize_state_from_values(initial_state, initial_synapses)
        else:
                # Use default initialization
                
            self.N = cfg.N
            self.K = self.set_K_alpha(cfg)
            self._initialize_state_from_config()

    def _initialize_parameters(self, cfg: DictConfig):
        """Initialize model parameters from configuration."""
        self.beta = cfg.beta
        self.k = self.set_k(cfg)
        self.tau_s = cfg.tau_s
        self.u = cfg.u
        self.rho = cfg.get("rho", 0)
        if isinstance(cfg.tau_J, str) and cfg.tau_J.lower() == 'infinity':
            self.freeze_synapses = True
            self.tau_J = float('inf')
        else:
            self.freeze_synapses = False
            self.tau_J = float(cfg.tau_J)
        self.eps = self.tau_s / self.tau_J if not self.freeze_synapses else 0
        self.cfg = cfg

    def _initialize_state_from_config(self):
        """Initialize model state variables from configuration."""
        self.sigma = np.random.uniform(-1, 1, self.N)
        self.J = self.initialize_synapses()
        self.J_0 = self.J.copy()

    def _initialize_state_from_values(self, initial_state: Optional[np.ndarray] = None, initial_synapses: Optional[np.ndarray] = None):
        """
        Initialize model state variables from provided values.
        
        Args:
            initial_state: Initial neural state
            initial_synapses: Initial synaptic weights
        """
        if initial_state is not None:
            self.N = len(initial_state)
            self.sigma = initial_state.copy()
        else:
            self.sigma = np.random.uniform(-1, 1, self.N)
            
        if initial_synapses is not None:
            if initial_synapses.shape != (self.N, self.N):
                raise ValueError(f"Initial synapses shape {initial_synapses.shape} doesn't match network size {self.N}")
            self.J = initial_synapses.copy()
        else:
            self.J = self.initialize_synapses()
        self.J_0 = self.J.copy()      
        
        
    def initialize_synapses(self) -> np.ndarray:
        """
        Initialize synaptic weights based on configured pattern distribution.
        Returns a symmetric matrix with zero diagonal and variance scaled by k/sqrt(K).
        """
        if self.cfg.pattern_distribution == "rademacher":
            # Standard normal scaled by k/sqrt(K)
            J = np.random.normal(0, self.k / np.sqrt(self.K), (self.N, self.N))
            
        elif self.cfg.pattern_distribution == "gaussian":
            # Same as rademacher case
            J = np.random.normal(0, self.k / np.sqrt(self.K), (self.N, self.N))
            
        elif self.cfg.pattern_distribution == "uniform":
            # Scaled by k/(sqrt(9K)) to match variance of [-1,1] uniform distribution
            J = np.random.normal(0, self.k / np.sqrt(9 * self.K), (self.N, self.N))
            
        else:
            log.error(f"Unsupported pattern distribution: {self.cfg.pattern_distribution}")
            log.error("Setting pattern distribution to rademacher")
            J = np.random.normal(0, self.k / np.sqrt(self.K), (self.N, self.N))

        # Zero diagonal
        np.fill_diagonal(J, 0)
        
        # Symmetrize: (J + J.T)/2
        # Multiply by sqrt(2) to preserve variance after symmetrization
        # because Var((X + Y)/2) = (Var(X) + Var(Y))/4 when X,Y independent
        J = ((J + J.T) / 2) * np.sqrt(2)
        
        return J
        
    def set_k(self, cfg: DictConfig) -> float:
        if cfg.k is None:
            # log.info("k is None, setting it to tanh(beta)")
            return np.tanh(self.beta)
        log.info(f"k is set to {cfg.k}")
        return cfg.k

    def set_K_alpha(self, cfg: DictConfig) -> float:
        if cfg.K < 1:
            log.info("K, num of patterns, is less than 1, treating it as alpha")
            return int(cfg.K * self.N)
        return cfg.K

    def g(self, x: np.ndarray) -> np.ndarray:
        if "activation" in self.cfg:
            if self.cfg.activation == "sigmoid":
                return 1 / (1 + np.exp(-x))
            if self.cfg.activation == "relu":
                return np.maximum(0, x)
            if self.cfg.activation == "tanh":
                return np.tanh(x)
        return x

    def update_neurons(self, h: np.ndarray) -> None:
        field = np.dot(self.J, self.sigma) + h * self.u
        self.sigma += (1 / self.tau_s) * (-self.sigma + np.tanh(self.beta * field))

    def update_synapses(self) -> None:
        if self.freeze_synapses:
            return  # Don't update synapses if they're frozen
        g_sigma = self.g(self.sigma)
        
        self.J += (1 / self.tau_J) * (-self.J + self.k * np.outer(g_sigma, g_sigma))
        np.fill_diagonal(self.J, 0)

    def step(self, h: np.ndarray) -> None:
        self.update_neurons(h)
        self.update_synapses()

    def get_state(self) -> dict:
        return {"sigmas": self.sigma.copy(), "Js": self.J.copy()}

    def get_sigma(self) -> np.ndarray:
        return self.sigma.copy()

    def get_J(self) -> np.ndarray:
        return self.J.copy()



    def compute_theoretical_std(
            self, Jhebb: np.ndarray
        ) -> float:
        return compute_theoretical_std(self.N, self.K, self.tau_J, rho = self.rho )
        
    

class AssociativeMemory(BaseAssociativeMemory):
    """Main associative memory model implementation with theoretical estimates."""

    def get_estimates_Fnorm2_old(self, i, Jhebb: np.ndarray = None):
        # update_simulation_dict is called now before the update of coupling matrix (model.step)
        # and the initial value of J is not stored there
        if Jhebb is None:
            log.error("Jhebb is not provided")
            return None
        eps = self.tau_s / self.tau_J
        norm_factor = 1 / (self.N * (self.N - 1))
        var_JHebb = norm_factor * np.sum(Jhebb**2)
        varJHebb_factor = ((4 - eps) * (1 - eps) ** (2 * i) - eps) / (2 - eps)
        independent_from_ij_term = (eps - eps * (1 - eps) ** (2 * i)) / (2 - eps)

        return varJHebb_factor * var_JHebb + independent_from_ij_term

    def get_estimates_Fnorm2(self, i: int, Jhebb: np.ndarray) -> float:
        """Calculate theoretical estimate of squared Frobenius norm."""
        if Jhebb is None:
            log.error("Jhebb not provided")
            return None
        step = i
        eps = self.tau_s / self.tau_J
        norm_factor = 1 / (self.N * (self.N - 1))

        # Calculate sample statistics
        stats = self._calculate_sample_statistics(Jhebb, norm_factor)

        # Calculate factors
        factors = self._calculate_estimate_factors(eps, step)

        return (
            factors["var_JHebb"] * stats["var_JHebb"]
            + factors["var_J0"] * stats["var_J0"]
            + factors["cov"] * stats["cov"]
            + factors["independent"]
        ) / self.k**2

    def _calculate_sample_statistics(
        self, Jhebb: np.ndarray, norm_factor: float
    ) -> Dict[str, float]:
        """Calculate sample statistics for the estimate."""
        return {
            "var_JHebb": norm_factor * np.sum(Jhebb**2),
            "var_J0": norm_factor * np.sum(self.J_0**2),
            "cov": norm_factor * np.sum(Jhebb * self.J_0),
        }

    def _calculate_estimate_factors(self, eps: float, step: int) -> Dict[str, float]:
        """Calculate factors used in the Frobenius norm estimate."""
        return {
            "var_JHebb": self.k**2
            * ((2) * (1 - eps) ** (2 * step) - eps)
            / (2 - eps),
            "var_J0": (1 - eps) ** (2 * step),
            "cov": -2 * self.k * (1 - eps) ** (2 * step),
            "independent": self.k**2
            * (eps - eps * (1 - eps) ** (2 * step))
            / (2 - eps),
        }


    def get_horizonal_line_y(
        self, Jhebb: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """Calculate the horizontal line (asymptotic) value."""
        if Jhebb is None:
            log.error("Jhebb not provided")
            return None

        eps = self.tau_s / self.tau_J
        norm_factor = 1 / (self.N * (self.N - 1))
        var_JHebb = norm_factor * np.sum(Jhebb**2)

        return eps / (2 - eps) * (1 - var_JHebb)


class MemoryMatrixAssociativeMemory(BaseAssociativeMemory):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.Xi = np.random.uniform(-1, 1, (self.K, self.N))
        self.Tra = np.eye(self.K)
        self.beta_Xi = cfg.get("beta_Xi", 1)
        self.beta_T = cfg.get("beta_T", cfg.get("beta", 1))

        self.tau_T = cfg.get("tau_T", self.set_tau_T(cfg))

    def update_memory(self) -> None:
        similarities = np.dot(self.Xi, self.sigma)
        tau_inv = softmax(self.beta_Xi * similarities)

        for k in range(self.K):
            self.Xi[k] += (1 / tau_inv[k]) * (-self.Xi[k] + self.sigma)

    def set_tau_T(self, cfg) -> float:
        if self.cfg.tau_J is None:
            log.info("tau_J")
            return 10
        if self.cfg.tau_s is None:
            log.info("tau_s")
            return 10

        log.info("tau_T is the average of tau_J and tau_s")
        return (self.cfg.tau_J + self.cfg.tau_s) / 2

    def update_transition_matrix(
        self, sigma_conf1: np.ndarray, sigma_conf2: np.ndarray
    ) -> None:
        similarity1 = np.dot(self.Xi, sigma_conf1)
        similarity2 = np.dot(self.Xi, sigma_conf2)

        exp_term1 = np.exp(self.beta_T * similarity1)
        exp_term2 = np.exp(self.beta_T * similarity2)

        transition_increment = np.outer(exp_term1, exp_term2)
        transition_increment *= self.Tra

        normalization = np.sum(transition_increment)

        delta_T = transition_increment / normalization - self.Tra
        self.Tra += self.tau_T * delta_T

        self.Tra /= np.sum(self.Tra, axis=1, keepdims=True)

    def step(self, h: np.ndarray) -> None:
        super().step(h)
        self.update_memory()

    def get_state(self) -> dict:
        base_state = super().get_state()
        return {**base_state, "Xis": self.Xi.copy(), "Tras": self.Tra.copy()}

    def get_memory(self) -> np.ndarray:
        return self.Xi.copy()

    def get_transition_matrix(self) -> np.ndarray:
        return self.Tra.copy()
