import numpy as np
from typing import Optional
import logging
from dataclasses import dataclass
from utils import generate_hebb


@dataclass
class ProbabilityInferenceResult:
    """Container for probability inference results"""
    probabilities: np.ndarray
    reconstruction_error: float
    determinant: float
    reconstructed_J: np.ndarray
    delta_Hebbian_recon: float
    normalized: bool = True
    positive: bool = True

class PatternProbabilityInference:
    def __init__(self, patterns: np.ndarray):
        """
        Initialize with known patterns.
        
        Args:
            patterns: Array of shape (K, N) containing K patterns of length N
        """
        self.patterns = patterns
        self.K, self.N = patterns.shape
        self.logger = logging.getLogger(__name__)
        
        # Validate patterns are binary
        if not np.all(np.abs(patterns) == 1):
            raise ValueError("Patterns must be ±1")
            
    def _build_pattern_matrix_binary(self) -> np.ndarray:
        """
        Build system matrix exploiting binary nature of patterns.
        For ±1 patterns, outer product elements are also ±1.
        """
        n_elements = (self.N * (self.N - 1)) // 2
        A = np.zeros((n_elements, self.K))
        
        # Get indices for upper triangular elements (excluding diagonal)
        rows, cols = np.triu_indices(self.N, k=1)
        
        # For each pattern, directly compute sign of outer product
        for k in range(self.K):
            pattern = self.patterns[k]
            # For binary patterns, outer product element = product of pattern elements
            A[:, k] = pattern[rows] * pattern[cols]
            
        return A
        
    def _extract_upper_triangular(self, J: np.ndarray) -> np.ndarray:
        """Extract upper triangular elements from J (excluding diagonal)"""
        rows, cols = np.triu_indices(self.N, k=1)
        return J[rows, cols]

    def infer_probabilities(
        self, 
        J: np.ndarray, 
        k: float = 1.0,
        force_normalization: bool = True,
        force_positive: bool = True
    ) -> ProbabilityInferenceResult:
        """
        Infer pattern probabilities from synaptic matrix J.
        
        Args:
            J: Observed synaptic matrix of shape (N, N)
            k: Scaling factor used in the synaptic dynamics (default=1.0)
            force_normalization: Whether to force probabilities to sum to 1
            force_positive: Whether to force probabilities to be non-negative
            
        Returns:
            ProbabilityInferenceResult object containing inferred probabilities and analysis
        """
        if J.shape != (self.N, self.N):
            raise ValueError(f"J must have shape ({self.N}, {self.N})")
            
        # Scale J by 1/k since J ≈ k * sum(p_μ * outer(ξ_μ, ξ_μ))
        J_scaled = J / k
        
        # Build system matrix
        A = self._build_pattern_matrix_binary()
        
        # Extract upper triangular elements from scaled J
        b = self._extract_upper_triangular(J_scaled)
        
        n_equations = A.shape[0]  # N(N-1)/2
        n_unknowns = self.K
        
        try:
            if n_equations == n_unknowns:
                # Square system - use direct inverse
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    self.logger.warning(f"Matrix nearly singular. Determinant: {det:.2e}")
                probs = np.linalg.solve(A, b)
                
            elif n_equations > n_unknowns:
                # Overdetermined system - use normal equations
                ATA = A.T @ A
                ATb = A.T @ b
                det = np.linalg.det(ATA)
                if abs(det) < 1e-10:
                    self.logger.warning(f"Normal equations matrix nearly singular. Determinant: {det:.2e}")
                probs = np.linalg.solve(ATA, ATb)
                
            else:
                # Underdetermined system - use minimum norm solution
                AAT = A @ A.T
                det = np.linalg.det(AAT)
                if abs(det) < 1e-10:
                    self.logger.warning(f"AAT matrix nearly singular. Determinant: {det:.2e}")
                probs = A.T @ np.linalg.solve(AAT, b)
            
            # Force non-negative probabilities if requested
            if force_positive:
                probs = np.maximum(probs, 0)
            
            # Normalize probabilities if requested
            if force_normalization:
                probs = probs / np.sum(probs)
            
            # Reconstruct full matrix using inferred probabilities
            J_reconstructed = self.reconstruct_J(probs, k)
            
            
            # Compute reconstruction error, see diff_Fnorm2 function in utils.py
            error = k**2  * np.linalg.norm(J - J_reconstructed, 'fro') ** 2 / (self.N * (self.N - 1))
            
            
            # Compute Delta using the hebbian matrix, see diff_Fnorm2 function in utils.py
            Hebb = generate_hebb(self.patterns, probabilities = None)
            delta_Hebbian_recon =  np.linalg.norm(Hebb - J_reconstructed, 'fro') ** 2 / (self.N * (self.N - 1))

            
            self.logger.info(
                f"Probability inference completed:\n"
                f"  Reconstruction error (mod Frob norm): {error:.2e}\n"
                f"  Delta Hebbian reconstruction: {delta_Hebbian_recon:.2e}\n"
                f"  Determinant: {det:.2e}\n"
                f"  System type: {n_equations} equations, {n_unknowns} unknowns\n"
                f"  Sum of probabilities: {np.sum(probs):.3f}\n"
                f"  Min probability: {np.min(probs):.3e}"
            )
            
            return ProbabilityInferenceResult(
                probabilities=probs,
                reconstruction_error=error,
                delta_Hebbian_recon=delta_Hebbian_recon,
                determinant=det,
                reconstructed_J=J_reconstructed,
                normalized=force_normalization,
                positive=force_positive
            )
            
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Failed to solve system: {str(e)}")
            raise
            
    def reconstruct_J(self, probabilities: np.ndarray, k: float = 1.0) -> np.ndarray:
        """
        Reconstruct synaptic matrix from inferred probabilities.
        
        Args:
            probabilities: Array of inferred pattern probabilities
            k: Scaling factor used in synaptic dynamics
        """
        J = np.zeros((self.N, self.N))
        
        # J = k * sum(p_μ * outer(ξ_μ, ξ_μ))
        for p, pattern in zip(probabilities, self.patterns):
            J += k * p * np.outer(pattern, pattern)
            
        # Ensure symmetry and zero diagonal
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        
        return J

def example_usage():
    """Example showing how to infer probabilities from a synaptic matrix"""
    # Generate test data
    N = 10  # number of neurons
    K = 4   # number of patterns
    k = 1.5 # scaling factor
    
    # Generate random ±1 patterns
    patterns = np.random.choice([-1, 1], size=(K, N))
    
    # Generate true probabilities that sum to 1
    true_probs = np.random.uniform(0, 1, K)
    true_probs /= true_probs.sum()
    
    # Create true J matrix
    J_true = np.zeros((N, N))
    for prob, pattern in zip(true_probs, patterns):
        J_true += k * prob * np.outer(pattern, pattern)
    J_true = (J_true + J_true.T) / 2
    np.fill_diagonal(J_true, 0)
    
    # Add small noise
    noise_level = 0.01
    J_obs = J_true + noise_level * np.random.randn(N, N)
    J_obs = (J_obs + J_obs.T) / 2
    np.fill_diagonal(J_obs, 0)
    
    # Infer probabilities
    try:
        inferrer = PatternProbabilityInference(patterns)
        result = inferrer.infer_probabilities(J_obs, k=k)
        
        print("\nTrue probabilities:", true_probs)
        print("Inferred probabilities:", result.probabilities)
        print(f"Reconstruction error (mod Frob norm): {result.reconstruction_error:.2e}")
        print(f"Determinant: {result.determinant:.2e}")
        
    except Exception as e:
        print(f"Failed: {str(e)}")

if __name__ == "__main__":
    example_usage()