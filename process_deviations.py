import os
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from utils import generate_hebb
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt
import json
from model import BaseAssociativeMemory

from setup_matplotlib_style import setup_matplotlib_style

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def generate_noisy_pattern(pattern: np.ndarray, r: float, M: int) -> np.ndarray:
    """
    Generate M noisy copies of a pattern
    
    P(η_i^(μ,a)|ξ_i^μ) = (1-r)/2 δ_{η,-ξ} + (1+r)/2 δ_{η,ξ}
    
    Args:
        pattern: Original pattern (ξ)
        r: Noise parameter (r=1 means perfect copy, r=0 means orthogonal)
        M: Number of copies to generate
    
    Returns:
        Array of shape (M, len(pattern)) containing noisy copies (η)
    """
    N = len(pattern)
    noisy_patterns = np.zeros((M, N))
    
    for m in range(M):
        # For each entry, probability (1+r)/2 to match pattern, (1-r)/2 to be opposite
        p_same = (1 + r) / 2
        rand = np.random.random(N)
        noisy_patterns[m] = np.where(rand < p_same, pattern, -pattern)
                
    return noisy_patterns

def compute_overlaps(state: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    """Compute overlaps between state and all patterns."""
    return np.dot(patterns, state) / len(state)

def run_simulation_from_state(
    initial_state: np.ndarray,
    synaptic_matrix: np.ndarray,
    patterns: np.ndarray,
    cfg: DictConfig
) -> pd.DataFrame:
    """
    Run simulation from a given initial state and track overlaps.
    
    Args:
        initial_state: Initial neural state
        synaptic_matrix: Initial synaptic weights
        patterns: Set of stored patterns
        cfg: Configuration object
        
    Returns:
        DataFrame with overlap trajectories for each pattern
    """
    # Initialize model with provided initial state and synapses
    model = BaseAssociativeMemory(
        cfg=cfg.model,
        initial_state=initial_state,
        initial_synapses=synaptic_matrix
    )
    
    # Track overlaps
    steps = int(cfg.simulation.T)
    K = len(patterns)
    overlaps = np.zeros((steps, K))
    
    # Run simulation
    for t in range(steps):
        overlaps[t] = compute_overlaps(model.sigma, patterns)
        model.step(h=np.zeros(len(initial_state)))  # No external field
        
    return pd.DataFrame(overlaps, columns=[f'pattern_{i}' for i in range(K)])

class DeviationAnalyzer:
    def __init__(self, cfg: DictConfig):
        """Initialize analyzer with configuration."""
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.results_dir = self.output_dir / 'deviation_analysis'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(f"{__name__}.DeviationAnalyzer")
        setup_matplotlib_style(medium_fonts=True)
        
    
    def load_event_data(self, deviation_dir: Path) -> Tuple[List[Dict], np.ndarray, List[Dict]]:
        """
        Load all event data, patterns, and matrices.
        
        Args:
            deviation_dir: Directory containing the large deviations data
            
        Returns:
            Tuple containing:
                - List of event metadata dictionaries
                - Pattern array
                - List of dictionaries containing matrices for each event
        """
        self.log.info(f"Loading data from {deviation_dir}")
        
        # Load patterns
        patterns = np.load(deviation_dir / 'patterns.npy')
        self.log.info(f"Loaded patterns with shape {patterns.shape}")
        
        # Load all event files
        events = []
        matrices = []
        
        # Sort files to ensure consistent ordering
        meta_files = sorted(deviation_dir.glob('deviation_*.json'))
        
        for meta_file in meta_files:
            # Get timestamp from filename
            timestamp = meta_file.stem.split('_', 1)[1]
            
            # Load metadata
            with open(meta_file) as f:
                event = json.load(f)
            events.append(event)
            
            # Load corresponding matrices
            matrix_file = deviation_dir / f"matrix_{timestamp}.npz"
            if not matrix_file.exists():
                raise FileNotFoundError(
                    f"Matrix file {matrix_file} not found for metadata {meta_file}"
                )
                
            with np.load(matrix_file) as data:
                matrices.append({
                    'synaptic_matrix': data['synaptic_matrix'],
                    'neural_state': data['neural_state'],
                    'overlaps': data['overlaps']
                })
                
        self.log.info(f"Loaded {len(events)} events")
        return events, patterns, matrices
        
    def analyze_deviation_event(
        self,
        synaptic_matrix: np.ndarray,
        patterns: np.ndarray,
        event_overlaps: np.ndarray,
        dataset_quality_levels: List[float],
        chose_another_pattern: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        Analyze a single deviation event with different noise levels.
        
        Args:
            synaptic_matrix: Synaptic weights at the event
            patterns: Stored patterns
            event_overlaps: Pattern overlaps at the event
            noise_levels: List of noise parameters to test
            
        Returns:
            DataFrame containing analysis results
        """
        results = []
        
        pattern_idx = np.argmax(abs(event_overlaps))
        if chose_another_pattern:
            pattern_idx = np.argmin(abs(event_overlaps))
            
        base_pattern = patterns[pattern_idx]
        self.log.info(f"Analyzing deviation event with highest overlap to pattern {pattern_idx}")
        
        for r in dataset_quality_levels:
            self.log.info(f"Processing quality level r={r}")
            
            # Generate M noisy versions of the highest-overlap pattern
            noisy_patterns = generate_noisy_pattern(
                base_pattern, 
                r, 
                self.cfg.M
            )
            
            # Run simulations for each noisy pattern
            all_overlaps = []
            for idx, noisy_initial_state in enumerate(noisy_patterns):
                overlaps_df = run_simulation_from_state(
                    initial_state=noisy_initial_state,
                    synaptic_matrix=synaptic_matrix,
                    patterns=patterns,
                    cfg=self.cfg
                )
                all_overlaps.append(overlaps_df)
            
            # Compute statistics
            mean_overlaps = pd.concat(all_overlaps).groupby(level=0).mean()
            std_overlaps = pd.concat(all_overlaps).groupby(level=0).std()
            
            # Store results
            results.append({
                'noise_level': r,
                'mean_overlaps': mean_overlaps.iloc[-1].values,
                'std_overlaps': std_overlaps.iloc[-1].values,
                'pattern_idx': pattern_idx,
                'full_trajectory': {
                    'mean': mean_overlaps,
                    'std': std_overlaps
                }
            })
            
        return pd.DataFrame(results)
    
    def analyze_pattern_probabilities(self, events, patterns, matrices, output_file: str = 'probability_analysis.txt'):
        """
        Analyze pattern probabilities for each large deviation event and save results to a text file.
        
        Args:
            events: List of event metadata dictionaries
            patterns: Array of stored patterns
            matrices: List of dictionaries containing matrices for each event
            output_file: Name of output file for analysis results
        """
        # Initialize probability inferrer
        from pattern_probabilities_inference import PatternProbabilityInference
        inferrer = PatternProbabilityInference(patterns)
        
        # Open output file
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            f.write("Pattern Probability Analysis for Large Deviation Events\n")
            f.write("=" * 80 + "\n\n")
            
            # Analyze each event
            for idx, (event, matrix_data) in enumerate(zip(events, matrices)):
                f.write(f"Event {idx+1} Analysis\n")
                f.write("-" * 40 + "\n")
                
                # Write event metadata
                f.write(f"Timestep: {event['step']}\n")
                f.write(f"Absolute difference: {event['abs_diff']:.6f}\n")
                f.write(f"Theoretical prediction: {event['theoretical_prediction']:.6f}\n")
                f.write(f"Empirical value (Modified squared F-norm of the synaptic matrix at timestep): {event['empirical_value']:.6f}\n")
                
                # Current overlaps
                f.write("\nPattern Overlaps:\n")
                for i, overlap in enumerate(matrix_data['overlaps']):
                    f.write(f"  Pattern {i}: {overlap:.6f}\n")
                
                # Infer probabilities from current J
                try:
                    # Try inference with both k=1 and actual scaling
                    k_list = [1.0, self.cfg.model.k] if self.cfg.model.k is not None else [1.0]
                    for k in k_list:
                        f.write(f"\nInferred Probabilities (k={k:.3f}):\n")
                        result = inferrer.infer_probabilities(
                            matrix_data['synaptic_matrix'],
                            k=k,
                            force_normalization=True,
                            force_positive=True
                        )
                        
                        # Write probability results
                        for i, prob in enumerate(result.probabilities):
                            f.write(f"  Pattern {i}: {prob:.6f}\n")
                            
                        f.write(f"\nReconstruction Analysis:\n")
                        f.write(f"  Error: {result.reconstruction_error:.6e}\n")
                        f.write(f"  Determinant: {result.determinant:.6e}\n")
                        f.write(f"  Modified squared Frobenius norm between J_empirical and J_Reconstruction: {result.delta_Hebbian_recon:.6e}\n")
                        # Compute relative error
                        rel_error = (result.delta_Hebbian_recon - event['empirical_value']) / event['empirical_value']
                        f.write(f"  Relative Error (D_hebb_recon - D_empirical)/ D_empirical: {rel_error:.6e}\n")
                
                except Exception as e:
                    f.write(f"\nError in probability inference: {str(e)}\n")
                
                # Compare with Hebbian matrix
                f.write("\nComparison with Hebbian Matrix:\n")
                Hebb = generate_hebb(patterns, probabilities=None)  # Uniform probabilities
                hebb_diff = np.linalg.norm(matrix_data['synaptic_matrix'] - Hebb, 'fro')
                hebb_rel_diff = hebb_diff / np.linalg.norm(Hebb, 'fro')
                f.write(f"  Frobenius norm difference: {hebb_diff:.6e}\n")
                f.write(f"  Relative difference: {hebb_rel_diff:.6e}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
                
            self.log.info(f"Saved probability analysis to {output_path}")    
    def process_events(self, events_path: str):
        """
        Process all deviation events and generate analysis.
        
        Args:
            events_path: Path to directory containing deviation events
        """
        events_dir = Path(events_path)
        events, patterns, matrices = self.load_event_data(events_dir)
        
        self.analyze_pattern_probabilities(events, patterns, matrices)
        if self.cfg.get("generate_plots", True):
            self.generate_and_plot_results(events, patterns, matrices, use_hebb=False, title_suffix = ' - Target pattern')
            self.generate_and_plot_results(events, patterns, matrices, use_hebb=False, title_suffix = ' - Another pattern', chose_another_pattern=True)
            self.generate_and_plot_results(events, patterns, matrices, use_hebb=True, title_suffix = ' - Hebbian - Target pattern')
            self.generate_and_plot_results(events, patterns, matrices, use_hebb=True, title_suffix = ' - Hebbian - Another pattern', chose_another_pattern=True)
        
        

    def generate_and_plot_results(self, events, patterns, matrices, use_hebb=False, title_suffix = '',
                                  chose_another_pattern: Optional[bool] = False):
        Hebb = generate_hebb(patterns, probabilities=None)
        
        all_results = []
        for idx, (event, matrix_data) in enumerate(zip(events, matrices)):
            self.log.info(f"Processing event {idx+1}/{len(events)} at step {event['step']}")
            
            #Using the synaptic matrix and patterns from the event
            results = self.analyze_deviation_event(
                matrix_data['synaptic_matrix'] if not use_hebb else Hebb,
                patterns,
                matrix_data['overlaps'],
                self.cfg.noise_levels,
                chose_another_pattern=chose_another_pattern
            )
            results['event_step'] = event['step']
            results['abs_diff'] = event['abs_diff']
            results['theoretical_prediction'] = event['theoretical_prediction']
            results['empirical_value'] = event['empirical_value']
            
            all_results.append(results)
       
        # Save combined results
        combined_results = pd.concat(all_results, ignore_index=True)
        results_file = self.results_dir / 'analysis_results.csv'
        combined_results.to_csv(results_file)
        self.log.info(f"Saved combined results to {results_file}")
        
        
        # Plot results
        self.plot_results(combined_results, title_suffix=title_suffix)
        return event,matrix_data


    def plot_results(self, results: pd.DataFrame, title_suffix: str=''):
        """Plot analysis results for each event separately."""
        # Process each event
        for event_step in results['event_step'].unique():
            event_results = results[results['event_step'] == event_step]
            event_number = event_results.index[0]
            K = len(event_results['mean_overlaps'].iloc[0])
            N = len(event_results['pattern_idx'].unique())
            
            # Plot 1: Final overlap with all patterns
            pattern_idx = event_results['pattern_idx'].iloc[0]
            self.plot_core( event_results, pattern_idx, K, event_number, event_step , title_suffix)

    def plot_core(self, event_results, pattern_idx, K, event_number, event_step , title_suffix ):
        plt.figure(figsize=(8, 6))
        for p in range(K):
            data = event_results
            mean_overlaps = [m[p] for m in data['mean_overlaps']]
            std_overlaps = [s[p] for s in data['std_overlaps']]
                
            plt.errorbar(
                    data['noise_level'],
                    mean_overlaps,
                    yerr=std_overlaps,
                    label=f'Pattern {p}',
                    capsize=5,
                    marker='o',
                    linestyle='-'
                )
            
        plt.xlabel('r')
        plt.ylabel('Final Overlap')
        plt.title(f'Pattern Retrieval' + title_suffix)
        plt.legend()
        plt.grid(True)
            
        stats_text_minimal = (
                f"$\|\\Delta\|_{{F^2}} = {event_results.iloc[0]['empirical_value']:.5f}$\n"
                f"$E(\|\\Delta\|_{{F^2}}) = {event_results.iloc[0]['theoretical_prediction']:.5f}$\n"
                f"$| \|\\Delta\|_{{F^2}} - "
                f"E(\|\\Delta\|_{{F^2}})| = " +
                f"{event_results.iloc[0]['abs_diff']:.5f}$"
            )
        stats_text_full = (
                f"$\|\\Delta^{{({int(event_step)})}}\|_{{F^2}} = {event_results.iloc[0]['empirical_value']:.5f}$\n"
                f"$E(\|\\Delta^{{({int(event_step)})}}\|_{{F^2}}) = {event_results.iloc[0]['theoretical_prediction']:.5f}$\n"
                f"$| \|\\Delta^{{({int(event_step)})}}\|_{{F^2}} - "
                f"E(\|\\Delta^{{({int(event_step)})}}\|_{{F^2}})| = " +
                f"{event_results.iloc[0]['abs_diff']:.5f}$"
            )
        
        if self.cfg.get("minimal_stats", False):
            stats_text = stats_text_minimal    
            plt.text(0.05, 0.05, stats_text,
                        transform=plt.gca().transAxes,
                        bbox=dict(facecolor='white', alpha=0.8),
                        fontsize=8,
                        verticalalignment='bottom',
                        horizontalalignment='left')
            
        plot_file = self.results_dir / f'pattern_retrieval_E{event_number}_n{int(event_step)}{title_suffix}.pdf'
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
            
            # Plot 2: Evolution of target pattern only
        plt.figure(figsize=(8, 6))
            
        # Get sorted unique noise levels
        noise_levels = sorted(event_results['noise_level'].unique())
        n_desired = 7  # Number of noise levels we want to plot

        if len(noise_levels) > n_desired:
            # Use array indexing that guarantees unique indices
            indices = np.round(np.linspace(0, len(noise_levels)-1, n_desired)).astype(int)
            # Remove any duplicates that might occur due to rounding
            indices = np.unique(indices)
            # Select noise levels at those indices
            noise_levels = [noise_levels[i] for i in indices]

        for r in noise_levels:
            data = event_results[event_results['noise_level'] == r]
            if len(data) == 0:
                continue
                    
            data = data.iloc[0]
            traj = data['full_trajectory']['mean']
            pattern_idx = data['pattern_idx']
            times = np.arange(len(traj))
                
            plt.plot(times, traj[f'pattern_{pattern_idx}'], 
                        label=f'r={r:.1f}',
                        linestyle='-')
            
        plt.xlabel('Time Steps')
        plt.ylabel('Overlap')
        plt.title(f'Retrieval Dynamics'+ title_suffix)
            
            # Add the same deviation information to the retrieval plot
        if self.cfg.get("minimal_stats", False):
            stats_text = stats_text_minimal
            plt.text(0.05, 0.05, stats_text,
                        transform=plt.gca().transAxes,
                        bbox=dict(facecolor='white', alpha=0.8),
                        fontsize=8,
                        verticalalignment='bottom',
                        horizontalalignment='left')
            
        plt.legend()
        plt.grid(True)
            
        trajectories_file = self.results_dir / f'target_retrieval_E{event_number}_n{int(event_step)}{title_suffix}.pdf'
        plt.savefig(trajectories_file, bbox_inches='tight')
        plt.close()
            
        self.log.info(f"Saved plots for event {event_number} at n={int(event_step)} to {trajectories_file}")
                

@hydra.main(config_path="config_dev", config_name="config_deviations")
def main(cfg: DictConfig):
    # Change to original working directory
    os.chdir(get_original_cwd())
    analyzer = DeviationAnalyzer(cfg)
    analyzer.process_events(cfg.deviations_path)

if __name__ == "__main__":
    main()