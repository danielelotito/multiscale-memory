import numpy as np
import os
import logging
from typing import Dict, Any
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class DeviationEvent:
    """Class to store information about a large deviation event"""
    step: int
    abs_diff: float
    synaptic_matrix: np.ndarray
    neural_state: np.ndarray
    overlaps: np.ndarray
    theoretical_prediction: float
    empirical_value: float
    offset_from_theoretical: float
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays"""
        return {
            'step': self.step,
            'abs_diff': float(self.abs_diff),  # Convert numpy float to Python float
            'theoretical_prediction': float(self.theoretical_prediction),
            'empirical_value': float(self.empirical_value),
            'offset_from_theoretical': float(self.offset_from_theoretical),
            'timestamp': self.timestamp
            # We don't include the arrays in the JSON, they're saved separately
        }

class LargeDeviationsTracker:
    """Class to track and manage large deviations events"""
    
    def __init__(self, cfg, patterns):
        """Initialize the tracker with configuration parameters"""
        # Initialize logger
        self.log = logging.getLogger(__name__ + '.LargeDeviationsTracker')
        
        self.cfg = cfg.large_deviations_callback
        self.max_points = self.cfg.get('max_points', 10)
        self.cooldown = self.cfg.get('cooldown', 100)
        self.simulation_length = cfg.simulation.T
        
        # Internal state
        self.saved_points = 0
        self.last_save_step = -self.cooldown
        self.events = []
        
        # Create output directory  
        self.base_dir = os.path.join(cfg.save.folders.data, 'large_deviations')
        os.makedirs(self.base_dir, exist_ok=True)
        
        
        # Save patterns
        self.save_patterns(patterns)

    def save_patterns(self, patterns: np.ndarray):
        """Save patterns to file"""
        patterns_file = os.path.join(self.base_dir, 'patterns.npy')
        np.save(patterns_file, patterns)
        self.log.info(f"Saved patterns to {patterns_file}")
        
    def save_event(self, event: DeviationEvent):
        """Save a deviation event to file"""
        # Get dimensions from matrices
        N = len(event.neural_state)
        K = event.synaptic_matrix.shape[0]  # or from patterns.shape[0]
        
        # Create filename with event info
        base_name = f"deviation_E-no{self.saved_points}_t{event.step}_N{N}_K{K}"
        base_name_matrix = f"matrix_E-no{self.saved_points}_t{event.step}_N{N}_K{K}"
        
        # Save metadata to JSON
        meta_filename = f"{base_name}.json"
        meta_filepath = os.path.join(self.base_dir, meta_filename)
        with open(meta_filepath, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)
        
        # Save arrays to NPY file
        arrays_filename = f"{base_name_matrix}.npz"
        arrays_filepath = os.path.join(self.base_dir, arrays_filename)
        np.savez(
            arrays_filepath,
            synaptic_matrix=event.synaptic_matrix,
            neural_state=event.neural_state,
            overlaps=event.overlaps
        )
        
    def compute_overlaps(self, sigmas, patterns: np.ndarray) -> np.ndarray:
        """Compute overlap between current state and patterns"""
        return np.dot(patterns, sigmas) / len(sigmas)

    def should_save_event(self, abs_diff: float, offset: float, step: int) -> bool:
        """
        Determine if the current event should be saved.
        
        Args:
            abs_diff: Absolute difference between empirical and theoretical values
            offset: Theoretical offset/threshold
            step: Current simulation step
        
        Returns:
            bool: True if event should be saved
        """ 
        
        # Don't save if we're within cooldown period (unless it's the last step)
        if (step - self.last_save_step <= self.cooldown and step < self.simulation_length - 1):
            return False
        
        # Don't save if we've reached the maximum number of points
        if self.saved_points >= self.max_points:
            return False
        
        # Save if it's either:
        # 1. The last step of simulation OR
        # 2. A significant deviation (abs_diff > offset)
        if step == self.simulation_length - 1:
            return True
            
        
        return abs_diff > offset


def large_deviations_callback(
    diff_normF2: float,
    estimates_Fnorm2: float,
    offset_from_theoretical: float,
    current_state: Dict[str, Any],
    patterns: np.ndarray,
    step: int,
    cfg: Any,
    tracker: LargeDeviationsTracker = None
) -> LargeDeviationsTracker:
    """Callback function to track large deviations between empirical and theoretical predictions."""
    """Of the modified squared Frob norm"""
    # Initialized in simulation.py
    
    # Calculate absolute difference
    abs_diff = np.abs(diff_normF2 - estimates_Fnorm2)
    
    # Check if we should save this event
    if tracker.should_save_event(abs_diff, offset_from_theoretical, step):
        # Create event
        event = DeviationEvent(
            step=step,
            abs_diff=float(abs_diff),  # Convert to Python float
            synaptic_matrix=current_state['Js'].copy(),
            neural_state=current_state['sigmas'].copy(),
            overlaps=tracker.compute_overlaps(current_state['sigmas'], patterns),
            theoretical_prediction=float(estimates_Fnorm2),
            empirical_value=float(diff_normF2),
            offset_from_theoretical=float(offset_from_theoretical)
        )
        
        tracker.saved_points += 1
        tracker.last_save_step = step
        
        
        
        # Save event
        tracker.save_event(event)
        
    return tracker