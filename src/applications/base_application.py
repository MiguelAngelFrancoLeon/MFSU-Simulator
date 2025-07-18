"""
Base Application Class for MFSU Simulator

This module provides the abstract base class for all MFSU applications.
Each specific application (superconductivity, gas dynamics, cosmology) 
should inherit from this class and implement the required methods.

Author: MFSU Development Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import logging
from dataclasses import dataclass, field

from ..core.mfsu_equation import MFSUEquation
from ..utils.parameter_validation import ParameterValidator
from ..utils.logger import get_logger


@dataclass
class ApplicationParameters:
    """
    Data class to hold application-specific parameters
    """
    # Core MFSU parameters
    alpha: float = 0.5          # Fractional order parameter
    beta: float = 0.1           # Stochastic coupling strength
    gamma: float = 0.01         # Nonlinear term coefficient
    hurst: float = 0.7          # Hurst parameter for fractional noise
    
    # Numerical parameters
    dt: float = 0.01            # Time step
    dx: float = 0.1             # Spatial step
    grid_size: int = 100        # Grid size
    max_time: float = 10.0      # Maximum simulation time
    
    # Application-specific parameters (to be extended by subclasses)
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BaseApplication(ABC):
    """
    Abstract base class for MFSU applications.
    
    This class provides the common interface and functionality that all
    MFSU applications should implement. Each specific application should
    inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, name: str, parameters: ApplicationParameters):
        """
        Initialize the base application.
        
        Args:
            name: Name of the application
            parameters: Application parameters
        """
        self.name = name
        self.parameters = parameters
        self.logger = get_logger(f"MFSU.{name}")
        self.validator = ParameterValidator()
        
        # Initialize the MFSU equation solver
        self.mfsu_solver = None
        self.is_initialized = False
        
        # Simulation state
        self.current_time = 0.0
        self.simulation_data = {}
        self.analysis_results = {}
        
        self.logger.info(f"Initialized {name} application")
    
    def initialize(self) -> bool:
        """
        Initialize the application and validate parameters.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Validate parameters
            if not self._validate_parameters():
                return False
            
            # Initialize MFSU equation solver
            self.mfsu_solver = MFSUEquation(
                alpha=self.parameters.alpha,
                beta=self.parameters.beta,
                gamma=self.parameters.gamma,
                hurst=self.parameters.hurst
            )
            
            # Set up numerical grid
            self._setup_numerical_grid()
            
            # Initialize application-specific components
            self._initialize_application()
            
            self.is_initialized = True
            self.logger.info(f"{self.name} application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {str(e)}")
            return False
    
    def _validate_parameters(self) -> bool:
        """
        Validate the application parameters.
        
        Returns:
            bool: True if parameters are valid
        """
        try:
            # Validate core MFSU parameters
            self.validator.validate_range(self.parameters.alpha, 0.1, 2.0, "alpha")
            self.validator.validate_range(self.parameters.beta, 0.0, 1.0, "beta")
            self.validator.validate_range(self.parameters.gamma, 0.0, 1.0, "gamma")
            self.validator.validate_range(self.parameters.hurst, 0.0, 1.0, "hurst")
            
            # Validate numerical parameters
            self.validator.validate_positive(self.parameters.dt, "dt")
            self.validator.validate_positive(self.parameters.dx, "dx")
            self.validator.validate_positive(self.parameters.grid_size, "grid_size")
            self.validator.validate_positive(self.parameters.max_time, "max_time")
            
            # Validate application-specific parameters
            return self._validate_application_parameters()
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {str(e)}")
            return False
    
    def _setup_numerical_grid(self):
        """Set up the numerical grid for the simulation."""
        self.x_grid = np.linspace(0, self.parameters.grid_size * self.parameters.dx, 
                                 self.parameters.grid_size)
        self.time_steps = int(self.parameters.max_time / self.parameters.dt)
        self.time_grid = np.linspace(0, self.parameters.max_time, self.time_steps)
    
    def run_simulation(self, initial_conditions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run the complete simulation.
        
        Args:
            initial_conditions: Initial conditions for the simulation
            
        Returns:
            Dict containing simulation results
        """
        if not self.is_initialized:
            self.logger.error("Application not initialized. Call initialize() first.")
            return {}
        
        try:
            self.logger.info(f"Starting {self.name} simulation")
            
            # Set initial conditions
            if initial_conditions is None:
                initial_conditions = self._get_default_initial_conditions()
            
            # Run the simulation
            results = self._run_simulation_loop(initial_conditions)
            
            # Post-process results
            self.simulation_data = self._post_process_results(results)
            
            self.logger.info(f"{self.name} simulation completed successfully")
            return self.simulation_data
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            return {}
    
    def _run_simulation_loop(self, initial_conditions: np.ndarray) -> Dict[str, Any]:
        """
        Run the main simulation loop.
        
        Args:
            initial_conditions: Initial conditions
            
        Returns:
            Dict containing raw simulation results
        """
        psi = initial_conditions.copy()
        results = {
            'time': [],
            'psi': [],
            'energy': [],
            'observables': []
        }
        
        for i, t in enumerate(self.time_grid):
            # Update current time
            self.current_time = t
            
            # Evolve the system one time step
            psi = self.mfsu_solver.evolve_time_step(
                psi, self.parameters.dt, self.x_grid, t
            )
            
            # Calculate observables
            observables = self._calculate_observables(psi, t)
            
            # Store results
            if i % self._get_output_frequency() == 0:
                results['time'].append(t)
                results['psi'].append(psi.copy())
                results['energy'].append(self._calculate_energy(psi))
                results['observables'].append(observables)
            
            # Check convergence or stopping criteria
            if self._check_stopping_criteria(psi, t):
                break
        
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the simulation results.
        
        Returns:
            Dict containing analysis results
        """
        if not self.simulation_data:
            self.logger.warning("No simulation data available for analysis")
            return {}
        
        try:
            self.logger.info(f"Analyzing {self.name} results")
            
            # Perform general analysis
            general_analysis = self._perform_general_analysis()
            
            # Perform application-specific analysis
            specific_analysis = self._perform_specific_analysis()
            
            # Combine results
            self.analysis_results = {
                'general': general_analysis,
                'specific': specific_analysis,
                'metadata': {
                    'application': self.name,
                    'parameters': self.parameters.__dict__,
                    'timestamp': self.current_time
                }
            }
            
            self.logger.info(f"{self.name} analysis completed")
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}
    
    def _perform_general_analysis(self) -> Dict[str, Any]:
        """Perform general analysis common to all applications."""
        psi_array = np.array(self.simulation_data['psi'])
        time_array = np.array(self.simulation_data['time'])
        
        return {
            'final_state_norm': np.linalg.norm(psi_array[-1]),
            'energy_conservation': np.std(self.simulation_data['energy']),
            'simulation_time': time_array[-1],
            'total_steps': len(time_array),
            'average_amplitude': np.mean(np.abs(psi_array)),
            'max_amplitude': np.max(np.abs(psi_array))
        }
    
    def _calculate_energy(self, psi: np.ndarray) -> float:
        """
        Calculate the total energy of the system.
        
        Args:
            psi: Current state
            
        Returns:
            Total energy
        """
        # Kinetic energy (fractional Laplacian term)
        kinetic = self.mfsu_solver.calculate_kinetic_energy(psi, self.x_grid)
        
        # Potential energy (nonlinear term)
        potential = self.parameters.gamma * np.sum(np.abs(psi)**4) * self.parameters.dx
        
        return kinetic + potential
    
    def _get_output_frequency(self) -> int:
        """Get the frequency for storing output data."""
        return max(1, self.time_steps // 1000)  # Store ~1000 time points
    
    def _check_stopping_criteria(self, psi: np.ndarray, t: float) -> bool:
        """
        Check if simulation should stop early.
        
        Args:
            psi: Current state
            t: Current time
            
        Returns:
            True if simulation should stop
        """
        # Check for numerical instability
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            self.logger.warning(f"Numerical instability detected at t={t}")
            return True
        
        # Check for extremely large amplitudes
        if np.max(np.abs(psi)) > 1e10:
            self.logger.warning(f"Solution amplitude too large at t={t}")
            return True
        
        return False
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def _initialize_application(self):
        """Initialize application-specific components."""
        pass
    
    @abstractmethod
    def _validate_application_parameters(self) -> bool:
        """Validate application-specific parameters."""
        pass
    
    @abstractmethod
    def _get_default_initial_conditions(self) -> np.ndarray:
        """Get default initial conditions for the application."""
        pass
    
    @abstractmethod
    def _calculate_observables(self, psi: np.ndarray, t: float) -> Dict[str, Any]:
        """Calculate application-specific observables."""
        pass
    
    @abstractmethod
    def _perform_specific_analysis(self) -> Dict[str, Any]:
        """Perform application-specific analysis."""
        pass
    
    @abstractmethod
    def get_application_description(self) -> str:
        """Get a description of the application."""
        pass
    
    @abstractmethod
    def get_physical_interpretation(self, results: Dict[str, Any]) -> str:
        """Get physical interpretation of the results."""
        pass
    
    # Utility methods
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """
        Update application parameters.
        
        Args:
            new_parameters: Dictionary of new parameter values
        """
        for key, value in new_parameters.items():
            if hasattr(self.parameters, key):
                setattr(self.parameters, key, value)
            else:
                self.parameters.custom_params[key] = value
        
        self.logger.info(f"Updated parameters for {self.name}")
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of current parameters."""
        return {
            'core_parameters': {
                'alpha': self.parameters.alpha,
                'beta': self.parameters.beta,
                'gamma': self.parameters.gamma,
                'hurst': self.parameters.hurst
            },
            'numerical_parameters': {
                'dt': self.parameters.dt,
                'dx': self.parameters.dx,
                'grid_size': self.parameters.grid_size,
                'max_time': self.parameters.max_time
            },
            'custom_parameters': self.parameters.custom_params
        }
    
    def export_results(self, filename: str, format: str = 'hdf5'):
        """
        Export simulation results to file.
        
        Args:
            filename: Output filename
            format: Export format ('hdf5', 'json', 'csv')
        """
        # This would be implemented using the data_io module
        from ..utils.data_io import export_data
        
        export_data(
            data=self.simulation_data,
            filename=filename,
            format=format,
            metadata=self.get_parameter_summary()
        )
        
        self.logger.info(f"Results exported to {filename}")
    
    def __str__(self) -> str:
        """String representation of the application."""
        return f"MFSU {self.name} Application (α={self.parameters.alpha}, β={self.parameters.beta})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
