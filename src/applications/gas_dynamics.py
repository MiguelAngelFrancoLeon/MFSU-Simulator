"""
Gas Dynamics Application Module for MFSU Simulator
==================================================

This module implements the MFSU equation for gas dynamics applications,
including turbulence modeling, compressible flow, and non-linear phenomena.

The MFSU equation: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Where:
- ψ: Gas density or velocity field
- α: Fractional diffusion coefficient
- β: Stochastic noise amplitude
- γ: Non-linear interaction strength
- ξ_H: Fractional Brownian motion with Hurst parameter H
- f(x,t): External forcing term
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.mfsu_equation import MFSUEquation
from ..core.fractional_operators import FractionalOperator
from ..core.stochastic_processes import FractionalBrownianMotion
from ..core.nonlinear_dynamics import NonlinearTerm
from .base_application import BaseApplication

logger = logging.getLogger(__name__)


@dataclass
class GasDynamicsParameters:
    """Parameters specific to gas dynamics applications."""
    
    # Fluid properties
    reynolds_number: float = 1000.0
    mach_number: float = 0.3
    prandtl_number: float = 0.7
    viscosity: float = 1e-5
    density: float = 1.225  # kg/m³ at sea level
    
    # Thermodynamic properties
    temperature: float = 288.15  # K
    pressure: float = 101325.0   # Pa
    specific_heat_ratio: float = 1.4
    gas_constant: float = 287.0  # J/(kg·K)
    
    # Turbulence parameters
    turbulent_intensity: float = 0.05
    integral_length_scale: float = 0.1
    kolmogorov_scale: float = 1e-6
    
    # MFSU specific parameters
    alpha: float = 0.5           # Fractional diffusion order
    beta: float = 0.1            # Stochastic noise amplitude
    gamma: float = 0.01          # Non-linear interaction strength
    hurst_parameter: float = 0.7  # Hurst exponent for fractional noise
    
    # Numerical parameters
    cfl_number: float = 0.5
    artificial_viscosity: float = 0.0


class GasDynamicsApplication(BaseApplication):
    """
    Gas Dynamics application of the MFSU equation.
    
    This class implements various gas dynamics phenomena using the unified
    stochastic fractal model, including:
    - Compressible flow
    - Turbulence modeling
    - Shock wave propagation
    - Boundary layer dynamics
    """
    
    def __init__(self, parameters: GasDynamicsParameters):
        """
        Initialize the gas dynamics application.
        
        Args:
            parameters: Gas dynamics specific parameters
        """
        super().__init__()
        self.params = parameters
        self.name = "Gas Dynamics"
        self.description = "MFSU-based gas dynamics and turbulence modeling"
        
        # Initialize MFSU components
        self.mfsu_equation = MFSUEquation(
            alpha=parameters.alpha,
            beta=parameters.beta,
            gamma=parameters.gamma
        )
        
        self.fractional_operator = FractionalOperator(order=parameters.alpha)
        self.stochastic_process = FractionalBrownianMotion(
            hurst=parameters.hurst_parameter
        )
        self.nonlinear_term = NonlinearTerm(coefficient=parameters.gamma)
        
        # Flow field variables
        self.velocity_field = None
        self.density_field = None
        self.pressure_field = None
        self.temperature_field = None
        
        logger.info(f"Initialized {self.name} application")
    
    def initialize_flow_field(self, grid: np.ndarray, 
                            initial_condition: str = "poiseuille") -> Dict[str, np.ndarray]:
        """
        Initialize the flow field with specified initial conditions.
        
        Args:
            grid: Spatial grid for the simulation
            initial_condition: Type of initial condition
            
        Returns:
            Dictionary containing initialized flow fields
        """
        nx, ny = grid.shape[:2]
        
        if initial_condition == "poiseuille":
            # Poiseuille flow profile
            self.velocity_field = self._create_poiseuille_profile(nx, ny)
            
        elif initial_condition == "turbulent_channel":
            # Turbulent channel flow
            self.velocity_field = self._create_turbulent_channel(nx, ny)
            
        elif initial_condition == "shock_tube":
            # Shock tube problem
            self.velocity_field = self._create_shock_tube(nx, ny)
            
        elif initial_condition == "random_turbulence":
            # Random turbulent field
            self.velocity_field = self._create_random_turbulence(nx, ny)
            
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")
        
        # Initialize other fields
        self.density_field = np.ones((nx, ny)) * self.params.density
        self.pressure_field = np.ones((nx, ny)) * self.params.pressure
        self.temperature_field = np.ones((nx, ny)) * self.params.temperature
        
        return {
            "velocity": self.velocity_field,
            "density": self.density_field,
            "pressure": self.pressure_field,
            "temperature": self.temperature_field
        }
    
    def _create_poiseuille_profile(self, nx: int, ny: int) -> np.ndarray:
        """Create a Poiseuille flow velocity profile."""
        u = np.zeros((nx, ny, 2))  # 2D velocity field
        
        # Parabolic profile in y-direction
        y = np.linspace(0, 1, ny)
        u_profile = 6 * y * (1 - y)  # Normalized parabolic profile
        
        for i in range(nx):
            u[i, :, 0] = u_profile  # u-component
            u[i, :, 1] = 0.0        # v-component
        
        return u
    
    def _create_turbulent_channel(self, nx: int, ny: int) -> np.ndarray:
        """Create a turbulent channel flow with random perturbations."""
        u = self._create_poiseuille_profile(nx, ny)
        
        # Add turbulent fluctuations
        fluctuation_amplitude = self.params.turbulent_intensity
        
        # Random fluctuations with proper scaling
        u_fluctuations = np.random.normal(0, fluctuation_amplitude, (nx, ny, 2))
        
        # Apply energy spectrum filtering (simplified Kolmogorov spectrum)
        for component in range(2):
            u_fluctuations[:, :, component] = self._apply_energy_spectrum_filter(
                u_fluctuations[:, :, component]
            )
        
        return u + u_fluctuations
    
    def _create_shock_tube(self, nx: int, ny: int) -> np.ndarray:
        """Create initial conditions for shock tube problem."""
        u = np.zeros((nx, ny, 2))
        
        # Discontinuity in the middle
        mid_x = nx // 2
        
        # Left side: high pressure region
        u[:mid_x, :, 0] = 0.0
        u[:mid_x, :, 1] = 0.0
        
        # Right side: low pressure region
        u[mid_x:, :, 0] = 0.0
        u[mid_x:, :, 1] = 0.0
        
        return u
    
    def _create_random_turbulence(self, nx: int, ny: int) -> np.ndarray:
        """Create random turbulent field with proper energy spectrum."""
        u = np.zeros((nx, ny, 2))
        
        for component in range(2):
            # Generate random field
            random_field = np.random.normal(0, 1, (nx, ny))
            
            # Apply energy spectrum filtering
            u[:, :, component] = self._apply_energy_spectrum_filter(random_field)
        
        return u * self.params.turbulent_intensity
    
    def _apply_energy_spectrum_filter(self, field: np.ndarray) -> np.ndarray:
        """Apply Kolmogorov energy spectrum filtering to a field."""
        # Fourier transform
        field_fft = np.fft.fft2(field)
        
        # Create wavenumber grid
        kx = np.fft.fftfreq(field.shape[0])
        ky = np.fft.fftfreq(field.shape[1])
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(Kx**2 + Ky**2)
        
        # Avoid division by zero
        K[K == 0] = 1e-10
        
        # Apply Kolmogorov spectrum E(k) ∝ k^(-5/3)
        spectrum_filter = K**(-5/6)  # k^(-5/3) in 2D becomes k^(-5/6) for amplitude
        
        # Apply filter
        filtered_fft = field_fft * spectrum_filter
        
        # Inverse transform
        filtered_field = np.real(np.fft.ifft2(filtered_fft))
        
        return filtered_field
    
    def compute_forcing_term(self, x: np.ndarray, t: float, 
                           field: np.ndarray) -> np.ndarray:
        """
        Compute the external forcing term f(x,t) for gas dynamics.
        
        Args:
            x: Spatial coordinates
            t: Time
            field: Current field values
            
        Returns:
            Forcing term values
        """
        # Pressure gradient forcing
        pressure_gradient = self._compute_pressure_gradient(field)
        
        # Body forces (e.g., gravity)
        body_force = self._compute_body_forces(x)
        
        # Artificial viscosity for numerical stability
        artificial_viscosity = self._compute_artificial_viscosity(field)
        
        # Total forcing
        forcing = pressure_gradient + body_force + artificial_viscosity
        
        return forcing
    
    def _compute_pressure_gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute pressure gradient forcing term."""
        if self.pressure_field is None:
            return np.zeros_like(field)
        
        # Compute pressure gradient using central differences
        grad_p = np.gradient(self.pressure_field)
        
        # Convert to forcing term (negative gradient)
        forcing = np.zeros_like(field)
        if len(field.shape) == 3:  # 2D vector field
            forcing[:, :, 0] = -grad_p[0] / self.params.density
            forcing[:, :, 1] = -grad_p[1] / self.params.density
        else:  # Scalar field
            forcing = -np.sqrt(grad_p[0]**2 + grad_p[1]**2) / self.params.density
        
        return forcing
    
    def _compute_body_forces(self, x: np.ndarray) -> np.ndarray:
        """Compute body forces (e.g., gravity)."""
        # For simplicity, assume no body forces
        return np.zeros_like(x)
    
    def _compute_artificial_viscosity(self, field: np.ndarray) -> np.ndarray:
        """Compute artificial viscosity for numerical stability."""
        if self.params.artificial_viscosity == 0:
            return np.zeros_like(field)
        
        # Compute Laplacian for artificial viscosity
        if len(field.shape) == 3:  # Vector field
            laplacian = np.zeros_like(field)
            for i in range(field.shape[2]):
                laplacian[:, :, i] = self._compute_laplacian(field[:, :, i])
        else:  # Scalar field
            laplacian = self._compute_laplacian(field)
        
        return self.params.artificial_viscosity * laplacian
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian of a field."""
        laplacian = np.zeros_like(field)
        
        # Central differences for second derivatives
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]
        )
        
        return laplacian
    
    def compute_diagnostics(self, field: np.ndarray, t: float) -> Dict[str, float]:
        """
        Compute diagnostic quantities for gas dynamics.
        
        Args:
            field: Current field values
            t: Current time
            
        Returns:
            Dictionary of diagnostic quantities
        """
        diagnostics = {}
        
        # Kinetic energy
        if len(field.shape) == 3:  # Vector field
            kinetic_energy = 0.5 * np.mean(np.sum(field**2, axis=2))
        else:  # Scalar field
            kinetic_energy = 0.5 * np.mean(field**2)
        
        diagnostics["kinetic_energy"] = kinetic_energy
        
        # Enstrophy (for 2D flows)
        if len(field.shape) == 3 and field.shape[2] == 2:
            vorticity = self._compute_vorticity(field)
            enstrophy = 0.5 * np.mean(vorticity**2)
            diagnostics["enstrophy"] = enstrophy
            diagnostics["max_vorticity"] = np.max(np.abs(vorticity))
        
        # Turbulent kinetic energy
        if self.velocity_field is not None:
            mean_velocity = np.mean(field, axis=(0, 1))
            if len(field.shape) == 3:
                fluctuations = field - mean_velocity
                tke = 0.5 * np.mean(np.sum(fluctuations**2, axis=2))
            else:
                fluctuations = field - mean_velocity
                tke = 0.5 * np.mean(fluctuations**2)
            
            diagnostics["turbulent_kinetic_energy"] = tke
        
        # Energy dissipation rate
        if self.params.reynolds_number > 0:
            epsilon = self._compute_energy_dissipation(field)
            diagnostics["energy_dissipation"] = epsilon
        
        # Mach number distribution
        if self.velocity_field is not None:
            sound_speed = np.sqrt(
                self.params.specific_heat_ratio * self.params.gas_constant * 
                self.params.temperature
            )
            if len(field.shape) == 3:
                velocity_magnitude = np.sqrt(np.sum(field**2, axis=2))
            else:
                velocity_magnitude = np.abs(field)
            
            mach_local = velocity_magnitude / sound_speed
            diagnostics["max_mach"] = np.max(mach_local)
            diagnostics["mean_mach"] = np.mean(mach_local)
        
        return diagnostics
    
    def _compute_vorticity(self, velocity_field: np.ndarray) -> np.ndarray:
        """Compute vorticity for 2D velocity field."""
        if len(velocity_field.shape) != 3 or velocity_field.shape[2] != 2:
            raise ValueError("Vorticity computation requires 2D velocity field")
        
        u = velocity_field[:, :, 0]
        v = velocity_field[:, :, 1]
        
        # Compute derivatives
        du_dy = np.gradient(u, axis=1)
        dv_dx = np.gradient(v, axis=0)
        
        # Vorticity = dv/dx - du/dy
        vorticity = dv_dx - du_dy
        
        return vorticity
    
    def _compute_energy_dissipation(self, field: np.ndarray) -> float:
        """Compute energy dissipation rate."""
        # Simplified calculation based on velocity gradients
        if len(field.shape) == 3:
            # Compute velocity gradients
            grad_u = np.gradient(field[:, :, 0])
            grad_v = np.gradient(field[:, :, 1])
            
            # Strain rate tensor components
            S11 = grad_u[0]  # du/dx
            S12 = 0.5 * (grad_u[1] + grad_v[0])  # 0.5*(du/dy + dv/dx)
            S22 = grad_v[1]  # dv/dy
            
            # Dissipation rate
            epsilon = 2 * self.params.viscosity * np.mean(
                S11**2 + 2*S12**2 + S22**2
            )
        else:
            # For scalar field, use simple gradient-based approximation
            grad_field = np.gradient(field)
            epsilon = self.params.viscosity * np.mean(
                grad_field[0]**2 + grad_field[1]**2
            )
        
        return epsilon
    
    def update_thermodynamic_properties(self, field: np.ndarray) -> None:
        """
        Update thermodynamic properties based on current flow field.
        
        Args:
            field: Current field values
        """
        # For compressible flow, update density, pressure, and temperature
        # This is a simplified model - in practice, you'd solve the full
        # compressible Navier-Stokes equations
        
        if len(field.shape) == 3:  # Vector field
            velocity_magnitude = np.sqrt(np.sum(field**2, axis=2))
        else:  # Scalar field
            velocity_magnitude = np.abs(field)
        
        # Update density using continuity equation (simplified)
        # In practice, this would require solving the full continuity equation
        
        # Update pressure using equation of state
        # P = ρRT for ideal gas
        self.pressure_field = (
            self.density_field * self.params.gas_constant * self.temperature_field
        )
        
        # Update temperature using energy equation (simplified)
        # This would require solving the full energy equation in practice
        
        logger.debug("Updated thermodynamic properties")
    
    def export_results(self, field: np.ndarray, t: float, 
                      filename: str) -> None:
        """
        Export simulation results to file.
        
        Args:
            field: Current field values
            t: Current time
            filename: Output filename
        """
        import h5py
        
        with h5py.File(filename, 'w') as f:
            # Save field data
            f.create_dataset('velocity_field', data=field)
            f.create_dataset('density_field', data=self.density_field)
            f.create_dataset('pressure_field', data=self.pressure_field)
            f.create_dataset('temperature_field', data=self.temperature_field)
            
            # Save parameters
            params_group = f.create_group('parameters')
            params_group.attrs['reynolds_number'] = self.params.reynolds_number
            params_group.attrs['mach_number'] = self.params.mach_number
            params_group.attrs['alpha'] = self.params.alpha
            params_group.attrs['beta'] = self.params.beta
            params_group.attrs['gamma'] = self.params.gamma
            params_group.attrs['hurst_parameter'] = self.params.hurst_parameter
            
            # Save simulation metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['time'] = t
            metadata_group.attrs['application'] = self.name
            
            # Save diagnostics
            diagnostics = self.compute_diagnostics(field, t)
            diagnostics_group = f.create_group('diagnostics')
            for key, value in diagnostics.items():
                diagnostics_group.attrs[key] = value
        
        logger.info(f"Results exported to {filename}")
    
    def get_default_parameters(self) -> Dict:
        """Get default parameters for gas dynamics application."""
        return {
            'reynolds_number': 1000.0,
            'mach_number': 0.3,
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst_parameter': 0.7,
            'turbulent_intensity': 0.05,
            'cfl_number': 0.5
        }
    
    def validate_parameters(self, parameters: Dict) -> bool:
        """
        Validate gas dynamics parameters.
        
        Args:
            parameters: Parameter dictionary to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        required_params = [
            'reynolds_number', 'mach_number', 'alpha', 'beta', 'gamma'
        ]
        
        # Check required parameters
        for param in required_params:
            if param not in parameters:
                logger.error(f"Missing required parameter: {param}")
                return False
        
        # Validate parameter ranges
        if parameters['reynolds_number'] <= 0:
            logger.error("Reynolds number must be positive")
            return False
        
        if not (0 < parameters['alpha'] <= 2):
            logger.error("Alpha must be in range (0, 2]")
            return False
        
        if parameters['beta'] < 0:
            logger.error("Beta must be non-negative")
            return False
        
        if parameters['gamma'] < 0:
            logger.error("Gamma must be non-negative")
            return False
        
        if parameters['mach_number'] < 0:
            logger.error("Mach number must be non-negative")
            return False
        
        return True


# Factory functions for common gas dynamics scenarios
def create_turbulent_channel_flow(reynolds_number: float = 1000.0,
                                 alpha: float = 0.5,
                                 beta: float = 0.1) -> GasDynamicsApplication:
    """Create a turbulent channel flow application."""
    params = GasDynamicsParameters(
        reynolds_number=reynolds_number,
        mach_number=0.1,  # Low Mach number for incompressible approximation
        alpha=alpha,
        beta=beta,
        gamma=0.01,
        hurst_parameter=0.7,
        turbulent_intensity=0.1
    )
    
    return GasDynamicsApplication(params)


def create_compressible_flow(mach_number: float = 0.8,
                           alpha: float = 0.8,
                           beta: float = 0.05) -> GasDynamicsApplication:
    """Create a compressible flow application."""
    params = GasDynamicsParameters(
        reynolds_number=1000.0,
        mach_number=mach_number,
        alpha=alpha,
        beta=beta,
        gamma=0.02,
        hurst_parameter=0.6,
        turbulent_intensity=0.05
    )
    
    return GasDynamicsApplication(params)


def create_shock_tube_simulation(alpha: float = 1.0,
                                beta: float = 0.01) -> GasDynamicsApplication:
    """Create a shock tube simulation application."""
    params = GasDynamicsParameters(
        reynolds_number=10000.0,
        mach_number=2.0,
        alpha=alpha,
        beta=beta,
        gamma=0.001,
        hurst_parameter=0.5,
        turbulent_intensity=0.01,
        artificial_viscosity=0.01
    )
    
    return GasDynamicsApplication(params)
