"""
Parameter Validation Module for MFSU Simulator
=============================================

Validates parameters for the MFSU equation:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

This module ensures all simulation parameters are within valid ranges
and physically meaningful for different applications.
"""

import numpy as np
import yaml
from typing import Dict, Any, List, Union, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApplicationType(Enum):
    """Enumeration of supported MFSU applications"""
    SUPERCONDUCTIVITY = "superconductivity"
    GAS_DYNAMICS = "gas_dynamics"
    COSMOLOGY = "cosmology"
    GENERAL = "general"


@dataclass
class ValidationResult:
    """Result of parameter validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_params: Optional[Dict[str, Any]] = None


@dataclass
class ParameterBounds:
    """Parameter bounds for validation"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    excluded_values: Optional[List[float]] = None
    must_be_positive: bool = False
    must_be_integer: bool = False
    
    def validate(self, value: Union[float, int], param_name: str) -> Tuple[bool, List[str]]:
        """Validate a single parameter value"""
        errors = []
        
        # Type validation
        if self.must_be_integer and not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                errors.append(f"{param_name} must be an integer")
                return False, errors
        
        # Positivity check
        if self.must_be_positive and value <= 0:
            errors.append(f"{param_name} must be positive (got {value})")
        
        # Range checks
        if self.min_value is not None and value < self.min_value:
            errors.append(f"{param_name} must be >= {self.min_value} (got {value})")
        
        if self.max_value is not None and value > self.max_value:
            errors.append(f"{param_name} must be <= {self.max_value} (got {value})")
        
        # Excluded values
        if self.excluded_values and value in self.excluded_values:
            errors.append(f"{param_name} cannot be {value}")
        
        return len(errors) == 0, errors


class MFSUParameterValidator:
    """
    Comprehensive parameter validator for MFSU simulations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize validator with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.parameter_bounds = self._setup_parameter_bounds()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for parameter validation"""
        return {
            'simulation': {
                'default_parameters': {
                    'alpha': 0.5,
                    'beta': 0.1,
                    'gamma': 0.01,
                    'hurst': 0.7
                },
                'numerical': {
                    'dt': 0.01,
                    'dx': 0.1,
                    'grid_size': 100,
                    'max_time': 10.0
                }
            },
            'validation': {
                'strict_mode': False,
                'auto_correct': True
            }
        }
    
    def _setup_parameter_bounds(self) -> Dict[str, ParameterBounds]:
        """Setup parameter bounds for validation"""
        return {
            # MFSU equation parameters
            'alpha': ParameterBounds(min_value=0.1, max_value=2.0, must_be_positive=True),
            'beta': ParameterBounds(min_value=0.0, max_value=10.0),
            'gamma': ParameterBounds(min_value=0.0, max_value=1.0),
            'hurst': ParameterBounds(min_value=0.0, max_value=1.0, excluded_values=[0.0, 1.0]),
            
            # Numerical parameters
            'dt': ParameterBounds(min_value=1e-6, max_value=1.0, must_be_positive=True),
            'dx': ParameterBounds(min_value=1e-6, max_value=10.0, must_be_positive=True),
            'grid_size': ParameterBounds(min_value=10, max_value=10000, must_be_positive=True, must_be_integer=True),
            'max_time': ParameterBounds(min_value=0.1, max_value=1000.0, must_be_positive=True),
            
            # Application-specific parameters
            'temperature': ParameterBounds(min_value=0.1, max_value=1000.0, must_be_positive=True),
            'reynolds_number': ParameterBounds(min_value=1.0, max_value=1e6, must_be_positive=True),
            'mach_number': ParameterBounds(min_value=0.0, max_value=10.0),
            'hubble_constant': ParameterBounds(min_value=50.0, max_value=100.0, must_be_positive=True),
            'omega_matter': ParameterBounds(min_value=0.0, max_value=1.0),
            
            # Grid and boundary conditions
            'boundary_type': ParameterBounds(),  # Handled separately
            'initial_condition_type': ParameterBounds(),  # Handled separately
        }
    
    def validate_mfsu_parameters(self, params: Dict[str, Any], 
                                application: ApplicationType = ApplicationType.GENERAL) -> ValidationResult:
        """
        Validate MFSU equation parameters
        
        Args:
            params: Dictionary of parameters to validate
            application: Type of application (affects validation rules)
            
        Returns:
            ValidationResult object with validation status and details
        """
        errors = []
        warnings = []
        corrected_params = params.copy()
        
        # Validate core MFSU parameters
        core_params = ['alpha', 'beta', 'gamma', 'hurst']
        for param in core_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
                continue
                
            bounds = self.parameter_bounds[param]
            is_valid, param_errors = bounds.validate(params[param], param)
            if not is_valid:
                errors.extend(param_errors)
        
        # Application-specific validation
        app_errors, app_warnings, app_corrections = self._validate_application_specific(
            params, application
        )
        errors.extend(app_errors)
        warnings.extend(app_warnings)
        corrected_params.update(app_corrections)
        
        # Numerical stability checks
        stability_errors, stability_warnings = self._check_numerical_stability(corrected_params)
        errors.extend(stability_errors)
        warnings.extend(stability_warnings)
        
        # Physical consistency checks
        physics_errors, physics_warnings = self._check_physical_consistency(
            corrected_params, application
        )
        errors.extend(physics_errors)
        warnings.extend(physics_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_params=corrected_params if corrected_params != params else None
        )
    
    def _validate_application_specific(self, params: Dict[str, Any], 
                                     application: ApplicationType) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate application-specific parameters"""
        errors = []
        warnings = []
        corrections = {}
        
        if application == ApplicationType.SUPERCONDUCTIVITY:
            # Superconductivity-specific validation
            if 'temperature' in params:
                if params['temperature'] > 300:
                    warnings.append("High temperature for superconductivity simulation")
            
            # Typical ranges for superconductivity
            if 'alpha' in params and params['alpha'] > 1.5:
                warnings.append("Alpha parameter unusually high for superconductivity")
                
        elif application == ApplicationType.GAS_DYNAMICS:
            # Gas dynamics validation
            if 'reynolds_number' in params:
                Re = params['reynolds_number']
                if Re < 2300:
                    warnings.append("Reynolds number suggests laminar flow")
                elif Re > 4000:
                    warnings.append("Reynolds number suggests turbulent flow")
            
            if 'mach_number' in params and params['mach_number'] > 0.3:
                warnings.append("Mach number > 0.3: compressibility effects important")
                
        elif application == ApplicationType.COSMOLOGY:
            # Cosmological parameter validation
            if 'hubble_constant' in params:
                H0 = params['hubble_constant']
                if not (65 <= H0 <= 75):
                    warnings.append(f"Hubble constant {H0} outside typical range 65-75 km/s/Mpc")
            
            if 'omega_matter' in params and 'omega_lambda' in params:
                total = params['omega_matter'] + params['omega_lambda']
                if abs(total - 1.0) > 0.1:
                    warnings.append("Ω_matter + Ω_Λ significantly differs from 1.0")
        
        return errors, warnings, corrections
    
    def _check_numerical_stability(self, params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Check numerical stability conditions"""
        errors = []
        warnings = []
        
        if 'dt' in params and 'dx' in params and 'alpha' in params:
            dt, dx, alpha = params['dt'], params['dx'], params['alpha']
            
            # CFL-like condition for fractional diffusion
            stability_factor = dt / (dx ** alpha)
            
            if stability_factor > 0.5:
                errors.append(f"Stability condition violated: dt/dx^α = {stability_factor:.3f} > 0.5")
            elif stability_factor > 0.1:
                warnings.append(f"Stability factor {stability_factor:.3f} may cause numerical issues")
        
        # Check grid resolution
        if 'grid_size' in params and 'dx' in params:
            domain_size = params['grid_size'] * params['dx']
            if domain_size < 1.0:
                warnings.append(f"Small domain size {domain_size:.2f} may not capture physics")
            elif domain_size > 100.0:
                warnings.append(f"Large domain size {domain_size:.1f} may be computationally expensive")
        
        return errors, warnings
    
    def _check_physical_consistency(self, params: Dict[str, Any], 
                                  application: ApplicationType) -> Tuple[List[str], List[str]]:
        """Check physical consistency of parameters"""
        errors = []
        warnings = []
        
        # Check parameter ratios
        if 'alpha' in params and 'beta' in params and 'gamma' in params:
            alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
            
            # Check if nonlinear term dominates
            if gamma > 10 * alpha:
                warnings.append("Nonlinear term γ may dominate over fractional diffusion")
            
            # Check stochastic vs deterministic balance
            if beta > alpha:
                warnings.append("Stochastic term β larger than diffusion term α")
        
        # Hurst parameter consistency
        if 'hurst' in params:
            H = params['hurst']
            if H < 0.3:
                warnings.append(f"Low Hurst parameter {H:.2f} indicates anti-persistent process")
            elif H > 0.8:
                warnings.append(f"High Hurst parameter {H:.2f} indicates strong persistence")
        
        return errors, warnings
    
    def validate_initial_conditions(self, initial_data: Dict[str, Any]) -> ValidationResult:
        """Validate initial conditions"""
        errors = []
        warnings = []
        
        required_fields = ['type', 'parameters']
        for field in required_fields:
            if field not in initial_data:
                errors.append(f"Missing required field in initial conditions: {field}")
        
        if 'type' in initial_data:
            ic_type = initial_data['type']
            valid_types = ['gaussian_packet', 'soliton_profile', 'random_field', 'custom']
            if ic_type not in valid_types:
                errors.append(f"Invalid initial condition type: {ic_type}")
        
        # Validate specific initial condition parameters
        if 'parameters' in initial_data:
            params = initial_data['parameters']
            
            if 'amplitude' in params and params['amplitude'] <= 0:
                errors.append("Initial condition amplitude must be positive")
            
            if 'width' in params and params['width'] <= 0:
                errors.append("Initial condition width must be positive")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_boundary_conditions(self, boundary_data: Dict[str, Any]) -> ValidationResult:
        """Validate boundary conditions"""
        errors = []
        warnings = []
        
        valid_types = ['periodic', 'dirichlet', 'neumann', 'absorbing']
        
        for boundary in ['left', 'right']:
            if boundary not in boundary_data:
                errors.append(f"Missing boundary condition for {boundary} boundary")
                continue
            
            bc = boundary_data[boundary]
            if 'type' not in bc:
                errors.append(f"Missing type for {boundary} boundary condition")
                continue
            
            if bc['type'] not in valid_types:
                errors.append(f"Invalid boundary type for {boundary}: {bc['type']}")
            
            # Check if values are provided for Dirichlet/Neumann
            if bc['type'] in ['dirichlet', 'neumann'] and 'value' not in bc:
                errors.append(f"Missing value for {bc['type']} boundary condition")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def suggest_parameters(self, application: ApplicationType, 
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suggest reasonable parameters for given application"""
        base_params = self.config['simulation']['default_parameters'].copy()
        
        # Application-specific suggestions
        if application == ApplicationType.SUPERCONDUCTIVITY:
            base_params.update({
                'alpha': 1.2,
                'beta': 0.05,
                'gamma': 0.1,
                'hurst': 0.6,
                'temperature': 77.0  # Liquid nitrogen temperature
            })
            
        elif application == ApplicationType.GAS_DYNAMICS:
            base_params.update({
                'alpha': 0.8,
                'beta': 0.2,
                'gamma': 0.02,
                'hurst': 0.7,
                'reynolds_number': 5000,
                'mach_number': 0.1
            })
            
        elif application == ApplicationType.COSMOLOGY:
            base_params.update({
                'alpha': 0.6,
                'beta': 0.1,
                'gamma': 0.001,
                'hurst': 0.8,
                'hubble_constant': 70.0,
                'omega_matter': 0.3
            })
        
        # Apply constraints if provided
        if constraints:
            base_params.update(constraints)
        
        return base_params
    
    def validate_full_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate complete simulation configuration"""
        errors = []
        warnings = []
        
        # Validate main parameters
        if 'parameters' in config:
            param_result = self.validate_mfsu_parameters(
                config['parameters'], 
                ApplicationType(config.get('application', 'general'))
            )
            errors.extend(param_result.errors)
            warnings.extend(param_result.warnings)
        
        # Validate initial conditions
        if 'initial_conditions' in config:
            ic_result = self.validate_initial_conditions(config['initial_conditions'])
            errors.extend(ic_result.errors)
            warnings.extend(ic_result.warnings)
        
        # Validate boundary conditions
        if 'boundary_conditions' in config:
            bc_result = self.validate_boundary_conditions(config['boundary_conditions'])
            errors.extend(bc_result.errors)
            warnings.extend(bc_result.warnings)
        
        # Validate numerical settings
        if 'numerical' in config:
            for param in ['dt', 'dx', 'grid_size', 'max_time']:
                if param in config['numerical']:
                    bounds = self.parameter_bounds[param]
                    is_valid, param_errors = bounds.validate(
                        config['numerical'][param], param
                    )
                    if not is_valid:
                        errors.extend(param_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


def validate_parameters(params: Dict[str, Any], 
                       application: str = "general",
                       config_path: Optional[str] = None) -> ValidationResult:
    """
    Convenience function for parameter validation
    
    Args:
        params: Parameters to validate
        application: Application type
        config_path: Path to configuration file
        
    Returns:
        ValidationResult object
    """
    validator = MFSUParameterValidator(config_path)
    app_type = ApplicationType(application.lower())
    return validator.validate_mfsu_parameters(params, app_type)


if __name__ == "__main__":
    # Example usage
    test_params = {
        'alpha': 0.5,
        'beta': 0.1,
        'gamma': 0.01,
        'hurst': 0.7,
        'dt': 0.01,
        'dx': 0.1,
        'grid_size': 100
    }
    
    result = validate_parameters(test_params, "superconductivity")
    
    print(f"Validation successful: {result.is_valid}")
    if result.errors:
        print("Errors:", result.errors)
    if result.warnings:
        print("Warnings:", result.warnings)
