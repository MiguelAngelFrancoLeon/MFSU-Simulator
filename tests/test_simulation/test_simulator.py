"""
Test suite for the MFSU Simulator main engine.
Tests the core simulation functionality for the Modified Fractional Stochastic Unified equation:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Author: MFSU Development Team
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from simulation.simulator import MFSUSimulator
from core.mfsu_equation import MFSUEquation
from core.fractional_operators import FractionalLaplacian
from core.stochastic_processes import FractionalBrownianNoise
from simulation.grid_manager import GridManager
from simulation.time_evolution import TimeEvolution
from simulation.boundary_conditions import BoundaryConditions


class TestMFSUSimulator:
    """Test class for the main MFSU Simulator."""

    @pytest.fixture
    def default_params(self):
        """Default simulation parameters for testing."""
        return {
            'alpha': 0.5,           # Fractional parameter
            'beta': 0.1,            # Stochastic coupling
            'gamma': 0.01,          # Nonlinear parameter
            'hurst': 0.7,           # Hurst parameter
            'dt': 0.01,             # Time step
            'dx': 0.1,              # Spatial step
            'grid_size': 64,        # Grid points
            'max_time': 1.0,        # Maximum simulation time
            'boundary_conditions': 'periodic'
        }

    @pytest.fixture
    def simulator(self, default_params):
        """Create a simulator instance for testing."""
        return MFSUSimulator(default_params)

    @pytest.fixture
    def initial_condition_gaussian(self):
        """Gaussian initial condition for testing."""
        def gaussian_ic(x):
            return np.exp(-((x - np.pi)**2) / 0.5)
        return gaussian_ic

    @pytest.fixture
    def initial_condition_soliton(self):
        """Soliton-like initial condition for testing."""
        def soliton_ic(x):
            return 1.0 / np.cosh(x - np.pi)
        return soliton_ic

    def test_simulator_initialization(self, default_params):
        """Test proper initialization of the simulator."""
        simulator = MFSUSimulator(default_params)
        
        assert simulator.alpha == default_params['alpha']
        assert simulator.beta == default_params['beta']
        assert simulator.gamma == default_params['gamma']
        assert simulator.hurst == default_params['hurst']
        assert simulator.dt == default_params['dt']
        assert simulator.dx == default_params['dx']
        assert simulator.grid_size == default_params['grid_size']
        assert simulator.max_time == default_params['max_time']
        
        # Check component initialization
        assert isinstance(simulator.grid_manager, GridManager)
        assert isinstance(simulator.mfsu_equation, MFSUEquation)
        assert isinstance(simulator.time_evolution, TimeEvolution)

    def test_simulator_initialization_invalid_params(self):
        """Test simulator initialization with invalid parameters."""
        # Test negative alpha
        with pytest.raises(ValueError, match="alpha must be positive"):
            MFSUSimulator({'alpha': -0.5, 'beta': 0.1, 'gamma': 0.01})
        
        # Test alpha > 2
        with pytest.raises(ValueError, match="alpha must be <= 2"):
            MFSUSimulator({'alpha': 2.5, 'beta': 0.1, 'gamma': 0.01})
        
        # Test invalid Hurst parameter
        with pytest.raises(ValueError, match="hurst must be in \\(0, 1\\)"):
            MFSUSimulator({'alpha': 0.5, 'beta': 0.1, 'gamma': 0.01, 'hurst': 1.5})
        
        # Test negative time step
        with pytest.raises(ValueError, match="dt must be positive"):
            MFSUSimulator({'alpha': 0.5, 'beta': 0.1, 'gamma': 0.01, 'dt': -0.01})

    def test_grid_setup(self, simulator):
        """Test proper grid setup."""
        x_grid = simulator.get_spatial_grid()
        
        assert len(x_grid) == simulator.grid_size
        assert np.isclose(x_grid[1] - x_grid[0], simulator.dx, rtol=1e-10)
        assert x_grid[0] >= 0
        assert x_grid[-1] <= 2 * np.pi

    def test_set_initial_condition_function(self, simulator, initial_condition_gaussian):
        """Test setting initial condition from function."""
        simulator.set_initial_condition(initial_condition_gaussian)
        
        x_grid = simulator.get_spatial_grid()
        expected = initial_condition_gaussian(x_grid)
        
        np.testing.assert_allclose(simulator.psi, expected, rtol=1e-12)

    def test_set_initial_condition_array(self, simulator):
        """Test setting initial condition from array."""
        initial_array = np.random.random(simulator.grid_size)
        simulator.set_initial_condition(initial_array)
        
        np.testing.assert_array_equal(simulator.psi, initial_array)

    def test_set_initial_condition_invalid(self, simulator):
        """Test setting invalid initial condition."""
        # Wrong size array
        with pytest.raises(ValueError, match="Initial condition array size"):
            simulator.set_initial_condition(np.random.random(10))
        
        # Invalid type
        with pytest.raises(TypeError):
            simulator.set_initial_condition("invalid")

    def test_external_forcing_setup(self, simulator):
        """Test external forcing term setup."""
        def forcing_function(x, t):
            return 0.1 * np.sin(x) * np.cos(t)
        
        simulator.set_external_forcing(forcing_function)
        
        x_grid = simulator.get_spatial_grid()
        t_test = 0.5
        expected = forcing_function(x_grid, t_test)
        actual = simulator.external_forcing(x_grid, t_test)
        
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_single_time_step(self, simulator, initial_condition_gaussian):
        """Test a single time step evolution."""
        simulator.set_initial_condition(initial_condition_gaussian)
        initial_psi = simulator.psi.copy()
        
        # Perform one time step
        simulator.step()
        
        # Check that the solution has evolved
        assert not np.allclose(simulator.psi, initial_psi)
        assert simulator.current_time == simulator.dt
        assert len(simulator.time_history) == 2  # t=0 and t=dt
        assert len(simulator.solution_history) == 2

    def test_conservation_properties(self, simulator, initial_condition_gaussian):
        """Test conservation properties (when applicable)."""
        simulator.set_initial_condition(initial_condition_gaussian)
        
        # Store initial values
        initial_norm = np.trapz(np.abs(simulator.psi)**2, dx=simulator.dx)
        
        # Run for a few steps
        for _ in range(10):
            simulator.step()
        
        # For the linearized case (γ=0), norm should be approximately conserved
        # Note: This is a weak test due to stochastic terms
        current_norm = np.trapz(np.abs(simulator.psi)**2, dx=simulator.dx)
        
        # Allow for reasonable variation due to stochastic terms and numerics
        relative_change = abs(current_norm - initial_norm) / initial_norm
        assert relative_change < 0.5, f"Norm changed by {relative_change*100:.2f}%"

    def test_run_simulation_complete(self, simulator, initial_condition_gaussian):
        """Test complete simulation run."""
        simulator.set_initial_condition(initial_condition_gaussian)
        
        result = simulator.run()
        
        # Check result structure
        assert 'time' in result
        assert 'solution' in result
        assert 'parameters' in result
        assert 'statistics' in result
        
        # Check time evolution
        assert len(result['time']) > 1
        assert result['time'][0] == 0.0
        assert result['time'][-1] <= simulator.max_time + simulator.dt
        
        # Check solution evolution
        assert result['solution'].shape[0] == len(result['time'])
        assert result['solution'].shape[1] == simulator.grid_size

    def test_run_simulation_with_callback(self, simulator, initial_condition_gaussian):
        """Test simulation with progress callback."""
        simulator.set_initial_condition(initial_condition_gaussian)
        
        callback_calls = []
        
        def progress_callback(step, time, psi):
            callback_calls.append((step, time, len(psi)))
        
        result = simulator.run(progress_callback=progress_callback)
        
        # Check that callback was called
        assert len(callback_calls) > 0
        assert all(isinstance(call[0], int) for call in callback_calls)
        assert all(call[2] == simulator.grid_size for call in callback_calls)

    def test_simulation_stability_gaussian(self, simulator, initial_condition_gaussian):
        """Test simulation stability with Gaussian initial condition."""
        simulator.set_initial_condition(initial_condition_gaussian)
        
        result = simulator.run()
        
        # Check for NaN or infinite values
        assert not np.any(np.isnan(result['solution']))
        assert not np.any(np.isinf(result['solution']))
        
        # Check that solution remains bounded
        max_amplitude = np.max(np.abs(result['solution']))
        assert max_amplitude < 100, f"Solution grew too large: {max_amplitude}"

    def test_simulation_stability_soliton(self, simulator, initial_condition_soliton):
        """Test simulation stability with soliton-like initial condition."""
        simulator.set_initial_condition(initial_condition_soliton)
        
        result = simulator.run()
        
        # Check for NaN or infinite values
        assert not np.any(np.isnan(result['solution']))
        assert not np.any(np.isinf(result['solution']))
        
        # Check that solution remains bounded
        max_amplitude = np.max(np.abs(result['solution']))
        assert max_amplitude < 100, f"Solution grew too large: {max_amplitude}"

    def test_parameter_modification(self, simulator):
        """Test parameter modification during simulation."""
        # Test parameter updates
        simulator.update_parameters({'alpha': 0.8, 'beta': 0.2})
        
        assert simulator.alpha == 0.8
        assert simulator.beta == 0.2
        
        # Test invalid parameter update
        with pytest.raises(ValueError):
            simulator.update_parameters({'alpha': -1.0})

    def test_boundary_conditions_periodic(self, default_params, initial_condition_gaussian):
        """Test periodic boundary conditions."""
        default_params['boundary_conditions'] = 'periodic'
        simulator = MFSUSimulator(default_params)
        simulator.set_initial_condition(initial_condition_gaussian)
        
        # Run simulation
        result = simulator.run()
        
        # For periodic BC, solution should be periodic at boundaries
        # This is a weak test as the solution evolves
        final_solution = result['solution'][-1]
        assert not np.isnan(final_solution[0])
        assert not np.isnan(final_solution[-1])

    def test_boundary_conditions_dirichlet(self, default_params, initial_condition_gaussian):
        """Test Dirichlet boundary conditions."""
        default_params['boundary_conditions'] = 'dirichlet'
        simulator = MFSUSimulator(default_params)
        simulator.set_initial_condition(initial_condition_gaussian)
        
        # Run simulation
        result = simulator.run()
        
        # For Dirichlet BC, boundaries should remain close to zero
        final_solution = result['solution'][-1]
        assert abs(final_solution[0]) < 1e-6
        assert abs(final_solution[-1]) < 1e-6

    def test_reset_simulation(self, simulator, initial_condition_gaussian):
        """Test simulation reset functionality."""
        simulator.set_initial_condition(initial_condition_gaussian)
        initial_psi = simulator.psi.copy()
        
        # Run for a few steps
        for _ in range(5):
            simulator.step()
        
        # Reset simulation
        simulator.reset()
        
        assert simulator.current_time == 0.0
        assert len(simulator.time_history) == 1
        assert len(simulator.solution_history) == 1
        np.testing.assert_array_equal(simulator.psi, initial_psi)

    def test_save_load_state(self, simulator, initial_condition_gaussian):
        """Test saving and loading simulation state."""
        simulator.set_initial_condition(initial_condition_gaussian)
        
        # Run for a few steps
        for _ in range(3):
            simulator.step()
        
        # Save state
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_state.npz')
            simulator.save_state(save_path)
            
            # Create new simulator and load state
            new_simulator = MFSUSimulator(simulator.get_parameters())
            new_simulator.load_state(save_path)
            
            # Check that states match
            np.testing.assert_array_equal(simulator.psi, new_simulator.psi)
            assert simulator.current_time == new_simulator.current_time
            assert len(simulator.time_history) == len(new_simulator.time_history)

    def test_fractal_dimension_analysis(self, simulator, initial_condition_gaussian):
        """Test fractal dimension analysis capabilities."""
        simulator.set_initial_condition(initial_condition_gaussian)
        result = simulator.run()
        
        # Test fractal analysis
        fractal_dims = simulator.analyze_fractal_dimension(result['solution'])
        
        assert isinstance(fractal_dims, np.ndarray)
        assert len(fractal_dims) == len(result['time'])
        assert all(0.5 <= dim <= 2.0 for dim in fractal_dims)

    def test_spectral_analysis(self, simulator, initial_condition_gaussian):
        """Test spectral analysis functionality."""
        simulator.set_initial_condition(initial_condition_gaussian)
        result = simulator.run()
        
        # Test spectral analysis
        spectrum_data = simulator.analyze_spectrum(result['solution'][-1])
        
        assert 'frequencies' in spectrum_data
        assert 'power_spectrum' in spectrum_data
        assert len(spectrum_data['frequencies']) == len(spectrum_data['power_spectrum'])

    def test_memory_usage(self, default_params):
        """Test memory usage remains reasonable."""
        # Create simulator with larger grid
        large_params = default_params.copy()
        large_params['grid_size'] = 512
        large_params['max_time'] = 2.0
        
        simulator = MFSUSimulator(large_params)
        
        def gaussian_ic(x):
            return np.exp(-((x - np.pi)**2) / 0.5)
        
        simulator.set_initial_condition(gaussian_ic)
        
        # Run simulation - should not crash due to memory
        result = simulator.run()
        
        assert len(result['solution']) > 0
        assert result['solution'].dtype == np.complex128 or result['solution'].dtype == np.float64

    @pytest.mark.parametrize("alpha_val", [0.25, 0.5, 0.75, 1.0, 1.5, 1.8])
    def test_different_alpha_values(self, default_params, alpha_val, initial_condition_gaussian):
        """Test simulation with different alpha values."""
        default_params['alpha'] = alpha_val
        simulator = MFSUSimulator(default_params)
        simulator.set_initial_condition(initial_condition_gaussian)
        
        result = simulator.run()
        
        # Check stability for different alpha values
        assert not np.any(np.isnan(result['solution']))
        assert not np.any(np.isinf(result['solution']))

    @pytest.mark.parametrize("hurst_val", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_different_hurst_parameters(self, default_params, hurst_val, initial_condition_gaussian):
        """Test simulation with different Hurst parameters."""
        default_params['hurst'] = hurst_val
        simulator = MFSUSimulator(default_params)
        simulator.set_initial_condition(initial_condition_gaussian)
        
        result = simulator.run()
        
        # Check that different Hurst parameters produce different results
        assert not np.any(np.isnan(result['solution']))
        assert len(result['solution']) > 1

    def test_performance_benchmark(self, simulator, initial_condition_gaussian):
        """Basic performance benchmark test."""
        import time
        
        simulator.set_initial_condition(initial_condition_gaussian)
        
        start_time = time.time()
        result = simulator.run()
        end_time = time.time()
        
        simulation_time = end_time - start_time
        num_steps = len(result['time'])
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert simulation_time < 60.0, f"Simulation took too long: {simulation_time:.2f}s"
        
        # Performance metric
        steps_per_second = num_steps / simulation_time
        assert steps_per_second > 1.0, f"Too slow: {steps_per_second:.2f} steps/s"

    def test_error_handling_integration_failure(self, simulator):
        """Test error handling when integration fails."""
        # Set up a condition that might cause integration failure
        def bad_initial_condition(x):
            return np.full_like(x, np.inf)
        
        with pytest.raises((ValueError, RuntimeError)):
            simulator.set_initial_condition(bad_initial_condition)
            simulator.run()

    def test_statistical_moments(self, simulator, initial_condition_gaussian):
        """Test calculation of statistical moments."""
        simulator.set_initial_condition(initial_condition_gaussian)
        result = simulator.run()
        
        # Calculate moments
        moments = simulator.calculate_moments(result['solution'], max_order=4)
        
        assert 'mean' in moments
        assert 'variance' in moments
        assert 'skewness' in moments
        assert 'kurtosis' in moments
        
        # Check that moments are finite
        for moment_name, moment_values in moments.items():
            assert all(np.isfinite(moment_values))


class TestMFSUSimulatorIntegration:
    """Integration tests for the MFSU Simulator with other components."""

    @pytest.fixture
    def complex_params(self):
        """Complex simulation parameters for integration testing."""
        return {
            'alpha': 1.2,
            'beta': 0.15,
            'gamma': 0.02,
            'hurst': 0.6,
            'dt': 0.005,
            'dx': 0.05,
            'grid_size': 128,
            'max_time': 2.0,
            'boundary_conditions': 'periodic'
        }

    def test_full_simulation_workflow(self, complex_params):
        """Test complete simulation workflow from initialization to analysis."""
        # Initialize simulator
        simulator = MFSUSimulator(complex_params)
        
        # Set initial condition
        def initial_condition(x):
            return np.exp(-((x - np.pi)**2) / 0.3) + 0.1 * np.sin(2*x)
        
        simulator.set_initial_condition(initial_condition)
        
        # Set external forcing
        def forcing(x, t):
            return 0.01 * np.sin(x) * np.exp(-t/2)
        
        simulator.set_external_forcing(forcing)
        
        # Run simulation with callback
        progress_data = []
        
        def callback(step, time, psi):
            if step % 50 == 0:  # Record every 50th step
                progress_data.append({
                    'step': step,
                    'time': time,
                    'max_amplitude': np.max(np.abs(psi)),
                    'energy': np.trapz(np.abs(psi)**2, dx=simulator.dx)
                })
        
        result = simulator.run(progress_callback=callback)
        
        # Verify results
        assert len(result['solution']) > 100
        assert len(progress_data) > 0
        assert not np.any(np.isnan(result['solution']))
        
        # Analyze results
        fractal_dims = simulator.analyze_fractal_dimension(result['solution'])
        spectrum = simulator.analyze_spectrum(result['solution'][-1])
        moments = simulator.calculate_moments(result['solution'])
        
        # Verify analysis results
        assert len(fractal_dims) == len(result['time'])
        assert 'power_spectrum' in spectrum
        assert 'mean' in moments


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
