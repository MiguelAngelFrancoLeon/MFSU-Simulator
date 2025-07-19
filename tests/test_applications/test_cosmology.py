"""
Test suite for cosmological applications of the MFSU simulator.

This module contains comprehensive tests for the cosmological implementation
of the Unified Stochastic Fractal Model (MFSU) equation:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Testing cosmological phenomena including:
- Dark matter structure formation
- Cosmic microwave background fluctuations
- Large-scale structure evolution
- Inflationary dynamics
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Import modules from the MFSU simulator
try:
    from src.applications.cosmology import (
        CosmologicalMFSU,
        DarkMatterSimulation,
        CMBFluctuations,
        LargeScaleStructure,
        InflationaryModel
    )
    from src.core.mfsu_equation import MFSUEquation
    from src.core.fractional_operators import FractionalLaplacian
    from src.core.stochastic_processes import FractionalNoise
    from src.simulation.simulator import MFSUSimulator
    from src.utils.constants import COSMOLOGICAL_CONSTANTS
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestCosmologicalMFSU:
    """Test suite for the main cosmological MFSU implementation."""
    
    @pytest.fixture
    def cosmological_params(self):
        """Standard cosmological parameters for testing."""
        return {
            'hubble_constant': 70.0,  # km/s/Mpc
            'omega_matter': 0.3,
            'omega_lambda': 0.7,
            'omega_baryon': 0.05,
            'sigma_8': 0.8,
            'n_s': 0.96,  # Spectral index
            'alpha': 1.8,  # Fractional derivative order
            'beta': 0.15,  # Stochastic coupling
            'gamma': 0.005,  # Nonlinear term
            'hurst': 0.8,  # Hurst parameter
            'redshift_initial': 1000.0,
            'redshift_final': 0.0
        }
    
    @pytest.fixture
    def cosmological_mfsu(self, cosmological_params):
        """Create a CosmologicalMFSU instance for testing."""
        return CosmologicalMFSU(cosmological_params)
    
    def test_cosmological_mfsu_initialization(self, cosmological_mfsu, cosmological_params):
        """Test proper initialization of cosmological MFSU model."""
        assert cosmological_mfsu.hubble_constant == cosmological_params['hubble_constant']
        assert cosmological_mfsu.omega_matter == cosmological_params['omega_matter']
        assert cosmological_mfsu.alpha == cosmological_params['alpha']
        assert hasattr(cosmological_mfsu, 'mfsu_equation')
        assert hasattr(cosmological_mfsu, 'grid_manager')
    
    def test_scale_factor_evolution(self, cosmological_mfsu):
        """Test scale factor evolution from redshift."""
        z_values = np.array([1000, 100, 10, 1, 0])
        a_values = cosmological_mfsu.scale_factor_from_redshift(z_values)
        
        expected_a = 1.0 / (1.0 + z_values)
        np.testing.assert_allclose(a_values, expected_a, rtol=1e-10)
        
        # Test monotonic increase
        assert np.all(np.diff(a_values) > 0)
    
    def test_hubble_parameter_evolution(self, cosmological_mfsu):
        """Test Hubble parameter as function of redshift."""
        z_values = np.linspace(0, 10, 100)
        H_z = cosmological_mfsu.hubble_parameter(z_values)
        
        # Test present-day value
        H_0 = cosmological_mfsu.hubble_parameter(0.0)
        assert abs(H_0 - cosmological_mfsu.hubble_constant) < 1e-10
        
        # Test monotonic increase with redshift
        assert np.all(np.diff(H_z) > 0)
    
    def test_comoving_distance_calculation(self, cosmological_mfsu):
        """Test comoving distance calculations."""
        z_test = np.array([0.1, 0.5, 1.0, 2.0])
        distances = cosmological_mfsu.comoving_distance(z_test)
        
        # Test positive distances
        assert np.all(distances > 0)
        
        # Test monotonic increase with redshift
        assert np.all(np.diff(distances) > 0)
        
        # Test zero distance at z=0
        d_zero = cosmological_mfsu.comoving_distance(0.0)
        assert abs(d_zero) < 1e-10
    
    def test_growth_factor_calculation(self, cosmological_mfsu):
        """Test linear growth factor calculation."""
        z_values = np.linspace(0, 5, 50)
        D_z = cosmological_mfsu.linear_growth_factor(z_values)
        
        # Test normalization at z=0
        D_0 = cosmological_mfsu.linear_growth_factor(0.0)
        assert abs(D_0 - 1.0) < 1e-10
        
        # Test monotonic decrease with redshift
        assert np.all(np.diff(D_z) < 0)


class TestDarkMatterSimulation:
    """Test suite for dark matter structure formation simulations."""
    
    @pytest.fixture
    def dm_params(self):
        """Parameters for dark matter simulation."""
        return {
            'box_size': 100.0,  # Mpc/h
            'n_particles': 64**3,
            'mass_resolution': 1e8,  # M_sun/h
            'alpha': 1.8,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.75,
            'initial_redshift': 100.0,
            'final_redshift': 0.0
        }
    
    @pytest.fixture
    def dm_simulation(self, dm_params):
        """Create dark matter simulation instance."""
        return DarkMatterSimulation(dm_params)
    
    def test_dm_simulation_initialization(self, dm_simulation, dm_params):
        """Test dark matter simulation initialization."""
        assert dm_simulation.box_size == dm_params['box_size']
        assert dm_simulation.n_particles == dm_params['n_particles']
        assert hasattr(dm_simulation, 'particle_positions')
        assert hasattr(dm_simulation, 'particle_velocities')
    
    def test_initial_conditions_generation(self, dm_simulation):
        """Test generation of initial conditions for dark matter."""
        dm_simulation.generate_initial_conditions()
        
        # Check particle arrays exist and have correct shape
        assert dm_simulation.particle_positions.shape == (dm_simulation.n_particles, 3)
        assert dm_simulation.particle_velocities.shape == (dm_simulation.n_particles, 3)
        
        # Check particles are within box
        positions = dm_simulation.particle_positions
        assert np.all(positions >= 0) and np.all(positions <= dm_simulation.box_size)
    
    def test_power_spectrum_calculation(self, dm_simulation):
        """Test power spectrum calculation from density field."""
        # Create mock density field
        grid_size = 64
        density_field = np.random.normal(0, 1, (grid_size, grid_size, grid_size))
        
        k_values, P_k = dm_simulation.compute_power_spectrum(density_field)
        
        # Test output shapes and values
        assert len(k_values) == len(P_k)
        assert np.all(k_values > 0)
        assert np.all(P_k >= 0)
    
    def test_halo_finding(self, dm_simulation):
        """Test halo identification in dark matter field."""
        # Create mock density field with overdense regions
        grid_size = 32
        density_field = np.random.normal(0, 0.5, (grid_size, grid_size, grid_size))
        
        # Add some high-density peaks
        density_field[15:17, 15:17, 15:17] = 5.0
        density_field[8:10, 8:10, 8:10] = 3.0
        
        halos = dm_simulation.find_halos(density_field, threshold=2.0)
        
        # Test halo detection
        assert len(halos) >= 1  # Should find at least one halo
        for halo in halos:
            assert 'position' in halo
            assert 'mass' in halo
            assert 'radius' in halo


class TestCMBFluctuations:
    """Test suite for Cosmic Microwave Background fluctuations."""
    
    @pytest.fixture
    def cmb_params(self):
        """Parameters for CMB fluctuation analysis."""
        return {
            'l_max': 2000,
            'temperature_rms': 2.725,  # Kelvin
            'alpha': 1.9,
            'beta': 0.05,
            'hurst': 0.85,
            'n_s': 0.96,
            'tau_reionization': 0.06
        }
    
    @pytest.fixture
    def cmb_fluctuations(self, cmb_params):
        """Create CMB fluctuations instance."""
        return CMBFluctuations(cmb_params)
    
    def test_cmb_initialization(self, cmb_fluctuations, cmb_params):
        """Test CMB fluctuations initialization."""
        assert cmb_fluctuations.l_max == cmb_params['l_max']
        assert cmb_fluctuations.temperature_rms == cmb_params['temperature_rms']
        assert hasattr(cmb_fluctuations, 'angular_power_spectrum')
    
    def test_angular_power_spectrum_calculation(self, cmb_fluctuations):
        """Test calculation of angular power spectrum."""
        l_values, C_l = cmb_fluctuations.compute_angular_power_spectrum()
        
        # Test output format
        assert len(l_values) == len(C_l)
        assert np.all(l_values >= 2)  # l starts from 2
        assert np.all(C_l > 0)  # Power spectrum should be positive
        
        # Test characteristic features
        # Find acoustic peaks
        peaks = cmb_fluctuations.find_acoustic_peaks(l_values, C_l)
        assert len(peaks) >= 3  # Should find multiple acoustic peaks
    
    def test_temperature_fluctuation_map(self, cmb_fluctuations):
        """Test generation of temperature fluctuation maps."""
        nside = 64  # HEALPix resolution parameter
        temp_map = cmb_fluctuations.generate_temperature_map(nside)
        
        # Test map properties
        expected_npix = 12 * nside**2
        assert len(temp_map) == expected_npix
        
        # Test statistical properties
        mean_temp = np.mean(temp_map)
        std_temp = np.std(temp_map)
        assert abs(mean_temp) < 1e-6  # Should be zero mean
        assert std_temp > 0  # Should have fluctuations
    
    def test_polarization_analysis(self, cmb_fluctuations):
        """Test polarization component analysis."""
        l_values = np.arange(2, 1000)
        C_l_EE, C_l_BB, C_l_TE = cmb_fluctuations.compute_polarization_spectra(l_values)
        
        # Test shapes
        assert len(C_l_EE) == len(l_values)
        assert len(C_l_BB) == len(l_values)
        assert len(C_l_TE) == len(l_values)
        
        # Test physical constraints
        assert np.all(C_l_EE >= 0)  # E-mode power should be positive
        assert np.all(C_l_BB >= 0)  # B-mode power should be positive


class TestLargeScaleStructure:
    """Test suite for large-scale structure evolution."""
    
    @pytest.fixture
    def lss_params(self):
        """Parameters for large-scale structure simulation."""
        return {
            'box_size': 500.0,  # Mpc/h
            'grid_size': 128,
            'alpha': 1.7,
            'beta': 0.12,
            'gamma': 0.008,
            'hurst': 0.7,
            'bias_factor': 1.5,
            'redshift_range': [5.0, 0.0]
        }
    
    @pytest.fixture
    def lss_model(self, lss_params):
        """Create large-scale structure model."""
        return LargeScaleStructure(lss_params)
    
    def test_lss_initialization(self, lss_model, lss_params):
        """Test large-scale structure model initialization."""
        assert lss_model.box_size == lss_params['box_size']
        assert lss_model.grid_size == lss_params['grid_size']
        assert hasattr(lss_model, 'density_field')
    
    def test_correlation_function_calculation(self, lss_model):
        """Test two-point correlation function calculation."""
        # Generate mock galaxy positions
        n_galaxies = 1000
        galaxy_positions = np.random.uniform(0, lss_model.box_size, (n_galaxies, 3))
        
        r_bins, xi_r = lss_model.compute_correlation_function(galaxy_positions)
        
        # Test output format
        assert len(r_bins) == len(xi_r)
        assert np.all(r_bins > 0)
        
        # Test correlation function properties
        # Should be positive at small scales, approach zero at large scales
        assert xi_r[0] > 0  # Positive correlation at small scales
    
    def test_void_identification(self, lss_model):
        """Test cosmic void identification."""
        # Create mock density field with underdense regions
        density_field = np.random.lognormal(0, 0.5, 
                                          (lss_model.grid_size,) * 3)
        
        # Create some voids
        center = lss_model.grid_size // 2
        density_field[center-5:center+5, center-5:center+5, center-5:center+5] = 0.1
        
        voids = lss_model.identify_voids(density_field, threshold=0.3)
        
        # Test void properties
        assert len(voids) >= 1
        for void in voids:
            assert 'center' in void
            assert 'radius' in void
            assert 'volume' in void
            assert void['radius'] > 0
    
    def test_galaxy_clustering(self, lss_model):
        """Test galaxy clustering analysis."""
        # Generate mock galaxy catalog
        n_galaxies = 5000
        galaxy_positions = np.random.uniform(0, lss_model.box_size, (n_galaxies, 3))
        galaxy_masses = np.random.lognormal(12, 1, n_galaxies)  # log(M/M_sun)
        
        clustering_stats = lss_model.analyze_galaxy_clustering(
            galaxy_positions, galaxy_masses
        )
        
        # Test clustering statistics
        assert 'correlation_length' in clustering_stats
        assert 'bias_factor' in clustering_stats
        assert 'fractal_dimension' in clustering_stats
        assert clustering_stats['correlation_length'] > 0


class TestInflationaryModel:
    """Test suite for inflationary cosmology applications."""
    
    @pytest.fixture
    def inflation_params(self):
        """Parameters for inflationary model."""
        return {
            'phi_initial': 15.0,  # Initial inflaton field value
            'phi_final': 0.1,
            'alpha': 1.95,  # Close to 2 for inflation
            'beta': 0.02,   # Small stochastic term
            'gamma': 1e-6,  # Tiny nonlinear term
            'hurst': 0.9,   # High correlation
            'potential_type': 'quadratic',
            'mass_inflaton': 1e-5,  # Planck units
            'n_efolds': 60
        }
    
    @pytest.fixture
    def inflation_model(self, inflation_params):
        """Create inflationary model."""
        return InflationaryModel(inflation_params)
    
    def test_inflation_initialization(self, inflation_model, inflation_params):
        """Test inflationary model initialization."""
        assert inflation_model.phi_initial == inflation_params['phi_initial']
        assert inflation_model.n_efolds == inflation_params['n_efolds']
        assert hasattr(inflation_model, 'potential_function')
    
    def test_inflaton_evolution(self, inflation_model):
        """Test inflaton field evolution during inflation."""
        time_steps = np.linspace(0, 1, 1000)
        phi_evolution = inflation_model.evolve_inflaton_field(time_steps)
        
        # Test field evolution
        assert len(phi_evolution) == len(time_steps)
        assert phi_evolution[0] == inflation_model.phi_initial
        
        # Test slow-roll behavior (field should decrease slowly)
        assert np.all(np.diff(phi_evolution) <= 0)  # Monotonic decrease
    
    def test_primordial_power_spectrum(self, inflation_model):
        """Test primordial power spectrum calculation."""
        k_values = np.logspace(-4, 2, 100)  # Wavenumbers
        P_R, P_tensor = inflation_model.compute_primordial_spectra(k_values)
        
        # Test output format
        assert len(P_R) == len(k_values)
        assert len(P_tensor) == len(k_values)
        assert np.all(P_R > 0)
        assert np.all(P_tensor >= 0)
        
        # Test scale invariance (approximately)
        n_s = inflation_model.compute_spectral_index()
        assert 0.9 < n_s < 1.1  # Should be close to 1
    
    def test_tensor_to_scalar_ratio(self, inflation_model):
        """Test tensor-to-scalar ratio calculation."""
        r_tensor = inflation_model.compute_tensor_to_scalar_ratio()
        
        # Test physical bounds
        assert 0 <= r_tensor <= 1  # Physical constraint
        
        # For quadratic potential, should give specific prediction
        if inflation_model.potential_type == 'quadratic':
            expected_r = 16 * inflation_model.compute_slow_roll_epsilon()
            assert abs(r_tensor - expected_r) < 0.01
    
    def test_reheating_analysis(self, inflation_model):
        """Test post-inflationary reheating phase."""
        reheating_results = inflation_model.analyze_reheating()
        
        # Test reheating parameters
        assert 'reheating_temperature' in reheating_results
        assert 'equation_of_state' in reheating_results
        assert 'duration' in reheating_results
        
        # Test physical constraints
        T_reh = reheating_results['reheating_temperature']
        assert T_reh > 0  # Positive temperature


class TestCosmologicalIntegration:
    """Integration tests for cosmological applications."""
    
    @pytest.fixture
    def full_cosmological_simulation(self):
        """Create a full cosmological simulation setup."""
        params = {
            'simulation_type': 'full_cosmology',
            'box_size': 100.0,
            'grid_size': 64,
            'alpha': 1.8,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.75,
            'initial_redshift': 10.0,
            'final_redshift': 0.0,
            'time_steps': 100
        }
        return MFSUSimulator(params)
    
    def test_end_to_end_cosmological_simulation(self, full_cosmological_simulation):
        """Test complete cosmological simulation workflow."""
        # Setup initial conditions
        full_cosmological_simulation.setup_initial_conditions()
        assert hasattr(full_cosmological_simulation, 'initial_field')
        
        # Run simulation
        results = full_cosmological_simulation.run_simulation()
        assert 'final_field' in results
        assert 'evolution_data' in results
        
        # Analyze results
        analysis = full_cosmological_simulation.analyze_results(results)
        assert 'power_spectrum' in analysis
        assert 'correlation_function' in analysis
        assert 'fractal_dimension' in analysis
    
    @patch('src.utils.data_io.save_simulation_data')
    def test_data_export_functionality(self, mock_save, full_cosmological_simulation):
        """Test data export for cosmological simulations."""
        # Generate mock results
        results = {
            'density_field': np.random.random((64, 64, 64)),
            'power_spectrum': (np.logspace(-2, 2, 50), np.random.random(50)),
            'parameters': {'alpha': 1.8, 'beta': 0.1}
        }
        
        # Test export
        full_cosmological_simulation.export_results(results, 'test_cosmology')
        mock_save.assert_called_once()
    
    def test_parameter_validation(self):
        """Test parameter validation for cosmological applications."""
        # Test invalid parameters
        invalid_params = {
            'hubble_constant': -70,  # Should be positive
            'omega_matter': 1.5,     # Should be < 1
            'alpha': 0.5,            # Too small for cosmological applications
        }
        
        with pytest.raises(ValueError):
            CosmologicalMFSU(invalid_params)
    
    def test_benchmark_against_analytical_solutions(self):
        """Test simulation results against known analytical solutions."""
        # Test Einstein-de Sitter universe (Omega_m = 1, Omega_Lambda = 0)
        params = {
            'hubble_constant': 70.0,
            'omega_matter': 1.0,
            'omega_lambda': 0.0,
            'alpha': 2.0,  # Standard case
            'beta': 0.0,   # No stochastic term
            'gamma': 0.0   # No nonlinear term
        }
        
        cosmological_model = CosmologicalMFSU(params)
        
        # Test analytical growth factor
        z_test = 1.0
        D_analytical = 1.0 / (1.0 + z_test)  # Einstein-de Sitter solution
        D_numerical = cosmological_model.linear_growth_factor(z_test)
        
        # Allow for small numerical errors
        assert abs(D_numerical - D_analytical) < 0.01


class TestCosmologicalDataHandling:
    """Test data handling for cosmological applications."""
    
    def test_load_observational_data(self):
        """Test loading of observational cosmological data."""
        # Create mock observational data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("redshift,luminosity_distance,error\n")
            f.write("0.1,500.0,10.0\n")
            f.write("0.5,2000.0,50.0\n")
            f.write("1.0,3500.0,100.0\n")
            temp_file = f.name
        
        try:
            # Test data loading
            from src.applications.cosmology import load_observational_data
            data = load_observational_data(temp_file)
            
            assert 'redshift' in data.columns
            assert 'luminosity_distance' in data.columns
            assert len(data) == 3
            
        finally:
            os.unlink(temp_file)
    
    def test_cosmic_background_data_processing(self):
        """Test processing of cosmic background data."""
        # Create mock CMB data
        mock_cmb_data = {
            'l': np.arange(2, 2000),
            'C_l': np.random.random(1998) * 1000,
            'error': np.random.random(1998) * 50
        }
        
        from src.applications.cosmology import process_cmb_data
        processed_data = process_cmb_data(mock_cmb_data)
        
        assert 'multipole' in processed_data
        assert 'power_spectrum' in processed_data
        assert len(processed_data['multipole']) == len(mock_cmb_data['l'])


# Performance and memory tests
class TestCosmologicalPerformance:
    """Performance tests for cosmological simulations."""
    
    def test_memory_usage_large_grid(self):
        """Test memory usage for large grid simulations."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large simulation
        params = {
            'box_size': 200.0,
            'grid_size': 128,  # Large grid
            'alpha': 1.8,
            'beta': 0.1,
            'gamma': 0.01
        }
        
        dm_sim = DarkMatterSimulation(params)
        dm_sim.generate_initial_conditions()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should not exceed reasonable bounds (e.g., 2GB)
        assert memory_increase < 2 * 1024**3  # 2GB in bytes
        
        # Cleanup
        del dm_sim
        gc.collect()
    
    @pytest.mark.slow
    def test_simulation_speed_benchmark(self):
        """Benchmark simulation speed for cosmological applications."""
        import time
        
        params = {
            'box_size': 50.0,
            'grid_size': 32,
            'alpha': 1.8,
            'beta': 0.1,
            'gamma': 0.01,
            'time_steps': 100
        }
        
        simulator = MFSUSimulator(params)
        
        start_time = time.time()
        simulator.run_simulation()
        end_time = time.time()
        
        simulation_time = end_time - start_time
        
        # Should complete within reasonable time (adjust based on hardware)
        assert simulation_time < 300  # 5 minutes max


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])
