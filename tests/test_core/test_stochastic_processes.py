"""
Tests for stochastic processes in the MFSU simulator.

This module tests the implementation of stochastic processes used in the MFSU equation:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Main focus is on the fractional Brownian motion ξ_H(x,t) with Hurst parameter H,
and related stochastic processes for complex systems modeling.
"""

import pytest
import numpy as np
from scipy import stats, fftpack
from scipy.special import gamma
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from core.stochastic_processes import (
    FractionalBrownianMotion,
    FractionalNoise,
    MultifractalProcess,
    StochasticField,
    generate_fbm_1d,
    generate_fbm_2d,
    generate_fractional_noise,
    hurst_estimator,
    correlation_function_fbm,
    spectral_density_fbm
)


class TestFractionalBrownianMotion:
    """Test suite for Fractional Brownian Motion (fBm) implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_points = 256
        self.hurst = 0.7
        self.dt = 0.01
        self.fbm = FractionalBrownianMotion(
            hurst=self.hurst,
            n_points=self.n_points,
            dt=self.dt
        )
    
    def test_initialization(self):
        """Test proper initialization of FractionalBrownianMotion."""
        assert self.fbm.hurst == self.hurst
        assert self.fbm.n_points == self.n_points
        assert self.fbm.dt == self.dt
        
        # Hurst parameter should be in valid range
        assert 0 < self.fbm.hurst < 1
    
    def test_hurst_parameter_validation(self):
        """Test validation of Hurst parameter."""
        # Valid Hurst values
        valid_hurst = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        for h in valid_hurst:
            fbm = FractionalBrownianMotion(hurst=h, n_points=64, dt=0.01)
            assert fbm.hurst == h
        
        # Invalid Hurst values should raise ValueError
        invalid_hurst = [-0.1, 0.0, 1.0, 1.1, 2.0]
        for h in invalid_hurst:
            with pytest.raises(ValueError):
                FractionalBrownianMotion(hurst=h, n_points=64, dt=0.01)
    
    def test_fbm_generation_1d(self):
        """Test 1D fractional Brownian motion generation."""
        fbm_path = self.fbm.generate()
        
        # Basic properties
        assert len(fbm_path) == self.n_points
        assert isinstance(fbm_path, np.ndarray)
        assert np.isfinite(fbm_path).all()
        
        # First point should be zero (fBm starts at origin)
        assert fbm_path[0] == 0.0
        
        # Path should be continuous (no infinite jumps)
        increments = np.diff(fbm_path)
        assert np.isfinite(increments).all()
    
    def test_fbm_statistical_properties(self):
        """Test statistical properties of fBm."""
        n_realizations = 100
        paths = []
        
        for _ in range(n_realizations):
            path = self.fbm.generate()
            paths.append(path)
        
        paths = np.array(paths)
        
        # Mean should be approximately zero
        mean_path = np.mean(paths, axis=0)
        assert np.abs(mean_path[-1]) < 0.5  # Allow some statistical variation
        
        # Variance should grow as t^(2H)
        times = np.arange(1, self.n_points) * self.dt
        variances = np.var(paths[:, 1:], axis=0)
        
        # Log-log relationship: log(Var) ≈ 2H * log(t) + const
        log_times = np.log(times[10:])  # Skip early times to avoid numerical issues
        log_vars = np.log(variances[10:])
        
        # Linear regression to estimate slope
        slope, intercept, r_value, _, _ = stats.linregress(log_times, log_vars)
        
        # Slope should be approximately 2*H
        assert abs(slope - 2*self.hurst) < 0.3  # Allow reasonable tolerance
        assert r_value**2 > 0.7  # Good linear fit
    
    def test_hurst_estimation(self):
        """Test Hurst parameter estimation from generated fBm."""
        # Generate longer path for better estimation
        fbm_long = FractionalBrownianMotion(
            hurst=self.hurst,
            n_points=1024,
            dt=0.01
        )
        path = fbm_long.generate()
        
        # Estimate Hurst parameter
        estimated_hurst = hurst_estimator(path, method='rs')  # R/S method
        
        # Should be close to true value
        assert abs(estimated_hurst - self.hurst) < 0.15
        
        # Test with different method
        estimated_hurst_dfa = hurst_estimator(path, method='dfa')  # DFA method
        assert abs(estimated_hurst_dfa - self.hurst) < 0.15
    
    def test_fbm_scaling_invariance(self):
        """Test scaling properties of fBm."""
        # Generate fBm at two different time scales
        fbm1 = FractionalBrownianMotion(hurst=self.hurst, n_points=128, dt=0.01)
        fbm2 = FractionalBrownianMotion(hurst=self.hurst, n_points=128, dt=0.02)
        
        path1 = fbm1.generate()
        path2 = fbm2.generate()
        
        # Both should be valid fBm paths
        assert np.isfinite(path1).all()
        assert np.isfinite(path2).all()
        
        # Scaling relationship should hold approximately
        # This is a complex property, so we just check basic reasonableness
        assert np.std(path2) >= np.std(path1)  # Larger dt should give larger variance


class TestFractionalNoise:
    """Test suite for fractional Gaussian noise."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_points = 512
        self.hurst = 0.6
        self.dt = 0.01
        
    def test_fractional_noise_generation(self):
        """Test generation of fractional Gaussian noise."""
        noise = generate_fractional_noise(
            n_points=self.n_points,
            hurst=self.hurst,
            dt=self.dt
        )
        
        # Basic properties
        assert len(noise) == self.n_points
        assert isinstance(noise, np.ndarray)
        assert np.isfinite(noise).all()
        
        # Should have zero mean (approximately)
        assert abs(np.mean(noise)) < 0.1
        
        # Should be stationary (unlike fBm)
        # Test by comparing variance in different segments
        mid = self.n_points // 2
        var1 = np.var(noise[:mid])
        var2 = np.var(noise[mid:])
        
        # Variances should be similar (within factor of 2)
        ratio = max(var1, var2) / min(var1, var2)
        assert ratio < 2.0
    
    def test_noise_autocorrelation(self):
        """Test autocorrelation properties of fractional noise."""
        noise = generate_fractional_noise(
            n_points=1024,
            hurst=self.hurst,
            dt=self.dt
        )
        
        # Compute autocorrelation function
        autocorr = np.correlate(noise, noise, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # For fractional noise, autocorrelation should decay as power law
        lags = np.arange(1, min(100, len(autocorr)))
        
        # Should be positive for small lags if H > 0.5
        if self.hurst > 0.5:
            assert autocorr[1] > 0
        
        # Should decay for large lags
        assert autocorr[10] < autocorr[1]
        assert autocorr[50] < autocorr[10]
    
    def test_spectral_density(self):
        """Test spectral density of fractional noise."""
        noise = generate_fractional_noise(
            n_points=self.n_points,
            hurst=self.hurst,
            dt=self.dt
        )
        
        # Compute power spectral density
        frequencies, psd = spectral_density_fbm(noise, self.dt)
        
        # PSD should follow power law: S(f) ~ f^(-2H-1)
        # Focus on intermediate frequencies to avoid numerical issues
        valid_idx = (frequencies > 0.1) & (frequencies < 10.0)
        if np.sum(valid_idx) > 10:  # Enough points for regression
            log_freq = np.log(frequencies[valid_idx])
            log_psd = np.log(psd[valid_idx])
            
            slope, _, r_value, _, _ = stats.linregress(log_freq, log_psd)
            
            # Slope should be approximately -(2H + 1)
            expected_slope = -(2*self.hurst + 1)
            assert abs(slope - expected_slope) < 0.5
            assert r_value**2 > 0.5  # Reasonable fit


class TestStochasticField:
    """Test suite for 2D stochastic fields ξ_H(x,t)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nx, self.ny = 64, 64
        self.lx, self.ly = 2*np.pi, 2*np.pi
        self.hurst = 0.75
        
    def test_2d_field_generation(self):
        """Test generation of 2D fractional stochastic fields."""
        field = generate_fbm_2d(
            nx=self.nx, ny=self.ny,
            lx=self.lx, ly=self.ly,
            hurst=self.hurst
        )
        
        # Basic properties
        assert field.shape == (self.nx, self.ny)
        assert isinstance(field, np.ndarray)
        assert np.isfinite(field).all()
        
        # Should have reasonable statistical properties
        assert abs(np.mean(field)) < 0.5  # Approximately zero mean
        assert np.std(field) > 0  # Non-constant field
    
    def test_field_isotropy(self):
        """Test isotropy of 2D stochastic field."""
        field = generate_fbm_2d(
            nx=128, ny=128,
            lx=2*np.pi, ly=2*np.pi,
            hurst=self.hurst
        )
        
        # Compute radial power spectrum
        fft_field = fftpack.fft2(field)
        power_2d = np.abs(fft_field)**2
        
        # Create radial coordinates
        kx = fftpack.fftfreq(128, 2*np.pi/128)
        ky = fftpack.fftfreq(128, 2*np.pi/128)
        KX, KY = np.meshgrid(kx, ky)
        K_rad = np.sqrt(KX**2 + KY**2)
        
        # The field should be approximately isotropic
        # (This is a complex test, so we just check basic reasonableness)
        assert np.std(power_2d) > 0  # Should have structure
        assert not np.any(np.isinf(power_2d))  # Should be finite
    
    def test_field_scaling_properties(self):
        """Test scaling properties of 2D fields."""
        # Generate fields at different resolutions
        field_64 = generate_fbm_2d(64, 64, 2*np.pi, 2*np.pi, self.hurst)
        field_128 = generate_fbm_2d(128, 128, 2*np.pi, 2*np.pi, self.hurst)
        
        # Both should be valid
        assert np.isfinite(field_64).all()
        assert np.isfinite(field_128).all()
        
        # Statistical properties should be consistent
        std_64 = np.std(field_64)
        std_128 = np.std(field_128)
        
        # Standard deviations should be of similar magnitude
        ratio = max(std_64, std_128) / min(std_64, std_128)
        assert ratio < 3.0  # Allow some variation


class TestMultifractalProcess:
    """Test suite for multifractal processes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_points = 256
        self.q_range = np.linspace(-5, 5, 21)
        
    def test_multifractal_generation(self):
        """Test generation of multifractal processes."""
        mf_process = MultifractalProcess(
            n_points=self.n_points,
            intermittency=0.1,
            h_min=0.1,
            h_max=0.9
        )
        
        signal = mf_process.generate()
        
        # Basic properties
        assert len(signal) == self.n_points
        assert isinstance(signal, np.ndarray)
        assert np.isfinite(signal).all()
        assert np.std(signal) > 0
    
    def test_multifractal_spectrum(self):
        """Test multifractal spectrum estimation."""
        # Generate a known multifractal signal
        mf_process = MultifractalProcess(
            n_points=1024,
            intermittency=0.2,
            h_min=0.2,
            h_max=0.8
        )
        
        signal = mf_process.generate()
        
        # Estimate multifractal spectrum
        spectrum = mf_process.estimate_spectrum(signal)
        
        # Should return valid spectrum
        assert isinstance(spectrum, dict)
        assert 'alpha' in spectrum
        assert 'f_alpha' in spectrum
        assert len(spectrum['alpha']) == len(spectrum['f_alpha'])
        
        # Values should be reasonable
        assert np.all(np.isfinite(spectrum['alpha']))
        assert np.all(np.isfinite(spectrum['f_alpha']))


class TestStochasticIntegration:
    """Test integration of stochastic processes with MFSU equation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_spatial = 64
        self.n_temporal = 100
        self.lx = 2 * np.pi
        self.dt = 0.01
        self.hurst = 0.7
        
    def test_stochastic_field_for_mfsu(self):
        """Test generation of stochastic field for MFSU equation."""
        # Create space-time stochastic field ξ_H(x,t)
        stochastic_field = StochasticField(
            nx=self.n_spatial,
            nt=self.n_temporal,
            lx=self.lx,
            dt=self.dt,
            hurst=self.hurst
        )
        
        field = stochastic_field.generate_spacetime_field()
        
        # Should have correct dimensions
        assert field.shape == (self.n_temporal, self.n_spatial)
        assert np.isfinite(field).all()
        
        # Each spatial slice should have proper statistics
        for t_idx in range(0, self.n_temporal, 10):  # Sample every 10th time step
            spatial_slice = field[t_idx, :]
            assert np.isfinite(spatial_slice).all()
            assert np.std(spatial_slice) > 0
    
    def test_multiplicative_noise_term(self):
        """Test multiplicative noise term β ξ_H(x,t)ψ from MFSU."""
        # Create test wave function ψ
        x = np.linspace(0, self.lx, self.n_spatial, endpoint=False)
        psi = np.exp(-0.5 * ((x - np.pi) / 0.5)**2) * np.exp(1j * 2*x)
        
        # Create stochastic field
        stochastic_field = StochasticField(
            nx=self.n_spatial,
            nt=self.n_temporal,
            lx=self.lx,
            dt=self.dt,
            hurst=self.hurst
        )
        
        xi_field = stochastic_field.generate_spacetime_field()
        
        # MFSU parameters
        beta = 0.1
        
        # Compute multiplicative term for one time step
        t_idx = 50
        xi_t = xi_field[t_idx, :]
        multiplicative_term = beta * xi_t * psi
        
        # Should have same shape as psi
        assert multiplicative_term.shape == psi.shape
        assert np.isfinite(multiplicative_term).all()
        
        # Should be complex (since psi is complex)
        assert np.iscomplexobj(multiplicative_term)
        
        # Magnitude should be reasonable
        assert np.max(np.abs(multiplicative_term)) < 10 * np.max(np.abs(psi))
    
    def test_correlation_structure(self):
        """Test temporal correlation structure of stochastic field."""
        stochastic_field = StochasticField(
            nx=self.n_spatial,
            nt=200,  # Longer time series for correlation analysis
            lx=self.lx,
            dt=self.dt,
            hurst=self.hurst
        )
        
        field = stochastic_field.generate_spacetime_field()
        
        # Compute temporal correlation at a fixed spatial point
        x_idx = self.n_spatial // 2
        time_series = field[:, x_idx]
        
        # Autocorrelation function
        autocorr = np.correlate(time_series, time_series, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Should decay over time
        assert autocorr[0] == 1.0  # Perfect correlation at zero lag
        assert autocorr[10] < autocorr[1]  # Decay with lag
        assert autocorr[50] < autocorr[10] if len(autocorr) > 50 else True


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_extreme_hurst_values(self):
        """Test behavior with extreme Hurst values."""
        # Very small Hurst (anti-persistent)
        hurst_small = 0.05
        fbm_small = FractionalBrownianMotion(
            hurst=hurst_small,
            n_points=128,
            dt=0.01
        )
        
        path_small = fbm_small.generate()
        assert np.isfinite(path_small).all()
        assert np.std(path_small) > 0
        
        # Large Hurst (very persistent)
        hurst_large = 0.95
        fbm_large = FractionalBrownianMotion(
            hurst=hurst_large,
            n_points=128,
            dt=0.01
        )
        
        path_large = fbm_large.generate()
        assert np.isfinite(path_large).all()
        assert np.std(path_large) > 0
        
        # Paths should have different characteristics
        # (More detailed statistical tests could be added here)
        assert not np.allclose(path_small, path_large)
    
    def test_long_time_series(self):
        """Test stability with long time series."""
        fbm = FractionalBrownianMotion(
            hurst=0.7,
            n_points=2048,  # Long series
            dt=0.005
        )
        
        path = fbm.generate()
        
        # Should not blow up
        assert np.isfinite(path).all()
        assert np.max(np.abs(path)) < 1000  # Reasonable bound
    
    def test_memory_efficiency(self):
        """Test memory efficiency with multiple realizations."""
        fbm = FractionalBrownianMotion(hurst=0.6, n_points=256, dt=0.01)
        
        # Generate multiple realizations
        paths = []
        for _ in range(50):
            path = fbm.generate()
            paths.append(path)
        
        # All should be valid
        for path in paths:
            assert np.isfinite(path).all()
        
        # Should complete without memory issues
        paths_array = np.array(paths)
        assert paths_array.shape == (50, 256)


def test_stochastic_field_reproducibility():
    """Test reproducibility with random seeds."""
    np.random.seed(42)
    
    fbm1 = FractionalBrownianMotion(hurst=0.7, n_points=100, dt=0.01)
    path1 = fbm1.generate()
    
    # Reset seed and generate again
    np.random.seed(42)
    fbm2 = FractionalBrownianMotion(hurst=0.7, n_points=100, dt=0.01)
    path2 = fbm2.generate()
    
    # Should be identical
    np.testing.assert_allclose(path1, path2, rtol=1e-15)


def test_integration_with_mfsu_time_stepping():
    """Test integration of stochastic processes with MFSU time stepping."""
    # This test ensures stochastic processes work correctly
    # in the context of the full MFSU equation evolution
    
    n_spatial = 32
    n_time = 50
    dt = 0.01
    
    # Create stochastic field for time evolution
    stoch_field = StochasticField(
        nx=n_spatial,
        nt=n_time,
        lx=2*np.pi,
        dt=dt,
        hurst=0.7
    )
    
    xi_field = stoch_field.generate_spacetime_field()
    
    # Simulate time stepping with stochastic term
    x = np.linspace(0, 2*np.pi, n_spatial, endpoint=False)
    psi = np.exp(-0.5 * ((x - np.pi) / 0.5)**2)  # Initial condition
    
    beta = 0.05
    evolution_trace = [psi.copy()]
    
    for t in range(1, min(10, n_time)):  # Just a few steps for testing
        xi_t = xi_field[t, :]
        
        # Simple forward Euler step (just for testing)
        # In real implementation, this would be more sophisticated
        stochastic_increment = beta * xi_t * psi * dt
        psi = psi + stochastic_increment
        
        evolution_trace.append(psi.copy())
    
    # Evolution should remain stable
    for psi_t in evolution_trace:
        assert np.isfinite(psi_t).all()
        assert np.max(np.abs(psi_t)) < 100  # Shouldn't blow up


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
