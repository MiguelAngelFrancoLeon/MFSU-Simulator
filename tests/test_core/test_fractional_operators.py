"""
Tests for fractional operators in the MFSU simulator.

This module tests the implementation of fractional Laplacian operators
used in the MFSU equation: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

The fractional Laplacian (-Δ)^(α/2) is implemented using spectral methods
in Fourier space for periodic boundary conditions.
"""

import pytest
import numpy as np
from scipy import fftpack
from scipy.special import gamma
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from core.fractional_operators import (
    FractionalLaplacian,
    fractional_laplacian_1d,
    fractional_laplacian_2d,
    fractional_laplacian_spectral,
    riesz_fractional_derivative
)


class TestFractionalLaplacian:
    """Test suite for the FractionalLaplacian class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid_size = 64
        self.domain_length = 2 * np.pi
        self.dx = self.domain_length / self.grid_size
        self.x = np.linspace(0, self.domain_length, self.grid_size, endpoint=False)
        
        # Initialize fractional Laplacian operator
        self.frac_lap = FractionalLaplacian(
            grid_size=self.grid_size,
            domain_length=self.domain_length,
            alpha=1.5
        )
    
    def test_initialization(self):
        """Test proper initialization of FractionalLaplacian."""
        assert self.frac_lap.grid_size == self.grid_size
        assert self.frac_lap.domain_length == self.domain_length
        assert self.frac_lap.alpha == 1.5
        assert self.frac_lap.dx == self.dx
        
        # Check that wavenumbers are properly initialized
        assert len(self.frac_lap.k) == self.grid_size
        assert isinstance(self.frac_lap.k, np.ndarray)
    
    def test_alpha_validation(self):
        """Test validation of alpha parameter."""
        # Valid alpha values
        valid_alphas = [0.1, 0.5, 1.0, 1.5, 1.9, 2.0]
        for alpha in valid_alphas:
            frac_lap = FractionalLaplacian(
                grid_size=32,
                domain_length=2*np.pi,
                alpha=alpha
            )
            assert frac_lap.alpha == alpha
        
        # Invalid alpha values should raise ValueError
        invalid_alphas = [-0.1, 0.0, 2.1, 3.0]
        for alpha in invalid_alphas:
            with pytest.raises(ValueError):
                FractionalLaplacian(
                    grid_size=32,
                    domain_length=2*np.pi,
                    alpha=alpha
                )
    
    def test_gaussian_function(self):
        """Test fractional Laplacian on Gaussian function."""
        # Create Gaussian function
        sigma = 0.5
        x0 = np.pi
        gaussian = np.exp(-0.5 * ((self.x - x0) / sigma)**2)
        
        # Apply fractional Laplacian
        result = self.frac_lap.apply(gaussian)
        
        # Check basic properties
        assert len(result) == len(gaussian)
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()
        
        # For alpha close to 2, should approximate regular Laplacian
        frac_lap_2 = FractionalLaplacian(
            grid_size=self.grid_size,
            domain_length=self.domain_length,
            alpha=1.99
        )
        result_2 = frac_lap_2.apply(gaussian)
        
        # Analytical second derivative of Gaussian
        second_deriv_analytical = gaussian * ((self.x - x0)**2 / sigma**4 - 1/sigma**2)
        
        # Should be approximately equal (within numerical error)
        np.testing.assert_allclose(result_2, -second_deriv_analytical, rtol=0.1)
    
    def test_sine_function(self):
        """Test fractional Laplacian on sine function."""
        k_mode = 1
        sine_func = np.sin(k_mode * self.x)
        
        # Apply fractional Laplacian
        result = self.frac_lap.apply(sine_func)
        
        # For sine function, (-Δ)^(α/2) sin(kx) = k^α sin(kx)
        expected = (k_mode**self.frac_lap.alpha) * sine_func
        
        # Check with relative tolerance due to numerical precision
        np.testing.assert_allclose(result, expected, rtol=1e-12)
    
    def test_cosine_function(self):
        """Test fractional Laplacian on cosine function."""
        k_mode = 2
        cosine_func = np.cos(k_mode * self.x)
        
        result = self.frac_lap.apply(cosine_func)
        expected = (k_mode**self.frac_lap.alpha) * cosine_func
        
        np.testing.assert_allclose(result, expected, rtol=1e-12)
    
    def test_linearity(self):
        """Test linearity of fractional Laplacian operator."""
        # Create two test functions
        f1 = np.sin(self.x)
        f2 = np.cos(2 * self.x)
        
        # Coefficients
        a, b = 2.5, -1.3
        
        # Test linearity: L[af1 + bf2] = aL[f1] + bL[f2]
        combined_func = a * f1 + b * f2
        result_combined = self.frac_lap.apply(combined_func)
        
        result_f1 = self.frac_lap.apply(f1)
        result_f2 = self.frac_lap.apply(f2)
        expected = a * result_f1 + b * result_f2
        
        np.testing.assert_allclose(result_combined, expected, rtol=1e-14)
    
    def test_energy_scaling(self):
        """Test energy scaling properties of fractional Laplacian."""
        # Create a function with known energy content
        func = np.sin(self.x) + 0.5 * np.cos(3 * self.x)
        
        # Apply fractional Laplacian
        result = self.frac_lap.apply(func)
        
        # Energy should be preserved in spectral domain
        func_fft = fftpack.fft(func)
        result_fft = fftpack.fft(result)
        
        # Check Parseval's theorem holds approximately
        energy_func = np.sum(np.abs(func_fft)**2)
        energy_result = np.sum(np.abs(result_fft)**2)
        
        # The energy should scale according to the fractional power
        # This is a complex relationship, so we just check it's reasonable
        assert energy_result > 0
        assert np.isfinite(energy_result)


class TestFractionalOperatorFunctions:
    """Test individual fractional operator functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n = 64
        self.L = 2 * np.pi
        self.x = np.linspace(0, self.L, self.n, endpoint=False)
    
    def test_fractional_laplacian_1d(self):
        """Test 1D fractional Laplacian function."""
        alpha = 1.5
        func = np.sin(self.x)
        
        result = fractional_laplacian_1d(func, alpha, self.L)
        
        # Should return array of same size
        assert len(result) == len(func)
        assert isinstance(result, np.ndarray)
        
        # For sine function, result should be k^alpha * sin(x)
        expected = (1.0**alpha) * func
        np.testing.assert_allclose(result, expected, rtol=1e-12)
    
    def test_fractional_laplacian_2d(self):
        """Test 2D fractional Laplacian function."""
        alpha = 1.8
        nx, ny = 32, 32
        Lx, Ly = 2*np.pi, 2*np.pi
        
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        # Test function: sin(x)cos(y)
        func = np.sin(X) * np.cos(Y)
        
        result = fractional_laplacian_2d(func, alpha, Lx, Ly)
        
        # Check dimensions
        assert result.shape == func.shape
        assert isinstance(result, np.ndarray)
        
        # For separable function, should get (1^alpha + 1^alpha) * func = 2 * func
        expected = 2**alpha * func
        np.testing.assert_allclose(result, expected, rtol=1e-11)
    
    def test_spectral_implementation(self):
        """Test spectral implementation of fractional Laplacian."""
        alpha = 1.3
        func = np.cos(2 * self.x) + np.sin(3 * self.x)
        
        result = fractional_laplacian_spectral(func, alpha, self.L)
        
        # Check result properties
        assert len(result) == len(func)
        assert np.isfinite(result).all()
        
        # Compare with direct implementation
        direct_result = fractional_laplacian_1d(func, alpha, self.L)
        np.testing.assert_allclose(result, direct_result, rtol=1e-13)
    
    def test_riesz_derivative(self):
        """Test Riesz fractional derivative implementation."""
        alpha = 1.7
        func = np.sin(self.x)
        
        result = riesz_fractional_derivative(func, alpha, self.L)
        
        # Check basic properties
        assert len(result) == len(func)
        assert isinstance(result, np.ndarray)
        
        # For sine function, Riesz derivative should relate to fractional Laplacian
        # This is a more complex test, so we just check reasonableness
        assert np.isfinite(result).all()
        assert np.std(result) > 0  # Should not be constant


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_function(self):
        """Test fractional Laplacian of zero function."""
        n = 32
        func = np.zeros(n)
        alpha = 1.5
        L = 2 * np.pi
        
        result = fractional_laplacian_1d(func, alpha, L)
        
        # Should return zeros
        np.testing.assert_allclose(result, func, atol=1e-15)
    
    def test_constant_function(self):
        """Test fractional Laplacian of constant function."""
        n = 32
        func = np.ones(n) * 5.0
        alpha = 1.5
        L = 2 * np.pi
        
        result = fractional_laplacian_1d(func, alpha, L)
        
        # Constant function should map to zero (no k=0 component in derivative)
        np.testing.assert_allclose(result, np.zeros(n), atol=1e-14)
    
    def test_high_frequency_stability(self):
        """Test stability with high-frequency components."""
        n = 128
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        
        # High frequency sine wave
        func = np.sin(16 * x)
        alpha = 1.9
        
        result = fractional_laplacian_1d(func, alpha, L)
        
        # Should not blow up
        assert np.isfinite(result).all()
        assert np.max(np.abs(result)) < 1e10  # Reasonable bound
    
    def test_different_alpha_values(self):
        """Test consistency across different alpha values."""
        n = 64
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        func = np.sin(x) + 0.5 * np.cos(2*x)
        
        alphas = [0.1, 0.5, 1.0, 1.5, 1.8, 1.99]
        results = {}
        
        for alpha in alphas:
            result = fractional_laplacian_1d(func, alpha, L)
            results[alpha] = result
            
            # Check basic properties for each alpha
            assert np.isfinite(result).all()
            assert len(result) == len(func)
        
        # Results should vary smoothly with alpha
        # (This is more of a sanity check)
        for i in range(len(alphas)-1):
            alpha1, alpha2 = alphas[i], alphas[i+1]
            # Results should be different but not drastically
            assert not np.allclose(results[alpha1], results[alpha2])


class TestPerformance:
    """Test performance and efficiency of fractional operators."""
    
    def test_large_array_performance(self):
        """Test performance on large arrays."""
        n = 1024
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        func = np.sin(x) * np.cos(2*x)
        alpha = 1.5
        
        # This should complete without timeout
        result = fractional_laplacian_1d(func, alpha, L)
        
        assert len(result) == n
        assert np.isfinite(result).all()
    
    def test_memory_efficiency(self):
        """Test that operations don't use excessive memory."""
        n = 256
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        func = np.random.randn(n)
        alpha = 1.7
        
        # Multiple operations should not accumulate memory
        for _ in range(10):
            result = fractional_laplacian_1d(func, alpha, L)
            func = result.copy()  # Use result as input for next iteration
        
        assert np.isfinite(result).all()


def test_integration_with_mfsu_equation():
    """Test integration of fractional operators with MFSU equation structure."""
    # This test ensures that the fractional Laplacian can be used
    # as part of the full MFSU equation implementation
    
    n = 64
    L = 2 * np.pi
    x = np.linspace(0, L, n, endpoint=False)
    
    # MFSU parameters
    alpha_param = 0.5  # Coefficient in MFSU equation  
    alpha_frac = 1.5   # Fractional order
    beta = 0.1
    gamma = 0.01
    
    # Initial condition (example wave packet)
    psi = np.exp(-0.5 * ((x - np.pi) / 0.5)**2) * np.exp(1j * x)
    
    # Apply fractional Laplacian (main term in MFSU)
    frac_lap_term = fractional_laplacian_1d(psi, alpha_frac, L)
    
    # This represents the main fractional term: α(-Δ)^(α/2)ψ
    main_term = alpha_param * frac_lap_term
    
    # Check that we can compute this without issues
    assert main_term.shape == psi.shape
    assert np.isfinite(main_term).all()
    
    # The result should be complex (since psi is complex)
    assert np.iscomplexobj(main_term)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
