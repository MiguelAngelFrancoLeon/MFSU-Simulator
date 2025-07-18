"""
Stochastic Processes Module for MFSU Simulator

This module implements stochastic processes and fractal noise generation
for the MFSU (Modelo Fractal Estocástico Unificado) equation:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Specifically handles the β ξ_H(x,t)ψ term where ξ_H(x,t) is fractional noise
with Hurst exponent H.
"""

import numpy as np
from scipy import fftpack
from scipy.stats import norm
from scipy.special import gamma
from numba import jit, njit
import warnings
from typing import Tuple, Optional, Union, Dict, Any

__all__ = [
    'FractionalBrownianMotion',
    'FractionalGaussianNoise',
    'MultifractalNoise',
    'StochasticForcing',
    'NoiseGenerator',
    'CorrelatedNoise'
]


class FractionalBrownianMotion:
    """
    Generates fractional Brownian motion (fBm) with specified Hurst exponent.
    
    Uses the Cholesky decomposition method for accurate correlation structure.
    """
    
    def __init__(self, hurst: float = 0.5, random_seed: Optional[int] = None):
        """
        Initialize fractional Brownian motion generator.
        
        Parameters:
        -----------
        hurst : float, default=0.5
            Hurst exponent (0 < H < 1)
        random_seed : int, optional
            Random seed for reproducibility
        """
        if not 0 < hurst < 1:
            raise ValueError("Hurst exponent must be between 0 and 1")
        
        self.hurst = hurst
        self.rng = np.random.RandomState(random_seed)
        
    def generate_1d(self, n: int, T: float = 1.0) -> np.ndarray:
        """
        Generate 1D fractional Brownian motion.
        
        Parameters:
        -----------
        n : int
            Number of time steps
        T : float
            Total time
            
        Returns:
        --------
        np.ndarray
            fBm path of length n
        """
        dt = T / n
        t = np.arange(n) * dt
        
        # Covariance matrix for fBm
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = 0.5 * (np.abs(t[i])**(2*self.hurst) + 
                                np.abs(t[j])**(2*self.hurst) - 
                                np.abs(t[i] - t[j])**(2*self.hurst))
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Add small regularization if needed
            C += 1e-10 * np.eye(n)
            L = np.linalg.cholesky(C)
        
        # Generate correlated Gaussian variables
        Z = self.rng.randn(n)
        return L @ Z
    
    def generate_2d(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
        """
        Generate 2D fractional Brownian motion field.
        
        Parameters:
        -----------
        nx, ny : int
            Grid dimensions
        Lx, Ly : float
            Domain size
            
        Returns:
        --------
        np.ndarray
            2D fBm field of shape (nx, ny)
        """
        return self._generate_2d_fft(nx, ny, Lx, Ly)
    
    def _generate_2d_fft(self, nx: int, ny: int, Lx: float, Ly: float) -> np.ndarray:
        """FFT-based 2D fBm generation for efficiency."""
        # Wavenumber grids
        kx = 2 * np.pi * fftpack.fftfreq(nx, Lx/nx)
        ky = 2 * np.pi * fftpack.fftfreq(ny, Ly/ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        
        # Avoid division by zero
        K[0, 0] = 1e-10
        
        # Power spectral density for fBm
        PSD = K**(-1 - 2*self.hurst)
        PSD[0, 0] = 0  # Remove DC component
        
        # Generate complex Gaussian noise
        noise_real = self.rng.randn(nx, ny)
        noise_imag = self.rng.randn(nx, ny)
        noise_complex = noise_real + 1j * noise_imag
        
        # Apply spectral filtering
        noise_filtered = np.sqrt(PSD) * noise_complex
        
        # Inverse FFT to get real-space field
        field = np.real(fftpack.ifft2(noise_filtered))
        
        return field


class FractionalGaussianNoise:
    """
    Generates fractional Gaussian noise (fGn) - the increments of fBm.
    
    This represents the ξ_H(x,t) term in the MFSU equation.
    """
    
    def __init__(self, hurst: float = 0.5, random_seed: Optional[int] = None):
        """
        Initialize fractional Gaussian noise generator.
        
        Parameters:
        -----------
        hurst : float
            Hurst exponent (0 < H < 1)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.fbm = FractionalBrownianMotion(hurst, random_seed)
        self.hurst = hurst
        
    def generate_1d(self, n: int, T: float = 1.0) -> np.ndarray:
        """Generate 1D fractional Gaussian noise."""
        # Generate fBm and take increments
        fbm_path = self.fbm.generate_1d(n + 1, T)
        return np.diff(fbm_path)
    
    def generate_2d(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
        """Generate 2D fractional Gaussian noise field."""
        return self.fbm.generate_2d(nx, ny, Lx, Ly)
    
    def generate_3d(self, nx: int, ny: int, nt: int, 
                   Lx: float = 1.0, Ly: float = 1.0, T: float = 1.0) -> np.ndarray:
        """
        Generate 3D (2D space + time) fractional Gaussian noise.
        
        Parameters:
        -----------
        nx, ny, nt : int
            Grid dimensions (space and time)
        Lx, Ly, T : float
            Domain sizes
            
        Returns:
        --------
        np.ndarray
            3D noise field of shape (nx, ny, nt)
        """
        noise_field = np.zeros((nx, ny, nt))
        
        # Generate time-correlated noise for each spatial point
        for i in range(nx):
            for j in range(ny):
                noise_field[i, j, :] = self.generate_1d(nt, T)
        
        return noise_field


@njit
def _multifractal_cascade(field: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Numba-optimized multifractal cascade multiplication.
    """
    result = field.copy()
    for _ in range(iterations):
        multiplier = np.exp(alpha * np.random.randn(*field.shape))
        result *= multiplier
    return result


class MultifractalNoise:
    """
    Generates multifractal noise using cascade processes.
    
    Useful for modeling turbulent-like fluctuations in the stochastic term.
    """
    
    def __init__(self, alpha: float = 0.1, cascade_levels: int = 3, 
                 random_seed: Optional[int] = None):
        """
        Initialize multifractal noise generator.
        
        Parameters:
        -----------
        alpha : float
            Cascade parameter controlling multifractality
        cascade_levels : int
            Number of cascade levels
        random_seed : int, optional
            Random seed
        """
        self.alpha = alpha
        self.cascade_levels = cascade_levels
        self.rng = np.random.RandomState(random_seed)
        
    def generate_2d(self, nx: int, ny: int) -> np.ndarray:
        """Generate 2D multifractal noise field."""
        # Start with Gaussian noise
        base_field = self.rng.randn(nx, ny)
        
        # Apply cascade multiplications
        np.random.seed(self.rng.randint(0, 2**31))
        multifractal_field = _multifractal_cascade(
            base_field, self.alpha, self.cascade_levels
        )
        
        return multifractal_field


class CorrelatedNoise:
    """
    Generates spatially and temporally correlated noise with specified correlation functions.
    """
    
    def __init__(self, correlation_length: float = 1.0, 
                 correlation_time: float = 1.0, 
                 random_seed: Optional[int] = None):
        """
        Initialize correlated noise generator.
        
        Parameters:
        -----------
        correlation_length : float
            Spatial correlation length
        correlation_time : float
            Temporal correlation time
        random_seed : int, optional
            Random seed
        """
        self.correlation_length = correlation_length
        self.correlation_time = correlation_time
        self.rng = np.random.RandomState(random_seed)
        
    def exponential_kernel(self, r: np.ndarray, correlation_length: float) -> np.ndarray:
        """Exponential correlation kernel."""
        return np.exp(-r / correlation_length)
    
    def gaussian_kernel(self, r: np.ndarray, correlation_length: float) -> np.ndarray:
        """Gaussian correlation kernel."""
        return np.exp(-0.5 * (r / correlation_length)**2)
    
    def generate_correlated_2d(self, nx: int, ny: int, 
                              kernel_type: str = 'exponential') -> np.ndarray:
        """
        Generate 2D correlated noise field.
        
        Parameters:
        -----------
        nx, ny : int
            Grid dimensions
        kernel_type : str
            Type of correlation kernel ('exponential' or 'gaussian')
            
        Returns:
        --------
        np.ndarray
            Correlated noise field
        """
        # Create distance matrix
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Reference point (center)
        x0, y0 = nx // 2, ny // 2
        R = np.sqrt((X - x0)**2 + (Y - y0)**2)
        
        # Choose correlation kernel
        if kernel_type == 'exponential':
            C = self.exponential_kernel(R, self.correlation_length)
        elif kernel_type == 'gaussian':
            C = self.gaussian_kernel(R, self.correlation_length)
        else:
            raise ValueError("kernel_type must be 'exponential' or 'gaussian'")
        
        # Generate correlated field using FFT convolution
        noise = self.rng.randn(nx, ny)
        
        # FFT-based convolution
        noise_fft = fftpack.fft2(noise)
        kernel_fft = fftpack.fft2(C)
        
        correlated_noise = np.real(fftpack.ifft2(noise_fft * kernel_fft))
        
        # Normalize
        correlated_noise = correlated_noise / np.std(correlated_noise)
        
        return correlated_noise


class StochasticForcing:
    """
    Main class for generating stochastic forcing terms in the MFSU equation.
    
    Handles the β ξ_H(x,t)ψ term with various noise types.
    """
    
    def __init__(self, noise_type: str = 'fractional_gaussian',
                 hurst: float = 0.7, beta: float = 0.1,
                 noise_params: Optional[Dict[str, Any]] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize stochastic forcing generator.
        
        Parameters:
        -----------
        noise_type : str
            Type of noise ('fractional_gaussian', 'multifractal', 'correlated')
        hurst : float
            Hurst exponent for fractional noise
        beta : float
            Noise intensity parameter
        noise_params : dict, optional
            Additional parameters for noise generation
        random_seed : int, optional
            Random seed
        """
        self.noise_type = noise_type
        self.hurst = hurst
        self.beta = beta
        self.noise_params = noise_params or {}
        self.random_seed = random_seed
        
        # Initialize noise generator
        if noise_type == 'fractional_gaussian':
            self.noise_generator = FractionalGaussianNoise(hurst, random_seed)
        elif noise_type == 'multifractal':
            alpha = self.noise_params.get('alpha', 0.1)
            levels = self.noise_params.get('cascade_levels', 3)
            self.noise_generator = MultifractalNoise(alpha, levels, random_seed)
        elif noise_type == 'correlated':
            corr_length = self.noise_params.get('correlation_length', 1.0)
            corr_time = self.noise_params.get('correlation_time', 1.0)
            self.noise_generator = CorrelatedNoise(corr_length, corr_time, random_seed)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def generate_forcing(self, psi: np.ndarray, grid_params: Dict[str, Any]) -> np.ndarray:
        """
        Generate stochastic forcing term β ξ_H(x,t)ψ.
        
        Parameters:
        -----------
        psi : np.ndarray
            Current field values
        grid_params : dict
            Grid parameters (nx, ny, nt, Lx, Ly, T)
            
        Returns:
        --------
        np.ndarray
            Stochastic forcing term
        """
        if psi.ndim == 1:
            return self._generate_1d_forcing(psi, grid_params)
        elif psi.ndim == 2:
            return self._generate_2d_forcing(psi, grid_params)
        elif psi.ndim == 3:
            return self._generate_3d_forcing(psi, grid_params)
        else:
            raise ValueError("psi must be 1D, 2D, or 3D array")
    
    def _generate_1d_forcing(self, psi: np.ndarray, grid_params: Dict[str, Any]) -> np.ndarray:
        """Generate 1D stochastic forcing."""
        n = len(psi)
        T = grid_params.get('T', 1.0)
        
        if self.noise_type == 'fractional_gaussian':
            noise = self.noise_generator.generate_1d(n, T)
        else:
            # For other noise types, use simple white noise in 1D
            noise = np.random.randn(n)
        
        return self.beta * noise * psi
    
    def _generate_2d_forcing(self, psi: np.ndarray, grid_params: Dict[str, Any]) -> np.ndarray:
        """Generate 2D stochastic forcing."""
        nx, ny = psi.shape
        Lx = grid_params.get('Lx', 1.0)
        Ly = grid_params.get('Ly', 1.0)
        
        if self.noise_type == 'fractional_gaussian':
            noise = self.noise_generator.generate_2d(nx, ny, Lx, Ly)
        elif self.noise_type == 'multifractal':
            noise = self.noise_generator.generate_2d(nx, ny)
        elif self.noise_type == 'correlated':
            kernel_type = self.noise_params.get('kernel_type', 'exponential')
            noise = self.noise_generator.generate_correlated_2d(nx, ny, kernel_type)
        else:
            noise = np.random.randn(nx, ny)
        
        return self.beta * noise * psi
    
    def _generate_3d_forcing(self, psi: np.ndarray, grid_params: Dict[str, Any]) -> np.ndarray:
        """Generate 3D (2D space + time) stochastic forcing."""
        nx, ny, nt = psi.shape
        Lx = grid_params.get('Lx', 1.0)
        Ly = grid_params.get('Ly', 1.0)
        T = grid_params.get('T', 1.0)
        
        if self.noise_type == 'fractional_gaussian':
            noise = self.noise_generator.generate_3d(nx, ny, nt, Lx, Ly, T)
        else:
            # For other noise types, generate time-independent 2D noise
            noise = np.zeros((nx, ny, nt))
            for t in range(nt):
                noise[:, :, t] = self._generate_2d_forcing(
                    psi[:, :, t], {'Lx': Lx, 'Ly': Ly}
                ) / self.beta  # Remove beta multiplication to avoid double application
        
        return self.beta * noise * psi
    
    def get_noise_statistics(self, noise_field: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical properties of generated noise.
        
        Parameters:
        -----------
        noise_field : np.ndarray
            Generated noise field
            
        Returns:
        --------
        dict
            Statistical properties
        """
        stats = {
            'mean': np.mean(noise_field),
            'std': np.std(noise_field),
            'variance': np.var(noise_field),
            'skewness': float(np.mean(((noise_field - np.mean(noise_field)) / np.std(noise_field))**3)),
            'kurtosis': float(np.mean(((noise_field - np.mean(noise_field)) / np.std(noise_field))**4)),
            'min': np.min(noise_field),
            'max': np.max(noise_field)
        }
        return stats


class NoiseGenerator:
    """
    Factory class for creating different types of noise generators.
    """
    
    @staticmethod
    def create_generator(noise_type: str, **kwargs) -> Union[FractionalGaussianNoise, 
                                                            MultifractalNoise, 
                                                            CorrelatedNoise]:
        """
        Create a noise generator of specified type.
        
        Parameters:
        -----------
        noise_type : str
            Type of noise generator
        **kwargs : dict
            Generator-specific parameters
            
        Returns:
        --------
        Noise generator instance
        """
        if noise_type == 'fractional_gaussian':
            return FractionalGaussianNoise(**kwargs)
        elif noise_type == 'multifractal':
            return MultifractalNoise(**kwargs)
        elif noise_type == 'correlated':
            return CorrelatedNoise(**kwargs)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    @staticmethod
    def validate_hurst_exponent(hurst: float) -> bool:
        """Validate Hurst exponent value."""
        return 0 < hurst < 1
    
    @staticmethod
    def estimate_hurst_exponent(time_series: np.ndarray, 
                               method: str = 'rs') -> float:
        """
        Estimate Hurst exponent from time series data.
        
        Parameters:
        -----------
        time_series : np.ndarray
            Time series data
        method : str
            Estimation method ('rs' for R/S analysis)
            
        Returns:
        --------
        float
            Estimated Hurst exponent
        """
        if method == 'rs':
            return _estimate_hurst_rs(time_series)
        else:
            raise ValueError(f"Unknown estimation method: {method}")


@njit
def _estimate_hurst_rs(time_series: np.ndarray) -> float:
    """
    Estimate Hurst exponent using R/S analysis (Rescaled Range).
    Optimized with Numba for performance.
    """
    n = len(time_series)
    if n < 10:
        return 0.5  # Default value for short series
    
    # Compute cumulative sum
    Y = np.cumsum(time_series - np.mean(time_series))
    
    # Compute R/S for different window sizes
    window_sizes = np.logspace(1, np.log10(n//4), 10).astype(np.int32)
    rs_values = np.zeros(len(window_sizes))
    
    for i, w in enumerate(window_sizes):
        if w >= n:
            continue
        
        # Compute R/S for current window size
        rs_sum = 0.0
        count = 0
        
        for j in range(0, n - w + 1, w):
            window_data = Y[j:j+w]
            R = np.max(window_data) - np.min(window_data)
            S = np.std(time_series[j:j+w])
            if S > 0:
                rs_sum += R / S
                count += 1
        
        if count > 0:
            rs_values[i] = rs_sum / count
    
    # Fit log-log slope
    valid_indices = rs_values > 0
    if np.sum(valid_indices) < 2:
        return 0.5
    
    log_w = np.log(window_sizes[valid_indices])
    log_rs = np.log(rs_values[valid_indices])
    
    # Simple linear regression
    n_valid = len(log_w)
    slope = (n_valid * np.sum(log_w * log_rs) - np.sum(log_w) * np.sum(log_rs)) / \
            (n_valid * np.sum(log_w**2) - np.sum(log_w)**2)
    
    return max(0.01, min(0.99, slope))  # Clamp to valid range


# Utility functions for noise generation
def power_law_noise(n: int, beta: float = 1.0, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate power-law noise with spectral density ~ 1/f^β.
    
    Parameters:
    -----------
    n : int
        Number of samples
    beta : float
        Power-law exponent
    random_seed : int, optional
        Random seed
        
    Returns:
    --------
    np.ndarray
        Power-law noise
    """
    rng = np.random.RandomState(random_seed)
    
    # Generate white noise
    white_noise = rng.randn(n)
    
    # Create frequency array
    freqs = np.fft.fftfreq(n)[1:n//2]  # Exclude DC and Nyquist
    
    # Power-law spectrum
    spectrum = np.zeros(n//2, dtype=complex)
    spectrum[1:] = (freqs ** (-beta/2)) * (rng.randn(len(freqs)) + 1j * rng.randn(len(freqs)))
    
    # Mirror for real signal
    full_spectrum = np.concatenate([spectrum, np.conj(spectrum[::-1])])
    
    # Inverse FFT
    colored_noise = np.real(np.fft.ifft(full_spectrum))
    
    return colored_noise


def pink_noise(n: int, random_seed: Optional[int] = None) -> np.ndarray:
    """Generate pink noise (1/f noise)."""
    return power_law_noise(n, beta=1.0, random_seed=random_seed)


def brown_noise(n: int, random_seed: Optional[int] = None) -> np.ndarray:
    """Generate brown noise (1/f² noise)."""
    return power_law_noise(n, beta=2.0, random_seed=random_seed)


def white_noise(n: int, random_seed: Optional[int] = None) -> np.ndarray:
    """Generate white noise (flat spectrum)."""
    rng = np.random.RandomState(random_seed)
    return rng.randn(n)


# Example usage and testing
if __name__ == "__main__":
    # Test fractional Gaussian noise generation
    print("Testing MFSU Stochastic Processes...")
    
    # Test 1D fractional Gaussian noise
    fgn = FractionalGaussianNoise(hurst=0.7)
    noise_1d = fgn.generate_1d(100, T=1.0)
    print(f"1D FGN statistics: mean={np.mean(noise_1d):.4f}, std={np.std(noise_1d):.4f}")
    
    # Test 2D fractional Gaussian noise
    noise_2d = fgn.generate_2d(32, 32, Lx=1.0, Ly=1.0)
    print(f"2D FGN statistics: mean={np.mean(noise_2d):.4f}, std={np.std(noise_2d):.4f}")
    
    # Test stochastic forcing
    psi = np.ones((32, 32))  # Test field
    forcing = StochasticForcing(noise_type='fractional_gaussian', hurst=0.7, beta=0.1)
    stoch_term = forcing.generate_forcing(psi, {'Lx': 1.0, 'Ly': 1.0})
    print(f"Stochastic forcing statistics: mean={np.mean(stoch_term):.4f}, std={np.std(stoch_term):.4f}")
    
    # Test Hurst exponent estimation
    test_series = fgn.generate_1d(1000, T=10.0)
    estimated_hurst = NoiseGenerator.estimate_hurst_exponent(test_series)
    print(f"Estimated Hurst exponent: {estimated_hurst:.3f} (true: 0.7)")
    
    print("All tests completed successfully!")
