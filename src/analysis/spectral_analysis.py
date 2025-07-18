"""
Análisis Espectral para el Simulador MFSU
==========================================

Este módulo implementa herramientas de análisis espectral para estudiar las
propiedades frecuenciales de las soluciones de la ecuación MFSU:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Incluye análisis de:
- Espectro de potencia
- Densidad espectral de potencia (PSD)
- Análisis de Fourier multidimensional
- Correlaciones espectrales
- Análisis de escalamiento espectral
"""

import numpy as np
import scipy.fft as fft
import scipy.signal as signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum

try:
    import pyfftw
    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    warnings.warn("PyFFTW not available, using scipy.fft")


class WindowType(Enum):
    """Tipos de ventanas para análisis espectral"""
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    TUKEY = "tukey"
    RECTANGULAR = "boxcar"


@dataclass
class SpectralResult:
    """Resultado del análisis espectral"""
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    phase_spectrum: np.ndarray
    coherence: Optional[np.ndarray] = None
    cross_spectrum: Optional[np.ndarray] = None
    metadata: Dict = None


class SpectralAnalyzer:
    """
    Analizador espectral para soluciones MFSU
    
    Proporciona herramientas completas para el análisis frecuencial
    de las soluciones de la ecuación MFSU, incluyendo análisis de
    potencia, correlaciones espectrales y escalamiento.
    """
    
    def __init__(self, 
                 use_pyfftw: bool = True,
                 fftw_wisdom_file: Optional[str] = None,
                 n_threads: int = None):
        """
        Inicializa el analizador espectral
        
        Parameters:
        -----------
        use_pyfftw : bool
            Usar PyFFTW para FFTs optimizadas
        fftw_wisdom_file : str, optional
            Archivo de wisdom para FFTW
        n_threads : int, optional
            Número de hilos para cálculos paralelos
        """
        self.use_pyfftw = use_pyfftw and HAS_PYFFTW
        self.n_threads = n_threads or 1
        
        if self.use_pyfftw:
            pyfftw.config.NUM_THREADS = self.n_threads
            if fftw_wisdom_file:
                try:
                    pyfftw.import_wisdom(fftw_wisdom_file)
                except:
                    warnings.warn(f"No se pudo cargar wisdom desde {fftw_wisdom_file}")
    
    def power_spectrum_1d(self, 
                         signal_data: np.ndarray,
                         dt: float = 1.0,
                         window: Union[str, WindowType] = WindowType.HANN,
                         detrend: str = 'linear',
                         nperseg: Optional[int] = None) -> SpectralResult:
        """
        Calcula el espectro de potencia 1D
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Señal temporal 1D
        dt : float
            Paso temporal
        window : str or WindowType
            Tipo de ventana
        detrend : str
            Método de detrending
        nperseg : int, optional
            Longitud de segmento para Welch
            
        Returns:
        --------
        SpectralResult
            Resultado del análisis espectral
        """
        if isinstance(window, WindowType):
            window = window.value
            
        # Usar método de Welch para estimación robusta
        if nperseg is None:
            nperseg = len(signal_data) // 8
            
        freqs, psd = signal.welch(
            signal_data, 
            fs=1/dt, 
            window=window,
            nperseg=nperseg,
            detrend=detrend,
            return_onesided=True
        )
        
        # Calcular espectro de fase
        fft_data = fft.fft(signal_data)
        phase_spectrum = np.angle(fft_data[:len(freqs)])
        
        metadata = {
            'dt': dt,
            'window': window,
            'detrend': detrend,
            'nperseg': nperseg,
            'signal_length': len(signal_data)
        }
        
        return SpectralResult(
            frequencies=freqs,
            power_spectrum=psd,
            phase_spectrum=phase_spectrum,
            metadata=metadata
        )
    
    def power_spectrum_2d(self,
                         field_data: np.ndarray,
                         dx: float = 1.0,
                         dy: float = 1.0,
                         window: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula el espectro de potencia 2D radial
        
        Parameters:
        -----------
        field_data : np.ndarray
            Campo 2D
        dx, dy : float
            Resolución espacial
        window : bool
            Aplicar ventana para reducir aliasing
            
        Returns:
        --------
        k_radial : np.ndarray
            Números de onda radiales
        power_radial : np.ndarray
            Espectro de potencia radial
        power_2d : np.ndarray
            Espectro de potencia 2D completo
        """
        ny, nx = field_data.shape
        
        # Aplicar ventana si se solicita
        if window:
            win_x = signal.windows.hann(nx)
            win_y = signal.windows.hann(ny)
            window_2d = np.outer(win_y, win_x)
            field_windowed = field_data * window_2d
        else:
            field_windowed = field_data
        
        # FFT 2D
        if self.use_pyfftw:
            fft_data = pyfftw.interfaces.scipy_fft.fft2(field_windowed)
        else:
            fft_data = fft.fft2(field_windowed)
        
        # Espectro de potencia 2D
        power_2d = np.abs(fft_data)**2
        power_2d = fft.fftshift(power_2d)
        
        # Frecuencias espaciales
        kx = fft.fftfreq(nx, dx)
        ky = fft.fftfreq(ny, dy)
        kx = fft.fftshift(kx)
        ky = fft.fftshift(ky)
        
        # Calcular espectro radial
        k_radial, power_radial = self._radial_average(power_2d, kx, ky)
        
        return k_radial, power_radial, power_2d
    
    def _radial_average(self, 
                       power_2d: np.ndarray,
                       kx: np.ndarray,
                       ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el promedio radial del espectro 2D
        """
        ny, nx = power_2d.shape
        center_x, center_y = nx // 2, ny // 2
        
        # Crear grilla de distancias radiales
        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Definir bins radiales
        r_max = min(center_x, center_y)
        r_bins = np.arange(0, r_max + 1, 1)
        
        # Calcular promedio en cada bin
        power_radial = np.zeros(len(r_bins) - 1)
        k_radial = np.zeros(len(r_bins) - 1)
        
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if np.any(mask):
                power_radial[i] = np.mean(power_2d[mask])
                k_radial[i] = (r_bins[i] + r_bins[i + 1]) / 2
        
        # Convertir a números de onda físicos
        dk = min(abs(kx[1] - kx[0]), abs(ky[1] - ky[0]))
        k_radial = k_radial * dk
        
        return k_radial, power_radial
    
    def spectral_scaling_analysis(self,
                                 frequencies: np.ndarray,
                                 power_spectrum: np.ndarray,
                                 freq_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Analiza el escalamiento espectral (ley de potencia)
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Frecuencias
        power_spectrum : np.ndarray
            Espectro de potencia
        freq_range : tuple, optional
            Rango de frecuencias para el ajuste
            
        Returns:
        --------
        Dict
            Resultados del análisis de escalamiento
        """
        # Filtrar frecuencias válidas (no cero)
        valid_mask = (frequencies > 0) & (power_spectrum > 0)
        freqs = frequencies[valid_mask]
        power = power_spectrum[valid_mask]
        
        if freq_range:
            range_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs = freqs[range_mask]
            power = power[range_mask]
        
        # Ajuste de ley de potencia: P(f) = A * f^(-β)
        def power_law(f, A, beta):
            return A * f**(-beta)
        
        try:
            # Ajuste en escala logarítmica
            log_freqs = np.log10(freqs)
            log_power = np.log10(power)
            
            # Regresión lineal en escala log
            coeffs = np.polyfit(log_freqs, log_power, 1)
            beta = -coeffs[0]  # Exponente espectral
            log_A = coeffs[1]
            A = 10**log_A
            
            # Calcular R²
            log_power_fit = np.polyval(coeffs, log_freqs)
            ss_res = np.sum((log_power - log_power_fit)**2)
            ss_tot = np.sum((log_power - np.mean(log_power))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calcular errores
            residuals = log_power - log_power_fit
            std_error = np.std(residuals)
            
            return {
                'spectral_exponent': beta,
                'amplitude': A,
                'r_squared': r_squared,
                'std_error': std_error,
                'freq_range': (freqs.min(), freqs.max()),
                'n_points': len(freqs)
            }
            
        except Exception as e:
            warnings.warn(f"Error en análisis de escalamiento: {e}")
            return {
                'spectral_exponent': np.nan,
                'amplitude': np.nan,
                'r_squared': np.nan,
                'std_error': np.nan,
                'freq_range': None,
                'n_points': 0
            }
    
    def coherence_analysis(self,
                          signal1: np.ndarray,
                          signal2: np.ndarray,
                          dt: float = 1.0,
                          nperseg: Optional[int] = None) -> SpectralResult:
        """
        Análisis de coherencia entre dos señales
        
        Parameters:
        -----------
        signal1, signal2 : np.ndarray
            Señales a comparar
        dt : float
            Paso temporal
        nperseg : int, optional
            Longitud de segmento
            
        Returns:
        --------
        SpectralResult
            Resultado incluyendo coherencia y espectro cruzado
        """
        if nperseg is None:
            nperseg = len(signal1) // 8
        
        # Coherencia
        freqs, coherence = signal.coherence(
            signal1, signal2, 
            fs=1/dt, 
            nperseg=nperseg
        )
        
        # Espectro cruzado
        _, cross_spectrum = signal.csd(
            signal1, signal2,
            fs=1/dt,
            nperseg=nperseg
        )
        
        # Espectros individuales
        _, psd1 = signal.welch(signal1, fs=1/dt, nperseg=nperseg)
        _, psd2 = signal.welch(signal2, fs=1/dt, nperseg=nperseg)
        
        # Fase del espectro cruzado
        phase_spectrum = np.angle(cross_spectrum)
        
        metadata = {
            'dt': dt,
            'nperseg': nperseg,
            'signal1_length': len(signal1),
            'signal2_length': len(signal2)
        }
        
        return SpectralResult(
            frequencies=freqs,
            power_spectrum=(psd1 + psd2) / 2,  # Promedio de espectros
            phase_spectrum=phase_spectrum,
            coherence=coherence,
            cross_spectrum=cross_spectrum,
            metadata=metadata
        )
    
    def time_frequency_analysis(self,
                               signal_data: np.ndarray,
                               dt: float = 1.0,
                               window: str = 'hann',
                               nperseg: Optional[int] = None,
                               noverlap: Optional[int] = None) -> Dict:
        """
        Análisis tiempo-frecuencia usando espectrograma
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Señal temporal
        dt : float
            Paso temporal
        window : str
            Tipo de ventana
        nperseg : int, optional
            Longitud de segmento
        noverlap : int, optional
            Superposición entre segmentos
            
        Returns:
        --------
        Dict
            Espectrograma y metadatos
        """
        if nperseg is None:
            nperseg = len(signal_data) // 16
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, times, Sxx = signal.spectrogram(
            signal_data,
            fs=1/dt,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        return {
            'frequencies': freqs,
            'times': times,
            'spectrogram': Sxx,
            'dt': dt,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap
        }
    
    def fractional_spectrum(self,
                           signal_data: np.ndarray,
                           alpha: float,
                           dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Espectro asociado al operador fraccionario (-Δ)^(α/2)
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Señal o campo
        alpha : float
            Orden fraccionario
        dt : float
            Resolución temporal/espacial
            
        Returns:
        --------
        frequencies : np.ndarray
            Frecuencias
        fractional_spectrum : np.ndarray
            Espectro del operador fraccionario
        """
        # FFT de la señal
        fft_data = fft.fft(signal_data)
        frequencies = fft.fftfreq(len(signal_data), dt)
        
        # Espectro del operador fraccionario: |k|^α
        k_alpha_spectrum = np.abs(2 * np.pi * frequencies)**alpha
        
        # Espectro de la señal con operador fraccionario
        fractional_spectrum = np.abs(fft_data)**2 * k_alpha_spectrum
        
        return frequencies, fractional_spectrum
    
    def save_spectral_analysis(self,
                              results: Dict,
                              filename: str,
                              format: str = 'npz') -> None:
        """
        Guarda resultados del análisis espectral
        
        Parameters:
        -----------
        results : Dict
            Resultados del análisis
        filename : str
            Nombre del archivo
        format : str
            Formato de guardado ('npz', 'mat', 'hdf5')
        """
        if format == 'npz':
            np.savez_compressed(filename, **results)
        elif format == 'mat':
            from scipy.io import savemat
            savemat(filename, results)
        elif format == 'hdf5':
            import h5py
            with h5py.File(filename, 'w') as f:
                for key, value in results.items():
                    f.create_dataset(key, data=value)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def plot_spectral_results(self,
                             result: SpectralResult,
                             log_scale: bool = True,
                             show_phase: bool = True,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualiza resultados del análisis espectral
        
        Parameters:
        -----------
        result : SpectralResult
            Resultado del análisis
        log_scale : bool
            Usar escala logarítmica
        show_phase : bool
            Mostrar espectro de fase
        figsize : tuple
            Tamaño de la figura
            
        Returns:
        --------
        plt.Figure
            Figura de matplotlib
        """
        if show_phase:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        
        # Espectro de potencia
        if log_scale:
            ax1.loglog(result.frequencies[1:], result.power_spectrum[1:])
            ax1.set_xlabel('Frecuencia (Hz)')
            ax1.set_ylabel('Densidad Espectral de Potencia')
        else:
            ax1.plot(result.frequencies, result.power_spectrum)
            ax1.set_xlabel('Frecuencia (Hz)')
            ax1.set_ylabel('Densidad Espectral de Potencia')
        
        ax1.set_title('Espectro de Potencia')
        ax1.grid(True, alpha=0.3)
        
        # Espectro de fase
        if show_phase:
            ax2.plot(result.frequencies, result.phase_spectrum)
            ax2.set_xlabel('Frecuencia (Hz)')
            ax2.set_ylabel('Fase (radianes)')
            ax2.set_title('Espectro de Fase')
            ax2.grid(True, alpha=0.3)
        
        # Coherencia si está disponible
        if result.coherence is not None and show_phase:
            ax3 = ax2.twinx()
            ax3.plot(result.frequencies, result.coherence, 'r--', alpha=0.7)
            ax3.set_ylabel('Coherencia', color='r')
            ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig


def analyze_mfsu_spectrum(field_data: np.ndarray,
                         time_series: np.ndarray,
                         dt: float,
                         dx: float,
                         alpha: float = 0.5,
                         save_results: bool = False,
                         output_dir: str = './') -> Dict:
    """
    Análisis espectral completo para soluciones MFSU
    
    Parameters:
    -----------
    field_data : np.ndarray
        Campo espacial 2D
    time_series : np.ndarray
        Serie temporal en un punto
    dt : float
        Paso temporal
    dx : float
        Paso espacial
    alpha : float
        Parámetro fraccionario
    save_results : bool
        Guardar resultados
    output_dir : str
        Directorio de salida
        
    Returns:
    --------
    Dict
        Resultados completos del análisis espectral
    """
    analyzer = SpectralAnalyzer()
    
    # Análisis temporal
    temporal_result = analyzer.power_spectrum_1d(time_series, dt)
    
    # Análisis espacial
    k_radial, power_radial, power_2d = analyzer.power_spectrum_2d(field_data, dx, dx)
    
    # Análisis de escalamiento
    scaling_temporal = analyzer.spectral_scaling_analysis(
        temporal_result.frequencies, 
        temporal_result.power_spectrum
    )
    
    scaling_spatial = analyzer.spectral_scaling_analysis(
        k_radial, 
        power_radial
    )
    
    # Espectro fraccionario
    freqs_frac, frac_spectrum = analyzer.fractional_spectrum(
        time_series, alpha, dt
    )
    
    results = {
        'temporal': {
            'frequencies': temporal_result.frequencies,
            'power_spectrum': temporal_result.power_spectrum,
            'phase_spectrum': temporal_result.phase_spectrum,
            'scaling': scaling_temporal
        },
        'spatial': {
            'k_radial': k_radial,
            'power_radial': power_radial,
            'power_2d': power_2d,
            'scaling': scaling_spatial
        },
        'fractional': {
            'frequencies': freqs_frac,
            'spectrum': frac_spectrum
        },
        'parameters': {
            'dt': dt,
            'dx': dx,
            'alpha': alpha
        }
    }
    
    if save_results:
        analyzer.save_spectral_analysis(
            results, 
            f"{output_dir}/mfsu_spectral_analysis.npz"
        )
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    print("MFSU Spectral Analysis Module")
    print("=============================")
    
    # Crear datos de prueba
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    
    # Señal de prueba con ruido fractal
    signal_test = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    signal_test += np.random.normal(0, 0.1, len(t))
    
    # Campo 2D de prueba
    x = np.linspace(0, 10, 64)
    y = np.linspace(0, 10, 64)
    X, Y = np.meshgrid(x, y)
    field_test = np.sin(2 * np.pi * X / 5) * np.cos(2 * np.pi * Y / 3)
    
    # Análisis completo
    results = analyze_mfsu_spectrum(
        field_test, signal_test, dt, x[1]-x[0], alpha=0.7
    )
    
    print(f"Exponente espectral temporal: {results['temporal']['scaling']['spectral_exponent']:.3f}")
    print(f"Exponente espectral espacial: {results['spatial']['scaling']['spectral_exponent']:.3f}")
    print("Análisis espectral completado exitosamente!")
