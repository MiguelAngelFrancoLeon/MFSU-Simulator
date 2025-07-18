"""
Módulo de Análisis Fractal para el Simulador MFSU
================================================

Este módulo implementa herramientas para el análisis de dimensión fractal
y propiedades fractales de las soluciones de la ecuación MFSU:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Incluye métodos para:
- Cálculo de dimensión fractal (box-counting, Hausdorff)
- Análisis de autosimilaridad
- Detección de patrones fractales
- Correlaciones de largo alcance
- Análisis multiescala

Author: MFSU Development Team
Date: 2025
"""

import numpy as np
import scipy.signal as signal
from scipy import optimize
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
import warnings
from numba import jit, prange


class FractalAnalyzer:
    """
    Clase principal para análisis fractal de campos ψ(x,t) generados por MFSU.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el analizador fractal.
        
        Parameters:
        -----------
        config : dict, optional
            Configuración del analizador con parámetros como:
            - box_sizes: rangos para box-counting
            - hurst_window: ventana para análisis de Hurst
            - correlation_max_lag: lag máximo para correlaciones
        """
        self.config = config or self._default_config()
        self.results_cache = {}
        
    def _default_config(self) -> Dict:
        """Configuración por defecto del analizador."""
        return {
            'box_sizes': np.logspace(0, 2, 20).astype(int),
            'hurst_window': 1024,
            'correlation_max_lag': 100,
            'detrending_order': 1,
            'confidence_level': 0.95,
            'min_points_fit': 5
        }
    
    def box_counting_dimension(self, field: np.ndarray, 
                             box_sizes: Optional[np.ndarray] = None) -> Dict:
        """
        Calcula la dimensión fractal usando el método box-counting.
        
        Parameters:
        -----------
        field : np.ndarray
            Campo 2D o 3D a analizar (puede ser |ψ|² o Re(ψ))
        box_sizes : np.ndarray, optional
            Tamaños de caja para el conteo
            
        Returns:
        --------
        dict : Resultados con dimensión fractal, coeficientes de ajuste y R²
        """
        if box_sizes is None:
            box_sizes = self.config['box_sizes']
            
        # Asegurar que el campo es 2D para box-counting
        if field.ndim > 2:
            field = np.mean(field, axis=0)
            
        # Normalizar el campo
        field_norm = (field - field.min()) / (field.max() - field.min())
        
        # Binarizar usando umbral de Otsu
        threshold = self._otsu_threshold(field_norm)
        binary_field = field_norm > threshold
        
        counts = []
        valid_sizes = []
        
        for box_size in box_sizes:
            if box_size >= min(binary_field.shape):
                continue
                
            count = self._count_boxes(binary_field, box_size)
            if count > 0:
                counts.append(count)
                valid_sizes.append(box_size)
        
        if len(counts) < self.config['min_points_fit']:
            raise ValueError("Insuficientes puntos para ajuste confiable")
        
        # Ajuste lineal en escala log-log
        log_sizes = np.log(valid_sizes)
        log_counts = np.log(counts)
        
        # Regresión robusta
        coeffs, r_squared, std_err = self._robust_linear_fit(log_sizes, log_counts)
        
        # La dimensión fractal es -pendiente
        fractal_dim = -coeffs[0]
        
        return {
            'fractal_dimension': fractal_dim,
            'dimension_error': std_err[0],
            'r_squared': r_squared,
            'box_sizes': valid_sizes,
            'counts': counts,
            'fit_coefficients': coeffs,
            'threshold_used': threshold
        }
    
    @staticmethod
    @jit(nopython=True)
    def _count_boxes(binary_field: np.ndarray, box_size: int) -> int:
        """Cuenta cajas ocupadas para un tamaño dado."""
        rows, cols = binary_field.shape
        count = 0
        
        for i in range(0, rows, box_size):
            for j in range(0, cols, box_size):
                # Verificar si la caja contiene al menos un pixel activo
                box_end_i = min(i + box_size, rows)
                box_end_j = min(j + box_size, cols)
                
                box_sum = 0
                for ii in range(i, box_end_i):
                    for jj in range(j, box_end_j):
                        box_sum += binary_field[ii, jj]
                        if box_sum > 0:  # Early exit si ya encontramos uno
                            count += 1
                            break
                    if box_sum > 0:
                        break
        
        return count
    
    def hurst_exponent(self, time_series: np.ndarray, 
                       method: str = 'dfa') -> Dict:
        """
        Calcula el exponente de Hurst usando diferentes métodos.
        
        Parameters:
        -----------
        time_series : np.ndarray
            Serie temporal 1D
        method : str
            Método: 'dfa' (Detrended Fluctuation Analysis), 
                   'rs' (R/S analysis), 'variogram'
            
        Returns:
        --------
        dict : Exponente de Hurst y estadísticas de ajuste
        """
        if method == 'dfa':
            return self._hurst_dfa(time_series)
        elif method == 'rs':
            return self._hurst_rs(time_series)
        elif method == 'variogram':
            return self._hurst_variogram(time_series)
        else:
            raise ValueError(f"Método {method} no soportado")
    
    def _hurst_dfa(self, time_series: np.ndarray) -> Dict:
        """Detrended Fluctuation Analysis para exponente de Hurst."""
        N = len(time_series)
        
        # Integrar la serie
        y = np.cumsum(time_series - np.mean(time_series))
        
        # Escalas a analizar
        scales = np.unique(np.logspace(1, np.log10(N//4), 20).astype(int))
        fluctuations = []
        
        for scale in scales:
            # Dividir en ventanas
            n_windows = N // scale
            if n_windows < 2:
                continue
                
            local_fluct = []
            
            for i in range(n_windows):
                start_idx = i * scale
                end_idx = (i + 1) * scale
                y_window = y[start_idx:end_idx]
                
                # Ajuste polinomial local
                x_window = np.arange(len(y_window))
                coeffs = np.polyfit(x_window, y_window, 
                                  self.config['detrending_order'])
                y_fit = np.polyval(coeffs, x_window)
                
                # Fluctuación local
                local_fluct.append(np.sqrt(np.mean((y_window - y_fit)**2)))
            
            fluctuations.append(np.mean(local_fluct))
        
        # Ajuste log-log
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(fluctuations)
        
        coeffs, r_squared, std_err = self._robust_linear_fit(log_scales, log_fluct)
        hurst = coeffs[0]  # La pendiente es el exponente de Hurst
        
        return {
            'hurst_exponent': hurst,
            'hurst_error': std_err[0],
            'r_squared': r_squared,
            'scales': scales[:len(fluctuations)],
            'fluctuations': fluctuations,
            'method': 'DFA'
        }
    
    def _hurst_rs(self, time_series: np.ndarray) -> Dict:
        """R/S Analysis para exponente de Hurst."""
        N = len(time_series)
        scales = np.unique(np.logspace(1, np.log10(N//2), 15).astype(int))
        rs_values = []
        
        for scale in scales:
            n_windows = N // scale
            if n_windows < 2:
                continue
                
            rs_window = []
            
            for i in range(n_windows):
                start_idx = i * scale
                end_idx = (i + 1) * scale
                window = time_series[start_idx:end_idx]
                
                # Desviaciones acumuladas
                mean_window = np.mean(window)
                deviations = np.cumsum(window - mean_window)
                
                # Rango
                R = np.max(deviations) - np.min(deviations)
                
                # Desviación estándar
                S = np.std(window)
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        # Ajuste log-log
        log_scales = np.log(scales[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        coeffs, r_squared, std_err = self._robust_linear_fit(log_scales, log_rs)
        hurst = coeffs[0]
        
        return {
            'hurst_exponent': hurst,
            'hurst_error': std_err[0],
            'r_squared': r_squared,
            'scales': scales[:len(rs_values)],
            'rs_values': rs_values,
            'method': 'R/S'
        }
    
    def multifractal_spectrum(self, field: np.ndarray, 
                            q_range: np.ndarray = None) -> Dict:
        """
        Calcula el espectro multifractal usando MFDFA.
        
        Parameters:
        -----------
        field : np.ndarray
            Campo a analizar
        q_range : np.ndarray
            Rango de momentos q para análisis multifractal
            
        Returns:
        --------
        dict : Espectro multifractal y parámetros característicos
        """
        if q_range is None:
            q_range = np.linspace(-5, 5, 21)
        
        # Convertir a serie 1D si es necesario
        if field.ndim > 1:
            field_1d = field.flatten()
        else:
            field_1d = field.copy()
        
        N = len(field_1d)
        scales = np.unique(np.logspace(1, np.log10(N//10), 15).astype(int))
        
        # Integrar la serie
        y = np.cumsum(field_1d - np.mean(field_1d))
        
        tau_q = []  # Función de escalamiento τ(q)
        
        for q in q_range:
            fluctuations_q = []
            
            for scale in scales:
                n_windows = N // scale
                if n_windows < 2:
                    continue
                
                local_fluct = []
                
                for i in range(n_windows):
                    start_idx = i * scale
                    end_idx = (i + 1) * scale
                    y_window = y[start_idx:end_idx]
                    
                    # Detrending polinomial
                    x_window = np.arange(len(y_window))
                    coeffs = np.polyfit(x_window, y_window, 1)
                    y_fit = np.polyval(coeffs, x_window)
                    
                    # Fluctuación local
                    fluct = np.sqrt(np.mean((y_window - y_fit)**2))
                    if fluct > 0:
                        local_fluct.append(fluct)
                
                if local_fluct:
                    # Promedio generalizado de orden q
                    if q != 0:
                        fq = np.mean(np.array(local_fluct)**q)**(1/q)
                    else:
                        fq = np.exp(np.mean(np.log(local_fluct)))
                    
                    fluctuations_q.append(fq)
            
            # Ajuste log-log para obtener h(q)
            if len(fluctuations_q) >= self.config['min_points_fit']:
                log_scales_q = np.log(scales[:len(fluctuations_q)])
                log_fluct_q = np.log(fluctuations_q)
                
                coeffs, _, _ = self._robust_linear_fit(log_scales_q, log_fluct_q)
                h_q = coeffs[0]
                tau_q.append(q * h_q - 1)
            else:
                tau_q.append(np.nan)
        
        # Calcular espectro f(α)
        tau_q = np.array(tau_q)
        valid_idx = ~np.isnan(tau_q)
        
        if np.sum(valid_idx) < 3:
            raise ValueError("Insuficientes datos válidos para espectro multifractal")
        
        # Derivada numérica para obtener α(q)
        alpha = np.gradient(tau_q[valid_idx])
        f_alpha = q_range[valid_idx] * alpha - tau_q[valid_idx]
        
        # Parámetros característicos
        alpha_0 = alpha[np.argmax(f_alpha)]  # α en el máximo
        width = np.max(alpha) - np.min(alpha)  # Ancho del espectro
        asymmetry = self._calculate_asymmetry(alpha, f_alpha)
        
        return {
            'q_range': q_range[valid_idx],
            'tau_q': tau_q[valid_idx],
            'alpha': alpha,
            'f_alpha': f_alpha,
            'alpha_0': alpha_0,
            'spectrum_width': width,
            'asymmetry': asymmetry,
            'is_multifractal': width > 0.1  # Criterio simple
        }
    
    def correlation_dimension(self, field: np.ndarray, 
                            embedding_dim: int = 5) -> Dict:
        """
        Calcula la dimensión de correlación usando el algoritmo de Grassberger-Procaccia.
        
        Parameters:
        -----------
        field : np.ndarray
            Campo o serie temporal
        embedding_dim : int
            Dimensión de embebido para reconstrucción del espacio de fases
            
        Returns:
        --------
        dict : Dimensión de correlación y estadísticas
        """
        # Convertir a serie 1D
        if field.ndim > 1:
            series = field.flatten()
        else:
            series = field.copy()
        
        # Reconstrucción del espacio de fases
        N = len(series)
        tau = self._optimal_delay(series)  # Delay óptimo
        
        # Crear vectores de estado embebidos
        M = N - (embedding_dim - 1) * tau
        embedded = np.zeros((M, embedding_dim))
        
        for i in range(M):
            for j in range(embedding_dim):
                embedded[i, j] = series[i + j * tau]
        
        # Rangos de distancia
        distances = pdist(embedded)
        r_min, r_max = np.percentile(distances, [5, 95])
        r_range = np.logspace(np.log10(r_min), np.log10(r_max), 20)
        
        # Función de correlación C(r)
        correlation_sums = []
        
        for r in r_range:
            count = np.sum(distances < r)
            total_pairs = len(distances)
            correlation_sums.append(count / total_pairs)
        
        # Ajuste log-log para obtener dimensión
        valid_idx = np.array(correlation_sums) > 0
        if np.sum(valid_idx) < self.config['min_points_fit']:
            raise ValueError("Insuficientes puntos para cálculo de dimensión de correlación")
        
        log_r = np.log(r_range[valid_idx])
        log_c = np.log(np.array(correlation_sums)[valid_idx])
        
        coeffs, r_squared, std_err = self._robust_linear_fit(log_r, log_c)
        correlation_dim = coeffs[0]
        
        return {
            'correlation_dimension': correlation_dim,
            'dimension_error': std_err[0],
            'r_squared': r_squared,
            'embedding_dimension': embedding_dim,
            'optimal_delay': tau,
            'r_range': r_range[valid_idx],
            'correlation_sums': np.array(correlation_sums)[valid_idx]
        }
    
    def lacunarity_analysis(self, field: np.ndarray, 
                          box_sizes: Optional[np.ndarray] = None) -> Dict:
        """
        Calcula la lacunaridad para caracterizar la heterogeneidad espacial.
        
        Parameters:
        -----------
        field : np.ndarray
            Campo 2D a analizar
        box_sizes : np.ndarray, optional
            Tamaños de caja para análisis
            
        Returns:
        --------
        dict : Lacunaridad por escala y parámetros derivados
        """
        if field.ndim != 2:
            raise ValueError("Lacunaridad requiere campo 2D")
        
        if box_sizes is None:
            max_size = min(field.shape) // 4
            box_sizes = np.arange(2, max_size, max(1, max_size//20))
        
        lacunarities = []
        valid_sizes = []
        
        for box_size in box_sizes:
            if box_size >= min(field.shape):
                continue
            
            # Obtener todas las sumas de cajas posibles
            box_sums = self._sliding_box_sums(field, box_size)
            
            if len(box_sums) > 1:
                # Lacunaridad = (momento segundo)/(momento primero)² 
                mean_sum = np.mean(box_sums)
                var_sum = np.var(box_sums)
                
                if mean_sum > 0:
                    lacunarity = 1 + var_sum / (mean_sum**2)
                    lacunarities.append(lacunarity)
                    valid_sizes.append(box_size)
        
        return {
            'box_sizes': valid_sizes,
            'lacunarity': lacunarities,
            'lacunarity_slope': self._calculate_slope(valid_sizes, lacunarities),
            'mean_lacunarity': np.mean(lacunarities) if lacunarities else 0
        }
    
    def _sliding_box_sums(self, field: np.ndarray, box_size: int) -> np.ndarray:
        """Calcula sumas de cajas deslizantes."""
        rows, cols = field.shape
        sums = []
        
        for i in range(rows - box_size + 1):
            for j in range(cols - box_size + 1):
                box_sum = np.sum(field[i:i+box_size, j:j+box_size])
                sums.append(box_sum)
        
        return np.array(sums)
    
    def _otsu_threshold(self, image: np.ndarray) -> float:
        """Calcula umbral de Otsu para binarización."""
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
        hist = hist.astype(float)
        
        # Normalizar histograma
        hist /= hist.sum()
        
        # Centros de bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Buscar umbral óptimo
        max_variance = 0
        threshold = 0
        
        for i in range(1, len(hist)):
            # Pesos de las clases
            w1 = np.sum(hist[:i])
            w2 = np.sum(hist[i:])
            
            if w1 == 0 or w2 == 0:
                continue
            
            # Medias de las clases
            mu1 = np.sum(bin_centers[:i] * hist[:i]) / w1
            mu2 = np.sum(bin_centers[i:] * hist[i:]) / w2
            
            # Varianza entre clases
            variance = w1 * w2 * (mu1 - mu2)**2
            
            if variance > max_variance:
                max_variance = variance
                threshold = bin_centers[i]
        
        return threshold
    
    def _robust_linear_fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Ajuste lineal robusto con estadísticas."""
        # Eliminar NaN e Inf
        valid_idx = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[valid_idx], y[valid_idx]
        
        if len(x_clean) < 2:
            raise ValueError("Insuficientes puntos válidos para ajuste")
        
        # Ajuste por mínimos cuadrados
        coeffs = np.polyfit(x_clean, y_clean, 1)
        y_pred = np.polyval(coeffs, x_clean)
        
        # R²
        ss_res = np.sum((y_clean - y_pred)**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Errores estándar
        mse = ss_res / (len(x_clean) - 2)
        x_var = np.sum((x_clean - np.mean(x_clean))**2)
        std_err = np.sqrt([mse / x_var, mse * (1/len(x_clean) + np.mean(x_clean)**2/x_var)])
        
        return coeffs, r_squared, std_err
    
    def _optimal_delay(self, series: np.ndarray, max_delay: int = 50) -> int:
        """Calcula delay óptimo usando autocorrelación."""
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalizar
        
        # Primer cero de la autocorrelación
        zero_crossings = np.where(np.diff(np.signbit(autocorr)))[0]
        
        if len(zero_crossings) > 0:
            return min(zero_crossings[0], max_delay)
        else:
            return max_delay // 4  # Valor por defecto
    
    def _calculate_asymmetry(self, alpha: np.ndarray, f_alpha: np.ndarray) -> float:
        """Calcula asimetría del espectro multifractal."""
        peak_idx = np.argmax(f_alpha)
        
        if peak_idx == 0 or peak_idx == len(alpha) - 1:
            return 0
        
        left_width = alpha[peak_idx] - alpha[0]
        right_width = alpha[-1] - alpha[peak_idx]
        
        if left_width + right_width == 0:
            return 0
        
        return (right_width - left_width) / (right_width + left_width)
    
    def _calculate_slope(self, x: List, y: List) -> float:
        """Calcula pendiente de ajuste lineal."""
        if len(x) < 2:
            return 0
        
        try:
            coeffs, _, _ = self._robust_linear_fit(np.array(x), np.array(y))
            return coeffs[0]
        except:
            return 0
    
    def comprehensive_analysis(self, field: np.ndarray, 
                             time_series: Optional[np.ndarray] = None) -> Dict:
        """
        Realiza análisis fractal completo del campo MFSU.
        
        Parameters:
        -----------
        field : np.ndarray
            Campo espacial ψ(x) o ψ(x,y)
        time_series : np.ndarray, optional
            Serie temporal ψ(t) en un punto específico
            
        Returns:
        --------
        dict : Resultados completos del análisis fractal
        """
        results = {}
        
        # Análisis de dimensión fractal espacial
        try:
            results['box_counting'] = self.box_counting_dimension(field)
        except Exception as e:
            results['box_counting'] = {'error': str(e)}
        
        # Análisis de lacunaridad
        if field.ndim == 2:
            try:
                results['lacunarity'] = self.lacunarity_analysis(field)
            except Exception as e:
                results['lacunarity'] = {'error': str(e)}
        
        # Análisis de correlación espacial
        try:
            results['correlation_dimension'] = self.correlation_dimension(field)
        except Exception as e:
            results['correlation_dimension'] = {'error': str(e)}
        
        # Análisis multifractal
        try:
            results['multifractal'] = self.multifractal_spectrum(field)
        except Exception as e:
            results['multifractal'] = {'error': str(e)}
        
        # Análisis temporal si se proporciona serie
        if time_series is not None:
            try:
                results['hurst_dfa'] = self.hurst_exponent(time_series, 'dfa')
                results['hurst_rs'] = self.hurst_exponent(time_series, 'rs')
            except Exception as e:
                results['temporal_analysis'] = {'error': str(e)}
        
        return results


def plot_fractal_results(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualiza resultados del análisis fractal.
    
    Parameters:
    -----------
    results : dict
        Resultados del análisis fractal
    save_path : str, optional
        Ruta para guardar las figuras
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Box-counting
    if 'box_counting' in results and 'error' not in results['box_counting']:
        bc = results['box_counting']
        axes[0].loglog(bc['box_sizes'], bc['counts'], 'bo-', alpha=0.7)
        axes[0].set_xlabel('Box Size')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Box-Counting D={bc["fractal_dimension"]:.3f}')
        axes[0].grid(True, alpha=0.3)
    
    # Hurst DFA
    if 'hurst_dfa' in results and 'error' not in results['hurst_dfa']:
        hurst = results['hurst_dfa']
        axes[1].loglog(hurst['scales'], hurst['fluctuations'], 'ro-', alpha=0.7)
        axes[1].set_xlabel('Scale')
        axes[1].set_ylabel('Fluctuation')
        axes[1].set_title(f'DFA H={hurst["hurst_exponent"]:.3f}')
        axes[1].grid(True, alpha=0.3)
    
    # Multifractal spectrum
    if 'multifractal' in results and 'error' not in results['multifractal']:
        mf = results['multifractal']
        axes[2].plot(mf['alpha'], mf['f_alpha'], 'go-', alpha=0.7)
        axes[2].set_xlabel('α')
        axes[2].set_ylabel('f(α)')
        axes[2].set_title(f'Multifractal W={mf["spectrum_width"]:.3f}')
        axes[2].grid(True, alpha=0.3)
    
    # Lacunarity
    if 'lacunarity' in results and 'error' not in results['lacunarity']:
        lac = results['lacunarity']
        axes[3].semilogx(lac['box_sizes'], lac['lacunarity'], 'mo-', alpha=0.7)
        axes[3].set_xlabel('Box Size')
        axes[3].set_ylabel('Lacunarity')
        axes[3].set_title(f'Lacunarity μ={lac["mean_lacunarity"]:.3f}')
        axes[3].grid(True, alpha=0.3)
    
    # Correlation dimension
    if 'correlation_dimension' in results and 'error' not in results['correlation_dimension']:
        cd = results['correlation_dimension']
        axes[4].loglog(cd['r_range'], cd['correlation_sums'], 'co-', alpha=0.7)
        axes[4].set_xlabel('Distance r')
        axes[4].set_ylabel('C(r)')
        axes[4].set_title(f'Correlation D={cd["correlation_dimension"]:.3f}')
        axes[4].grid(True, alpha=0.3)
    
    # Resumen de dimensiones
    dimensions = []
    labels = []
    
    for key, name in [('box_counting', 'Box-Count'), ('correlation_dimension', 'Correlation'), 
                     ('hurst_dfa', 'Hurst DFA'), ('hurst_rs', 'Hurst R/S')]:
        if key in results and 'error' not in results[key]:
            if key.startswith('hurst'):
                dimensions.append(results[key]['hurst_exponent'])
            else:
                dim_key = 'fractal_dimension' if key == 'box_counting' else 'correlation_dimension'
                dimensions.append(results[key][dim_key])
            labels.append(name)
    
    if dimensions:
        axes[5].bar(range(len(dimensions)), dimensions, alpha=0.7, 
                   color=['blue', 'red', 'green', 'orange'][:len(dimensions)])
        axes[5].set_xticks(range(len(dimensions)))
        axes[5].set_xticklabels(labels, rotation=45)
        axes[5].set_ylabel('Dimension/Exponent')
        axes[5].set_title('Fractal Dimensions Summary')
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


class FractalValidation:
    """
    Clase para validar y verificar resultados del análisis fractal.
    """
    
    def __init__(self):
        self.known_fractals = {
            'cantor_set': {'box_counting': 0.631, 'hausdorff': 0.631},
            'sierpinski_triangle': {'box_counting': 1.585, 'hausdorff': 1.585},
            'koch_curve': {'box_counting': 1.262, 'hausdorff': 1.262},
            'brownian_motion': {'hurst': 0.5, 'correlation': 2.0}
        }
    
    def validate_dimension(self, computed_dim: float, 
                         expected_dim: float, 
                         tolerance: float = 0.1) -> Dict:
        """
        Valida dimensión fractal calculada contra valor esperado.
        
        Parameters:
        -----------
        computed_dim : float
            Dimensión calculada
        expected_dim : float
            Dimensión esperada
        tolerance : float
            Tolerancia para validación
            
        Returns:
        --------
        dict : Resultado de validación
        """
        error = abs(computed_dim - expected_dim)
        relative_error = error / expected_dim if expected_dim != 0 else float('inf')
        
        is_valid = error <= tolerance
        
        return {
            'is_valid': is_valid,
            'computed_dimension': computed_dim,
            'expected_dimension': expected_dim,
            'absolute_error': error,
            'relative_error': relative_error,
            'tolerance': tolerance
        }
    
    def generate_test_fractal(self, fractal_type: str, 
                            size: int = 256, 
                            **kwargs) -> np.ndarray:
        """
        Genera fractales conocidos para pruebas.
        
        Parameters:
        -----------
        fractal_type : str
            Tipo de fractal: 'cantor', 'sierpinski', 'mandelbrot', 'fbm'
        size : int
            Tamaño del fractal generado
        **kwargs : dict
            Parámetros específicos del fractal
            
        Returns:
        --------
        np.ndarray : Fractal generado
        """
        if fractal_type == 'cantor':
            return self._generate_cantor_set(size, kwargs.get('iterations', 5))
        elif fractal_type == 'sierpinski':
            return self._generate_sierpinski_triangle(size, kwargs.get('iterations', 6))
        elif fractal_type == 'mandelbrot':
            return self._generate_mandelbrot_set(size, kwargs.get('max_iter', 100))
        elif fractal_type == 'fbm':
            return self._generate_fbm(size, kwargs.get('hurst', 0.7))
        else:
            raise ValueError(f"Fractal tipo {fractal_type} no soportado")
    
    def _generate_cantor_set(self, size: int, iterations: int) -> np.ndarray:
        """Genera conjunto de Cantor 1D."""
        line = np.ones(size)
        
        for i in range(iterations):
            new_line = line.copy()
            segment_size = size // (3**i)
            
            for j in range(3**i):
                start = j * segment_size
                middle_start = start + segment_size // 3
                middle_end = start + 2 * segment_size // 3
                
                if middle_end < size:
                    new_line[middle_start:middle_end] = 0
            
            line = new_line
        
        return line
    
    def _generate_sierpinski_triangle(self, size: int, iterations: int) -> np.ndarray:
        """Genera triángulo de Sierpinski."""
        triangle = np.zeros((size, size))
        
        # Triángulo inicial
        for i in range(size):
            for j in range(i + 1):
                triangle[size - 1 - i, size//2 - j + i//2] = 1
        
        # Aplicar regla de Sierpinski
        for _ in range(iterations):
            new_triangle = triangle.copy()
            rows, cols = triangle.shape
            
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    neighbors = (triangle[i-1, j-1] + triangle[i-1, j] + 
                               triangle[i-1, j+1])
                    if neighbors == 1:
                        new_triangle[i, j] = 1
                    elif neighbors == 2:
                        new_triangle[i, j] = 0
            
            triangle = new_triangle
        
        return triangle
    
    def _generate_mandelbrot_set(self, size: int, max_iter: int) -> np.ndarray:
        """Genera conjunto de Mandelbrot."""
        x = np.linspace(-2.5, 1.5, size)
        y = np.linspace(-2.0, 2.0, size)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        Z = np.zeros_like(C)
        mandelbrot = np.zeros(C.shape)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            mandelbrot[mask] = i
        
        return mandelbrot
    
    def _generate_fbm(self, size: int, hurst: float) -> np.ndarray:
        """Genera movimiento Browniano fraccionario."""
        # Método de Cholesky para fBm
        n = size
        t = np.arange(n)
        
        # Matriz de covarianza
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = 0.5 * (abs(i)**(2*hurst) + 
                                         abs(j)**(2*hurst) - 
                                         abs(i-j)**(2*hurst))
        
        # Descomposición de Cholesky
        L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(n))
        
        # Generar ruido blanco
        white_noise = np.random.randn(n)
        
        # Generar fBm
        fbm = L @ white_noise
        
        return fbm
    
    def benchmark_performance(self, analyzer: FractalAnalyzer, 
                            field_sizes: List[int] = None) -> Dict:
        """
        Evalúa rendimiento del analizador fractal.
        
        Parameters:
        -----------
        analyzer : FractalAnalyzer
            Instancia del analizador
        field_sizes : list
            Tamaños de campo para benchmarking
            
        Returns:
        --------
        dict : Resultados de rendimiento
        """
        import time
        
        if field_sizes is None:
            field_sizes = [64, 128, 256, 512]
        
        results = {
            'field_sizes': field_sizes,
            'box_counting_times': [],
            'hurst_times': [],
            'multifractal_times': [],
            'total_times': []
        }
        
        for size in field_sizes:
            print(f"Benchmarking tamaño {size}x{size}...")
            
            # Generar campo de prueba
            test_field = self.generate_test_fractal('mandelbrot', size)
            test_series = self.generate_test_fractal('fbm', size)
            
            total_start = time.time()
            
            # Box-counting
            start_time = time.time()
            try:
                analyzer.box_counting_dimension(test_field)
                bc_time = time.time() - start_time
            except:
                bc_time = -1
            results['box_counting_times'].append(bc_time)
            
            # Hurst exponent
            start_time = time.time()
            try:
                analyzer.hurst_exponent(test_series)
                hurst_time = time.time() - start_time
            except:
                hurst_time = -1
            results['hurst_times'].append(hurst_time)
            
            # Multifractal
            start_time = time.time()
            try:
                analyzer.multifractal_spectrum(test_field)
                mf_time = time.time() - start_time
            except:
                mf_time = -1
            results['multifractal_times'].append(mf_time)
            
            total_time = time.time() - total_start
            results['total_times'].append(total_time)
        
        return results


def analyze_mfsu_field(field_data: np.ndarray, 
                      time_series: Optional[np.ndarray] = None,
                      config: Optional[Dict] = None,
                      save_plots: bool = True,
                      output_dir: str = './analysis_output') -> Dict:
    """
    Función de conveniencia para análisis fractal completo de campos MFSU.
    
    Parameters:
    -----------
    field_data : np.ndarray
        Campo ψ(x,y) o |ψ(x,y)|² del simulador MFSU
    time_series : np.ndarray, optional
        Serie temporal ψ(t) en punto específico
    config : dict, optional
        Configuración del análisis
    save_plots : bool
        Si guardar gráficos de resultados
    output_dir : str
        Directorio para guardar resultados
        
    Returns:
    --------
    dict : Resultados completos del análisis fractal
    """
    import os
    
    # Crear directorio de salida
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Inicializar analizador
    analyzer = FractalAnalyzer(config)
    
    # Realizar análisis completo
    print("Iniciando análisis fractal completo...")
    results = analyzer.comprehensive_analysis(field_data, time_series)
    
    # Agregar metadatos
    results['metadata'] = {
        'field_shape': field_data.shape,
        'field_stats': {
            'mean': np.mean(field_data),
            'std': np.std(field_data),
            'min': np.min(field_data),
            'max': np.max(field_data)
        },
        'analysis_timestamp': time.time()
    }
    
    if time_series is not None:
        results['metadata']['time_series_length'] = len(time_series)
    
    # Generar plots si se solicita
    if save_plots:
        plot_path = os.path.join(output_dir, 'fractal_analysis.png')
        plot_fractal_results(results, plot_path)
        print(f"Gráficos guardados en: {plot_path}")
    
    # Guardar resultados
    if save_plots:
        import json
        results_path = os.path.join(output_dir, 'fractal_results.json')
        
        # Convertir arrays numpy a listas para serialización JSON
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_results[key][subkey] = subvalue.tolist()
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Resultados guardados en: {results_path}")
    
    return results


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de análisis con datos sintéticos
    print("=== Ejemplo de Análisis Fractal MFSU ===")
    
    # Generar campo de prueba (simulando solución MFSU)
    np.random.seed(42)
    x = np.linspace(-10, 10, 128)
    y = np.linspace(-10, 10, 128)
    X, Y = np.meshgrid(x, y)
    
    # Campo tipo solitón con ruido fractal
    field = np.exp(-(X**2 + Y**2)/4) * (1 + 0.1*np.random.randn(128, 128))
    
    # Serie temporal
    t = np.linspace(0, 10, 1000)
    time_series = np.exp(-t/5) * np.sin(2*np.pi*t) + 0.1*np.random.randn(1000)
    
    # Realizar análisis
    results = analyze_mfsu_field(field, time_series, save_plots=False)
    
    # Mostrar resultados clave
    print("\n=== Resultados del Análisis ===")
    
    if 'box_counting' in results and 'error' not in results['box_counting']:
        bc = results['box_counting']
        print(f"Dimensión Fractal (Box-Counting): {bc['fractal_dimension']:.3f} ± {bc['dimension_error']:.3f}")
        print(f"R² del ajuste: {bc['r_squared']:.3f}")
    
    if 'hurst_dfa' in results and 'error' not in results['hurst_dfa']:
        hurst = results['hurst_dfa']
        print(f"Exponente de Hurst (DFA): {hurst['hurst_exponent']:.3f} ± {hurst['hurst_error']:.3f}")
        print(f"R² del ajuste: {hurst['r_squared']:.3f}")
    
    if 'multifractal' in results and 'error' not in results['multifractal']:
        mf = results['multifractal']
        print(f"Ancho espectro multifractal: {mf['spectrum_width']:.3f}")
        print(f"¿Es multifractal?: {mf['is_multifractal']}")
        print(f"Asimetría: {mf['asymmetry']:.3f}")
    
    print("\n=== Análisis Completo ===")
    print("El análisis fractal está listo para integrar con el simulador MFSU.")
    print("Módulos disponibles:")
    print("- FractalAnalyzer: Análisis completo de dimensiones fractales")
    print("- FractalValidation: Validación y benchmarking") 
    print("- analyze_mfsu_field(): Función de conveniencia para análisis rápido")
