"""
Módulo de Análisis Estadístico para el Simulador MFSU
====================================================

Este módulo proporciona herramientas para el análisis estadístico de los resultados
de simulaciones del Modelo Fractal Estocástico Unificado (MFSU).

Ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Autor: MFSU Development Team
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import periodogram, welch
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Imports locales
from ..utils.logger import get_logger
from ..utils.constants import PHYSICAL_CONSTANTS

logger = get_logger(__name__)

@dataclass
class StatisticalResults:
    """Clase para almacenar resultados de análisis estadístico"""
    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    range_val: float
    median: float
    q25: float
    q75: float
    iqr: float
    coefficient_variation: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convierte los resultados a diccionario"""
        return {
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'min': self.min_val,
            'max': self.max_val,
            'range': self.range_val,
            'median': self.median,
            'q25': self.q25,
            'q75': self.q75,
            'iqr': self.iqr,
            'cv': self.coefficient_variation
        }

class StatisticalAnalyzer:
    """
    Clase principal para análisis estadístico de simulaciones MFSU
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Inicializa el analizador estadístico
        
        Args:
            confidence_level: Nivel de confianza para intervalos (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def basic_statistics(self, data: np.ndarray) -> StatisticalResults:
        """
        Calcula estadísticas básicas de los datos
        
        Args:
            data: Array de datos a analizar
            
        Returns:
            StatisticalResults con estadísticas básicas
        """
        data_flat = data.flatten()
        
        # Estadísticas básicas
        mean = np.mean(data_flat)
        std = np.std(data_flat, ddof=1)
        variance = np.var(data_flat, ddof=1)
        
        # Momentos estadísticos
        skewness = stats.skew(data_flat)
        kurtosis = stats.kurtosis(data_flat)
        
        # Valores extremos
        min_val = np.min(data_flat)
        max_val = np.max(data_flat)
        range_val = max_val - min_val
        
        # Percentiles
        median = np.median(data_flat)
        q25 = np.percentile(data_flat, 25)
        q75 = np.percentile(data_flat, 75)
        iqr = q75 - q25
        
        # Coeficiente de variación
        cv = std / mean if mean != 0 else np.inf
        
        return StatisticalResults(
            mean=mean, std=std, variance=variance,
            skewness=skewness, kurtosis=kurtosis,
            min_val=min_val, max_val=max_val, range_val=range_val,
            median=median, q25=q25, q75=q75, iqr=iqr,
            coefficient_variation=cv
        )
    
    def confidence_interval(self, data: np.ndarray, 
                          statistic: str = 'mean') -> Tuple[float, float]:
        """
        Calcula intervalos de confianza para estadísticas
        
        Args:
            data: Array de datos
            statistic: Estadística a calcular ('mean', 'median', 'std')
            
        Returns:
            Tuple con límites inferior y superior del intervalo
        """
        data_flat = data.flatten()
        n = len(data_flat)
        
        if statistic == 'mean':
            mean = np.mean(data_flat)
            se = stats.sem(data_flat)
            h = se * stats.t.ppf((1 + self.confidence_level) / 2., n-1)
            return mean - h, mean + h
            
        elif statistic == 'median':
            # Bootstrap para mediana
            n_bootstrap = 1000
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data_flat, size=n, replace=True)
                bootstrap_medians.append(np.median(bootstrap_sample))
            
            lower = np.percentile(bootstrap_medians, 100 * self.alpha / 2)
            upper = np.percentile(bootstrap_medians, 100 * (1 - self.alpha / 2))
            return lower, upper
            
        elif statistic == 'std':
            # Chi-cuadrado para desviación estándar
            std = np.std(data_flat, ddof=1)
            chi2_lower = stats.chi2.ppf(self.alpha / 2, n-1)
            chi2_upper = stats.chi2.ppf(1 - self.alpha / 2, n-1)
            
            lower = std * np.sqrt((n-1) / chi2_upper)
            upper = std * np.sqrt((n-1) / chi2_lower)
            return lower, upper
            
        else:
            raise ValueError(f"Estadística '{statistic}' no soportada")
    
    def normality_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Realiza pruebas de normalidad en los datos
        
        Args:
            data: Array de datos
            
        Returns:
            Diccionario con resultados de pruebas de normalidad
        """
        data_flat = data.flatten()
        
        results = {}
        
        # Test de Shapiro-Wilk (para muestras pequeñas)
        if len(data_flat) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data_flat)
            results['shapiro'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(data_flat, 'norm', 
                                    args=(np.mean(data_flat), np.std(data_flat)))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > 0.05
        }
        
        # Test de Anderson-Darling
        ad_stat, ad_critical, ad_significance = stats.anderson(data_flat, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_stat,
            'critical_values': ad_critical,
            'significance_levels': ad_significance,
            'is_normal': ad_stat < ad_critical[2]  # 5% significance level
        }
        
        # Test de Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(data_flat)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > 0.05
        }
        
        return results
    
    def correlation_analysis(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """
        Análisis de correlación entre dos conjuntos de datos
        
        Args:
            data1: Primer conjunto de datos
            data2: Segundo conjunto de datos
            
        Returns:
            Diccionario con resultados de correlación
        """
        data1_flat = data1.flatten()
        data2_flat = data2.flatten()
        
        # Asegurar que ambos arrays tengan la misma longitud
        min_len = min(len(data1_flat), len(data2_flat))
        data1_flat = data1_flat[:min_len]
        data2_flat = data2_flat[:min_len]
        
        results = {}
        
        # Correlación de Pearson
        pearson_r, pearson_p = stats.pearsonr(data1_flat, data2_flat)
        results['pearson'] = {
            'correlation': pearson_r,
            'p_value': pearson_p,
            'is_significant': pearson_p < 0.05
        }
        
        # Correlación de Spearman
        spearman_r, spearman_p = stats.spearmanr(data1_flat, data2_flat)
        results['spearman'] = {
            'correlation': spearman_r,
            'p_value': spearman_p,
            'is_significant': spearman_p < 0.05
        }
        
        # Correlación de Kendall
        kendall_tau, kendall_p = stats.kendalltau(data1_flat, data2_flat)
        results['kendall'] = {
            'correlation': kendall_tau,
            'p_value': kendall_p,
            'is_significant': kendall_p < 0.05
        }
        
        return results
    
    def time_series_analysis(self, data: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
        """
        Análisis de series temporales
        
        Args:
            data: Series temporal
            dt: Paso temporal
            
        Returns:
            Diccionario con resultados del análisis
        """
        results = {}
        
        # Autocorrelación
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalizar
        
        results['autocorrelation'] = autocorr
        
        # Tiempo de correlación (donde autocorr cae a 1/e)
        target = 1/np.e
        try:
            corr_time_idx = np.where(autocorr < target)[0][0]
            corr_time = corr_time_idx * dt
        except IndexError:
            corr_time = len(data) * dt
        
        results['correlation_time'] = corr_time
        
        # Análisis espectral
        freqs, psd = welch(data, fs=1/dt, nperseg=min(len(data)//4, 256))
        results['power_spectral_density'] = {
            'frequencies': freqs,
            'psd': psd,
            'dominant_frequency': freqs[np.argmax(psd)]
        }
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(data)
            results['stationarity'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        except ImportError:
            logger.warning("statsmodels no disponible para test de estacionariedad")
        
        return results
    
    def distribution_fitting(self, data: np.ndarray, 
                           distributions: List[str] = None) -> Dict[str, Any]:
        """
        Ajuste de distribuciones estadísticas a los datos
        
        Args:
            data: Array de datos
            distributions: Lista de distribuciones a probar
            
        Returns:
            Diccionario con resultados del ajuste
        """
        if distributions is None:
            distributions = ['norm', 'lognorm', 'gamma', 'beta', 'weibull_min', 'exponweib']
        
        data_flat = data.flatten()
        results = {}
        
        for dist_name in distributions:
            try:
                # Obtener distribución
                dist = getattr(stats, dist_name)
                
                # Ajustar parámetros
                params = dist.fit(data_flat)
                
                # Calcular AIC y BIC
                log_likelihood = np.sum(dist.logpdf(data_flat, *params))
                k = len(params)
                n = len(data_flat)
                
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                # Test de bondad de ajuste
                ks_stat, ks_p = stats.kstest(data_flat, dist.cdf, args=params)
                
                results[dist_name] = {
                    'parameters': params,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'good_fit': ks_p > 0.05
                }
                
            except Exception as e:
                logger.warning(f"Error ajustando distribución {dist_name}: {e}")
                continue
        
        # Encontrar mejor ajuste (menor AIC)
        if results:
            best_dist = min(results.keys(), key=lambda x: results[x]['aic'])
            results['best_fit'] = {
                'distribution': best_dist,
                'parameters': results[best_dist]['parameters'],
                'aic': results[best_dist]['aic']
            }
        
        return results
    
    def spatial_statistics(self, data: np.ndarray, 
                         coordinates: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Análisis estadístico espacial
        
        Args:
            data: Array de datos espaciales
            coordinates: Coordenadas espaciales (si no se proporcionan, se usan índices)
            
        Returns:
            Diccionario con estadísticas espaciales
        """
        if data.ndim == 1:
            # Datos 1D
            if coordinates is None:
                coordinates = np.arange(len(data)).reshape(-1, 1)
            
        elif data.ndim == 2:
            # Datos 2D
            if coordinates is None:
                y, x = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
                coordinates = np.column_stack((x.flatten(), y.flatten()))
            data = data.flatten()
            
        results = {}
        
        # Índice de Moran (autocorrelación espacial)
        try:
            distances = pdist(coordinates)
            # Matriz de pesos (inverso de la distancia)
            weights = 1 / (distances + 1e-10)  # Evitar división por cero
            weight_matrix = squareform(weights)
            
            # Normalizar pesos
            weight_matrix = weight_matrix / np.sum(weight_matrix, axis=1, keepdims=True)
            
            # Calcular índice de Moran
            n = len(data)
            mean_data = np.mean(data)
            numerator = 0
            denominator = 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        numerator += weight_matrix[i, j] * (data[i] - mean_data) * (data[j] - mean_data)
                
                denominator += (data[i] - mean_data) ** 2
            
            moran_i = (n / np.sum(weight_matrix)) * (numerator / denominator)
            results['moran_index'] = moran_i
            
        except Exception as e:
            logger.warning(f"Error calculando índice de Moran: {e}")
        
        # Variograma experimental
        try:
            max_distance = np.max(pdist(coordinates)) / 3  # Usar hasta 1/3 de la distancia máxima
            distance_bins = np.linspace(0, max_distance, 20)
            
            variogram = []
            for i in range(len(distance_bins) - 1):
                bin_distances = []
                bin_values = []
                
                for idx1 in range(len(coordinates)):
                    for idx2 in range(idx1 + 1, len(coordinates)):
                        dist = np.linalg.norm(coordinates[idx1] - coordinates[idx2])
                        if distance_bins[i] <= dist < distance_bins[i + 1]:
                            bin_distances.append(dist)
                            bin_values.append(0.5 * (data[idx1] - data[idx2]) ** 2)
                
                if bin_values:
                    variogram.append(np.mean(bin_values))
                else:
                    variogram.append(0)
            
            results['variogram'] = {
                'distances': distance_bins[:-1],
                'values': variogram
            }
            
        except Exception as e:
            logger.warning(f"Error calculando variograma: {e}")
        
        return results
    
    def parameter_sensitivity_analysis(self, simulation_results: Dict[str, np.ndarray],
                                     parameter_values: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Análisis de sensibilidad de parámetros
        
        Args:
            simulation_results: Resultados de simulaciones para diferentes parámetros
            parameter_values: Valores de parámetros correspondientes
            
        Returns:
            Diccionario con análisis de sensibilidad
        """
        results = {}
        
        for param_name, param_vals in parameter_values.items():
            if param_name in simulation_results:
                sim_results = simulation_results[param_name]
                
                # Calcular estadísticas para cada valor de parámetro
                param_stats = []
                for i, param_val in enumerate(param_vals):
                    if i < len(sim_results):
                        stats_result = self.basic_statistics(sim_results[i])
                        param_stats.append(stats_result.mean)
                
                # Calcular sensibilidad (derivada numérica)
                if len(param_stats) > 1:
                    sensitivity = np.gradient(param_stats, param_vals)
                    
                    results[param_name] = {
                        'parameter_values': param_vals,
                        'mean_response': param_stats,
                        'sensitivity': sensitivity,
                        'max_sensitivity': np.max(np.abs(sensitivity)),
                        'sensitive_range': param_vals[np.argmax(np.abs(sensitivity))]
                    }
        
        return results
    
    def generate_report(self, data: np.ndarray, 
                       output_file: Optional[str] = None) -> str:
        """
        Genera un reporte estadístico completo
        
        Args:
            data: Array de datos a analizar
            output_file: Archivo de salida (opcional)
            
        Returns:
            String con el reporte
        """
        report = []
        report.append("=" * 60)
        report.append("REPORTE ESTADÍSTICO - SIMULACIÓN MFSU")
        report.append("=" * 60)
        report.append("")
        
        # Estadísticas básicas
        basic_stats = self.basic_statistics(data)
        report.append("ESTADÍSTICAS BÁSICAS")
        report.append("-" * 20)
        for key, value in basic_stats.to_dict().items():
            report.append(f"{key.capitalize()}: {value:.6f}")
        report.append("")
        
        # Intervalos de confianza
        report.append("INTERVALOS DE CONFIANZA (95%)")
        report.append("-" * 30)
        
        for stat in ['mean', 'median', 'std']:
            try:
                lower, upper = self.confidence_interval(data, stat)
                report.append(f"{stat.capitalize()}: [{lower:.6f}, {upper:.6f}]")
            except Exception as e:
                report.append(f"{stat.capitalize()}: Error - {e}")
        report.append("")
        
        # Pruebas de normalidad
        normality_results = self.normality_test(data)
        report.append("PRUEBAS DE NORMALIDAD")
        report.append("-" * 20)
        
        for test_name, test_result in normality_results.items():
            report.append(f"{test_name.replace('_', ' ').title()}:")
            report.append(f"  Estadística: {test_result['statistic']:.6f}")
            if 'p_value' in test_result:
                report.append(f"  P-valor: {test_result['p_value']:.6f}")
            report.append(f"  ¿Normal?: {test_result['is_normal']}")
            report.append("")
        
        # Análisis de series temporales (si es unidimensional)
        if data.ndim == 1:
            ts_results = self.time_series_analysis(data)
            report.append("ANÁLISIS DE SERIES TEMPORALES")
            report.append("-" * 30)
            report.append(f"Tiempo de correlación: {ts_results['correlation_time']:.6f}")
            if 'power_spectral_density' in ts_results:
                psd_info = ts_results['power_spectral_density']
                report.append(f"Frecuencia dominante: {psd_info['dominant_frequency']:.6f}")
            report.append("")
        
        # Ajuste de distribuciones
        dist_results = self.distribution_fitting(data)
        if 'best_fit' in dist_results:
            report.append("MEJOR AJUSTE DE DISTRIBUCIÓN")
            report.append("-" * 30)
            best_fit = dist_results['best_fit']
            report.append(f"Distribución: {best_fit['distribution']}")
            report.append(f"Parámetros: {best_fit['parameters']}")
            report.append(f"AIC: {best_fit['aic']:.6f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Reporte guardado en: {output_file}")
        
        return report_text

# Clases especializadas para diferentes tipos de análisis

class MFSUStochasticAnalyzer(StatisticalAnalyzer):
    """
    Analizador especializado para procesos estocásticos del MFSU
    """
    
    def __init__(self, hurst_exponent: float = 0.5, **kwargs):
        """
        Inicializa el analizador estocástico
        
        Args:
            hurst_exponent: Exponente de Hurst esperado
        """
        super().__init__(**kwargs)
        self.hurst_exponent = hurst_exponent
    
    def estimate_hurst_exponent(self, data: np.ndarray) -> Dict[str, float]:
        """
        Estima el exponente de Hurst usando múltiples métodos
        
        Args:
            data: Serie temporal
            
        Returns:
            Diccionario con estimaciones del exponente de Hurst
        """
        results = {}
        
        # Método R/S (Rescaled Range)
        try:
            rs_hurst = self._rs_hurst(data)
            results['rs_method'] = rs_hurst
        except Exception as e:
            logger.warning(f"Error en método R/S: {e}")
        
        # Método DFA (Detrended Fluctuation Analysis)
        try:
            dfa_hurst = self._dfa_hurst(data)
            results['dfa_method'] = dfa_hurst
        except Exception as e:
            logger.warning(f"Error en método DFA: {e}")
        
        # Método Varianza-Tiempo
        try:
            vt_hurst = self._variance_time_hurst(data)
            results['variance_time_method'] = vt_hurst
        except Exception as e:
            logger.warning(f"Error en método varianza-tiempo: {e}")
        
        # Promedio de métodos
        if results:
            results['average'] = np.mean(list(results.values()))
        
        return results
    
    def _rs_hurst(self, data: np.ndarray) -> float:
        """Método R/S para estimar exponente de Hurst"""
        n = len(data)
        rs_values = []
        
        for size in range(10, n//4):
            segments = n // size
            rs_segment = []
            
            for i in range(segments):
                segment = data[i*size:(i+1)*size]
                mean_seg = np.mean(segment)
                cumsum = np.cumsum(segment - mean_seg)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                if S > 0:
                    rs_segment.append(R/S)
            
            if rs_segment:
                rs_values.append((size, np.mean(rs_segment)))
        
        if len(rs_values) > 5:
            sizes, rs_vals = zip(*rs_values)
            log_sizes = np.log(sizes)
            log_rs = np.log(rs_vals)
            
            # Regresión lineal
            slope, _ = np.polyfit(log_sizes, log_rs, 1)
            return slope
        
        return 0.5  # Valor por defecto
    
    def _dfa_hurst(self, data: np.ndarray) -> float:
        """Método DFA para estimar exponente de Hurst"""
        n = len(data)
        
        # Integrar la serie
        y = np.cumsum(data - np.mean(data))
        
        fluctuations = []
        window_sizes = np.logspace(1, np.log10(n//4), 20).astype(int)
        
        for window_size in window_sizes:
            # Dividir en ventanas
            n_windows = n // window_size
            if n_windows < 4:
                continue
                
            detrended = []
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                segment = y[start:end]
                
                # Ajustar tendencia polinomial
                t = np.arange(len(segment))
                poly_coeff = np.polyfit(t, segment, 1)
                trend = np.polyval(poly_coeff, t)
                
                detrended.extend(segment - trend)
            
            if detrended:
                fluctuation = np.sqrt(np.mean(np.array(detrended)**2))
                fluctuations.append((window_size, fluctuation))
        
        if len(fluctuations) > 5:
            sizes, flucts = zip(*fluctuations)
            log_sizes = np.log(sizes)
            log_flucts = np.log(flucts)
            
            # Regresión lineal
            slope, _ = np.polyfit(log_sizes, log_flucts, 1)
            return slope
        
        return 0.5
    
    def _variance_time_hurst(self, data: np.ndarray) -> float:
        """Método varianza-tiempo para estimar exponente de Hurst"""
        n = len(data)
        variances = []
        
        for m in range(2, n//10):
            # Agregación temporal
            aggregated = []
            for i in range(0, n-m+1, m):
                aggregated.append(np.mean(data[i:i+m]))
            
            if len(aggregated) > 5:
                var = np.var(aggregated)
                variances.append((m, var))
        
        if len(variances) > 5:
            ms, vars = zip(*variances)
            log_ms = np.log(ms)
            log_vars = np.log(vars)
            
            # Regresión lineal
            slope, _ = np.polyfit(log_ms, log_vars, 1)
            return 1 + slope/2
        
        return 0.5

class MFSUFractalAnalyzer(StatisticalAnalyzer):
    """
    Analizador especializado para propiedades fractales del MFSU
    """
    
    def multifractal_analysis(self, data: np.ndarray, 
                            q_range: np.ndarray = None) -> Dict[str, Any]:
        """
        Análisis multifractal de los datos
        
        Args:
            data: Array de datos
            q_range: Rango de momentos q para análisis
            
        Returns:
            Diccionario con resultados del análisis multifractal
        """
        if q_range is None:
            q_range = np.linspace(-10, 10, 41)
        
        results = {}
        
        # Análisis MFDFA (Multifractal Detrended Fluctuation Analysis)
        try:
            mfdfa_results = self._mfdfa_analysis(data, q_range)
            results['mfdfa'] = mfdfa_results
        except Exception as e:
            logger.warning(f"Error en análisis MFDFA: {e}")
        
        # Análisis de singularidades
        try:
            singularity_spectrum = self._singularity_spectrum(data)
            results['singularity_spectrum'] = singularity_spectrum
        except Exception as e:
            logger.warning(f"Error en espectro de singularidades: {e}")
        
        return results
    
    def _mfdfa_analysis(self, data: np.ndarray, q_range: np.ndarray) -> Dict[str, Any]:
        """Análisis Multifractal Detrended Fluctuation Analysis"""
        n = len(data)
        
        # Integrar la serie
        y = np.cumsum(data - np.mean(data))
        
        # Rangos de escala
        scales = np.logspace(1, np.log10(n//4), 20).astype(int)
        
        fluctuations = np.zeros((len(q_range), len(scales)))
        
        for i, scale in enumerate(scales):
            n_segments = n // scale
            if n_segments < 4:
                continue
            
            # Calcular fluctuaciones para cada segmento
            segment_fluctuations = []
            
            for j in range(n_segments):
                start = j * scale
                end = start + scale
                segment = y[start:end]
                
                # Detrending polinomial (orden 1)
                t = np.arange(len(segment))
                poly_coeff = np.polyfit(t, segment, 1)
                trend = np.polyval(poly_coeff, t)
                
                fluctuation = np.sqrt(np.mean((segment - trend)**2))
                segment_fluctuations.append(fluctuation)
            
            # Calcular F_q(s) para cada q
            for k, q in enumerate(q_range):
                if q == 0:
                    # Caso especial para q=0
                    log_flucts = np.log(segment_fluctuations)
                    fluctuations[k, i] = np.exp(np.mean(log_flucts))
                else:
                    mean_fluct_q = np.mean(np.array(segment_fluctuations)**q)
                    fluctuations[k, i] = mean_fluct_q**(1/q)
        
        # Calcular exponentes de scaling h(q)
        h_q = np.zeros(len(q_range))
        for k in range(len(q_range)):
            valid_indices = fluctuations[k, :] > 0
            if np.sum(valid_indices) > 5:
                log_scales = np.log(scales[valid_indices])
                log_flucts = np.log(fluctuations[k, valid_indices])
                h_q[k], _ = np.polyfit(log_scales, log_flucts, 1)
        
        # Calcular tau(q) y dimensiones fractales
        tau_q = q_range * h_q - 1
        
        # Espectro multifractal f(alpha)
        alpha = np.gradient(tau_q, q_range)
        f_alpha = q_range * alpha - tau_q
        
        return {
            'q_range': q_range,
            'h_q': h_q,
            'tau_q': tau_q,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'fluctuations': fluctuations,
            'scales': scales,
            'width_multifractal': np.max(alpha) - np.min(alpha)
        }
    
    def _singularity_spectrum(self, data: np.ndarray) -> Dict[str, Any]:
        """Calcula el espectro de singularidades usando wavelets"""
        try:
            import pywt
            
            # Transformada wavelet continua
            scales = np.arange(1, 64)
            coefficients, _ = pywt.cwt(data, scales, 'morl')
            
            # Módulo de los coeficientes
            modulus = np.abs(coefficients)
            
            # Exponentes de Hölder locales
            alpha_local = np.zeros_like(modulus)
            
            for i in range(len(scales)):
                if i > 0:
                    alpha_local[i, :] = np.log(modulus[i, :] / modulus[i-1, :]) / np.log(scales[i] / scales[i-1])
            
            # Histograma de exponentes
            alpha_hist, alpha_bins = np.histogram(alpha_local.flatten(), bins=50, density=True)
            
            return {
                'alpha_values': alpha_bins[:-1],
                'f_alpha': alpha_hist,
                'alpha_range': (np.min(alpha_local), np.max(alpha_local))
            }
            
        except ImportError:
            logger.warning("PyWavelets no disponible para análisis de singularidades")
            return {}
    
    def fractal_dimension(self, data: np.ndarray, method: str = 'box_counting') -> float:
        """
        Calcula la dimensión fractal usando diferentes métodos
        
        Args:
            data: Array de datos
            method: Método a usar ('box_counting', 'correlation', 'higuchi')
            
        Returns:
            Dimensión fractal estimada
        """
        if method == 'box_counting':
            return self._box_counting_dimension(data)
        elif method == 'correlation':
            return self._correlation_dimension(data)
        elif method == 'higuchi':
            return self._higuchi_dimension(data)
        else:
            raise ValueError(f"Método '{method}' no soportado")
    
    def _box_counting_dimension(self, data: np.ndarray) -> float:
        """Dimensión fractal por conteo de cajas"""
        if data.ndim == 1:
            # Para series temporales, crear representación 2D
            x = np.arange(len(data))
            y = data
            
            # Normalizar coordenadas
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
            
            points = np.column_stack((x_norm, y_norm))
        else:
            points = data
        
        # Rangos de tamaños de caja
        box_sizes = np.logspace(-3, 0, 20)
        counts = []
        
        for box_size in box_sizes:
            # Crear grilla de cajas
            n_boxes = int(1 / box_size)
            occupied_boxes = set()
            
            for point in points:
                box_x = int(point[0] * n_boxes)
                box_y = int(point[1] * n_boxes)
                occupied_boxes.add((box_x, box_y))
            
            counts.append(len(occupied_boxes))
        
        # Regresión lineal en escala log-log
        valid_counts = np.array(counts) > 0
        if np.sum(valid_counts) > 5:
            log_sizes = np.log(box_sizes[valid_counts])
            log_counts = np.log(np.array(counts)[valid_counts])
            
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            return -slope
        
        return 1.0
    
    def _correlation_dimension(self, data: np.ndarray) -> float:
        """Dimensión de correlación usando algoritmo de Grassberger-Procaccia"""
        if data.ndim == 1:
            # Reconstrucción del espacio de fases
            embedding_dim = 3
            delay = 1
            
            n = len(data)
            embedded = np.zeros((n - (embedding_dim - 1) * delay, embedding_dim))
            
            for i in range(embedding_dim):
                embedded[:, i] = data[i * delay:n - (embedding_dim - 1 - i) * delay]
            
            points = embedded
        else:
            points = data
        
        n_points = len(points)
        if n_points > 1000:
            # Submuestrear para eficiencia
            indices = np.random.choice(n_points, 1000, replace=False)
            points = points[indices]
            n_points = 1000
        
        # Calcular distancias
        distances = pdist(points)
        
        # Rangos de radios
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)
        
        correlations = []
        for r in radii:
            count = np.sum(distances <= r)
            correlation = count / (n_points * (n_points - 1) / 2)
            correlations.append(correlation)
        
        # Regresión lineal en región de escalamiento
        log_radii = np.log(radii)
        log_corr = np.log(np.array(correlations) + 1e-10)
        
        # Encontrar región lineal
        if len(log_radii) > 5:
            slope, _ = np.polyfit(log_radii, log_corr, 1)
            return slope
        
        return 1.0
    
    def _higuchi_dimension(self, data: np.ndarray) -> float:
        """Dimensión fractal usando método de Higuchi"""
        n = len(data)
        k_max = min(n // 4, 20)
        
        curve_lengths = []
        k_values = range(1, k_max + 1)
        
        for k in k_values:
            lengths = []
            
            for m in range(1, k + 1):
                # Subsecuencia con paso k comenzando en m
                subsequence = data[m-1::k]
                
                if len(subsequence) > 1:
                    # Longitud de la curva
                    length = np.sum(np.abs(np.diff(subsequence)))
                    length = length * (n - 1) / (k * len(subsequence))
                    lengths.append(length)
            
            if lengths:
                curve_lengths.append(np.mean(lengths))
        
        if len(curve_lengths) > 5:
            log_k = np.log(k_values)
            log_lengths = np.log(curve_lengths)
            
            slope, _ = np.polyfit(log_k, log_lengths, 1)
            return -slope
        
        return 1.0

# Funciones de utilidad para análisis estadístico

def compare_distributions(data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
    """
    Compara dos distribuciones estadísticamente
    
    Args:
        data1: Primera distribución
        data2: Segunda distribución
        
    Returns:
        Diccionario con resultados de comparación
    """
    results = {}
    
    # Test de Mann-Whitney U
    mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(
        data1.flatten(), data2.flatten(), alternative='two-sided'
    )
    results['mann_whitney'] = {
        'statistic': mannwhitney_stat,
        'p_value': mannwhitney_p,
        'significant_difference': mannwhitney_p < 0.05
    }
    
    # Test de Kolmogorov-Smirnov
    ks_stat, ks_p = stats.ks_2samp(data1.flatten(), data2.flatten())
    results['kolmogorov_smirnov'] = {
        'statistic': ks_stat,
        'p_value': ks_p,
        'significant_difference': ks_p < 0.05
    }
    
    # Test t de Student (si las distribuciones son normales)
    try:
        t_stat, t_p = stats.ttest_ind(data1.flatten(), data2.flatten())
        results['t_test'] = {
            'statistic': t_stat,
            'p_value': t_p,
            'significant_difference': t_p < 0.05
        }
    except Exception as e:
        logger.warning(f"Error en test t: {e}")
    
    # Test de Levene para igualdad de varianzas
    levene_stat, levene_p = stats.levene(data1.flatten(), data2.flatten())
    results['levene_test'] = {
        'statistic': levene_stat,
        'p_value': levene_p,
        'equal_variances': levene_p > 0.05
    }
    
    return results

def bootstrap_confidence_interval(data: np.ndarray, 
                                statistic_func: callable,
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calcula intervalos de confianza usando bootstrap
    
    Args:
        data: Array de datos
        statistic_func: Función que calcula la estadística
        confidence_level: Nivel de confianza
        n_bootstrap: Número de muestras bootstrap
        
    Returns:
        Tuple con límites inferior y superior
    """
    data_flat = data.flatten()
    n = len(data_flat)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data_flat, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return lower, upper

def monte_carlo_test(data: np.ndarray, 
                    null_hypothesis_func: callable,
                    test_statistic_func: callable,
                    n_simulations: int = 1000) -> Dict[str, Any]:
    """
    Realiza test de Monte Carlo
    
    Args:
        data: Datos observados
        null_hypothesis_func: Función que genera datos bajo hipótesis nula
        test_statistic_func: Función que calcula estadística de prueba
        n_simulations: Número de simulaciones
        
    Returns:
        Diccionario con resultados del test
    """
    # Estadística observada
    observed_stat = test_statistic_func(data)
    
    # Generar distribución nula
    null_stats = []
    for _ in range(n_simulations):
        null_data = null_hypothesis_func(data)
        null_stat = test_statistic_func(null_data)
        null_stats.append(null_stat)
    
    # Calcular p-valor
    p_value = np.mean(np.array(null_stats) >= observed_stat)
    
    return {
        'observed_statistic': observed_stat,
        'null_statistics': null_stats,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'critical_value_95': np.percentile(null_stats, 95)
    }

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo
    np.random.seed(42)
    
    # Simulación de datos MFSU
    t = np.linspace(0, 10, 1000)
    alpha, beta, gamma = 0.5, 0.1, 0.01
    
    # Datos con ruido fractal y no linealidad
    noise = np.random.randn(len(t))
    data = np.sin(t) + 0.1 * noise + 0.05 * np.sin(t)**3
    
    # Crear analizador
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Análisis básico
    print("=== ANÁLISIS ESTADÍSTICO BÁSICO ===")
    basic_stats = analyzer.basic_statistics(data)
    print(f"Media: {basic_stats.mean:.4f}")
    print(f"Desviación estándar: {basic_stats.std:.4f}")
    print(f"Asimetría: {basic_stats.skewness:.4f}")
    print(f"Curtosis: {basic_stats.kurtosis:.4f}")
    
    # Pruebas de normalidad
    print("\n=== PRUEBAS DE NORMALIDAD ===")
    normality = analyzer.normality_test(data)
    for test, result in normality.items():
        print(f"{test}: p-valor = {result['p_value']:.4f}, Normal = {result['is_normal']}")
    
    # Análisis de series temporales
    print("\n=== ANÁLISIS DE SERIES TEMPORALES ===")
    ts_analysis = analyzer.time_series_analysis(data)
    print(f"Tiempo de correlación: {ts_analysis['correlation_time']:.4f}")
    
    # Análisis estocástico
    print("\n=== ANÁLISIS ESTOCÁSTICO ===")
    stoch_analyzer = MFSUStochasticAnalyzer()
    hurst_results = stoch_analyzer.estimate_hurst_exponent(data)
    print(f"Exponente de Hurst (promedio): {hurst_results.get('average', 'N/A'):.4f}")
    
    # Análisis fractal
    print("\n=== ANÁLISIS FRACTAL ===")
    fractal_analyzer = MFSUFractalAnalyzer()
    fractal_dim = fractal_analyzer.fractal_dimension(data, method='higuchi')
    print(f"Dimensión fractal (Higuchi): {fractal_dim:.4f}")
    
    # Generar reporte completo
    print("\n=== GENERANDO REPORTE COMPLETO ===")
    report = analyzer.generate_report(data)
    print("Reporte generado exitosamente")
    
    print("\nAnálisis estadístico completado.")
