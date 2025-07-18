"""
Aplicación cosmológica del modelo MFSU para simulaciones de estructura 
del universo a gran escala, formación de galaxias y análisis de materia oscura.

La ecuación MFSU aplicada a cosmología:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Donde:
- ψ(x,t): Campo cosmológico (densidad de materia, potencial gravitatorio)
- α: Parámetro de difusión anómala relacionado con la expansión del universo
- β: Intensidad del ruido fractal (fluctuaciones cuánticas primordiales)
- γ: Coeficiente no lineal (autogravedad, interacciones materia oscura)
- ξ_H: Ruido fractal con exponente de Hurst H
- f(x,t): Fuerza externa (inflación, energía oscura)
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import quad
from typing import Dict, Any, Tuple, Optional, List
import logging

from .base_application import BaseApplication
from ..core.mfsu_equation import MFSUEquation
from ..core.stochastic_processes import StochasticProcess
from ..utils.constants import PLANCK_CONSTANT, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

logger = logging.getLogger(__name__)


class CosmologyApplication(BaseApplication):
    """
    Aplicación del modelo MFSU para simulaciones cosmológicas.
    
    Simula la evolución de estructuras cósmicas usando la ecuación MFSU
    con parámetros calibrados para escalas cosmológicas.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa la aplicación cosmológica.
        
        Args:
            config: Configuración con parámetros cosmológicos
        """
        super().__init__(config)
        self.application_name = "Cosmology"
        
        # Parámetros cosmológicos estándar
        self.hubble_constant = config.get('hubble_constant', 70.0)  # km/s/Mpc
        self.omega_matter = config.get('omega_matter', 0.3)
        self.omega_lambda = config.get('omega_lambda', 0.7)
        self.omega_baryon = config.get('omega_baryon', 0.05)
        self.omega_dark_matter = config.get('omega_dark_matter', 0.25)
        
        # Parámetros de inflación
        self.inflation_amplitude = config.get('inflation_amplitude', 1e-5)
        self.spectral_index = config.get('spectral_index', 0.965)
        
        # Escalas características
        self.box_size = config.get('box_size', 100.0)  # Mpc/h
        self.redshift_initial = config.get('redshift_initial', 100.0)
        self.redshift_final = config.get('redshift_final', 0.0)
        
        # Inicializar ecuación MFSU para cosmología
        self._setup_mfsu_parameters()
        
        logger.info(f"Aplicación cosmológica inicializada con H0={self.hubble_constant}")
    
    def _setup_mfsu_parameters(self):
        """Configura los parámetros MFSU para aplicaciones cosmológicas."""
        
        # Parámetro α: relacionado con la difusión anómala cosmológica
        # Calibrado para reproducir el espectro de potencia observado
        self.alpha = self.config.get('alpha', 1.5)
        
        # Parámetro β: intensidad del ruido fractal (fluctuaciones primordiales)
        # Relacionado con la amplitud de las fluctuaciones de densidad
        self.beta = self.config.get('beta', self.inflation_amplitude * 1e3)
        
        # Parámetro γ: no linealidad gravitatoria
        # Relacionado con la formación de estructuras
        self.gamma = self.config.get('gamma', 0.01)
        
        # Exponente de Hurst para ruido fractal cosmológico
        self.hurst_exponent = self.config.get('hurst', 0.7)
        
        # Configuración de la grilla espacial
        self.grid_params = {
            'nx': self.config.get('grid_size', 128),
            'ny': self.config.get('grid_size', 128),
            'nz': self.config.get('grid_size', 128),
            'dx': self.box_size / self.config.get('grid_size', 128),
            'dy': self.box_size / self.config.get('grid_size', 128),
            'dz': self.box_size / self.config.get('grid_size', 128)
        }
    
    def generate_initial_conditions(self, condition_type: str = 'gaussian_random') -> np.ndarray:
        """
        Genera condiciones iniciales para la simulación cosmológica.
        
        Args:
            condition_type: Tipo de condición inicial
                - 'gaussian_random': Fluctuaciones gaussianas
                - 'power_law': Espectro de potencia tipo ley de potencias
                - 'inflation': Fluctuaciones inflacionarias
        
        Returns:
            Campo inicial ψ(x,y,z,t=0)
        """
        nx, ny, nz = self.grid_params['nx'], self.grid_params['ny'], self.grid_params['nz']
        
        if condition_type == 'gaussian_random':
            # Fluctuaciones gaussianas simples
            field = np.random.normal(0, self.inflation_amplitude, (nx, ny, nz))
            
        elif condition_type == 'power_law':
            # Espectro de potencia tipo ley de potencias
            field = self._generate_power_law_field()
            
        elif condition_type == 'inflation':
            # Fluctuaciones inflacionarias realistas
            field = self._generate_inflation_field()
            
        else:
            raise ValueError(f"Tipo de condición inicial no reconocida: {condition_type}")
        
        logger.info(f"Condiciones iniciales generadas: {condition_type}")
        return field
    
    def _generate_power_law_field(self) -> np.ndarray:
        """
        Genera un campo con espectro de potencia tipo ley de potencias.
        
        Returns:
            Campo con espectro P(k) ∝ k^n
        """
        nx, ny, nz = self.grid_params['nx'], self.grid_params['ny'], self.grid_params['nz']
        
        # Generar modos de Fourier
        kx = np.fft.fftfreq(nx, d=self.grid_params['dx'])
        ky = np.fft.fftfreq(ny, d=self.grid_params['dy'])
        kz = np.fft.fftfreq(nz, d=self.grid_params['dz'])
        
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # Evitar división por cero
        k_magnitude[k_magnitude == 0] = 1
        
        # Espectro de potencia P(k) ∝ k^n
        power_spectrum = k_magnitude**(self.spectral_index - 1)
        
        # Generar amplitudes aleatorias
        random_phases = np.random.random((nx, ny, nz)) * 2 * np.pi
        random_amplitudes = np.random.rayleigh(scale=1.0, size=(nx, ny, nz))
        
        # Campo en espacio de Fourier
        field_fourier = random_amplitudes * np.sqrt(power_spectrum) * np.exp(1j * random_phases)
        
        # Transformada inversa
        field = np.fft.ifftn(field_fourier).real
        
        # Normalizar
        field = field * self.inflation_amplitude / np.std(field)
        
        return field
    
    def _generate_inflation_field(self) -> np.ndarray:
        """
        Genera fluctuaciones inflacionarias realistas.
        
        Returns:
            Campo con fluctuaciones inflacionarias
        """
        nx, ny, nz = self.grid_params['nx'], self.grid_params['ny'], self.grid_params['nz']
        
        # Usar proceso estocástico fractal para generar fluctuaciones
        stochastic_process = StochasticProcess(
            process_type='fractional_brownian',
            hurst=self.hurst_exponent,
            grid_size=nx
        )
        
        # Generar campo 3D
        field = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                field[i, j, :] = stochastic_process.generate_realization(
                    length=nz,
                    dt=self.grid_params['dz']
                )
        
        # Normalizar con amplitud inflacionaria
        field = field * self.inflation_amplitude / np.std(field)
        
        return field
    
    def external_force(self, field: np.ndarray, t: float, x: np.ndarray, 
                      y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calcula la fuerza externa f(x,t) para aplicaciones cosmológicas.
        
        Args:
            field: Campo actual ψ(x,y,z,t)
            t: Tiempo actual
            x, y, z: Coordenadas espaciales
        
        Returns:
            Fuerza externa f(x,y,z,t)
        """
        # Conversión de tiempo a redshift
        redshift = self._time_to_redshift(t)
        
        # Término de expansión del universo (Hubble flow)
        hubble_term = self._hubble_expansion_term(field, redshift)
        
        # Término de energía oscura
        dark_energy_term = self._dark_energy_term(field, redshift)
        
        # Término de inflación (solo en épocas tempranas)
        inflation_term = self._inflation_term(field, redshift)
        
        return hubble_term + dark_energy_term + inflation_term
    
    def _time_to_redshift(self, t: float) -> float:
        """
        Convierte tiempo de simulación a redshift cosmológico.
        
        Args:
            t: Tiempo de simulación
        
        Returns:
            Redshift z
        """
        # Mapeo lineal simplificado
        t_normalized = t / self.config.get('max_time', 10.0)
        z = self.redshift_initial * (1 - t_normalized) + self.redshift_final * t_normalized
        return max(z, 0.0)
    
    def _hubble_expansion_term(self, field: np.ndarray, redshift: float) -> np.ndarray:
        """
        Término de expansión de Hubble.
        
        Args:
            field: Campo actual
            redshift: Redshift actual
        
        Returns:
            Término de expansión
        """
        # Factor de escala
        scale_factor = 1.0 / (1.0 + redshift)
        
        # Parámetro de Hubble a este redshift
        hubble_z = self.hubble_constant * np.sqrt(
            self.omega_matter * (1 + redshift)**3 + self.omega_lambda
        )
        
        # Término proporcional al campo
        return -hubble_z * scale_factor * field
    
    def _dark_energy_term(self, field: np.ndarray, redshift: float) -> np.ndarray:
        """
        Término de energía oscura.
        
        Args:
            field: Campo actual
            redshift: Redshift actual
        
        Returns:
            Término de energía oscura
        """
        # La energía oscura domina a redshifts bajos
        dark_energy_strength = self.omega_lambda / (1 + redshift)**2
        
        # Término repulsivo (acelera la expansión)
        return dark_energy_strength * field
    
    def _inflation_term(self, field: np.ndarray, redshift: float) -> np.ndarray:
        """
        Término de inflación para épocas tempranas.
        
        Args:
            field: Campo actual
            redshift: Redshift actual
        
        Returns:
            Término inflacionario
        """
        # Solo activo a redshifts muy altos
        if redshift < 10.0:
            return np.zeros_like(field)
        
        # Término exponencial característico de inflación
        inflation_strength = self.inflation_amplitude * np.exp(-(redshift - 100) / 10)
        
        return inflation_strength * field
    
    def compute_power_spectrum(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el espectro de potencia del campo.
        
        Args:
            field: Campo 3D
        
        Returns:
            Tuple (k, P(k)) con números de onda y espectro de potencia
        """
        nx, ny, nz = field.shape
        
        # Transformada de Fourier
        field_fourier = np.fft.fftn(field)
        
        # Espectro de potencia
        power_3d = np.abs(field_fourier)**2
        
        # Números de onda
        kx = np.fft.fftfreq(nx, d=self.grid_params['dx'])
        ky = np.fft.fftfreq(ny, d=self.grid_params['dy'])
        kz = np.fft.fftfreq(nz, d=self.grid_params['dz'])
        
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # Promediar en cáscaras esféricas
        k_bins = np.logspace(np.log10(np.min(k_magnitude[k_magnitude > 0])), 
                           np.log10(np.max(k_magnitude)), 50)
        
        power_spectrum = np.zeros(len(k_bins) - 1)
        k_centers = np.zeros(len(k_bins) - 1)
        
        for i in range(len(k_bins) - 1):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
            if np.any(mask):
                power_spectrum[i] = np.mean(power_3d[mask])
                k_centers[i] = np.mean(k_magnitude[mask])
        
        return k_centers, power_spectrum
    
    def compute_correlation_function(self, field: np.ndarray, 
                                   max_distance: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la función de correlación espacial.
        
        Args:
            field: Campo 3D
            max_distance: Distancia máxima para calcular correlaciones
        
        Returns:
            Tuple (r, ξ(r)) con distancias y función de correlación
        """
        if max_distance is None:
            max_distance = self.box_size / 4
        
        # Función de correlación usando FFT
        field_fourier = np.fft.fftn(field)
        power_3d = np.abs(field_fourier)**2
        correlation_3d = np.fft.ifftn(power_3d).real
        
        # Centrar la función de correlación
        correlation_3d = np.fft.fftshift(correlation_3d)
        
        # Coordenadas radiales
        nx, ny, nz = field.shape
        x = np.linspace(-self.box_size/2, self.box_size/2, nx)
        y = np.linspace(-self.box_size/2, self.box_size/2, ny)
        z = np.linspace(-self.box_size/2, self.box_size/2, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Promediar en cáscaras esféricas
        r_bins = np.linspace(0, max_distance, 50)
        correlation_function = np.zeros(len(r_bins) - 1)
        r_centers = np.zeros(len(r_bins) - 1)
        
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
            if np.any(mask):
                correlation_function[i] = np.mean(correlation_3d[mask])
                r_centers[i] = (r_bins[i] + r_bins[i + 1]) / 2
        
        # Normalizar
        correlation_function = correlation_function / correlation_function[0] - 1
        
        return r_centers, correlation_function
    
    def analyze_structure_formation(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Analiza la formación de estructuras en el campo.
        
        Args:
            field: Campo 3D
        
        Returns:
            Diccionario con métricas de estructura
        """
        # Estadísticas básicas
        mean_field = np.mean(field)
        std_field = np.std(field)
        
        # Función de correlación
        r, xi = self.compute_correlation_function(field)
        
        # Espectro de potencia
        k, pk = self.compute_power_spectrum(field)
        
        # Longitud de correlación
        # Definida como la escala donde ξ(r) = 0.5
        correlation_length = 0.0
        if len(r) > 0 and len(xi) > 0:
            idx = np.where(xi > 0.5)[0]
            if len(idx) > 0:
                correlation_length = r[idx[-1]]
        
        # Análisis de densidad
        density_contrast = (field - mean_field) / mean_field
        
        # Regiones de sobredensidad (potenciales halos)
        overdensity_threshold = 2.0 * std_field
        overdense_regions = np.sum(density_contrast > overdensity_threshold)
        
        # Análisis topológico simplificado
        # Genus estadístico (medida de conectividad)
        genus = self._compute_genus(field)
        
        return {
            'mean_field': mean_field,
            'field_variance': std_field**2,
            'correlation_length': correlation_length,
            'overdense_regions': overdense_regions,
            'genus': genus,
            'power_spectrum': {'k': k, 'P_k': pk},
            'correlation_function': {'r': r, 'xi': xi}
        }
    
    def _compute_genus(self, field: np.ndarray) -> float:
        """
        Calcula el genus estadístico del campo.
        
        Args:
            field: Campo 3D
        
        Returns:
            Valor del genus
        """
        # Implementación simplificada del genus
        # usando la fórmula de Euler-Poincaré
        
        threshold = np.mean(field)
        binary_field = field > threshold
        
        # Conteo de componentes conectadas (aproximación)
        # En una implementación completa se usaría algoritmos más sofisticados
        
        # Por ahora, retornamos una medida simplificada
        return np.sum(binary_field) / field.size
    
    def validate_parameters(self) -> List[str]:
        """
        Valida los parámetros cosmológicos.
        
        Returns:
            Lista de mensajes de validación
        """
        warnings = []
        
        # Validar parámetros cosmológicos estándar
        if self.omega_matter + self.omega_lambda != 1.0:
            warnings.append("Ω_m + Ω_Λ ≠ 1: Universo no plano")
        
        if self.hubble_constant < 50 or self.hubble_constant > 100:
            warnings.append(f"Constante de Hubble inusual: {self.hubble_constant}")
        
        if self.omega_baryon > self.omega_matter:
            warnings.append("Ω_b > Ω_m: Inconsistencia en componentes de materia")
        
        # Validar parámetros MFSU
        if self.alpha <= 0:
            warnings.append("α ≤ 0: Parámetro de difusión anómala inválido")
        
        if self.beta < 0:
            warnings.append("β < 0: Intensidad de ruido negativa")
        
        if self.hurst_exponent <= 0 or self.hurst_exponent >= 1:
            warnings.append("H no está en (0,1): Exponente de Hurst inválido")
        
        # Validar escalas
        if self.redshift_initial <= self.redshift_final:
            warnings.append("z_inicial ≤ z_final: Evolución temporal inconsistente")
        
        return warnings
    
    def get_application_info(self) -> Dict[str, Any]:
        """
        Información sobre la aplicación cosmológica.
        
        Returns:
            Diccionario con información de la aplicación
        """
        return {
            'name': self.application_name,
            'version': '1.0.0',
            'description': 'Simulación cosmológica usando el modelo MFSU',
            'parameters': {
                'cosmological': {
                    'H0': self.hubble_constant,
                    'Ω_m': self.omega_matter,
                    'Ω_Λ': self.omega_lambda,
                    'Ω_b': self.omega_baryon,
                    'n_s': self.spectral_index
                },
                'mfsu': {
                    'α': self.alpha,
                    'β': self.beta,
                    'γ': self.gamma,
                    'H': self.hurst_exponent
                },
                'simulation': {
                    'box_size': self.box_size,
                    'grid_size': self.grid_params['nx'],
                    'z_initial': self.redshift_initial,
                    'z_final': self.redshift_final
                }
            },
            'applications': [
                'Large-scale structure formation',
                'Galaxy clustering analysis',
                'Dark matter halo formation',
                'Cosmic web evolution',
                'Primordial fluctuation analysis'
            ]
        }
