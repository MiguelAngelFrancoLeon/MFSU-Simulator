"""
Constantes físicas y parámetros del simulador MFSU
Modelo Fractal Estocástico Unificado

Ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Autor: MFSU Development Team
Versión: 1.0.0
"""

import numpy as np
from typing import Dict, Any

# =============================================================================
# CONSTANTES FÍSICAS UNIVERSALES
# =============================================================================

# Constantes fundamentales
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
HBAR = PLANCK_CONSTANT / (2 * np.pi)  # ℏ = h/2π
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
SPEED_OF_LIGHT = 299792458  # m/s
ELECTRON_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹

# Constantes matemáticas
PI = np.pi
E = np.e
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# =============================================================================
# PARÁMETROS MFSU POR DEFECTO
# =============================================================================

class MFSUDefaults:
    """Parámetros por defecto para la ecuación MFSU"""
    
    # Parámetros principales de la ecuación MFSU
    ALPHA = 0.5          # Exponente fractal del operador fraccionario
    BETA = 0.1           # Intensidad del ruido estocástico
    GAMMA = 0.01         # Coeficiente no lineal cúbico
    HURST_EXPONENT = 0.7 # Exponente de Hurst para ruido fractal
    
    # Límites válidos para parámetros
    ALPHA_MIN = 0.1
    ALPHA_MAX = 2.0
    BETA_MIN = 0.0
    BETA_MAX = 1.0
    GAMMA_MIN = 0.0
    GAMMA_MAX = 0.1
    HURST_MIN = 0.0
    HURST_MAX = 1.0

# =============================================================================
# PARÁMETROS NUMÉRICOS
# =============================================================================

class NumericalDefaults:
    """Parámetros numéricos por defecto"""
    
    # Discretización temporal
    DT = 0.01            # Paso temporal
    MAX_TIME = 10.0      # Tiempo máximo de simulación
    
    # Discretización espacial
    DX = 0.1             # Paso espacial
    GRID_SIZE = 100      # Tamaño de grilla (NxN para 2D)
    DOMAIN_SIZE = 10.0   # Tamaño del dominio [-L/2, L/2]
    
    # Tolerancias numéricas
    TOLERANCE = 1e-8     # Tolerancia para convergencia
    MAX_ITERATIONS = 1000 # Máximo número de iteraciones
    
    # Parámetros de estabilidad
    CFL_NUMBER = 0.5     # Número de Courant-Friedrichs-Lewy

# =============================================================================
# CONSTANTES PARA APLICACIONES ESPECÍFICAS
# =============================================================================

class SuperconductivityConstants:
    """Constantes para aplicaciones en superconductividad"""
    
    # Temperaturas características
    ROOM_TEMPERATURE = 300.0      # K
    LIQUID_NITROGEN_TEMP = 77.0   # K
    LIQUID_HELIUM_TEMP = 4.2      # K
    
    # Parámetros de materiales superconductores típicos
    COOPER_PAIR_SIZE = 1e-6       # m (longitud de coherencia típica)
    PENETRATION_DEPTH = 1e-7      # m (profundidad de penetración)
    
    # Campos críticos típicos
    HC1_TYPICAL = 0.01           # T (campo crítico inferior)
    HC2_TYPICAL = 10.0           # T (campo crítico superior)

class FluidDynamicsConstants:
    """Constantes para dinámica de fluidos"""
    
    # Números adimensionales típicos
    REYNOLDS_LOW = 100
    REYNOLDS_MODERATE = 1000
    REYNOLDS_HIGH = 10000
    
    MACH_SUBSONIC = 0.3
    MACH_TRANSONIC = 1.0
    MACH_SUPERSONIC = 2.0
    
    # Propiedades del aire a condiciones estándar
    AIR_DENSITY = 1.225          # kg/m³
    AIR_VISCOSITY = 1.81e-5      # Pa⋅s
    SOUND_SPEED_AIR = 343        # m/s

class CosmologyConstants:
    """Constantes cosmológicas"""
    
    # Parámetros cosmológicos estándar (Planck 2018)
    HUBBLE_CONSTANT = 67.4       # km/s/Mpc
    OMEGA_MATTER = 0.315         # Densidad de materia
    OMEGA_LAMBDA = 0.685         # Constante cosmológica
    OMEGA_BARYON = 0.049         # Densidad bariónica
    
    # Escalas cosmológicas
    HUBBLE_TIME = 14.4e9         # años (tiempo de Hubble)
    PLANCK_LENGTH = 1.616e-35    # m
    PLANCK_TIME = 5.391e-44      # s

# =============================================================================
# CONSTANTES DE ANÁLISIS Y VISUALIZACIÓN
# =============================================================================

class AnalysisConstants:
    """Constantes para análisis de datos"""
    
    # Parámetros para análisis fractal
    FRACTAL_DIM_MIN = 1.0
    FRACTAL_DIM_MAX = 3.0
    
    # Ventanas espectrales
    SPECTRAL_WINDOWS = ['hann', 'blackman', 'hamming', 'bartlett']
    DEFAULT_WINDOW = 'hann'
    
    # Parámetros estadísticos
    CONFIDENCE_LEVEL = 0.95
    N_BOOTSTRAP = 1000

class VisualizationConstants:
    """Constantes para visualización"""
    
    # Mapas de colores por defecto
    DEFAULT_COLORMAP = 'viridis'
    PHASE_COLORMAP = 'hsv'
    MAGNITUDE_COLORMAP = 'plasma'
    
    # Resolución de figuras
    DPI = 300
    FIGURE_SIZE = (10, 8)
    
    # Colores para gráficos
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# =============================================================================
# CONFIGURACIONES DE ARCHIVOS Y FORMATOS
# =============================================================================

class FileFormats:
    """Formatos de archivo soportados"""
    
    # Formatos de datos
    DATA_FORMATS = ['.h5', '.hdf5', '.npz', '.csv', '.json']
    IMAGE_FORMATS = ['.png', '.jpg', '.pdf', '.svg']
    VIDEO_FORMATS = ['.mp4', '.avi', '.gif']
    
    # Configuraciones de exportación
    DEFAULT_DATA_FORMAT = 'h5'
    DEFAULT_IMAGE_FORMAT = 'png'
    
    # Compresión
    COMPRESSION_LEVEL = 6

# =============================================================================
# DICCIONARIOS DE CONFIGURACIÓN COMPLETOS
# =============================================================================

# Configuración completa por defecto
DEFAULT_CONFIG: Dict[str, Any] = {
    'mfsu_parameters': {
        'alpha': MFSUDefaults.ALPHA,
        'beta': MFSUDefaults.BETA,
        'gamma': MFSUDefaults.GAMMA,
        'hurst': MFSUDefaults.HURST_EXPONENT,
    },
    'numerical': {
        'dt': NumericalDefaults.DT,
        'dx': NumericalDefaults.DX,
        'grid_size': NumericalDefaults.GRID_SIZE,
        'max_time': NumericalDefaults.MAX_TIME,
        'tolerance': NumericalDefaults.TOLERANCE,
        'max_iterations': NumericalDefaults.MAX_ITERATIONS,
    },
    'analysis': {
        'confidence_level': AnalysisConstants.CONFIDENCE_LEVEL,
        'n_bootstrap': AnalysisConstants.N_BOOTSTRAP,
    },
    'visualization': {
        'colormap': VisualizationConstants.DEFAULT_COLORMAP,
        'dpi': VisualizationConstants.DPI,
        'figure_size': VisualizationConstants.FIGURE_SIZE,
    }
}

# Configuraciones específicas por aplicación
APPLICATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    'superconductivity': {
        'temperature_range': [1, SuperconductivityConstants.ROOM_TEMPERATURE],
        'field_range': [0, SuperconductivityConstants.HC2_TYPICAL],
        'coherence_length': SuperconductivityConstants.COOPER_PAIR_SIZE,
    },
    'fluid_dynamics': {
        'reynolds_number': FluidDynamicsConstants.REYNOLDS_MODERATE,
        'mach_number': FluidDynamicsConstants.MACH_SUBSONIC,
        'fluid_density': FluidDynamicsConstants.AIR_DENSITY,
    },
    'cosmology': {
        'hubble_constant': CosmologyConstants.HUBBLE_CONSTANT,
        'omega_matter': CosmologyConstants.OMEGA_MATTER,
        'omega_lambda': CosmologyConstants.OMEGA_LAMBDA,
    }
}

# =============================================================================
# FUNCIONES UTILITARIAS
# =============================================================================

def validate_mfsu_parameters(alpha: float, beta: float, gamma: float, hurst: float) -> bool:
    """
    Valida que los parámetros MFSU estén en rangos válidos
    
    Parameters:
    -----------
    alpha : float
        Exponente fractal
    beta : float
        Intensidad del ruido
    gamma : float
        Coeficiente no lineal
    hurst : float
        Exponente de Hurst
        
    Returns:
    --------
    bool
        True si todos los parámetros son válidos
    """
    return (
        MFSUDefaults.ALPHA_MIN <= alpha <= MFSUDefaults.ALPHA_MAX and
        MFSUDefaults.BETA_MIN <= beta <= MFSUDefaults.BETA_MAX and
        MFSUDefaults.GAMMA_MIN <= gamma <= MFSUDefaults.GAMMA_MAX and
        MFSUDefaults.HURST_MIN <= hurst <= MFSUDefaults.HURST_MAX
    )

def get_stability_condition(alpha: float, dt: float, dx: float) -> float:
    """
    Calcula la condición de estabilidad para la ecuación MFSU
    
    Parameters:
    -----------
    alpha : float
        Exponente fractal
    dt : float
        Paso temporal
    dx : float
        Paso espacial
        
    Returns:
    --------
    float
        Factor de estabilidad
    """
    return dt * (dx ** (-alpha))

def get_characteristic_scales(alpha: float, beta: float, gamma: float) -> Dict[str, float]:
    """
    Calcula escalas características del sistema MFSU
    
    Parameters:
    -----------
    alpha : float
        Exponente fractal
    beta : float
        Intensidad del ruido
    gamma : float
        Coeficiente no lineal
        
    Returns:
    --------
    Dict[str, float]
        Escalas características (longitud, tiempo, amplitud)
    """
    # Escalas aproximadas basadas en análisis dimensional
    length_scale = (alpha / gamma) ** (1 / alpha) if gamma > 0 else 1.0
    time_scale = (alpha / beta**2) ** (2 / alpha) if beta > 0 else 1.0
    amplitude_scale = np.sqrt(beta / gamma) if gamma > 0 else 1.0
    
    return {
        'length': length_scale,
        'time': time_scale,
        'amplitude': amplitude_scale
    }

# =============================================================================
# MENSAJES DE INFORMACIÓN
# =============================================================================

INFO_MESSAGES = {
    'welcome': "Bienvenido al Simulador MFSU v1.0.0",
    'equation': "Ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)",
    'applications': "Aplicaciones disponibles: Superconductividad, Dinámica de Fluidos, Cosmología",
    'github': "Proyecto disponible en: https://zenodo.org/records/15828185"
}

if __name__ == "__main__":
    # Pruebas básicas de validación
    print(INFO_MESSAGES['welcome'])
    print(INFO_MESSAGES['equation'])
    
    # Validar parámetros por defecto
    if validate_mfsu_parameters(
        MFSUDefaults.ALPHA, 
        MFSUDefaults.BETA, 
        MFSUDefaults.GAMMA, 
        MFSUDefaults.HURST_EXPONENT
    ):
        print("✓ Parámetros MFSU por defecto son válidos")
    
    # Mostrar escalas características
    scales = get_characteristic_scales(
        MFSUDefaults.ALPHA, 
        MFSUDefaults.BETA, 
        MFSUDefaults.GAMMA
    )
    print(f"Escalas características: {scales}")
