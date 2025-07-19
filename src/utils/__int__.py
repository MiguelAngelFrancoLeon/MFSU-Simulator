"""
MFSU Simulator - Utilities Module
==================================

Este módulo proporciona utilidades comunes para el simulador MFSU
(Modelo Fractal Estocástico Unificado).

Incluye:
- Entrada/salida de datos
- Validación de parámetros
- Sistema de logging
- Constantes físicas
- Funciones auxiliares comunes

Author: MFSU Development Team
Version: 1.0.0
License: MIT
"""

from .constants import *
from .data_io import (
    load_simulation_data,
    save_simulation_data,
    load_configuration,
    save_configuration,
    export_results,
    import_experimental_data,
    create_output_directory,
    get_data_path,
    DataFormat,
    SimulationData
)

from .parameter_validation import (
    validate_mfsu_parameters,
    validate_numerical_parameters,
    validate_boundary_conditions,
    validate_initial_conditions,
    ParameterError,
    ParameterValidator,
    get_parameter_bounds,
    normalize_parameters
)

from .logger import (
    setup_logging,
    get_logger,
    log_simulation_start,
    log_simulation_end,
    log_progress,
    LogLevel,
    MFSULogger
)

# Versión del módulo utils
__version__ = "1.0.0"

# Metadatos del proyecto
__project_info__ = {
    "name": "MFSU Simulator",
    "version": "1.0.0",
    "description": "Unified Stochastic Fractal Model Simulator",
    "equation": "∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)",
    "author": "MFSU Development Team",
    "license": "MIT"
}

# Configuración por defecto
DEFAULT_CONFIG = {
    "simulation": {
        "default_parameters": {
            "alpha": 0.5,    # Parámetro del operador fraccionario
            "beta": 0.1,     # Intensidad del ruido fractal
            "gamma": 0.01,   # Coeficiente no lineal
            "hurst": 0.7     # Exponente de Hurst
        },
        "numerical": {
            "dt": 0.01,      # Paso temporal
            "dx": 0.1,       # Resolución espacial
            "grid_size": 100, # Tamaño de la grilla
            "max_time": 10.0  # Tiempo máximo de simulación
        }
    }
}

# Funciones auxiliares comunes
def get_version():
    """
    Retorna la versión del simulador MFSU.
    
    Returns:
        str: Versión del simulador
    """
    return __version__

def get_project_info():
    """
    Retorna información del proyecto MFSU.
    
    Returns:
        dict: Diccionario con información del proyecto
    """
    return __project_info__.copy()

def print_banner():
    """
    Imprime el banner del simulador MFSU.
    """
    banner = f"""
    ╔════════════════════════════════════════════════════════════╗
    ║                    MFSU SIMULATOR v{__version__}                    ║
    ║            Modelo Fractal Estocástico Unificado           ║
    ║                                                            ║
    ║  Ecuación: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f  ║
    ║                                                            ║
    ║  Aplicaciones:                                             ║
    ║  • Superconductividad                                      ║
    ║  • Dinámica de gases                                       ║
    ║  • Cosmología                                              ║
    ║  • Análisis fractal                                        ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """
    Verifica que todas las dependencias estén instaladas.
    
    Returns:
        bool: True si todas las dependencias están disponibles
        
    Raises:
        ImportError: Si falta alguna dependencia crítica
    """
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'plotly', 
        'pandas', 'numba', 'pyfftw', 'h5py', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Faltan las siguientes dependencias: {', '.join(missing_packages)}\n"
            f"Instálalas con: pip install {' '.join(missing_packages)}"
        )
    
    return True

def initialize_mfsu_environment():
    """
    Inicializa el entorno del simulador MFSU.
    
    Esta función:
    - Verifica las dependencias
    - Configura el logging
    - Crea directorios necesarios
    - Imprime el banner
    """
    try:
        # Verificar dependencias
        check_dependencies()
        
        # Configurar logging
        setup_logging()
        logger = get_logger(__name__)
        
        # Imprimir banner
        print_banner()
        
        # Crear directorios de salida si no existen
        create_output_directory()
        
        logger.info("Entorno MFSU inicializado correctamente")
        return True
        
    except Exception as e:
        print(f"Error inicializando el entorno MFSU: {e}")
        return False

# Aliases para compatibilidad
load_data = load_simulation_data
save_data = save_simulation_data
validate_params = validate_mfsu_parameters

# Lista de elementos exportados
__all__ = [
    # Constantes
    'PHYSICAL_CONSTANTS', 'MATHEMATICAL_CONSTANTS', 'DEFAULT_TOLERANCES',
    
    # Data I/O
    'load_simulation_data', 'save_simulation_data', 'load_configuration',
    'save_configuration', 'export_results', 'import_experimental_data',
    'create_output_directory', 'get_data_path', 'DataFormat', 'SimulationData',
    
    # Parameter validation
    'validate_mfsu_parameters', 'validate_numerical_parameters', 
    'validate_boundary_conditions', 'validate_initial_conditions',
    'ParameterError', 'ParameterValidator', 'get_parameter_bounds',
    'normalize_parameters',
    
    # Logging
    'setup_logging', 'get_logger', 'log_simulation_start', 'log_simulation_end',
    'log_progress', 'LogLevel', 'MFSULogger',
    
    # Utilidades generales
    'get_version', 'get_project_info', 'print_banner', 'check_dependencies',
    'initialize_mfsu_environment', 'DEFAULT_CONFIG',
    
    # Aliases
    'load_data', 'save_data', 'validate_params'
]
