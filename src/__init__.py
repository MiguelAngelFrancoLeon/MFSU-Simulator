"""
MFSU Simulator - Unified Stochastic Fractal Model Simulator

Este paquete implementa un simulador completo para el Modelo Fractal Estocástico Unificado (MFSU)
basado en la ecuación diferencial parcial estocástica:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Donde:
- ψ(x,t): Campo principal
- α: Parámetro de difusión fractal
- β: Intensidad del ruido fractal
- γ: Coeficiente no lineal
- ξ_H(x,t): Ruido fractal con exponente de Hurst H
- f(x,t): Término de forzamiento externo

Aplicaciones:
- Superconductividad
- Dinámica de gases
- Cosmología
- Análisis fractal

Author: MFSU Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "MFSU Development Team"
__email__ = "mfsu@example.com"
__license__ = "MIT"
__description__ = "Unified Stochastic Fractal Model Simulator"

# Importaciones principales del core
from .core.mfsu_equation import MFSUEquation
from .core.fractional_operators import FractionalOperator
from .core.stochastic_processes import StochasticProcess, FractalNoise
from .core.nonlinear_dynamics import NonlinearTerm
from .core.numerical_methods import NumericalSolver

# Importaciones de aplicaciones
from .applications.base_application import BaseApplication
from .applications.superconductivity import SuperconductivityApp
from .applications.gas_dynamics import GasDynamicsApp
from .applications.cosmology import CosmologyApp

# Importaciones de simulación
from .simulation.simulator import MFSUSimulator
from .simulation.grid_manager import GridManager
from .simulation.time_evolution import TimeEvolution
from .simulation.boundary_conditions import BoundaryConditions

# Importaciones de análisis
from .analysis.fractal_analysis import FractalAnalyzer
from .analysis.spectral_analysis import SpectralAnalyzer
from .analysis.statistical_analysis import StatisticalAnalyzer
from .analysis.visualization import Visualizer

# Importaciones de utilidades
from .utils.data_io import DataIO
from .utils.parameter_validation import ParameterValidator
from .utils.logger import get_logger
from .utils.constants import PhysicalConstants

# Configuración de logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Exports públicos
__all__ = [
    # Core
    'MFSUEquation',
    'FractionalOperator',
    'StochasticProcess',
    'FractalNoise',
    'NonlinearTerm',
    'NumericalSolver',
    
    # Applications
    'BaseApplication',
    'SuperconductivityApp',
    'GasDynamicsApp',
    'CosmologyApp',
    
    # Simulation
    'MFSUSimulator',
    'GridManager',
    'TimeEvolution',
    'BoundaryConditions',
    
    # Analysis
    'FractalAnalyzer',
    'SpectralAnalyzer',
    'StatisticalAnalyzer',
    'Visualizer',
    
    # Utils
    'DataIO',
    'ParameterValidator',
    'get_logger',
    'PhysicalConstants',
    
    # Convenience functions
    'create_simulator',
    'run_simulation',
    'analyze_results',
]

# Funciones de conveniencia
def create_simulator(application_type='base', **kwargs):
    """
    Crea un simulador MFSU con configuración por defecto.
    
    Parameters:
    -----------
    application_type : str
        Tipo de aplicación ('base', 'superconductivity', 'gas_dynamics', 'cosmology')
    **kwargs : dict
        Parámetros adicionales para la simulación
        
    Returns:
    --------
    MFSUSimulator
        Instancia del simulador configurado
    """
    # Mapeo de tipos de aplicación
    app_mapping = {
        'base': BaseApplication,
        'superconductivity': SuperconductivityApp,
        'gas_dynamics': GasDynamicsApp,
        'cosmology': CosmologyApp
    }
    
    if application_type not in app_mapping:
        raise ValueError(f"Tipo de aplicación no válido: {application_type}")
    
    # Crear aplicación
    app_class = app_mapping[application_type]
    application = app_class(**kwargs)
    
    # Crear simulador
    simulator = MFSUSimulator(application=application)
    
    return simulator

def run_simulation(simulator=None, application_type='base', **kwargs):
    """
    Ejecuta una simulación MFSU completa.
    
    Parameters:
    -----------
    simulator : MFSUSimulator, optional
        Simulador pre-configurado. Si None, se crea uno nuevo.
    application_type : str
        Tipo de aplicación si se crea un simulador nuevo
    **kwargs : dict
        Parámetros para la simulación
        
    Returns:
    --------
    dict
        Resultados de la simulación
    """
    if simulator is None:
        simulator = create_simulator(application_type, **kwargs)
    
    # Ejecutar simulación
    results = simulator.run(**kwargs)
    
    return results

def analyze_results(results, analysis_type='full', **kwargs):
    """
    Analiza los resultados de una simulación MFSU.
    
    Parameters:
    -----------
    results : dict
        Resultados de la simulación
    analysis_type : str
        Tipo de análisis ('fractal', 'spectral', 'statistical', 'full')
    **kwargs : dict
        Parámetros adicionales para el análisis
        
    Returns:
    --------
    dict
        Resultados del análisis
    """
    analysis_results = {}
    
    if analysis_type in ['fractal', 'full']:
        fractal_analyzer = FractalAnalyzer()
        analysis_results['fractal'] = fractal_analyzer.analyze(results, **kwargs)
    
    if analysis_type in ['spectral', 'full']:
        spectral_analyzer = SpectralAnalyzer()
        analysis_results['spectral'] = spectral_analyzer.analyze(results, **kwargs)
    
    if analysis_type in ['statistical', 'full']:
        statistical_analyzer = StatisticalAnalyzer()
        analysis_results['statistical'] = statistical_analyzer.analyze(results, **kwargs)
    
    return analysis_results

# Información del paquete
def get_package_info():
    """
    Retorna información del paquete MFSU.
    
    Returns:
    --------
    dict
        Información del paquete
    """
    return {
        'name': 'mfsu-simulator',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'description': __description__,
        'equation': '∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)',
        'applications': ['superconductivity', 'gas_dynamics', 'cosmology'],
        'features': [
            'Operadores fraccionarios',
            'Procesos estocásticos',
            'Dinámicas no lineales',
            'Análisis fractal',
            'Visualización avanzada'
        ]
    }

# Validación de dependencias
def check_dependencies():
    """
    Verifica que todas las dependencias estén instaladas correctamente.
    
    Returns:
    --------
    dict
        Estado de las dependencias
    """
    dependencies = {
        'numpy': False,
        'scipy': False,
        'matplotlib': False,
        'plotly': False,
        'pandas': False,
        'numba': False,
        'pyfftw': False,
        'h5py': False,
        'pyyaml': False,
        'tqdm': False,
        'scikit-learn': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

# Configuración por defecto
DEFAULT_CONFIG = {
    'simulation': {
        'default_parameters': {
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.7
        },
        'numerical': {
            'dt': 0.01,
            'dx': 0.1,
            'grid_size': 100,
            'max_time': 10.0
        }
    },
    'applications': {
        'superconductivity': {
            'temperature_range': [1, 300],
            'material_parameters': {}
        },
        'gas_dynamics': {
            'reynolds_number': 1000,
            'mach_number': 0.3
        },
        'cosmology': {
            'hubble_constant': 70,
            'omega_matter': 0.3
        }
    }
}

# Función de inicialización
def initialize_mfsu(config=None, verbose=True):
    """
    Inicializa el entorno MFSU con configuración personalizada.
    
    Parameters:
    -----------
    config : dict, optional
        Configuración personalizada
    verbose : bool
        Si imprimir información de inicialización
        
    Returns:
    --------
    dict
        Estado de la inicialización
    """
    if verbose:
        print(f"Inicializando MFSU Simulator v{__version__}")
        print(f"Ecuación: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)")
    
    # Verificar dependencias
    deps = check_dependencies()
    missing_deps = [dep for dep, status in deps.items() if not status]
    
    if missing_deps and verbose:
        print(f"Advertencia: Dependencias faltantes: {missing_deps}")
    
    # Configurar logging
    logger = get_logger('mfsu')
    logger.info(f"MFSU Simulator v{__version__} inicializado")
    
    # Aplicar configuración
    if config:
        # Merge con configuración por defecto
        import copy
        merged_config = copy.deepcopy(DEFAULT_CONFIG)
        merged_config.update(config)
        config = merged_config
    else:
        config = DEFAULT_CONFIG
    
    return {
        'status': 'initialized',
        'version': __version__,
        'dependencies': deps,
        'config': config,
        'missing_dependencies': missing_deps
    }

# Ejecutar inicialización automática
_init_status = initialize_mfsu(verbose=False)
