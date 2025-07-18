"""
Módulo de Simulación MFSU - Modelo Fractal Estocástico Unificado

Este módulo contiene las clases y funciones principales para la simulación
de la ecuación MFSU:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Donde:
- α: parámetro de difusión fraccional
- β: intensidad del ruido fraccional
- γ: parámetro de no linealidad
- ξ_H(x,t): proceso estocástico fractal con exponente de Hurst H
- f(x,t): término de forzamiento externo

Componentes principales:
- Simulator: Motor principal de simulación
- GridManager: Gestión de grillas espaciales
- TimeEvolution: Evolución temporal del sistema
- BoundaryConditions: Condiciones de frontera
"""

from .simulator import (
    MFSUSimulator,
    SimulationConfig,
    SimulationResult,
    SimulationStatus
)

from .grid_manager import (
    GridManager,
    Grid1D,
    Grid2D,
    Grid3D,
    GridType,
    BoundaryType
)

from .time_evolution import (
    TimeEvolution,
    AdaptiveTimeStep,
    FixedTimeStep,
    RungeKutta4,
    CrankNicolson,
    FractionalEuler
)

from .boundary_conditions import (
    BoundaryConditions,
    DirichletBC,
    NeumannBC,
    PeriodicBC,
    AbsorbingBC,
    ReflectingBC
)

# Versión del módulo de simulación
__version__ = "1.0.0"

# Exportar las clases principales
__all__ = [
    # Simulador principal
    'MFSUSimulator',
    'SimulationConfig',
    'SimulationResult',
    'SimulationStatus',
    
    # Gestión de grillas
    'GridManager',
    'Grid1D',
    'Grid2D', 
    'Grid3D',
    'GridType',
    'BoundaryType',
    
    # Evolución temporal
    'TimeEvolution',
    'AdaptiveTimeStep',
    'FixedTimeStep',
    'RungeKutta4',
    'CrankNicolson',
    'FractionalEuler',
    
    # Condiciones de frontera
    'BoundaryConditions',
    'DirichletBC',
    'NeumannBC',
    'PeriodicBC',
    'AbsorbingBC',
    'ReflectingBC',
    
    # Funciones de utilidad
    'create_default_config',
    'validate_parameters',
    'run_simulation',
    'run_batch_simulation'
]

# Configuración por defecto para simulaciones
DEFAULT_CONFIG = {
    'parameters': {
        'alpha': 0.5,      # Parámetro de difusión fraccional
        'beta': 0.1,       # Intensidad del ruido fraccional
        'gamma': 0.01,     # Parámetro de no linealidad
        'hurst': 0.7,      # Exponente de Hurst para el ruido fractal
    },
    'numerical': {
        'dt': 0.01,        # Paso temporal
        'dx': 0.1,         # Resolución espacial
        'grid_size': 100,  # Tamaño de la grilla
        'max_time': 10.0,  # Tiempo máximo de simulación
        'method': 'rk4',   # Método numérico
    },
    'boundary': {
        'type': 'periodic',  # Tipo de condición de frontera
        'value': 0.0,        # Valor para condiciones Dirichlet
    },
    'output': {
        'save_interval': 10,  # Intervalo para guardar resultados
        'output_format': 'hdf5',  # Formato de salida
        'compression': True,   # Compresión de datos
    }
}

def create_default_config():
    """
    Crea una configuración por defecto para simulaciones MFSU.
    
    Returns:
        SimulationConfig: Configuración por defecto
    """
    from .simulator import SimulationConfig
    return SimulationConfig(**DEFAULT_CONFIG)

def validate_parameters(config):
    """
    Valida los parámetros de configuración de la simulación.
    
    Args:
        config (dict or SimulationConfig): Configuración a validar
        
    Returns:
        bool: True si la configuración es válida
        
    Raises:
        ValueError: Si algún parámetro no es válido
    """
    from ..utils.parameter_validation import validate_simulation_config
    return validate_simulation_config(config)

def run_simulation(config=None, initial_condition=None, **kwargs):
    """
    Ejecuta una simulación MFSU con los parámetros especificados.
    
    Args:
        config (SimulationConfig, optional): Configuración de la simulación
        initial_condition (callable or array, optional): Condición inicial
        **kwargs: Parámetros adicionales para la simulación
        
    Returns:
        SimulationResult: Resultado de la simulación
    """
    if config is None:
        config = create_default_config()
    
    # Actualizar configuración con kwargs
    if kwargs:
        config.update(kwargs)
    
    # Crear simulador
    simulator = MFSUSimulator(config)
    
    # Establecer condición inicial si se proporciona
    if initial_condition is not None:
        simulator.set_initial_condition(initial_condition)
    
    # Ejecutar simulación
    return simulator.run()

def run_batch_simulation(configs, n_jobs=1, verbose=True):
    """
    Ejecuta múltiples simulaciones en paralelo.
    
    Args:
        configs (list): Lista de configuraciones de simulación
        n_jobs (int): Número de procesos paralelos
        verbose (bool): Mostrar progreso
        
    Returns:
        list: Lista de resultados de simulación
    """
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    
    if n_jobs == 1:
        # Ejecución secuencial
        results = []
        iterator = tqdm(configs, desc="Ejecutando simulaciones") if verbose else configs
        
        for config in iterator:
            result = run_simulation(config)
            results.append(result)
            
        return results
    else:
        # Ejecución paralela
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            if verbose:
                from tqdm import tqdm
                futures = [executor.submit(run_simulation, config) for config in configs]
                results = []
                for future in tqdm(futures, desc="Ejecutando simulaciones"):
                    results.append(future.result())
            else:
                futures = [executor.submit(run_simulation, config) for config in configs]
                results = [future.result() for future in futures]
                
        return results

# Información del módulo
__author__ = "MFSU Development Team"
__email__ = "mfsu-dev@example.com"
__description__ = "Módulo de simulación para el Modelo Fractal Estocástico Unificado"
__license__ = "MIT"

# Configuración de logging para el módulo
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Verificar importaciones al cargar el módulo
try:
    from ..core import mfsu_equation, fractional_operators, stochastic_processes
    from ..utils import constants, logger as utils_logger
    logger.info("Módulo de simulación MFSU cargado correctamente")
except ImportError as e:
    logger.warning(f"Advertencia al cargar dependencias: {e}")
    logger.warning("Algunas funcionalidades pueden no estar disponibles")

# Función para obtener información del módulo
def get_module_info():
    """
    Retorna información sobre el módulo de simulación.
    
    Returns:
        dict: Información del módulo
    """
    return {
        'name': 'MFSU Simulation Module',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'license': __license__,
        'components': __all__
    }
