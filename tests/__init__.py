"""
MFSU Simulator - Test Suite Initialization
==========================================

Sistema de pruebas para el simulador del Modelo Fractal Estocástico Unificado (MFSU).
Este módulo inicializa y configura el entorno de pruebas para todos los componentes
del simulador.

Ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Autor: MFSU Development Team
Versión: 1.0.0
Fecha: 2025
"""

import os
import sys
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import pytest
import warnings

# Configurar el path para importar módulos del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Importar módulos core del proyecto
try:
    from core.mfsu_equation import MFSUEquation
    from core.fractional_operators import FractionalOperator
    from core.stochastic_processes import StochasticProcess
    from core.numerical_methods import NumericalSolver
    from utils.constants import PHYSICAL_CONSTANTS
    from utils.logger import setup_logger
except ImportError as e:
    warnings.warn(f"No se pudieron importar algunos módulos core: {e}")

# Configuración global de pruebas
TEST_CONFIG = {
    'numerical_tolerance': 1e-10,
    'convergence_tolerance': 1e-8,
    'max_iterations': 1000,
    'default_grid_size': 64,
    'default_time_steps': 100,
    'temp_dir': None,
    'random_seed': 42,
    'verbose': False
}

# Parámetros de referencia para las pruebas
DEFAULT_MFSU_PARAMS = {
    'alpha': 0.5,    # Exponente fraccionario
    'beta': 0.1,     # Intensidad del ruido fractal
    'gamma': 0.01,   # Coeficiente no lineal
    'hurst': 0.7,    # Exponente de Hurst
    'dt': 0.01,      # Paso temporal
    'dx': 0.1,       # Paso espacial
    'grid_size': 64, # Tamaño de grilla
    'max_time': 1.0  # Tiempo máximo de simulación
}

# Configuración de aplicaciones específicas para pruebas
APPLICATION_TEST_PARAMS = {
    'superconductivity': {
        'temperature_range': [10, 100],
        'critical_field': 0.5,
        'coherence_length': 1.0
    },
    'gas_dynamics': {
        'reynolds_number': 100,
        'mach_number': 0.1,
        'viscosity': 0.01
    },
    'cosmology': {
        'hubble_constant': 70,
        'omega_matter': 0.3,
        'dark_energy': 0.7
    }
}


class TestEnvironment:
    """
    Configuración y gestión del entorno de pruebas.
    """
    
    def __init__(self):
        self.temp_dir = None
        self.logger = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Configura el entorno de pruebas."""
        # Configurar directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix='mfsu_test_')
        TEST_CONFIG['temp_dir'] = self.temp_dir
        
        # Configurar logging para pruebas
        self.logger = setup_logger(
            'mfsu_tests',
            level=logging.WARNING if not TEST_CONFIG['verbose'] else logging.DEBUG,
            log_file=os.path.join(self.temp_dir, 'test.log')
        )
        
        # Configurar numpy para reproducibilidad
        np.random.seed(TEST_CONFIG['random_seed'])
        
        # Suprimir warnings innecesarios durante las pruebas
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def cleanup(self):
        """Limpia el entorno de pruebas."""
        import shutil
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# Instancia global del entorno de pruebas
_test_env = TestEnvironment()


def get_test_config() -> Dict[str, Any]:
    """Retorna la configuración actual de pruebas."""
    return TEST_CONFIG.copy()


def get_default_mfsu_params() -> Dict[str, Any]:
    """Retorna parámetros por defecto para la ecuación MFSU."""
    return DEFAULT_MFSU_PARAMS.copy()


def get_application_params(app_name: str) -> Dict[str, Any]:
    """
    Retorna parámetros de prueba para una aplicación específica.
    
    Args:
        app_name: Nombre de la aplicación ('superconductivity', 'gas_dynamics', 'cosmology')
    
    Returns:
        Dict con parámetros de la aplicación
    """
    if app_name not in APPLICATION_TEST_PARAMS:
        raise ValueError(f"Aplicación {app_name} no reconocida")
    return APPLICATION_TEST_PARAMS[app_name].copy()


def setup_test_data(grid_size: int = None) -> Dict[str, np.ndarray]:
    """
    Genera datos de prueba estándar para las simulaciones.
    
    Args:
        grid_size: Tamaño de la grilla (usa default si no se especifica)
    
    Returns:
        Dict con arrays de datos de prueba
    """
    if grid_size is None:
        grid_size = TEST_CONFIG['default_grid_size']
    
    x = np.linspace(-5, 5, grid_size)
    t = np.linspace(0, 1, TEST_CONFIG['default_time_steps'])
    
    # Condición inicial: paquete gaussiano
    psi0 = np.exp(-x**2) * np.exp(1j * x)
    
    # Función de forzamiento de prueba
    f_test = np.zeros((len(t), len(x)), dtype=complex)
    
    # Ruido fractal de prueba
    xi_test = np.random.randn(len(t), len(x))
    
    return {
        'x': x,
        't': t,
        'psi0': psi0,
        'forcing': f_test,
        'noise': xi_test
    }


def assert_mfsu_solution_properties(psi: np.ndarray, params: Dict[str, Any]):
    """
    Verifica propiedades básicas de una solución MFSU.
    
    Args:
        psi: Solución de la ecuación MFSU
        params: Parámetros utilizados en la simulación
    """
    # Verificar que no hay NaN o Inf
    assert not np.any(np.isnan(psi)), "La solución contiene NaN"
    assert not np.any(np.isinf(psi)), "La solución contiene Inf"
    
    # Verificar conservación aproximada (dependiendo del tipo de ecuación)
    if params.get('gamma', 0) == 0:  # Caso lineal
        norm_initial = np.sum(np.abs(psi[0])**2)
        norm_final = np.sum(np.abs(psi[-1])**2)
        relative_change = abs(norm_final - norm_initial) / norm_initial
        assert relative_change < 0.1, f"Norma no conservada: cambio relativo {relative_change}"


def run_convergence_test(solver_func, params: Dict[str, Any], 
                        grid_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Ejecuta un test de convergencia variando el tamaño de grilla.
    
    Args:
        solver_func: Función que resuelve la ecuación MFSU
        params: Parámetros base para la simulación
        grid_sizes: Lista de tamaños de grilla para probar
    
    Returns:
        Dict con resultados del test de convergencia
    """
    if grid_sizes is None:
        grid_sizes = [32, 64, 128]
    
    errors = []
    reference_solution = None
    
    for i, grid_size in enumerate(grid_sizes):
        test_params = params.copy()
        test_params['grid_size'] = grid_size
        test_params['dx'] = 10.0 / grid_size  # Ajustar dx proporcionalmente
        
        solution = solver_func(test_params)
        
        if i == len(grid_sizes) - 1:  # Usar la solución más fina como referencia
            reference_solution = solution
        else:
            # Interpolar para comparar con la solución de referencia
            if reference_solution is not None:
                # Implementar interpolación y cálculo de error
                # (detalles específicos dependen del tipo de solución)
                error = np.mean(np.abs(solution - reference_solution))
                errors.append(error)
    
    return {
        'grid_sizes': grid_sizes[:-1],
        'errors': errors,
        'convergence_rate': _estimate_convergence_rate(grid_sizes[:-1], errors)
    }


def _estimate_convergence_rate(grid_sizes: List[int], errors: List[float]) -> float:
    """Estima la tasa de convergencia a partir de errores y tamaños de grilla."""
    if len(errors) < 2:
        return 0.0
    
    # Usar regresión lineal en escala log-log
    log_h = np.log([10.0/g for g in grid_sizes])
    log_errors = np.log(errors)
    
    # Ajuste lineal: log(error) = p * log(h) + c
    p = np.polyfit(log_h, log_errors, 1)[0]
    return p


# Fixtures de pytest para uso común
@pytest.fixture
def mfsu_params():
    """Fixture que proporciona parámetros por defecto del MFSU."""
    return get_default_mfsu_params()


@pytest.fixture
def test_data():
    """Fixture que proporciona datos de prueba estándar."""
    return setup_test_data()


@pytest.fixture
def temp_dir():
    """Fixture que proporciona un directorio temporal."""
    return TEST_CONFIG['temp_dir']


# Configuración de pytest
def pytest_configure(config):
    """Configuración personalizada de pytest."""
    # Agregar marcadores personalizados
    config.addinivalue_line("markers", "slow: marca las pruebas como lentas")
    config.addinivalue_line("markers", "integration: pruebas de integración")
    config.addinivalue_line("markers", "convergence: pruebas de convergencia")
    config.addinivalue_line("markers", "application: pruebas específicas de aplicaciones")


def pytest_unconfigure(config):
    """Limpieza después de las pruebas."""
    _test_env.cleanup()


# Exportar símbolos principales
__all__ = [
    'TEST_CONFIG',
    'DEFAULT_MFSU_PARAMS', 
    'APPLICATION_TEST_PARAMS',
    'TestEnvironment',
    'get_test_config',
    'get_default_mfsu_params',
    'get_application_params',
    'setup_test_data',
    'assert_mfsu_solution_properties',
    'run_convergence_test'
]


# Información del módulo
__version__ = '1.0.0'
__author__ = 'MFSU Development Team'
__email__ = 'mfsu-dev@example.com'

# Logging de inicialización
if _test_env.logger:
    _test_env.logger.info("Sistema de pruebas MFSU inicializado correctamente")
    _test_env.logger.info(f"Directorio temporal: {_test_env.temp_dir}")
    _test_env.logger.info(f"Configuración de pruebas: {TEST_CONFIG}")
