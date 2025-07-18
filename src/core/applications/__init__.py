"""
Módulo de Aplicaciones del Simulador MFSU
========================================

Este módulo contiene las implementaciones específicas de aplicaciones
del Modelo Fractal Estocástico Unificado (MFSU) para diferentes dominios:

- Superconductividad
- Dinámica de gases
- Cosmología

Fórmula MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Donde:
- α: parámetro de difusión fraccional
- β: intensidad del ruido fractal
- γ: término no lineal
- ξ_H: ruido fractal con exponente de Hurst H
- f(x,t): término de forzamiento externo

Autor: MFSU Development Team
Versión: 1.0.0
"""

from .base_application import BaseApplication
from .superconductivity import SuperconductivityApplication
from .gas_dynamics import GasDynamicsApplication
from .cosmology import CosmologyApplication

# Versión del módulo
__version__ = "1.0.0"

# Aplicaciones disponibles
__all__ = [
    'BaseApplication',
    'SuperconductivityApplication', 
    'GasDynamicsApplication',
    'CosmologyApplication',
    'ApplicationRegistry',
    'get_application',
    'list_applications'
]

# Registro de aplicaciones disponibles
APPLICATION_REGISTRY = {
    'superconductivity': {
        'class': SuperconductivityApplication,
        'name': 'Superconductivity',
        'description': 'Análisis de transiciones superconductoras usando MFSU',
        'domains': ['condensed_matter', 'material_science'],
        'parameters': {
            'temperature_range': [1, 300],
            'default_alpha': 0.5,
            'default_beta': 0.1,
            'default_gamma': 0.01
        }
    },
    'gas_dynamics': {
        'class': GasDynamicsApplication,
        'name': 'Gas Dynamics',
        'description': 'Simulación de dinámica de gases con efectos fractales',
        'domains': ['fluid_dynamics', 'aerodynamics'],
        'parameters': {
            'reynolds_number': 1000,
            'mach_number': 0.3,
            'default_alpha': 0.6,
            'default_beta': 0.15,
            'default_gamma': 0.02
        }
    },
    'cosmology': {
        'class': CosmologyApplication,
        'name': 'Cosmology',
        'description': 'Aplicaciones cosmológicas del modelo MFSU',
        'domains': ['astrophysics', 'cosmology'],
        'parameters': {
            'hubble_constant': 70,
            'omega_matter': 0.3,
            'default_alpha': 0.4,
            'default_beta': 0.05,
            'default_gamma': 0.005
        }
    }
}


class ApplicationRegistry:
    """
    Registro centralizado de aplicaciones MFSU disponibles.
    
    Esta clase gestiona el registro y acceso a todas las aplicaciones
    implementadas del simulador MFSU.
    """
    
    def __init__(self):
        self._applications = APPLICATION_REGISTRY.copy()
    
    def register(self, name, application_class, description=None, 
                 domains=None, parameters=None):
        """
        Registra una nueva aplicación en el sistema.
        
        Args:
            name (str): Nombre identificador de la aplicación
            application_class (class): Clase de la aplicación
            description (str, optional): Descripción de la aplicación
            domains (list, optional): Dominios científicos aplicables
            parameters (dict, optional): Parámetros por defecto
        """
        self._applications[name] = {
            'class': application_class,
            'name': name.replace('_', ' ').title(),
            'description': description or f"Aplicación {name}",
            'domains': domains or [],
            'parameters': parameters or {}
        }
    
    def get_application(self, name):
        """
        Obtiene una aplicación registrada.
        
        Args:
            name (str): Nombre de la aplicación
            
        Returns:
            class: Clase de la aplicación
            
        Raises:
            KeyError: Si la aplicación no existe
        """
        if name not in self._applications:
            raise KeyError(f"Aplicación '{name}' no encontrada. "
                          f"Disponibles: {list(self._applications.keys())}")
        return self._applications[name]['class']
    
    def list_applications(self):
        """
        Lista todas las aplicaciones disponibles.
        
        Returns:
            dict: Diccionario con información de las aplicaciones
        """
        return {name: {
            'name': info['name'],
            'description': info['description'],
            'domains': info['domains']
        } for name, info in self._applications.items()}
    
    def get_application_info(self, name):
        """
        Obtiene información detallada de una aplicación.
        
        Args:
            name (str): Nombre de la aplicación
            
        Returns:
            dict: Información completa de la aplicación
        """
        if name not in self._applications:
            raise KeyError(f"Aplicación '{name}' no encontrada")
        return self._applications[name].copy()
    
    def create_application(self, name, **kwargs):
        """
        Crea una instancia de la aplicación especificada.
        
        Args:
            name (str): Nombre de la aplicación
            **kwargs: Argumentos adicionales para la aplicación
            
        Returns:
            BaseApplication: Instancia de la aplicación
        """
        app_class = self.get_application(name)
        app_info = self.get_application_info(name)
        
        # Combinar parámetros por defecto con los proporcionados
        params = app_info.get('parameters', {}).copy()
        params.update(kwargs)
        
        return app_class(**params)


# Instancia global del registro
_registry = ApplicationRegistry()

# Funciones de conveniencia para acceso global
def get_application(name):
    """
    Obtiene una clase de aplicación por nombre.
    
    Args:
        name (str): Nombre de la aplicación
        
    Returns:
        class: Clase de la aplicación
    """
    return _registry.get_application(name)


def list_applications():
    """
    Lista todas las aplicaciones disponibles.
    
    Returns:
        dict: Diccionario con información de las aplicaciones
    """
    return _registry.list_applications()


def create_application(name, **kwargs):
    """
    Crea una instancia de aplicación.
    
    Args:
        name (str): Nombre de la aplicación
        **kwargs: Parámetros de configuración
        
    Returns:
        BaseApplication: Instancia de la aplicación
    """
    return _registry.create_application(name, **kwargs)


def get_application_info(name):
    """
    Obtiene información detallada de una aplicación.
    
    Args:
        name (str): Nombre de la aplicación
        
    Returns:
        dict: Información de la aplicación
    """
    return _registry.get_application_info(name)


def register_application(name, application_class, **kwargs):
    """
    Registra una nueva aplicación en el sistema.
    
    Args:
        name (str): Nombre de la aplicación
        application_class (class): Clase de la aplicación
        **kwargs: Información adicional de la aplicación
    """
    _registry.register(name, application_class, **kwargs)


# Validación de importaciones
def validate_applications():
    """
    Valida que todas las aplicaciones registradas puedan ser importadas.
    
    Returns:
        dict: Resultados de la validación
    """
    results = {}
    for name, info in APPLICATION_REGISTRY.items():
        try:
            app_class = info['class']
            # Verificar que hereda de BaseApplication
            if not issubclass(app_class, BaseApplication):
                results[name] = {
                    'status': 'error',
                    'message': f"Clase {app_class.__name__} no hereda de BaseApplication"
                }
            else:
                results[name] = {
                    'status': 'success',
                    'class': app_class.__name__,
                    'description': info['description']
                }
        except Exception as e:
            results[name] = {
                'status': 'error',
                'message': str(e)
            }
    
    return results


# Información del módulo
def get_module_info():
    """
    Obtiene información sobre el módulo de aplicaciones.
    
    Returns:
        dict: Información del módulo
    """
    return {
        'version': __version__,
        'applications_count': len(APPLICATION_REGISTRY),
        'applications': list(APPLICATION_REGISTRY.keys()),
        'mfsu_equation': "∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)",
        'description': __doc__.split('\n')[1] if __doc__ else "Módulo de aplicaciones MFSU"
    }


# Ejecutar validación al importar (opcional)
if __name__ == "__main__":
    print("=== Módulo de Aplicaciones MFSU ===")
    print(f"Versión: {__version__}")
    print(f"Aplicaciones disponibles: {len(APPLICATION_REGISTRY)}")
    
    for name, info in APPLICATION_REGISTRY.items():
        print(f"  - {info['name']}: {info['description']}")
    
    print("\n=== Validación de Aplicaciones ===")
    validation_results = validate_applications()
    for name, result in validation_results.items():
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"  {status} {name}: {result.get('message', result.get('class', 'OK'))}")
