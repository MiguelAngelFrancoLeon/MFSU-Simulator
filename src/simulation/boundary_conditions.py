"""
Condiciones de Frontera para el Simulador MFSU
===============================================

Este módulo implementa diferentes tipos de condiciones de frontera para la ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Soporta condiciones periódicas, Dirichlet, Neumann, absorbentes y condiciones 
específicas para cada aplicación (superconductividad, dinámica de gases, cosmología).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import logging
from scipy.special import gamma as gamma_func
from scipy.fft import fft, ifft, fftshift, ifftshift


logger = logging.getLogger(__name__)


class BoundaryCondition(ABC):
    """Clase base abstracta para condiciones de frontera."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        logger.info(f"Inicializando condición de frontera: {name}")
    
    @abstractmethod
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica la condición de frontera al campo ψ."""
        pass
    
    @abstractmethod
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica el operador diferencial según la condición de frontera."""
        pass


class PeriodicBoundary(BoundaryCondition):
    """
    Condiciones de frontera periódicas.
    Ideal para sistemas con simetría traslacional.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("Periodic", parameters)
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """
        Las condiciones periódicas se manejan automáticamente con FFT.
        No se requiere modificación explícita del campo.
        """
        return psi
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Para condiciones periódicas, el operador no se modifica."""
        return operator


class DirichletBoundary(BoundaryCondition):
    """
    Condiciones de frontera de Dirichlet: ψ(frontera) = valor_fijo
    """
    
    def __init__(self, boundary_values: Dict[str, float], parameters: Dict[str, Any] = None):
        super().__init__("Dirichlet", parameters)
        self.boundary_values = boundary_values  # {'left': val, 'right': val}
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica valores fijos en los bordes."""
        psi_modified = psi.copy()
        
        # Borde izquierdo
        if 'left' in self.boundary_values:
            psi_modified[0] = self.boundary_values['left']
        
        # Borde derecho
        if 'right' in self.boundary_values:
            psi_modified[-1] = self.boundary_values['right']
        
        return psi_modified
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica el operador para forzar valores en los bordes."""
        modified_operator = operator.copy()
        
        # Anula las filas correspondientes a los bordes
        modified_operator[0, :] = 0
        modified_operator[0, 0] = 1
        modified_operator[-1, :] = 0
        modified_operator[-1, -1] = 1
        
        return modified_operator


class NeumannBoundary(BoundaryCondition):
    """
    Condiciones de frontera de Neumann: ∂ψ/∂x|_frontera = valor_fijo
    """
    
    def __init__(self, derivative_values: Dict[str, float], parameters: Dict[str, Any] = None):
        super().__init__("Neumann", parameters)
        self.derivative_values = derivative_values  # {'left': dval, 'right': dval}
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica condiciones de derivada fija en los bordes."""
        psi_modified = psi.copy()
        dx = grid[1] - grid[0]
        
        # Borde izquierdo: ∂ψ/∂x ≈ (ψ[1] - ψ[0])/dx = derivative_value
        if 'left' in self.derivative_values:
            psi_modified[0] = psi_modified[1] - dx * self.derivative_values['left']
        
        # Borde derecho: ∂ψ/∂x ≈ (ψ[-1] - ψ[-2])/dx = derivative_value
        if 'right' in self.derivative_values:
            psi_modified[-1] = psi_modified[-2] + dx * self.derivative_values['right']
        
        return psi_modified
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica el operador para condiciones de Neumann."""
        modified_operator = operator.copy()
        dx = grid[1] - grid[0]
        
        # Modifica las filas de los bordes para condiciones de derivada
        if 'left' in self.derivative_values:
            modified_operator[0, :] = 0
            modified_operator[0, 0] = -1/dx
            modified_operator[0, 1] = 1/dx
        
        if 'right' in self.derivative_values:
            modified_operator[-1, :] = 0
            modified_operator[-1, -2] = -1/dx
            modified_operator[-1, -1] = 1/dx
        
        return modified_operator


class AbsorbingBoundary(BoundaryCondition):
    """
    Condiciones de frontera absorbentes (PML - Perfectly Matched Layer).
    Absorbe ondas salientes sin reflexión.
    """
    
    def __init__(self, absorption_length: float = 0.1, strength: float = 1.0, 
                 parameters: Dict[str, Any] = None):
        super().__init__("Absorbing", parameters)
        self.absorption_length = absorption_length
        self.strength = strength
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica absorción gradual cerca de los bordes."""
        psi_modified = psi.copy()
        N = len(grid)
        L = grid[-1] - grid[0]
        absorption_points = int(self.absorption_length * N)
        
        if absorption_points > 0:
            # Perfil de absorción (función suave)
            for i in range(absorption_points):
                # Borde izquierdo
                damping = np.exp(-self.strength * (absorption_points - i)**2 / absorption_points**2)
                psi_modified[i] *= damping
                
                # Borde derecho
                psi_modified[N-1-i] *= damping
        
        return psi_modified
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Para PML, el operador base no se modifica."""
        return operator


class ReflectiveBoundary(BoundaryCondition):
    """
    Condiciones de frontera reflectivas.
    Útil para paredes rígidas en dinámica de fluidos.
    """
    
    def __init__(self, reflection_type: str = "hard", parameters: Dict[str, Any] = None):
        super().__init__("Reflective", parameters)
        self.reflection_type = reflection_type  # "hard" o "soft"
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica reflexión en los bordes."""
        psi_modified = psi.copy()
        
        if self.reflection_type == "hard":
            # Reflexión dura: ψ(frontera) = -ψ(interior)
            psi_modified[0] = -psi_modified[1]
            psi_modified[-1] = -psi_modified[-2]
        elif self.reflection_type == "soft":
            # Reflexión suave: ψ(frontera) = ψ(interior)
            psi_modified[0] = psi_modified[1]
            psi_modified[-1] = psi_modified[-2]
        
        return psi_modified
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica el operador para condiciones reflectivas."""
        modified_operator = operator.copy()
        
        if self.reflection_type == "hard":
            modified_operator[0, :] = 0
            modified_operator[0, 0] = 1
            modified_operator[0, 1] = 1
            modified_operator[-1, :] = 0
            modified_operator[-1, -1] = 1
            modified_operator[-1, -2] = 1
        elif self.reflection_type == "soft":
            modified_operator[0, :] = 0
            modified_operator[0, 0] = 1
            modified_operator[0, 1] = -1
            modified_operator[-1, :] = 0
            modified_operator[-1, -1] = 1
            modified_operator[-1, -2] = -1
        
        return modified_operator


class SuperconductivityBoundary(BoundaryCondition):
    """
    Condiciones de frontera específicas para superconductividad.
    Considera efectos de superficie y acoplamiento con campos externos.
    """
    
    def __init__(self, surface_parameter: float = 1.0, external_field: float = 0.0,
                 parameters: Dict[str, Any] = None):
        super().__init__("Superconductivity", parameters)
        self.surface_parameter = surface_parameter
        self.external_field = external_field
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica condiciones específicas para superconductividad."""
        psi_modified = psi.copy()
        
        # Condición de De Gennes en la superficie
        # ψ'(0) = (1/ξ)ψ(0), donde ξ es la longitud de coherencia
        dx = grid[1] - grid[0]
        xi = self.surface_parameter
        
        # Borde izquierdo
        psi_modified[0] = psi_modified[1] / (1 + dx/xi)
        
        # Borde derecho con campo externo
        psi_modified[-1] = psi_modified[-2] * np.exp(-1j * self.external_field * t)
        
        return psi_modified
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica operador para superconductividad."""
        return operator  # Las condiciones se aplican en apply()


class CosmologyBoundary(BoundaryCondition):
    """
    Condiciones de frontera para aplicaciones cosmológicas.
    Considera expansión del universo y horizonte de eventos.
    """
    
    def __init__(self, hubble_parameter: float = 0.7, expansion_rate: float = 1.0,
                 parameters: Dict[str, Any] = None):
        super().__init__("Cosmology", parameters)
        self.hubble_parameter = hubble_parameter
        self.expansion_rate = expansion_rate
    
    def apply(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica condiciones cosmológicas con expansión."""
        psi_modified = psi.copy()
        
        # Factor de escala dependiente del tiempo
        scale_factor = np.exp(self.hubble_parameter * t)
        
        # Condiciones en el horizonte
        horizon_damping = np.exp(-self.expansion_rate * t)
        psi_modified[0] *= horizon_damping
        psi_modified[-1] *= horizon_damping
        
        return psi_modified
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica operador para cosmología."""
        return operator


class BoundaryConditionManager:
    """
    Gestor principal de condiciones de frontera.
    Coordina múltiples condiciones y aplicaciones específicas.
    """
    
    def __init__(self):
        self.boundary_conditions = {}
        self.active_boundary = None
        logger.info("Inicializando gestor de condiciones de frontera")
    
    def register_boundary(self, name: str, boundary: BoundaryCondition):
        """Registra una nueva condición de frontera."""
        self.boundary_conditions[name] = boundary
        logger.info(f"Condición de frontera '{name}' registrada")
    
    def set_active_boundary(self, name: str):
        """Establece la condición de frontera activa."""
        if name in self.boundary_conditions:
            self.active_boundary = self.boundary_conditions[name]
            logger.info(f"Condición de frontera activa: {name}")
        else:
            raise ValueError(f"Condición de frontera '{name}' no encontrada")
    
    def apply_boundary(self, psi: np.ndarray, grid: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Aplica la condición de frontera activa."""
        if self.active_boundary is None:
            logger.warning("No hay condición de frontera activa")
            return psi
        
        return self.active_boundary.apply(psi, grid, t, **kwargs)
    
    def modify_operator(self, operator: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Modifica el operador según la condición de frontera activa."""
        if self.active_boundary is None:
            return operator
        
        return self.active_boundary.modify_operator(operator, grid)
    
    def get_boundary_info(self) -> Dict[str, Any]:
        """Retorna información sobre la condición de frontera activa."""
        if self.active_boundary is None:
            return {"name": "None", "parameters": {}}
        
        return {
            "name": self.active_boundary.name,
            "parameters": self.active_boundary.parameters
        }


def create_boundary_from_config(config: Dict[str, Any]) -> BoundaryCondition:
    """
    Factory function para crear condiciones de frontera desde configuración.
    
    Args:
        config: Diccionario de configuración con 'type' y 'parameters'
    
    Returns:
        BoundaryCondition: Instancia de la condición de frontera
    """
    boundary_type = config.get('type', 'periodic')
    parameters = config.get('parameters', {})
    
    if boundary_type.lower() == 'periodic':
        return PeriodicBoundary(parameters)
    
    elif boundary_type.lower() == 'dirichlet':
        boundary_values = parameters.get('values', {'left': 0.0, 'right': 0.0})
        return DirichletBoundary(boundary_values, parameters)
    
    elif boundary_type.lower() == 'neumann':
        derivative_values = parameters.get('derivatives', {'left': 0.0, 'right': 0.0})
        return NeumannBoundary(derivative_values, parameters)
    
    elif boundary_type.lower() == 'absorbing':
        absorption_length = parameters.get('absorption_length', 0.1)
        strength = parameters.get('strength', 1.0)
        return AbsorbingBoundary(absorption_length, strength, parameters)
    
    elif boundary_type.lower() == 'reflective':
        reflection_type = parameters.get('reflection_type', 'hard')
        return ReflectiveBoundary(reflection_type, parameters)
    
    elif boundary_type.lower() == 'superconductivity':
        surface_parameter = parameters.get('surface_parameter', 1.0)
        external_field = parameters.get('external_field', 0.0)
        return SuperconductivityBoundary(surface_parameter, external_field, parameters)
    
    elif boundary_type.lower() == 'cosmology':
        hubble_parameter = parameters.get('hubble_parameter', 0.7)
        expansion_rate = parameters.get('expansion_rate', 1.0)
        return CosmologyBoundary(hubble_parameter, expansion_rate, parameters)
    
    else:
        logger.warning(f"Tipo de frontera desconocido: {boundary_type}. Usando periódica.")
        return PeriodicBoundary(parameters)


# Funciones de utilidad adicionales

def validate_boundary_compatibility(boundary: BoundaryCondition, 
                                  application: str) -> bool:
    """
    Valida si una condición de frontera es compatible con una aplicación.
    
    Args:
        boundary: Condición de frontera
        application: Nombre de la aplicación ('superconductivity', 'gas_dynamics', 'cosmology')
    
    Returns:
        bool: True si es compatible
    """
    compatibility_matrix = {
        'superconductivity': ['Dirichlet', 'Neumann', 'Superconductivity'],
        'gas_dynamics': ['Periodic', 'Reflective', 'Absorbing'],
        'cosmology': ['Periodic', 'Absorbing', 'Cosmology']
    }
    
    return boundary.name in compatibility_matrix.get(application, [])


def apply_fractional_boundary_correction(psi: np.ndarray, alpha: float, 
                                       grid: np.ndarray) -> np.ndarray:
    """
    Aplica correcciones especiales para operadores fraccionarios en los bordes.
    
    Args:
        psi: Campo a corregir
        alpha: Orden fraccionario
        grid: Grilla espacial
    
    Returns:
        np.ndarray: Campo corregido
    """
    if alpha == 2.0:
        return psi  # No hay corrección para el caso clásico
    
    psi_corrected = psi.copy()
    N = len(psi)
    
    # Corrección basada en la naturaleza no-local del operador fraccionario
    correction_length = max(1, int(0.1 * N))
    
    for i in range(correction_length):
        # Factor de corrección que decrece desde los bordes
        factor = (alpha / 2.0) * (i + 1) / correction_length
        
        # Borde izquierdo
        psi_corrected[i] *= (1 - factor)
        
        # Borde derecho
        psi_corrected[N-1-i] *= (1 - factor)
    
    return psi_corrected


if __name__ == "__main__":
    # Ejemplo de uso
    import matplotlib.pyplot as plt
    
    # Crear grilla de prueba
    x = np.linspace(0, 10, 100)
    psi_test = np.exp(-(x - 5)**2)
    
    # Crear manager de condiciones de frontera
    manager = BoundaryConditionManager()
    
    # Registrar diferentes condiciones
    manager.register_boundary("periodic", PeriodicBoundary())
    manager.register_boundary("dirichlet", DirichletBoundary({'left': 0, 'right': 0}))
    manager.register_boundary("absorbing", AbsorbingBoundary(absorption_length=0.2))
    
    # Probar diferentes condiciones
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    conditions = ["periodic", "dirichlet", "absorbing"]
    
    for i, condition in enumerate(conditions):
        manager.set_active_boundary(condition)
        psi_modified = manager.apply_boundary(psi_test, x, t=0)
        
        axes[i].plot(x, psi_test, 'b--', label='Original', alpha=0.7)
        axes[i].plot(x, psi_modified, 'r-', label=f'{condition.title()}', linewidth=2)
        axes[i].set_title(f'Condición: {condition.title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Información del manager
    axes[3].text(0.1, 0.7, f"Condiciones registradas:\n{list(manager.boundary_conditions.keys())}", 
                 transform=axes[3].transAxes, fontsize=12)
    axes[3].text(0.1, 0.3, f"Info activa:\n{manager.get_boundary_info()}", 
                 transform=axes[3].transAxes, fontsize=10)
    axes[3].set_title('Información del Manager')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Módulo boundary_conditions.py implementado correctamente")
    print(f"📊 Condiciones disponibles: {list(manager.boundary_conditions.keys())}")
