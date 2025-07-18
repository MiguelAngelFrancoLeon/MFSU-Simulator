"""
Módulo para términos no lineales en la ecuación MFSU.

Este módulo implementa los términos no lineales presentes en la ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Específicamente maneja:
- Término cúbico no lineal: -γψ³
- Término de fuerza externa: f(x,t)
- Otros términos no lineales generalizados

Author: MFSU Development Team
Version: 1.0.0
"""

import numpy as np
from typing import Union, Callable, Optional, Dict, Any
from numba import jit, njit
import warnings


class NonlinearDynamics:
    """
    Clase para manejar términos no lineales en la ecuación MFSU.
    
    Esta clase proporciona métodos para calcular y gestionar los términos
    no lineales que aparecen en la ecuación MFSU, incluyendo el término
    cúbico y fuerzas externas.
    """
    
    def __init__(self, gamma: float = 0.01, force_function: Optional[Callable] = None):
        """
        Inicializa el manejador de términos no lineales.
        
        Parameters:
        -----------
        gamma : float
            Coeficiente del término no lineal cúbico
        force_function : callable, optional
            Función de fuerza externa f(x,t)
        """
        self.gamma = gamma
        self.force_function = force_function
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Valida los parámetros de inicialización."""
        if not isinstance(self.gamma, (int, float)):
            raise TypeError("gamma debe ser un número real")
        
        if self.force_function is not None and not callable(self.force_function):
            raise TypeError("force_function debe ser una función callable")
    
    def cubic_nonlinearity(self, psi: np.ndarray) -> np.ndarray:
        """
        Calcula el término no lineal cúbico -γψ³.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo complejo ψ
            
        Returns:
        --------
        np.ndarray
            Término no lineal cúbico -γψ³
        """
        return -self.gamma * self._compute_cubic_term(psi)
    
    @staticmethod
    @njit
    def _compute_cubic_term(psi: np.ndarray) -> np.ndarray:
        """
        Cálculo optimizado del término cúbico |ψ|²ψ.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo complejo
            
        Returns:
        --------
        np.ndarray
            |ψ|²ψ
        """
        return np.abs(psi)**2 * psi
    
    def external_force(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Calcula la fuerza externa f(x,t).
        
        Parameters:
        -----------
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo actual
            
        Returns:
        --------
        np.ndarray
            Fuerza externa f(x,t)
        """
        if self.force_function is None:
            return np.zeros_like(x, dtype=complex)
        
        return self.force_function(x, t)
    
    def total_nonlinear_term(self, psi: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
        """
        Calcula el término no lineal total: -γψ³ + f(x,t).
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo complejo ψ
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo actual
            
        Returns:
        --------
        np.ndarray
            Término no lineal total
        """
        cubic_term = self.cubic_nonlinearity(psi)
        force_term = self.external_force(x, t)
        
        return cubic_term + force_term
    
    def update_gamma(self, new_gamma: float):
        """
        Actualiza el coeficiente del término no lineal.
        
        Parameters:
        -----------
        new_gamma : float
            Nuevo valor del coeficiente γ
        """
        if not isinstance(new_gamma, (int, float)):
            raise TypeError("gamma debe ser un número real")
        
        self.gamma = new_gamma
    
    def set_force_function(self, force_function: Callable):
        """
        Establece una nueva función de fuerza externa.
        
        Parameters:
        -----------
        force_function : callable
            Nueva función de fuerza f(x,t)
        """
        if not callable(force_function):
            raise TypeError("force_function debe ser una función callable")
        
        self.force_function = force_function


class PredefinedForces:
    """
    Clase que contiene fuerzas externas predefinidas comúnmente utilizadas.
    """
    
    @staticmethod
    def gaussian_pulse(x: np.ndarray, t: float, 
                      amplitude: float = 1.0, 
                      center: float = 0.0, 
                      width: float = 1.0,
                      frequency: float = 1.0) -> np.ndarray:
        """
        Pulso gaussiano modulado.
        
        Parameters:
        -----------
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo
        amplitude : float
            Amplitud del pulso
        center : float
            Centro del pulso
        width : float
            Ancho del pulso
        frequency : float
            Frecuencia de modulación
            
        Returns:
        --------
        np.ndarray
            Pulso gaussiano f(x,t) = A * exp(-(x-x₀)²/w²) * cos(ωt)
        """
        spatial_part = np.exp(-((x - center)**2) / (2 * width**2))
        temporal_part = np.cos(2 * np.pi * frequency * t)
        
        return amplitude * spatial_part * temporal_part
    
    @staticmethod
    def harmonic_drive(x: np.ndarray, t: float,
                      amplitude: float = 1.0,
                      frequency: float = 1.0,
                      phase: float = 0.0) -> np.ndarray:
        """
        Fuerza armónica uniforme.
        
        Parameters:
        -----------
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo
        amplitude : float
            Amplitud
        frequency : float
            Frecuencia
        phase : float
            Fase inicial
            
        Returns:
        --------
        np.ndarray
            Fuerza armónica f(x,t) = A * cos(ωt + φ)
        """
        return amplitude * np.cos(2 * np.pi * frequency * t + phase) * np.ones_like(x)
    
    @staticmethod
    def localized_oscillator(x: np.ndarray, t: float,
                           amplitude: float = 1.0,
                           center: float = 0.0,
                           width: float = 1.0,
                           frequency: float = 1.0) -> np.ndarray:
        """
        Oscilador localizado espacialmente.
        
        Parameters:
        -----------
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo
        amplitude : float
            Amplitud
        center : float
            Centro de localización
        width : float
            Ancho de localización
        frequency : float
            Frecuencia de oscilación
            
        Returns:
        --------
        np.ndarray
            Oscilador localizado
        """
        spatial_envelope = np.exp(-((x - center)**2) / (2 * width**2))
        temporal_oscillation = np.sin(2 * np.pi * frequency * t)
        
        return amplitude * spatial_envelope * temporal_oscillation
    
    @staticmethod
    def step_function(x: np.ndarray, t: float,
                     amplitude: float = 1.0,
                     turn_on_time: float = 1.0,
                     turn_off_time: Optional[float] = None) -> np.ndarray:
        """
        Función escalón temporal.
        
        Parameters:
        -----------
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo
        amplitude : float
            Amplitud
        turn_on_time : float
            Tiempo de encendido
        turn_off_time : float, optional
            Tiempo de apagado
            
        Returns:
        --------
        np.ndarray
            Función escalón
        """
        if turn_off_time is None:
            factor = 1.0 if t >= turn_on_time else 0.0
        else:
            factor = 1.0 if turn_on_time <= t < turn_off_time else 0.0
        
        return amplitude * factor * np.ones_like(x)


class NonlinearAnalyzer:
    """
    Clase para análisis de términos no lineales y sus efectos.
    """
    
    def __init__(self, nonlinear_dynamics: NonlinearDynamics):
        """
        Inicializa el analizador.
        
        Parameters:
        -----------
        nonlinear_dynamics : NonlinearDynamics
            Instancia de la dinámica no lineal
        """
        self.nonlinear_dynamics = nonlinear_dynamics
    
    def compute_nonlinear_strength(self, psi: np.ndarray) -> float:
        """
        Calcula la intensidad del término no lineal.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo complejo
            
        Returns:
        --------
        float
            Intensidad RMS del término no lineal
        """
        cubic_term = self.nonlinear_dynamics.cubic_nonlinearity(psi)
        return np.sqrt(np.mean(np.abs(cubic_term)**2))
    
    def nonlinear_energy(self, psi: np.ndarray, dx: float) -> float:
        """
        Calcula la energía asociada al término no lineal.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo complejo
        dx : float
            Espaciado de la grilla
            
        Returns:
        --------
        float
            Energía no lineal
        """
        density = 0.5 * self.nonlinear_dynamics.gamma * np.abs(psi)**4
        return np.trapz(density, dx=dx)
    
    def stability_analysis(self, psi: np.ndarray, perturbation_amplitude: float = 1e-6) -> Dict[str, Any]:
        """
        Análisis de estabilidad del término no lineal.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo base
        perturbation_amplitude : float
            Amplitud de perturbación para el análisis
            
        Returns:
        --------
        Dict[str, Any]
            Resultados del análisis de estabilidad
        """
        # Estado base
        base_nonlinear = self.nonlinear_dynamics.cubic_nonlinearity(psi)
        
        # Perturbación
        perturbation = perturbation_amplitude * np.random.randn(*psi.shape)
        perturbed_psi = psi + perturbation
        perturbed_nonlinear = self.nonlinear_dynamics.cubic_nonlinearity(perturbed_psi)
        
        # Respuesta lineal
        linear_response = perturbed_nonlinear - base_nonlinear
        
        # Métricas de estabilidad
        response_amplitude = np.max(np.abs(linear_response))
        growth_rate = response_amplitude / perturbation_amplitude
        
        return {
            'growth_rate': growth_rate,
            'response_amplitude': response_amplitude,
            'perturbation_amplitude': perturbation_amplitude,
            'stable': growth_rate < 1.0
        }


# Funciones auxiliares para casos de uso comunes
def create_cubic_nonlinearity(gamma: float) -> NonlinearDynamics:
    """
    Crea una instancia de NonlinearDynamics solo con término cúbico.
    
    Parameters:
    -----------
    gamma : float
        Coeficiente del término cúbico
        
    Returns:
    --------
    NonlinearDynamics
        Instancia configurada
    """
    return NonlinearDynamics(gamma=gamma)


def create_driven_system(gamma: float, force_function: Callable) -> NonlinearDynamics:
    """
    Crea un sistema no lineal con fuerza externa.
    
    Parameters:
    -----------
    gamma : float
        Coeficiente del término cúbico
    force_function : callable
        Función de fuerza externa
        
    Returns:
    --------
    NonlinearDynamics
        Instancia configurada
    """
    return NonlinearDynamics(gamma=gamma, force_function=force_function)


def validate_nonlinear_parameters(gamma: float, 
                                 psi_max: float,
                                 stability_threshold: float = 1.0) -> bool:
    """
    Valida parámetros no lineales para estabilidad numérica.
    
    Parameters:
    -----------
    gamma : float
        Coeficiente no lineal
    psi_max : float
        Amplitud máxima esperada del campo
    stability_threshold : float
        Umbral de estabilidad
        
    Returns:
    --------
    bool
        True si los parámetros son estables
    """
    nonlinear_scale = abs(gamma) * psi_max**2
    
    if nonlinear_scale > stability_threshold:
        warnings.warn(f"Parámetros no lineales pueden causar inestabilidad: "
                     f"γ|ψ|² = {nonlinear_scale:.3e} > {stability_threshold}")
        return False
    
    return True


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo básico
    nl_dynamics = NonlinearDynamics(gamma=0.01)
    
    # Crear campo de prueba
    x = np.linspace(-10, 10, 100)
    psi = np.exp(-x**2/4) * np.exp(1j * x)
    
    # Calcular término no lineal
    cubic_term = nl_dynamics.cubic_nonlinearity(psi)
    
    print(f"Intensidad del término cúbico: {np.max(np.abs(cubic_term)):.6f}")
    
    # Ejemplo con fuerza externa
    force_func = lambda x, t: PredefinedForces.gaussian_pulse(x, t, amplitude=0.1)
    nl_dynamics_driven = NonlinearDynamics(gamma=0.01, force_function=force_func)
    
    total_nl = nl_dynamics_driven.total_nonlinear_term(psi, x, t=1.0)
    print(f"Término no lineal total: {np.max(np.abs(total_nl)):.6f}")
