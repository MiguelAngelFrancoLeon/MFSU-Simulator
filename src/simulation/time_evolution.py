"""
Time Evolution para el Simulador MFSU
=====================================

Este módulo implementa diferentes esquemas numéricos para la evolución temporal
de la ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Esquemas implementados:
- Euler explícito e implícito
- Runge-Kutta de orden 2 y 4
- Métodos espectrales (ETDRK)
- Esquemas semi-implícitos optimizados
- Método de splitting para términos estocásticos
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings

# Importar módulos del proyecto
from .grid_manager import GridManager
from ..core.stochastic_processes import StochasticProcess
from ..utils.constants import MACHINE_EPSILON

logger = logging.getLogger(__name__)

class TimeScheme(Enum):
    """Esquemas de integración temporal"""
    EULER_EXPLICIT = "euler_explicit"
    EULER_IMPLICIT = "euler_implicit"
    RK2 = "runge_kutta_2"
    RK4 = "runge_kutta_4"
    ETDRK2 = "etdrk2"  # Exponential Time Differencing RK2
    ETDRK4 = "etdrk4"  # Exponential Time Differencing RK4
    SEMI_IMPLICIT = "semi_implicit"
    SPLITTING = "splitting"
    ADAMS_BASHFORTH = "adams_bashforth"

@dataclass
class TimeParameters:
    """Parámetros para evolución temporal"""
    dt: float = 0.01                    # Paso temporal
    t_final: float = 10.0               # Tiempo final
    scheme: TimeScheme = TimeScheme.RK4 # Esquema numérico
    
    # Parámetros de la ecuación MFSU
    alpha: float = 0.5                  # Coeficiente del término fraccionario
    beta: float = 0.1                   # Coeficiente del ruido estocástico
    gamma: float = 0.01                 # Coeficiente no lineal
    
    # Control de estabilidad
    adaptive_timestep: bool = False     # Paso adaptativo
    max_dt: float = 0.1                 # Paso máximo
    min_dt: float = 1e-6                # Paso mínimo
    tolerance: float = 1e-6             # Tolerancia para paso adaptativo
    
    # Parámetros específicos para métodos espectrales
    etd_contour_points: int = 32        # Puntos para integración en ETD
    
    # Control de memoria y output
    save_every: int = 1                 # Guardar cada N pasos
    max_memory_mb: float = 1000.0       # Límite de memoria

class TimeEvolutionError(Exception):
    """Excepción para errores en evolución temporal"""
    pass

class BaseTimeEvolver(ABC):
    """Clase base para evolucionadores temporales"""
    
    def __init__(self, grid: GridManager, params: TimeParameters):
        self.grid = grid
        self.params = params
        self.current_time = 0.0
        self.step_count = 0
        
        # Historial para métodos multi-paso
        self.history = []
        self.max_history = 4
        
        # Estadísticas
        self.stats = {
            'total_steps': 0,
            'rejected_steps': 0,
            'min_dt_used': params.dt,
            'max_dt_used': params.dt,
            'avg_dt_used': params.dt
        }
        
        self._setup_scheme()
    
    @abstractmethod
    def _setup_scheme(self):
        """Configura el esquema numérico específico"""
        pass
    
    @abstractmethod
    def _single_step(self, psi: np.ndarray, dt: float, 
                    stochastic_process: Optional[StochasticProcess] = None,
                    external_force: Optional[Callable] = None) -> np.ndarray:
        """Realiza un paso temporal"""
        pass
    
    def evolve(self, psi_initial: np.ndarray, 
               stochastic_process: Optional[StochasticProcess] = None,
               external_force: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evoluciona el sistema desde condiciones iniciales
        
        Args:
            psi_initial: Condición inicial
            stochastic_process: Proceso estocástico ξ_H(x,t)
            external_force: Fuerza externa f(x,t)
            
        Returns:
            Tuple (tiempos, soluciones)
        """
        # Validar entrada
        if psi_initial.shape != self.grid.get_shape():
            raise TimeEvolutionError(f"Forma incompatible: {psi_initial.shape} vs {self.grid.get_shape()}")
        
        # Inicializar arrays de salida
        n_steps = int(np.ceil(self.params.t_final / self.params.dt))
        n_saved = n_steps // self.params.save_every + 1
        
        times = np.zeros(n_saved)
        solutions = np.zeros((n_saved,) + psi_initial.shape, dtype=complex)
        
        # Condición inicial
        psi = psi_initial.copy()
        solutions[0] = psi
        times[0] = 0.0
        
        save_index = 1
        dt = self.params.dt
        
        logger.info(f"Iniciando evolución temporal: t_final={self.params.t_final}, dt={dt}")
        
        try:
            for step in range(n_steps):
                # Verificar límites de memoria
                if self._check_memory_usage():
                    logger.warning("Límite de memoria alcanzado, deteniendo simulación")
                    break
                
                # Paso adaptativo si está habilitado
                if self.params.adaptive_timestep:
                    dt, psi = self._adaptive_step(psi, dt, stochastic_process, external_force)
                else:
                    psi = self._single_step(psi, dt, stochastic_process, external_force)
                
                self.current_time += dt
                self.step_count += 1
                
                # Guardar datos
                if (step + 1) % self.params.save_every == 0:
                    if save_index < len(times):
                        times[save_index] = self.current_time
                        solutions[save_index] = psi
                        save_index += 1
                
                # Actualizar estadísticas
                self._update_stats(dt)
                
                # Verificar estabilidad
                if not self._check_stability(psi):
                    logger.warning(f"Inestabilidad detectada en paso {step}")
                    break
                
                # Log progreso
                if step % (n_steps // 10) == 0:
                    logger.info(f"Progreso: {step/n_steps*100:.1f}% (t={self.current_time:.3f})")
        
        except Exception as e:
            logger.error(f"Error en evolución temporal: {e}")
            raise TimeEvolutionError(f"Fallo en evolución: {e}")
        
        # Recortar arrays si es necesario
        if save_index < len(times):
            times = times[:save_index]
            solutions = solutions[:save_index]
        
        logger.info(f"Evolución completada: {self.step_count} pasos, t_final={self.current_time:.3f}")
        self._log_final_stats()
        
        return times, solutions
    
    def _adaptive_step(self, psi: np.ndarray, dt: float,
                      stochastic_process: Optional[StochasticProcess],
                      external_force: Optional[Callable]) -> Tuple[float, np.ndarray]:
        """Implementa paso temporal adaptativo"""
        # Intentar paso con dt
        psi1 = self._single_step(psi, dt, stochastic_process, external_force)
        
        # Paso con dt/2, dos veces
        psi_half = self._single_step(psi, dt/2, stochastic_process, external_force)
        psi2 = self._single_step(psi_half, dt/2, stochastic_process, external_force)
        
        # Estimar error
        error = np.max(np.abs(psi2 - psi1))
        
        # Decidir si aceptar el paso
        if error < self.params.tolerance:
            # Aceptar paso y posiblemente aumentar dt
            new_dt = min(dt * 1.2, self.params.max_dt)
            return new_dt, psi2
        else:
            # Rechazar paso y reducir dt
            new_dt = max(dt * 0.5, self.params.min_dt)
            self.stats['rejected_steps'] += 1
            return self._adaptive_step(psi, new_dt, stochastic_process, external_force)
    
    def _check_memory_usage(self) -> bool:
        """Verifica uso de memoria"""
        # Estimación simple del uso de memoria
        memory_mb = self.grid.get_memory_usage()
        return memory_mb > self.params.max_memory_mb
    
    def _check_stability(self, psi: np.ndarray) -> bool:
        """Verifica estabilidad numérica"""
        # Verificar NaN o infinitos
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            return False
        
        # Verificar crecimiento exponencial
        max_val = np.max(np.abs(psi))
        if max_val > 1e10:
            return False
        
        return True
    
    def _update_stats(self, dt: float):
        """Actualiza estadísticas"""
        self.stats['total_steps'] += 1
        self.stats['min_dt_used'] = min(self.stats['min_dt_used'], dt)
        self.stats['max_dt_used'] = max(self.stats['max_dt_used'], dt)
        
        # Promedio móvil
        n = self.stats['total_steps']
        old_avg = self.stats['avg_dt_used']
        self.stats['avg_dt_used'] = (old_avg * (n-1) + dt) / n
    
    def _log_final_stats(self):
        """Log estadísticas finales"""
        logger.info("Estadísticas de evolución temporal:")
        logger.info(f"  Pasos totales: {self.stats['total_steps']}")
        logger.info(f"  Pasos rechazados: {self.stats['rejected_steps']}")
        logger.info(f"  dt promedio: {self.stats['avg_dt_used']:.6f}")
        logger.info(f"  dt min/max: {self.stats['min_dt_used']:.6f}/{self.stats['max_dt_used']:.6f}")

class EulerExplicitEvolver(BaseTimeEvolver):
    """Evolucionador Euler explícito"""
    
    def _setup_scheme(self):
        """Configura el esquema Euler explícito"""
        logger.info("Configurando esquema Euler explícito")
        
        # Verificar estabilidad CFL
        if hasattr(self.grid, 'dx'):
            min_dx = min(self.grid.dx) if isinstance(self.grid.dx, list) else self.grid.dx
            cfl_limit = min_dx**2 / (2 * self.params.alpha)
            if self.params.dt > cfl_limit:
                warnings.warn(f"dt={self.params.dt} excede límite CFL={cfl_limit:.6f}")
    
    def _single_step(self, psi: np.ndarray, dt: float,
                    stochastic_process: Optional[StochasticProcess] = None,
                    external_force: Optional[Callable] = None) -> np.ndarray:
        """Paso Euler explícito"""
        # Términos de la ecuación MFSU
        # ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
        
        # Término fraccionario: α(-Δ)^(α/2)ψ
        fractional_term = self.params.alpha * self.grid.apply_fractional_laplacian(psi)
        
        # Término estocástico: β ξ_H(x,t)ψ
        stochastic_term = 0.0
        if stochastic_process is not None:
            noise = stochastic_process.generate_noise(self.current_time, self.grid.get_shape())
            stochastic_term = self.params.beta * noise * psi
        
        # Término no lineal: -γψ³
        nonlinear_term = -self.params.gamma * psi**3
        
        # Fuerza externa: f(x,t)
        external_term = 0.0
        if external_force is not None:
            external_term = external_force(self.current_time, self.grid.x)
        
        # Integración Euler explícita
        dpsi_dt = fractional_term + stochastic_term + nonlinear_term + external_term
        psi_new = psi + dt * dpsi_dt
        
        return psi_new

class RungeKutta4Evolver(BaseTimeEvolver):
    """Evolucionador Runge-Kutta de orden 4"""
    
    def _setup_scheme(self):
        """Configura el esquema RK4"""
        logger.info("Configurando esquema Runge-Kutta 4")
    
    def _single_step(self, psi: np.ndarray, dt: float,
                    stochastic_process: Optional[StochasticProcess] = None,
                    external_force: Optional[Callable] = None) -> np.ndarray:
        """Paso Runge-Kutta 4"""
        def rhs(psi_state, t):
            """Lado derecho de la ecuación MFSU"""
            # Término fraccionario
            fractional_term = self.params.alpha * self.grid.apply_fractional_laplacian(psi_state)
            
            # Término estocástico
            stochastic_term = 0.0
            if stochastic_process is not None:
                noise = stochastic_process.generate_noise(t, self.grid.get_shape())
                stochastic_term = self.params.beta * noise * psi_state
            
            # Término no lineal
            nonlinear_term = -self.params.gamma * psi_state**3
            
            # Fuerza externa
            external_term = 0.0
            if external_force is not None:
                external_term = external_force(t, self.grid.x)
            
            return fractional_term + stochastic_term + nonlinear_term + external_term
        
        # Pasos RK4
        t = self.current_time
        k1 = dt * rhs(psi, t)
        k2 = dt * rhs(psi + k1/2, t + dt/2)
        k3 = dt * rhs(psi + k2/2, t + dt/2)
        k4 = dt * rhs(psi + k3, t + dt)
        
        psi_new = psi + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return psi_new

class ETDRKEvolver(BaseTimeEvolver):
    """Evolucionador Exponential Time Differencing Runge-Kutta"""
    
    def _setup_scheme(self):
        """Configura el esquema ETDRK"""
        logger.info("Configurando esquema ETDRK")
        
        # Precomputar coeficientes exponenciales para términos lineales
        if hasattr(self.grid, 'fractional_laplacian_spectrum'):
            # Método espectral
            self.linear_spectrum = self.params.alpha * self.grid.fractional_laplacian_spectrum
            self._setup_etd_coefficients()
        else:
            # Método de diferencias finitas
            logger.warning("ETDRK no optimizado para diferencias finitas")
    
    def _setup_etd_coefficients(self):
        """Configura coeficientes ETD usando integración de contorno"""
        dt = self.params.dt
        L = self.linear_spectrum
        
        # Evitar división por cero
        L_safe = np.where(np.abs(L) < MACHINE_EPSILON, MACHINE_EPSILON, L)
        
        # Coeficientes ETD básicos
        self.etd_exp = np.exp(dt * L_safe)
        
        # Para ETDRK2
        self.etd_phi1 = np.where(np.abs(L) < MACHINE_EPSILON, 
                                dt, 
                                (np.exp(dt * L) - 1) / L)
        
        # Para ETDRK4 (requiere más coeficientes)
        if self.params.scheme == TimeScheme.ETDRK4:
            self.etd_phi2 = np.where(np.abs(L) < MACHINE_EPSILON,
                                    dt**2 / 2,
                                    (np.exp(dt * L) - 1 - dt * L) / L**2)
    
    def _single_step(self, psi: np.ndarray, dt: float,
                    stochastic_process: Optional[StochasticProcess] = None,
                    external_force: Optional[Callable] = None) -> np.ndarray:
        """Paso ETDRK"""
        if self.params.scheme == TimeScheme.ETDRK2:
            return self._etdrk2_step(psi, dt, stochastic_process, external_force)
        elif self.params.scheme == TimeScheme.ETDRK4:
            return self._etdrk4_step(psi, dt, stochastic_process, external_force)
        else:
            raise TimeEvolutionError(f"Esquema ETDRK no soportado: {self.params.scheme}")
    
    def _etdrk2_step(self, psi: np.ndarray, dt: float,
                    stochastic_process: Optional[StochasticProcess],
                    external_force: Optional[Callable]) -> np.ndarray:
        """Paso ETDRK2"""
        # Transformar a espacio de Fourier
        psi_hat = np.fft.fftn(psi)
        
        # Evaluar términos no lineales
        nonlinear = self._evaluate_nonlinear(psi, stochastic_process, external_force)
        nonlinear_hat = np.fft.fftn(nonlinear)
        
        # Primer paso
        psi_hat_half = self.etd_exp**(1/2) * psi_hat + self.etd_phi1/2 * nonlinear_hat
        psi_half = np.fft.ifftn(psi_hat_half)
        
        # Segundo paso
        nonlinear_half = self._evaluate_nonlinear(psi_half, stochastic_process, external_force)
        nonlinear_half_hat = np.fft.fftn(nonlinear_half)
        
        # Paso completo
        psi_hat_new = (self.etd_exp * psi_hat + 
                      self.etd_phi1 * nonlinear_half_hat)
        
        return np.fft.ifftn(psi_hat_new)
    
    def _evaluate_nonlinear(self, psi: np.ndarray,
                           stochastic_process: Optional[StochasticProcess],
                           external_force: Optional[Callable]) -> np.ndarray:
        """Evalúa términos no lineales y estocásticos"""
        # Término estocástico
        stochastic_term = 0.0
        if stochastic_process is not None:
            noise = stochastic_process.generate_noise(self.current_time, self.grid.get_shape())
            stochastic_term = self.params.beta * noise * psi
        
        # Término no lineal
        nonlinear_term = -self.params.gamma * psi**3
        
        # Fuerza externa
        external_term = 0.0
        if external_force is not None:
            external_term = external_force(self.current_time, self.grid.x)
        
        return stochastic_term + nonlinear_term + external_term

class SemiImplicitEvolver(BaseTimeEvolver):
    """Evolucionador semi-implícito (implícito para términos lineales)"""
    
    def _setup_scheme(self):
        """Configura el esquema semi-implícito"""
        logger.info("Configurando esquema semi-implícito")
        
        # Precomputar matriz de evolución implícita
        if hasattr(self.grid, 'fractional_laplacian_spectrum'):
            # Método espectral
            dt = self.params.dt
            self.implicit_factor = 1 / (1 - dt * self.params.alpha * self.grid.fractional_laplacian_spectrum)
        else:
            # Método de diferencias finitas
            logger.warning("Semi-implícito no implementado para diferencias finitas")
    
    def _single_step(self, psi: np.ndarray, dt: float,
                    stochastic_process: Optional[StochasticProcess] = None,
                    external_force: Optional[Callable] = None) -> np.ndarray:
        """Paso semi-implícito"""
        # Términos explícitos (no lineales y estocásticos)
        stochastic_term = 0.0
        if stochastic_process is not None:
            noise = stochastic_process.generate_noise(self.current_time, self.grid.get_shape())
            stochastic_term = self.params.beta * noise * psi
        
        nonlinear_term = -self.params.gamma * psi**3
        
        external_term = 0.0
        if external_force is not None:
            external_term = external_force(self.current_time, self.grid.x)
        
        explicit_terms = stochastic_term + nonlinear_term + external_term
        
        # Transformar a espacio de Fourier
        psi_hat = np.fft.fftn(psi)
        explicit_hat = np.fft.fftn(explicit_terms)
        
        # Evolución semi-implícita
        psi_hat_new = self.implicit_factor * (psi_hat + dt * explicit_hat)
        
        return np.fft.ifftn(psi_hat_new)

# Factory para crear evolucionadores
def create_time_evolver(scheme: TimeScheme, grid: GridManager, 
                       params: TimeParameters) -> BaseTimeEvolver:
    """Factory para crear evolucionadores temporales"""
    
    if scheme == TimeScheme.EULER_EXPLICIT:
        return EulerExplicitEvolver(grid, params)
    elif scheme == TimeScheme.RK4:
        return RungeKutta4Evolver(grid, params)
    elif scheme in [TimeScheme.ETDRK2, TimeScheme.ETDRK4]:
        return ETDRKEvolver(grid, params)
    elif scheme == TimeScheme.SEMI_IMPLICIT:
        return SemiImplicitEvolver(grid, params)
    else:
        raise TimeEvolutionError(f"Esquema no implementado: {scheme}")

# Funciones de conveniencia
def evolve_mfsu(psi_initial: np.ndarray, grid: GridManager,
               t_final: float = 10.0, dt: float = 0.01,
               alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.01,
               scheme: TimeScheme = TimeScheme.RK4,
               stochastic_process: Optional[StochasticProcess] = None,
               external_force: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función de conveniencia para evolucionar la ecuación MFSU
    
    Args:
        psi_initial: Condición inicial
        grid: Gestor de grilla
        t_final: Tiempo final
        dt: Paso temporal
        alpha, beta, gamma: Parámetros de la ecuación MFSU
        scheme: Esquema numérico
        stochastic_process: Proceso estocástico
        external_force: Fuerza externa
        
    Returns:
        Tuple (tiempos, soluciones)
    """
    params = TimeParameters(
        dt=dt,
        t_final=t_final,
        scheme=scheme,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    
    evolver = create_time_evolver(scheme, grid, params)
    return evolver.evolve(psi_initial, stochastic_process, external_force)

def stability_analysis(grid: GridManager, alpha: float = 0.5, dt: float = 0.01) -> Dict[str, Any]:
    """
    Análisis de estabilidad para diferentes esquemas
    
    Args:
        grid: Gestor de grilla
        alpha: Parámetro fraccionario
        dt: Paso temporal
        
    Returns:
        Diccionario con información de estabilidad
    """
    analysis = {}
    
    # CFL para Euler explícito
    if hasattr(grid, 'dx'):
        min_dx = min(grid.dx) if isinstance(grid.dx, list) else grid.dx
        cfl_limit = min_dx**2 / (2 * alpha)
        analysis['euler_explicit'] = {
            'cfl_limit': cfl_limit,
            'stable': dt <= cfl_limit,
            'recommended_dt': cfl_limit * 0.5
        }
    
    # Análisis espectral
    if hasattr(grid, 'fractional_laplacian_spectrum'):
        max_eigenvalue = np.max(np.real(alpha * grid.fractional_laplacian_spectrum))
        analysis['spectral'] = {
            'max_eigenvalue': max_eigenvalue,
            'stable_dt_limit': 2.0 / max_eigenvalue if max_eigenvalue > 0 else np.inf
        }
    
    return analysis

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear grilla de prueba
    from .grid_manager import create_spectral_grid_1d
    grid = create_spectral_grid_1d(L=10.0, N=128, alpha=0.5)
    
    # Condición inicial: paquete gaussiano
    x = grid.x[0]
    psi_initial = np.exp(-(x - 5)**2) + 0j
    
    # Parámetros de evolución
    params = TimeParameters(
        dt=0.01,
        t_final=2.0,
        scheme=TimeScheme.RK4,
        alpha=0.5,
        beta=0.1,
        gamma=0.01
    )
    
    # Crear evolucionador
    evolver = create_time_evolver(TimeScheme.RK4, grid, params)
    
    # Evolucionar
    print("Iniciando evolución MFSU...")
    times, solutions = evolver.evolve(psi_initial)
    
    print(f"Evolución completada:")
    print(f"  Tiempos: {len(times)} puntos")
    print(f"  Tiempo final: {times[-1]:.3f}")
    print(f"  Norma inicial: {np.linalg.norm(psi_initial):.6f}")
    print(f"  Norma final: {np.linalg.norm(solutions[-1]):.6f}")
    
    # Análisis de estabilidad
    stability = stability_analysis(grid, alpha=0.5, dt=0.01)
    print(f"\nAnálisis de estabilidad:")
    for scheme, info in stability.items():
        print(f"  {scheme}: {info}")
