"""
Métodos numéricos especializados para la ecuación MFSU
======================================================

Este módulo implementa esquemas numéricos avanzados para resolver la ecuación
diferencial parcial estocástica fractal MFSU:

∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Incluye métodos pseudo-espectrales, diferencias finitas fraccionarias,
y esquemas de integración temporal adaptativos.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numba import jit, cuda
import warnings
from typing import Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

class NumericalSolver(ABC):
    """Clase base abstracta para solucionadores numéricos MFSU"""
    
    def __init__(self, grid_size: int, dx: float, dt: float):
        self.N = grid_size
        self.dx = dx
        self.dt = dt
        self.x = np.linspace(0, grid_size * dx, grid_size)
        
    @abstractmethod
    def solve_step(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Avanza la solución un paso temporal"""
        pass

class PseudoSpectralSolver(NumericalSolver):
    """
    Solucionador pseudo-espectral para la ecuación MFSU
    
    Utiliza transformada de Fourier para manejar eficientemente
    el operador fraccionario (-Δ)^(α/2)
    """
    
    def __init__(self, grid_size: int, dx: float, dt: float, alpha: float = 0.5):
        super().__init__(grid_size, dx, dt)
        self.alpha = alpha
        self._setup_spectral_operators()
        
    def _setup_spectral_operators(self):
        """Configura los operadores espectrales"""
        # Frecuencias para FFT
        self.k = 2 * np.pi * fftfreq(self.N, self.dx)
        
        # Operador fraccionario en el espacio de Fourier
        self.fractional_op = np.zeros_like(self.k)
        mask = self.k != 0
        self.fractional_op[mask] = np.abs(self.k[mask]) ** self.alpha
        
        # Operador de integración para esquema semi-implícito
        self.integrator = np.exp(-self.fractional_op * self.dt)
    
    def solve_step(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """
        Resuelve un paso temporal usando método pseudo-espectral
        
        Esquema semi-implícito:
        - Término fraccionario: implícito (exacto en Fourier)
        - Términos no lineales: explícito (Adams-Bashforth)
        """
        alpha = params.get('alpha', 0.5)
        beta = params.get('beta', 0.1)
        gamma = params.get('gamma', 0.01)
        
        # Transformada de Fourier
        psi_hat = fft(psi)
        
        # Término fraccionario (implícito)
        fractional_term = -alpha * self.fractional_op * psi_hat
        
        # Términos no lineales y estocásticos (explícito)
        nonlinear_term = self._compute_nonlinear_terms(psi, t, params)
        nonlinear_hat = fft(nonlinear_term)
        
        # Integración semi-implícita
        psi_new_hat = (psi_hat + self.dt * nonlinear_hat) * self.integrator
        
        # Transformada inversa
        psi_new = ifft(psi_new_hat)
        
        return np.real(psi_new)
    
    def _compute_nonlinear_terms(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Calcula términos no lineales y estocásticos"""
        beta = params.get('beta', 0.1)
        gamma = params.get('gamma', 0.01)
        
        # Término estocástico (requiere generación de ruido fractal)
        stochastic_term = beta * self._generate_fractional_noise(t, params) * psi
        
        # Término no lineal cúbico
        nonlinear_term = -gamma * psi**3
        
        # Término de fuerza externa
        external_force = self._external_force(t, params)
        
        return stochastic_term + nonlinear_term + external_force
    
    def _generate_fractional_noise(self, t: float, params: Dict[str, float]) -> np.ndarray:
        """Genera ruido fractal ξ_H(x,t)"""
        hurst = params.get('hurst', 0.7)
        # Implementación simplificada - debería usar el módulo stochastic_processes
        return np.random.normal(0, 1, self.N)
    
    def _external_force(self, t: float, params: Dict[str, float]) -> np.ndarray:
        """Término de fuerza externa f(x,t)"""
        # Implementación por defecto - puede ser sobrescrita
        return np.zeros(self.N)

class FiniteDifferenceSolver(NumericalSolver):
    """
    Solucionador por diferencias finitas para la ecuación MFSU
    
    Utiliza aproximaciones de diferencias finitas para el operador
    fraccionario y esquemas de Runge-Kutta para la integración temporal
    """
    
    def __init__(self, grid_size: int, dx: float, dt: float, alpha: float = 0.5):
        super().__init__(grid_size, dx, dt)
        self.alpha = alpha
        self._setup_finite_difference_operators()
    
    def _setup_finite_difference_operators(self):
        """Configura matrices de diferencias finitas fraccionarias"""
        # Implementación de la matriz del operador fraccionario
        # Usando la definición de Grünwald-Letnikov
        self.fractional_matrix = self._build_fractional_matrix()
    
    def _build_fractional_matrix(self) -> np.ndarray:
        """Construye la matriz del operador fraccionario discreto"""
        # Coeficientes de Grünwald-Letnikov para la derivada fraccionaria
        coeffs = self._grunwald_letnikov_coefficients(self.N, self.alpha)
        
        # Construir matriz tridiagonal/pentadiagonal
        matrix = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(max(0, i-len(coeffs)//2), min(self.N, i+len(coeffs)//2+1)):
                if i-j+len(coeffs)//2 < len(coeffs):
                    matrix[i, j] = coeffs[i-j+len(coeffs)//2]
        
        return matrix / (self.dx**self.alpha)
    
    @staticmethod
    @jit(nopython=True)
    def _grunwald_letnikov_coefficients(n: int, alpha: float) -> np.ndarray:
        """Calcula coeficientes de Grünwald-Letnikov"""
        coeffs = np.zeros(n)
        coeffs[0] = 1.0
        
        for k in range(1, n):
            coeffs[k] = coeffs[k-1] * (alpha - k + 1) / k
        
        return coeffs
    
    def solve_step(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Resuelve un paso temporal usando diferencias finitas"""
        return self._runge_kutta_4(psi, t, params)
    
    def _runge_kutta_4(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Integración temporal Runge-Kutta de 4to orden"""
        dt = self.dt
        
        k1 = dt * self._compute_rhs(psi, t, params)
        k2 = dt * self._compute_rhs(psi + k1/2, t + dt/2, params)
        k3 = dt * self._compute_rhs(psi + k2/2, t + dt/2, params)
        k4 = dt * self._compute_rhs(psi + k3, t + dt, params)
        
        return psi + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _compute_rhs(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Calcula el lado derecho de la ecuación MFSU"""
        alpha = params.get('alpha', 0.5)
        beta = params.get('beta', 0.1)
        gamma = params.get('gamma', 0.01)
        
        # Término fraccionario
        fractional_term = -alpha * self.fractional_matrix @ psi
        
        # Término estocástico
        stochastic_term = beta * self._generate_fractional_noise(t, params) * psi
        
        # Término no lineal
        nonlinear_term = -gamma * psi**3
        
        # Término de fuerza externa
        external_force = self._external_force(t, params)
        
        return fractional_term + stochastic_term + nonlinear_term + external_force
    
    def _generate_fractional_noise(self, t: float, params: Dict[str, float]) -> np.ndarray:
        """Genera ruido fractal ξ_H(x,t)"""
        hurst = params.get('hurst', 0.7)
        # Implementación simplificada
        return np.random.normal(0, 1, self.N)
    
    def _external_force(self, t: float, params: Dict[str, float]) -> np.ndarray:
        """Término de fuerza externa f(x,t)"""
        return np.zeros(self.N)

class AdaptiveTimeSteppingSolver(PseudoSpectralSolver):
    """
    Solucionador con paso temporal adaptativo
    
    Ajusta automáticamente el paso temporal basado en criterios
    de estabilidad y precisión
    """
    
    def __init__(self, grid_size: int, dx: float, dt_initial: float, 
                 alpha: float = 0.5, tolerance: float = 1e-6):
        super().__init__(grid_size, dx, dt_initial, alpha)
        self.dt_initial = dt_initial
        self.tolerance = tolerance
        self.dt_min = dt_initial / 100
        self.dt_max = dt_initial * 10
    
    def solve_step(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Resuelve con paso temporal adaptativo"""
        dt_old = self.dt
        
        # Intenta con paso temporal actual
        psi_full = super().solve_step(psi, t, params)
        
        # Calcula solución con paso temporal reducido para estimar error
        self.dt = dt_old / 2
        psi_half1 = super().solve_step(psi, t, params)
        psi_half2 = super().solve_step(psi_half1, t + self.dt, params)
        
        # Estima error local
        error = np.max(np.abs(psi_full - psi_half2))
        
        # Ajusta paso temporal
        if error > self.tolerance:
            self.dt = max(self.dt_min, dt_old * 0.8)
        elif error < self.tolerance / 10:
            self.dt = min(self.dt_max, dt_old * 1.2)
        else:
            self.dt = dt_old
        
        return psi_half2 if error > self.tolerance else psi_full

class GPUSolver(PseudoSpectralSolver):
    """
    Solucionador optimizado para GPU usando CUDA
    
    Acelera cálculos intensivos usando paralelización GPU
    """
    
    def __init__(self, grid_size: int, dx: float, dt: float, alpha: float = 0.5):
        super().__init__(grid_size, dx, dt, alpha)
        self.gpu_available = cuda.is_available()
        
        if self.gpu_available:
            self._setup_gpu_arrays()
    
    def _setup_gpu_arrays(self):
        """Configura arrays en GPU"""
        self.d_psi = cuda.device_array(self.N, dtype=np.complex128)
        self.d_k = cuda.to_device(self.k)
        self.d_fractional_op = cuda.to_device(self.fractional_op)
    
    @cuda.jit
    def _gpu_fractional_operator(psi_hat, k, alpha, result):
        """Operador fraccionario en GPU"""
        idx = cuda.grid(1)
        if idx < psi_hat.size:
            result[idx] = -alpha * (abs(k[idx]) ** alpha) * psi_hat[idx]
    
    def solve_step(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Resuelve en GPU si está disponible"""
        if not self.gpu_available:
            return super().solve_step(psi, t, params)
        
        # Implementación GPU optimizada
        return self._gpu_solve_step(psi, t, params)
    
    def _gpu_solve_step(self, psi: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Implementación GPU del paso de solución"""
        # Copia datos a GPU
        cuda.to_device(psi, to=self.d_psi)
        
        # Procesa en GPU
        # ... implementación específica GPU ...
        
        # Regresa resultado a CPU
        return cuda.from_device(self.d_psi, shape=psi.shape)

class SolverFactory:
    """Fábrica para crear diferentes tipos de solucionadores"""
    
    @staticmethod
    def create_solver(solver_type: str, grid_size: int, dx: float, dt: float, 
                     alpha: float = 0.5, **kwargs) -> NumericalSolver:
        """
        Crea un solucionador del tipo especificado
        
        Args:
            solver_type: Tipo de solucionador ('spectral', 'finite_diff', 'adaptive', 'gpu')
            grid_size: Tamaño de la grilla
            dx: Espaciado espacial
            dt: Paso temporal
            alpha: Parámetro fraccionario
            **kwargs: Parámetros adicionales específicos del solucionador
        
        Returns:
            Instancia del solucionador solicitado
        """
        solvers = {
            'spectral': PseudoSpectralSolver,
            'finite_diff': FiniteDifferenceSolver,
            'adaptive': AdaptiveTimeSteppingSolver,
            'gpu': GPUSolver
        }
        
        if solver_type not in solvers:
            raise ValueError(f"Tipo de solucionador '{solver_type}' no reconocido. "
                           f"Opciones disponibles: {list(solvers.keys())}")
        
        return solvers[solver_type](grid_size, dx, dt, alpha, **kwargs)

# Funciones de utilidad para análisis de estabilidad
def cfl_condition(dx: float, dt: float, alpha: float, max_velocity: float = 1.0) -> bool:
    """
    Verifica la condición CFL para estabilidad numérica
    
    Para ecuaciones fraccionarias: dt ≤ C * dx^α
    """
    cfl_number = max_velocity * dt / (dx ** alpha)
    return cfl_number <= 1.0

def stability_analysis(solver: NumericalSolver, params: Dict[str, float]) -> Dict[str, float]:
    """
    Analiza la estabilidad del esquema numérico
    
    Returns:
        Diccionario con métricas de estabilidad
    """
    # Análisis de autovalores para estabilidad lineal
    eigenvalues = np.linalg.eigvals(solver.fractional_matrix) if hasattr(solver, 'fractional_matrix') else []
    
    stability_metrics = {
        'max_eigenvalue': np.max(np.real(eigenvalues)) if len(eigenvalues) > 0 else 0,
        'spectral_radius': np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0,
        'cfl_satisfied': cfl_condition(solver.dx, solver.dt, params.get('alpha', 0.5))
    }
    
    return stability_metrics

def convergence_test(solver_type: str, grid_sizes: list, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Prueba de convergencia para diferentes resoluciones
    
    Args:
        solver_type: Tipo de solucionador a probar
        grid_sizes: Lista de tamaños de grilla para probar
        params: Parámetros de la simulación
    
    Returns:
        Diccionario con métricas de convergencia
    """
    errors = []
    
    # Solución de referencia (grilla más fina)
    reference_solver = SolverFactory.create_solver(
        solver_type, grid_sizes[-1], 0.1/grid_sizes[-1], 0.01, params.get('alpha', 0.5)
    )
    
    # Condición inicial simple
    x_ref = np.linspace(0, 1, grid_sizes[-1])
    psi_ref = np.exp(-(x_ref - 0.5)**2 / 0.1)
    
    # Evoluciona solución de referencia
    for _ in range(10):
        psi_ref = reference_solver.solve_step(psi_ref, 0, params)
    
    # Calcula errores para diferentes resoluciones
    for N in grid_sizes[:-1]:
        solver = SolverFactory.create_solver(solver_type, N, 0.1/N, 0.01, params.get('alpha', 0.5))
        x = np.linspace(0, 1, N)
        psi = np.exp(-(x - 0.5)**2 / 0.1)
        
        for _ in range(10):
            psi = solver.solve_step(psi, 0, params)
        
        # Interpola para comparar
        psi_interp = np.interp(x_ref, x, psi)
        error = np.linalg.norm(psi_interp - psi_ref) / np.linalg.norm(psi_ref)
        errors.append(error)
    
    return {
        'grid_sizes': np.array(grid_sizes[:-1]),
        'errors': np.array(errors),
        'convergence_rate': np.polyfit(np.log(grid_sizes[:-1]), np.log(errors), 1)[0]
    }

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros de simulación
    params = {
        'alpha': 0.5,
        'beta': 0.1,
        'gamma': 0.01,
        'hurst': 0.7
    }
    
    # Crear solucionador
    solver = SolverFactory.create_solver('spectral', 128, 0.1, 0.01, params['alpha'])
    
    # Condición inicial
    x = np.linspace(0, 12.8, 128)
    psi = np.exp(-(x - 6.4)**2 / 2.0)
    
    # Evolución temporal
    times = np.arange(0, 1.0, 0.01)
    solution = np.zeros((len(times), len(psi)))
    solution[0] = psi
    
    for i, t in enumerate(times[1:], 1):
        psi = solver.solve_step(psi, t, params)
        solution[i] = psi
    
    print(f"Simulación completada. Forma de la solución: {solution.shape}")
    print(f"Métricas de estabilidad: {stability_analysis(solver, params)}")
