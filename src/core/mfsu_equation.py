"""
Implementación de la Ecuación MFSU (Modelo Fractal Estocástico Unificado)

La ecuación MFSU está definida como:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Donde:
- ψ(x,t): Campo principal
- α: Parámetro de difusión fraccionaria
- β: Intensidad del ruido fractal
- γ: Parámetro de no linealidad
- ξ_H(x,t): Ruido fractal con exponente de Hurst H
- f(x,t): Función de forzamiento externa
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.special import gamma
import warnings
from typing import Tuple, Optional, Union, Callable
from numba import jit, njit


class MFSUEquation:
    """
    Implementación de la Ecuación MFSU (Modelo Fractal Estocástico Unificado)
    
    Esta clase proporciona métodos para resolver la ecuación diferencial
    fraccionaria estocástica MFSU usando métodos espectrales y diferencias finitas.
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 0.1, 
                 gamma: float = 0.01,
                 hurst: float = 0.7,
                 grid_size: int = 256,
                 domain_length: float = 10.0,
                 dt: float = 0.01):
        """
        Inicializa la ecuación MFSU con parámetros dados.
        
        Parameters:
        -----------
        alpha : float
            Parámetro de difusión fraccionaria (0 < α ≤ 2)
        beta : float
            Intensidad del ruido fractal
        gamma : float
            Parámetro de no linealidad
        hurst : float
            Exponente de Hurst para el ruido fractal (0 < H < 1)
        grid_size : int
            Número de puntos en la grilla espacial
        domain_length : float
            Longitud del dominio espacial
        dt : float
            Paso temporal
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hurst = hurst
        self.grid_size = grid_size
        self.domain_length = domain_length
        self.dt = dt
        
        # Validar parámetros
        self._validate_parameters()
        
        # Configurar grilla espacial
        self.dx = domain_length / grid_size
        self.x = np.linspace(0, domain_length, grid_size, endpoint=False)
        
        # Configurar números de onda para operadores espectrales
        self.k = fftfreq(grid_size, d=self.dx) * 2 * np.pi
        self.k[0] = 1e-10  # Evitar división por cero
        
        # Precalcular operador fraccionario en el espacio de Fourier
        self._setup_fractional_operator()
        
        # Inicializar generador de ruido fractal
        self._initialize_noise_generator()
    
    def _validate_parameters(self):
        """Valida los parámetros de entrada."""
        if not (0 < self.alpha <= 2):
            raise ValueError("α debe estar en el rango (0, 2]")
        if not (0 < self.hurst < 1):
            raise ValueError("El exponente de Hurst debe estar en (0, 1)")
        if self.grid_size <= 0:
            raise ValueError("El tamaño de la grilla debe ser positivo")
        if self.dt <= 0:
            raise ValueError("El paso temporal debe ser positivo")
    
    def _setup_fractional_operator(self):
        """Configura el operador fraccionario (-Δ)^(α/2) en el espacio de Fourier."""
        # Para el operador fraccionario, usamos |k|^α
        self.fractional_operator = np.abs(self.k) ** self.alpha
        
        # Aplicar ventana para evitar problemas en altas frecuencias
        cutoff = 0.8 * np.max(np.abs(self.k))
        window = np.exp(-0.5 * (np.abs(self.k) / cutoff) ** 6)
        self.fractional_operator *= window
    
    def _initialize_noise_generator(self):
        """Inicializa el generador de ruido fractal."""
        self.noise_state = np.random.RandomState(42)  # Semilla fija para reproducibilidad
        self.noise_buffer = []
        self.noise_buffer_size = 100
    
    def generate_fractional_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Genera ruido fractal ξ_H(x,t) con exponente de Hurst H.
        
        Parameters:
        -----------
        shape : tuple
            Forma del array de ruido a generar
            
        Returns:
        --------
        np.ndarray
            Array de ruido fractal
        """
        if len(shape) == 1:
            # Ruido espacial 1D
            return self._generate_spatial_fractional_noise(shape[0])
        else:
            # Para formas más complejas, generar ruido gaussiano correlacionado
            return self._generate_correlated_noise(shape)
    
    def _generate_spatial_fractional_noise(self, n_points: int) -> np.ndarray:
        """
        Genera ruido fractal espacial usando el método espectral.
        
        Parameters:
        -----------
        n_points : int
            Número de puntos espaciales
            
        Returns:
        --------
        np.ndarray
            Array de ruido fractal espacial
        """
        # Generar ruido blanco en el espacio de Fourier
        white_noise = self.noise_state.randn(n_points) + 1j * self.noise_state.randn(n_points)
        
        # Aplicar filtrado fractal: |k|^(-H-0.5)
        k_abs = np.abs(self.k[:n_points])
        k_abs[0] = 1e-10  # Evitar división por cero
        
        # Filtro fractal
        fractal_filter = k_abs ** (-(self.hurst + 0.5))
        
        # Aplicar filtro y transformar de vuelta
        filtered_noise = white_noise * fractal_filter
        real_noise = np.real(ifft(filtered_noise))
        
        # Normalizar
        return real_noise / np.std(real_noise)
    
    def _generate_correlated_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Genera ruido correlacionado para casos multidimensionales.
        
        Parameters:
        -----------
        shape : tuple
            Forma del array de ruido
            
        Returns:
        --------
        np.ndarray
            Array de ruido correlacionado
        """
        # Implementación simplificada para casos generales
        noise = self.noise_state.randn(*shape)
        
        # Aplicar correlación temporal simple si es necesario
        if len(shape) > 1:
            # Suavizado temporal básico
            for i in range(1, shape[0]):
                noise[i] = 0.9 * noise[i-1] + 0.1 * noise[i]
        
        return noise
    
    def apply_fractional_operator(self, psi: np.ndarray) -> np.ndarray:
        """
        Aplica el operador fraccionario (-Δ)^(α/2) a ψ.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo de entrada
            
        Returns:
        --------
        np.ndarray
            Resultado de aplicar el operador fraccionario
        """
        # Transformada de Fourier
        psi_hat = fft(psi)
        
        # Aplicar operador fraccionario
        result_hat = -self.fractional_operator * psi_hat
        
        # Transformada inversa
        result = np.real(ifft(result_hat))
        
        return result
    
    def nonlinear_term(self, psi: np.ndarray) -> np.ndarray:
        """
        Calcula el término no lineal -γψ³.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo de entrada
            
        Returns:
        --------
        np.ndarray
            Término no lineal
        """
        return -self.gamma * psi**3
    
    def stochastic_term(self, psi: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Calcula el término estocástico β ξ_H(x,t)ψ.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo de entrada
        noise : np.ndarray
            Ruido fractal
            
        Returns:
        --------
        np.ndarray
            Término estocástico
        """
        return self.beta * noise * psi
    
    def external_forcing(self, x: np.ndarray, t: float, 
                        forcing_func: Optional[Callable] = None) -> np.ndarray:
        """
        Calcula la función de forzamiento externa f(x,t).
        
        Parameters:
        -----------
        x : np.ndarray
            Coordenadas espaciales
        t : float
            Tiempo actual
        forcing_func : callable, optional
            Función de forzamiento personalizada
            
        Returns:
        --------
        np.ndarray
            Forzamiento externo
        """
        if forcing_func is None:
            # Forzamiento por defecto: pulso gaussiano modulado
            return 0.01 * np.exp(-0.5 * ((x - 0.5 * self.domain_length) / 2.0)**2) * np.sin(0.1 * t)
        else:
            return forcing_func(x, t)
    
    def compute_rhs(self, psi: np.ndarray, t: float, 
                   forcing_func: Optional[Callable] = None) -> np.ndarray:
        """
        Calcula el lado derecho de la ecuación MFSU.
        
        ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo actual
        t : float
            Tiempo actual
        forcing_func : callable, optional
            Función de forzamiento personalizada
            
        Returns:
        --------
        np.ndarray
            Derivada temporal dψ/dt
        """
        # Término de difusión fraccionaria
        diffusion_term = self.alpha * self.apply_fractional_operator(psi)
        
        # Generar ruido fractal
        noise = self.generate_fractional_noise((len(psi),))
        
        # Término estocástico
        stochastic = self.stochastic_term(psi, noise)
        
        # Término no lineal
        nonlinear = self.nonlinear_term(psi)
        
        # Forzamiento externo
        forcing = self.external_forcing(self.x, t, forcing_func)
        
        # Combinar todos los términos
        dpsi_dt = diffusion_term + stochastic + nonlinear + forcing
        
        return dpsi_dt
    
    def time_step_euler(self, psi: np.ndarray, t: float, 
                       forcing_func: Optional[Callable] = None) -> np.ndarray:
        """
        Realiza un paso temporal usando el método de Euler.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo actual
        t : float
            Tiempo actual
        forcing_func : callable, optional
            Función de forzamiento personalizada
            
        Returns:
        --------
        np.ndarray
            Campo actualizado
        """
        dpsi_dt = self.compute_rhs(psi, t, forcing_func)
        return psi + self.dt * dpsi_dt
    
    def time_step_rk4(self, psi: np.ndarray, t: float,
                     forcing_func: Optional[Callable] = None) -> np.ndarray:
        """
        Realiza un paso temporal usando el método Runge-Kutta de 4to orden.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo actual
        t : float
            Tiempo actual
        forcing_func : callable, optional
            Función de forzamiento personalizada
            
        Returns:
        --------
        np.ndarray
            Campo actualizado
        """
        # Coeficientes RK4
        k1 = self.dt * self.compute_rhs(psi, t, forcing_func)
        k2 = self.dt * self.compute_rhs(psi + 0.5 * k1, t + 0.5 * self.dt, forcing_func)
        k3 = self.dt * self.compute_rhs(psi + 0.5 * k2, t + 0.5 * self.dt, forcing_func)
        k4 = self.dt * self.compute_rhs(psi + k3, t + self.dt, forcing_func)
        
        return psi + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def evolve(self, psi_initial: np.ndarray, t_final: float,
               method: str = 'rk4', forcing_func: Optional[Callable] = None,
               save_interval: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evoluciona el sistema desde t=0 hasta t=t_final.
        
        Parameters:
        -----------
        psi_initial : np.ndarray
            Condición inicial
        t_final : float
            Tiempo final
        method : str
            Método de integración ('euler' o 'rk4')
        forcing_func : callable, optional
            Función de forzamiento personalizada
        save_interval : int
            Intervalo para guardar datos
            
        Returns:
        --------
        tuple
            (tiempos, evolución_del_campo)
        """
        n_steps = int(t_final / self.dt)
        n_save = n_steps // save_interval + 1
        
        # Arrays para almacenar resultados
        times = np.zeros(n_save)
        evolution = np.zeros((n_save, len(psi_initial)))
        
        # Condición inicial
        psi = psi_initial.copy()
        evolution[0] = psi
        times[0] = 0.0
        
        # Seleccionar método de integración
        if method == 'euler':
            time_step = self.time_step_euler
        elif method == 'rk4':
            time_step = self.time_step_rk4
        else:
            raise ValueError("Método debe ser 'euler' o 'rk4'")
        
        # Evolución temporal
        save_counter = 1
        for i in range(1, n_steps + 1):
            t = i * self.dt
            psi = time_step(psi, t, forcing_func)
            
            # Guardar datos si es necesario
            if i % save_interval == 0 and save_counter < n_save:
                times[save_counter] = t
                evolution[save_counter] = psi
                save_counter += 1
        
        return times, evolution
    
    def get_energy(self, psi: np.ndarray) -> float:
        """
        Calcula la energía del sistema.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo actual
            
        Returns:
        --------
        float
            Energía total del sistema
        """
        # Energía cinética (aproximación usando diferencias finitas)
        psi_grad = np.gradient(psi, self.dx)
        kinetic = 0.5 * np.sum(psi_grad**2) * self.dx
        
        # Energía potencial
        potential = 0.25 * self.gamma * np.sum(psi**4) * self.dx
        
        return kinetic + potential
    
    def get_statistics(self, psi: np.ndarray) -> dict:
        """
        Calcula estadísticas del campo.
        
        Parameters:
        -----------
        psi : np.ndarray
            Campo actual
            
        Returns:
        --------
        dict
            Diccionario con estadísticas
        """
        return {
            'mean': np.mean(psi),
            'std': np.std(psi),
            'max': np.max(psi),
            'min': np.min(psi),
            'energy': self.get_energy(psi),
            'l2_norm': np.sqrt(np.sum(psi**2) * self.dx)
        }


def create_initial_condition(x: np.ndarray, condition_type: str = 'gaussian', 
                           **kwargs) -> np.ndarray:
    """
    Crea condiciones iniciales para la ecuación MFSU.
    
    Parameters:
    -----------
    x : np.ndarray
        Coordenadas espaciales
    condition_type : str
        Tipo de condición inicial
    **kwargs
        Parámetros adicionales
        
    Returns:
    --------
    np.ndarray
        Condición inicial
    """
    if condition_type == 'gaussian':
        center = kwargs.get('center', 0.5 * (x[-1] - x[0]))
        width = kwargs.get('width', 1.0)
        amplitude = kwargs.get('amplitude', 1.0)
        return amplitude * np.exp(-0.5 * ((x - center) / width)**2)
    
    elif condition_type == 'soliton':
        center = kwargs.get('center', 0.5 * (x[-1] - x[0]))
        width = kwargs.get('width', 1.0)
        amplitude = kwargs.get('amplitude', 1.0)
        return amplitude / np.cosh((x - center) / width)
    
    elif condition_type == 'random':
        amplitude = kwargs.get('amplitude', 0.1)
        return amplitude * np.random.randn(len(x))
    
    else:
        raise ValueError(f"Tipo de condición inicial '{condition_type}' no reconocido")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia de la ecuación MFSU
    mfsu = MFSUEquation(alpha=1.5, beta=0.05, gamma=0.01, hurst=0.7)
    
    # Crear condición inicial
    psi0 = create_initial_condition(mfsu.x, 'gaussian', width=2.0, amplitude=1.0)
    
    # Evolucionar el sistema
    times, evolution = mfsu.evolve(psi0, t_final=5.0, method='rk4')
    
    print(f"Simulación completada: {len(times)} pasos temporales")
    print(f"Estadísticas finales: {mfsu.get_statistics(evolution[-1])}")
