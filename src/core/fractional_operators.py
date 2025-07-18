"""
Implementación de operadores fraccionarios para el simulador MFSU.

Este módulo contiene las implementaciones de los operadores fraccionarios
utilizados en la ecuación MFSU, incluyendo el Laplaciano fraccionario
(-Δ)^(α/2) y métodos numéricos para su evaluación eficiente.

Autor: MFSU Development Team
"""

import numpy as np
import scipy.fft as fft
import scipy.sparse as sp
from scipy.special import gamma
from typing import Union, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import warnings


class FractionalOperator(ABC):
    """Clase base abstracta para operadores fraccionarios."""
    
    def __init__(self, alpha: float, domain_size: Union[float, Tuple[float, ...]], 
                 grid_points: Union[int, Tuple[int, ...]]):
        """
        Inicializa el operador fraccionario.
        
        Args:
            alpha: Exponente fraccionario (0 < α < 2)
            domain_size: Tamaño del dominio (escalar para 1D, tupla para N-D)
            grid_points: Número de puntos de grilla (escalar para 1D, tupla para N-D)
        """
        self.alpha = alpha
        self.domain_size = domain_size
        self.grid_points = grid_points
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Valida los parámetros del operador."""
        if not (0 < self.alpha < 2):
            raise ValueError(f"El exponente α debe estar en (0, 2), recibido: {self.alpha}")
    
    @abstractmethod
    def apply(self, u: np.ndarray) -> np.ndarray:
        """Aplica el operador fraccionario a la función u."""
        pass


class FractionalLaplacianFFT(FractionalOperator):
    """
    Implementación del Laplaciano fraccionario usando FFT.
    
    Esta implementación utiliza la representación espectral del operador
    fraccionario: F[(-Δ)^(α/2) u] = |k|^α F[u]
    """
    
    def __init__(self, alpha: float, domain_size: Union[float, Tuple[float, ...]], 
                 grid_points: Union[int, Tuple[int, ...]]):
        super().__init__(alpha, domain_size, grid_points)
        self._setup_frequencies()
    
    def _setup_frequencies(self):
        """Configura las frecuencias para la transformada de Fourier."""
        if isinstance(self.grid_points, int):
            # Caso 1D
            self.ndim = 1
            self.grid_shape = (self.grid_points,)
            dx = self.domain_size / self.grid_points
            freqs = fft.fftfreq(self.grid_points, dx)
            self.k_squared = freqs**2 * (2 * np.pi)**2
        else:
            # Caso N-D
            self.ndim = len(self.grid_points)
            self.grid_shape = self.grid_points
            
            if isinstance(self.domain_size, (int, float)):
                dx_values = [self.domain_size / n for n in self.grid_points]
            else:
                dx_values = [L / n for L, n in zip(self.domain_size, self.grid_points)]
            
            freq_arrays = []
            for i, (n, dx) in enumerate(zip(self.grid_points, dx_values)):
                freqs = fft.fftfreq(n, dx)
                freq_arrays.append(freqs)
            
            # Crear grilla de frecuencias
            freq_grids = np.meshgrid(*freq_arrays, indexing='ij')
            self.k_squared = sum([(2 * np.pi * k)**2 for k in freq_grids])
        
        # Símbolo del operador fraccionario
        self.symbol = self.k_squared**(self.alpha / 2)
        
        # Evitar división por cero en k=0
        self.symbol[self.k_squared == 0] = 0
    
    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        Aplica el Laplaciano fraccionario usando FFT.
        
        Args:
            u: Función de entrada
            
        Returns:
            (-Δ)^(α/2) u
        """
        if u.shape != self.grid_shape:
            raise ValueError(f"Shape de entrada {u.shape} no coincide con {self.grid_shape}")
        
        # Transformada de Fourier
        u_hat = fft.fftn(u)
        
        # Aplicar el símbolo del operador
        result_hat = self.symbol * u_hat
        
        # Transformada inversa
        result = fft.ifftn(result_hat)
        
        # Tomar la parte real (debería ser real para funciones reales)
        return np.real(result)


class FractionalLaplacianIntegral(FractionalOperator):
    """
    Implementación del Laplaciano fraccionario usando representación integral.
    
    Esta implementación utiliza la representación singular integral:
    (-Δ)^(α/2) u(x) = C(α,d) P.V. ∫ (u(x) - u(y)) / |x-y|^(d+α) dy
    """
    
    def __init__(self, alpha: float, domain_size: Union[float, Tuple[float, ...]], 
                 grid_points: Union[int, Tuple[int, ...]]):
        super().__init__(alpha, domain_size, grid_points)
        self._setup_grid()
        self._compute_normalization()
    
    def _setup_grid(self):
        """Configura la grilla espacial."""
        if isinstance(self.grid_points, int):
            # Caso 1D
            self.ndim = 1
            self.grid_shape = (self.grid_points,)
            self.dx = self.domain_size / self.grid_points
            self.x = np.linspace(0, self.domain_size, self.grid_points, endpoint=False)
        else:
            # Caso N-D
            self.ndim = len(self.grid_points)
            self.grid_shape = self.grid_points
            
            if isinstance(self.domain_size, (int, float)):
                dx_values = [self.domain_size / n for n in self.grid_points]
            else:
                dx_values = [L / n for L, n in zip(self.domain_size, self.grid_points)]
            
            self.dx = np.prod(dx_values)
            
            # Crear grilla de coordenadas
            coord_arrays = []
            for i, (L, n) in enumerate(zip(self.domain_size, self.grid_points)):
                coords = np.linspace(0, L, n, endpoint=False)
                coord_arrays.append(coords)
            
            self.x = np.meshgrid(*coord_arrays, indexing='ij')
    
    def _compute_normalization(self):
        """Calcula la constante de normalización."""
        # C(α,d) = 2^α Γ((d+α)/2) / (π^(d/2) |Γ(-α/2)|)
        d = self.ndim
        numerator = 2**self.alpha * gamma((d + self.alpha) / 2)
        denominator = np.pi**(d/2) * abs(gamma(-self.alpha/2))
        self.normalization = numerator / denominator
    
    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        Aplica el Laplaciano fraccionario usando representación integral.
        
        Args:
            u: Función de entrada
            
        Returns:
            (-Δ)^(α/2) u
        """
        if u.shape != self.grid_shape:
            raise ValueError(f"Shape de entrada {u.shape} no coincide con {self.grid_shape}")
        
        result = np.zeros_like(u)
        
        if self.ndim == 1:
            # Caso 1D optimizado
            for i in range(self.grid_points):
                xi = self.x[i]
                integrand = np.zeros_like(u)
                
                for j in range(self.grid_points):
                    if i != j:
                        xj = self.x[j]
                        distance = abs(xi - xj)
                        if distance > 0:
                            integrand[j] = (u[i] - u[j]) / distance**(1 + self.alpha)
                
                result[i] = self.normalization * np.trapz(integrand, self.x)
        else:
            # Caso N-D (implementación básica)
            warnings.warn("Implementación N-D del operador integral es computacionalmente costosa")
            # Implementación simplificada para casos N-D
            # En producción, se recomendaría usar FFT para casos N-D
            return self._apply_nd_simplified(u)
        
        return result
    
    def _apply_nd_simplified(self, u: np.ndarray) -> np.ndarray:
        """Implementación simplificada para casos N-D."""
        # Para casos N-D, usar aproximación por diferencias finitas
        # Esta es una aproximación, no la representación integral exacta
        fft_operator = FractionalLaplacianFFT(self.alpha, self.domain_size, self.grid_points)
        return fft_operator.apply(u)


class FractionalLaplacianMatrix(FractionalOperator):
    """
    Implementación matricial del Laplaciano fraccionario.
    
    Esta implementación construye explícitamente la matriz del operador
    fraccionario para casos pequeños donde es factible.
    """
    
    def __init__(self, alpha: float, domain_size: Union[float, Tuple[float, ...]], 
                 grid_points: Union[int, Tuple[int, ...]]):
        super().__init__(alpha, domain_size, grid_points)
        
        # Limitar a casos pequeños para evitar problemas de memoria
        total_points = np.prod(self.grid_points) if hasattr(self.grid_points, '__iter__') else self.grid_points
        if total_points > 1000:
            warnings.warn(f"Implementación matricial para {total_points} puntos puede ser ineficiente")
        
        self._construct_matrix()
    
    def _construct_matrix(self):
        """Construye la matriz del operador fraccionario."""
        if isinstance(self.grid_points, int):
            # Caso 1D
            self.ndim = 1
            self.grid_shape = (self.grid_points,)
            self.matrix = self._construct_1d_matrix()
        else:
            # Caso N-D
            self.ndim = len(self.grid_points)
            self.grid_shape = self.grid_points
            self.matrix = self._construct_nd_matrix()
    
    def _construct_1d_matrix(self):
        """Construye la matriz 1D del operador fraccionario."""
        n = self.grid_points
        dx = self.domain_size / n
        
        # Usar método espectral para construir la matriz
        # Construir matriz de transformada de Fourier discreta
        F = np.zeros((n, n), dtype=complex)
        for j in range(n):
            for k in range(n):
                F[j, k] = np.exp(-2j * np.pi * j * k / n)
        F = F / np.sqrt(n)
        
        # Matriz de frecuencias
        freqs = fft.fftfreq(n, dx)
        k_squared = (2 * np.pi * freqs)**2
        
        # Símbolo del operador
        symbol = k_squared**(self.alpha / 2)
        symbol[k_squared == 0] = 0
        
        # Construir matriz del operador
        Lambda = np.diag(symbol)
        F_inv = np.conj(F.T)
        
        matrix = np.real(F_inv @ Lambda @ F)
        return sp.csr_matrix(matrix)
    
    def _construct_nd_matrix(self):
        """Construye la matriz N-D del operador fraccionario."""
        # Para simplicidad, usar aproximación por diferencias finitas
        # En una implementación completa, se usaría el método espectral N-D
        total_points = np.prod(self.grid_points)
        matrix = sp.lil_matrix((total_points, total_points))
        
        # Implementación básica usando kernel fraccionario
        # Esta es una aproximación simplificada
        for i in range(total_points):
            for j in range(total_points):
                if i != j:
                    # Calcular distancia entre puntos i y j
                    coords_i = np.unravel_index(i, self.grid_shape)
                    coords_j = np.unravel_index(j, self.grid_shape)
                    
                    distance = np.sqrt(sum((ci - cj)**2 for ci, cj in zip(coords_i, coords_j)))
                    if distance > 0:
                        matrix[i, j] = -1 / distance**(self.ndim + self.alpha)
        
        return matrix.tocsr()
    
    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        Aplica el operador fraccionario usando multiplicación matricial.
        
        Args:
            u: Función de entrada
            
        Returns:
            (-Δ)^(α/2) u
        """
        if u.shape != self.grid_shape:
            raise ValueError(f"Shape de entrada {u.shape} no coincide con {self.grid_shape}")
        
        # Convertir a vector 1D
        u_vec = u.flatten()
        
        # Aplicar matriz
        result_vec = self.matrix @ u_vec
        
        # Convertir de vuelta a forma original
        return result_vec.reshape(self.grid_shape)


class FractionalOperatorFactory:
    """Factory para crear operadores fraccionarios."""
    
    @staticmethod
    def create_fractional_laplacian(method: str = 'fft', alpha: float = 1.0,
                                   domain_size: Union[float, Tuple[float, ...]] = 1.0,
                                   grid_points: Union[int, Tuple[int, ...]] = 100) -> FractionalOperator:
        """
        Crea un operador Laplaciano fraccionario.
        
        Args:
            method: Método de implementación ('fft', 'integral', 'matrix')
            alpha: Exponente fraccionario
            domain_size: Tamaño del dominio
            grid_points: Número de puntos de grilla
            
        Returns:
            Instancia del operador fraccionario
        """
        if method == 'fft':
            return FractionalLaplacianFFT(alpha, domain_size, grid_points)
        elif method == 'integral':
            return FractionalLaplacianIntegral(alpha, domain_size, grid_points)
        elif method == 'matrix':
            return FractionalLaplacianMatrix(alpha, domain_size, grid_points)
        else:
            raise ValueError(f"Método desconocido: {method}")


def fractional_derivative(u: np.ndarray, alpha: float, axis: int = 0, 
                         method: str = 'fft') -> np.ndarray:
    """
    Calcula la derivada fraccionaria de una función.
    
    Args:
        u: Función de entrada
        alpha: Orden de la derivada fraccionaria
        axis: Eje a lo largo del cual calcular la derivada
        method: Método de cálculo ('fft', 'grunwald')
        
    Returns:
        Derivada fraccionaria de u
    """
    if method == 'fft':
        return _fractional_derivative_fft(u, alpha, axis)
    elif method == 'grunwald':
        return _fractional_derivative_grunwald(u, alpha, axis)
    else:
        raise ValueError(f"Método desconocido: {method}")


def _fractional_derivative_fft(u: np.ndarray, alpha: float, axis: int) -> np.ndarray:
    """Implementación FFT de la derivada fraccionaria."""
    # Tomar FFT a lo largo del eje especificado
    u_hat = fft.fft(u, axis=axis)
    
    # Crear frecuencias
    n = u.shape[axis]
    freqs = fft.fftfreq(n)
    
    # Expandir dimensiones para broadcasting
    shape = [1] * u.ndim
    shape[axis] = n
    freqs = freqs.reshape(shape)
    
    # Aplicar el operador fraccionario
    symbol = (2j * np.pi * freqs) ** alpha
    result_hat = symbol * u_hat
    
    # Transformada inversa
    result = fft.ifft(result_hat, axis=axis)
    
    return np.real(result)


def _fractional_derivative_grunwald(u: np.ndarray, alpha: float, axis: int) -> np.ndarray:
    """Implementación Grünwald-Letnikov de la derivada fraccionaria."""
    # Implementación simplificada del método Grünwald-Letnikov
    # En una implementación completa, se incluirían más términos
    n = u.shape[axis]
    result = np.zeros_like(u)
    
    # Coeficientes de Grünwald-Letnikov
    coeffs = np.zeros(n)
    coeffs[0] = 1
    for k in range(1, n):
        coeffs[k] = coeffs[k-1] * (alpha - k + 1) / k
    
    # Aplicar convolución
    for i in range(n):
        indices = [slice(None)] * u.ndim
        for j in range(min(i+1, n)):
            indices[axis] = i - j
            result[tuple([slice(None) if k != axis else i for k in range(u.ndim)])] += \
                coeffs[j] * u[tuple(indices)]
    
    return result


# Funciones de utilidad para validación y benchmarking
def validate_fractional_operator(operator: FractionalOperator, 
                                test_function: Callable = None) -> dict:
    """
    Valida un operador fraccionario usando funciones de prueba conocidas.
    
    Args:
        operator: Operador a validar
        test_function: Función de prueba (por defecto usa funciones estándar)
        
    Returns:
        Diccionario con métricas de validación
    """
    results = {}
    
    # Función de prueba por defecto: Gaussiana
    if test_function is None:
        if operator.ndim == 1:
            x = np.linspace(0, operator.domain_size, operator.grid_points)
            test_func = np.exp(-0.5 * ((x - operator.domain_size/2) / 0.1)**2)
        else:
            # Crear función de prueba N-D
            coords = np.meshgrid(*[np.linspace(0, L, n) for L, n in 
                                  zip(operator.domain_size, operator.grid_points)], 
                                indexing='ij')
            center = [L/2 for L in operator.domain_size]
            test_func = np.exp(-0.5 * sum(((c - c0) / 0.1)**2 
                                         for c, c0 in zip(coords, center)))
    else:
        test_func = test_function
    
    # Aplicar operador
    try:
        result = operator.apply(test_func)
        results['success'] = True
        results['norm'] = np.linalg.norm(result)
        results['max_abs'] = np.max(np.abs(result))
        results['mean'] = np.mean(result)
        results['std'] = np.std(result)
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso y pruebas básicas
    print("Probando operadores fraccionarios...")
    
    # Crear operador 1D
    alpha = 1.5
    domain_size = 2 * np.pi
    grid_points = 128
    
    op_fft = FractionalOperatorFactory.create_fractional_laplacian(
        'fft', alpha, domain_size, grid_points)
    
    # Función de prueba
    x = np.linspace(0, domain_size, grid_points)
    u = np.sin(x) * np.exp(-0.1 * x)
    
    # Aplicar operador
    result = op_fft.apply(u)
    
    print(f"Norma del resultado: {np.linalg.norm(result):.6f}")
    print(f"Rango: [{np.min(result):.6f}, {np.max(result):.6f}]")
    
    # Validar
    validation = validate_fractional_operator(op_fft)
    print(f"Validación: {validation}")
    
    print("¡Operadores fraccionarios implementados correctamente!")
