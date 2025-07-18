"""
Grid Manager para el Simulador MFSU
====================================

Este módulo maneja la creación y gestión de grillas espaciales para la simulación
de la ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Características:
- Grillas 1D, 2D y 3D adaptativas
- Soporte para condiciones de frontera periódicas y no periódicas
- Optimización para operadores fraccionarios
- Gestión eficiente de memoria
- Interpolación y refinamiento de grilla
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

# Configuración del logger
logger = logging.getLogger(__name__)

class BoundaryType(Enum):
    """Tipos de condiciones de frontera"""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ABSORBING = "absorbing"

class GridType(Enum):
    """Tipos de grilla"""
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    SPECTRAL = "spectral"

@dataclass
class GridParameters:
    """Parámetros de configuración de la grilla"""
    dimensions: int = 1
    grid_type: GridType = GridType.UNIFORM
    boundary_type: BoundaryType = BoundaryType.PERIODIC
    
    # Parámetros espaciales
    L: Union[float, List[float]] = 10.0  # Longitud del dominio
    N: Union[int, List[int]] = 128       # Número de puntos
    dx: Optional[Union[float, List[float]]] = None  # Espaciado
    
    # Parámetros para grillas adaptativas
    refinement_threshold: float = 1e-3
    max_refinement_level: int = 3
    min_grid_points: int = 32
    
    # Parámetros para operadores fraccionarios
    alpha: float = 0.5  # Orden fraccionario
    spectral_accuracy: bool = True

class GridManager:
    """
    Gestor principal de grillas espaciales para simulaciones MFSU
    """
    
    def __init__(self, params: GridParameters):
        """
        Inicializa el gestor de grillas
        
        Args:
            params: Parámetros de configuración de la grilla
        """
        self.params = params
        self.dimensions = params.dimensions
        self._validate_parameters()
        
        # Inicializar arrays de coordenadas
        self.x = None
        self.dx = None
        self.k = None  # Números de onda para espacio de Fourier
        
        # Matrices para operadores diferenciales
        self.laplacian_matrix = None
        self.fractional_laplacian_matrix = None
        
        # Información de refinamiento (para grillas adaptativas)
        self.refinement_levels = None
        self.refinement_mask = None
        
        # Crear la grilla inicial
        self._create_grid()
        self._setup_operators()
        
        logger.info(f"GridManager inicializado: {self.dimensions}D, {self.params.grid_type.value}")

    def _validate_parameters(self):
        """Valida los parámetros de entrada"""
        if self.dimensions not in [1, 2, 3]:
            raise ValueError(f"Dimensiones no soportadas: {self.dimensions}")
        
        if self.dimensions > 1:
            if isinstance(self.params.L, (int, float)):
                self.params.L = [self.params.L] * self.dimensions
            if isinstance(self.params.N, int):
                self.params.N = [self.params.N] * self.dimensions
                
        if len(self.params.L) != self.dimensions:
            raise ValueError("Longitud L debe coincidir con las dimensiones")
        if len(self.params.N) != self.dimensions:
            raise ValueError("Número de puntos N debe coincidir con las dimensiones")

    def _create_grid(self):
        """Crea la grilla espacial según los parámetros"""
        if self.params.grid_type == GridType.UNIFORM:
            self._create_uniform_grid()
        elif self.params.grid_type == GridType.ADAPTIVE:
            self._create_adaptive_grid()
        elif self.params.grid_type == GridType.SPECTRAL:
            self._create_spectral_grid()

    def _create_uniform_grid(self):
        """Crea una grilla uniforme"""
        self.x = []
        self.dx = []
        
        for dim in range(self.dimensions):
            L = self.params.L[dim]
            N = self.params.N[dim]
            
            if self.params.boundary_type == BoundaryType.PERIODIC:
                # Grilla periódica: no incluir el último punto
                dx = L / N
                x = np.linspace(0, L - dx, N)
            else:
                # Grilla no periódica: incluir ambos extremos
                dx = L / (N - 1)
                x = np.linspace(0, L, N)
            
            self.x.append(x)
            self.dx.append(dx)
        
        # Para 2D y 3D, crear grillas meshgrid
        if self.dimensions > 1:
            self.X = np.meshgrid(*self.x, indexing='ij')

    def _create_adaptive_grid(self):
        """Crea una grilla adaptativa (inicialmente uniforme)"""
        # Comenzar con grilla uniforme
        self._create_uniform_grid()
        
        # Inicializar información de refinamiento
        self.refinement_levels = [np.zeros(N, dtype=int) for N in self.params.N]
        self.refinement_mask = [np.ones(N, dtype=bool) for N in self.params.N]
        
        logger.info("Grilla adaptativa inicializada (uniforme)")

    def _create_spectral_grid(self):
        """Crea una grilla espectral optimizada para operadores fraccionarios"""
        self._create_uniform_grid()
        
        # Crear números de onda para transformada de Fourier
        self.k = []
        for dim in range(self.dimensions):
            N = self.params.N[dim]
            L = self.params.L[dim]
            
            if self.params.boundary_type == BoundaryType.PERIODIC:
                # Frecuencias para FFT con condiciones periódicas
                k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
            else:
                # Frecuencias para transformada de coseno (DCT)
                k = np.pi * np.arange(N) / L
            
            self.k.append(k)
        
        logger.info("Grilla espectral creada")

    def _setup_operators(self):
        """Configura operadores diferenciales"""
        if self.params.grid_type == GridType.SPECTRAL:
            self._setup_spectral_operators()
        else:
            self._setup_finite_difference_operators()

    def _setup_spectral_operators(self):
        """Configura operadores espectrales para derivadas fraccionarias"""
        if self.dimensions == 1:
            k = self.k[0]
            # Operador Laplaciano fraccionario: (-Δ)^(α/2) = |k|^α en espacio de Fourier
            self.fractional_laplacian_spectrum = np.abs(k) ** self.params.alpha
        else:
            # Para dimensiones múltiples
            K = np.meshgrid(*self.k, indexing='ij')
            k_squared = sum(ki**2 for ki in K)
            self.fractional_laplacian_spectrum = k_squared ** (self.params.alpha / 2)

    def _setup_finite_difference_operators(self):
        """Configura operadores de diferencias finitas"""
        if self.dimensions == 1:
            self._setup_1d_operators()
        elif self.dimensions == 2:
            self._setup_2d_operators()
        else:
            self._setup_3d_operators()

    def _setup_1d_operators(self):
        """Configura operadores 1D"""
        N = self.params.N[0]
        dx = self.dx[0]
        
        # Matriz Laplaciana de segundo orden
        self.laplacian_matrix = self._create_laplacian_matrix_1d(N, dx)
        
        # Aproximación del Laplaciano fraccionario usando diferencias finitas
        self.fractional_laplacian_matrix = self._create_fractional_laplacian_1d(N, dx)

    def _create_laplacian_matrix_1d(self, N: int, dx: float) -> np.ndarray:
        """Crea matriz Laplaciana 1D"""
        A = np.zeros((N, N))
        
        # Coeficientes de diferencias finitas de segundo orden
        coeff = 1.0 / (dx * dx)
        
        if self.params.boundary_type == BoundaryType.PERIODIC:
            # Condiciones periódicas
            for i in range(N):
                A[i, (i-1) % N] = coeff
                A[i, i] = -2 * coeff
                A[i, (i+1) % N] = coeff
        elif self.params.boundary_type == BoundaryType.DIRICHLET:
            # Condiciones Dirichlet (ψ = 0 en fronteras)
            for i in range(1, N-1):
                A[i, i-1] = coeff
                A[i, i] = -2 * coeff
                A[i, i+1] = coeff
            # Fronteras
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
        elif self.params.boundary_type == BoundaryType.NEUMANN:
            # Condiciones Neumann (∂ψ/∂n = 0 en fronteras)
            for i in range(1, N-1):
                A[i, i-1] = coeff
                A[i, i] = -2 * coeff
                A[i, i+1] = coeff
            # Fronteras con derivada cero
            A[0, 0] = -coeff
            A[0, 1] = coeff
            A[-1, -2] = coeff
            A[-1, -1] = -coeff
        
        return A

    def _create_fractional_laplacian_1d(self, N: int, dx: float) -> np.ndarray:
        """
        Crea aproximación del Laplaciano fraccionario 1D
        Usando la aproximación de Grünwald-Letnikov
        """
        alpha = self.params.alpha
        A = np.zeros((N, N))
        
        # Coeficientes de Grünwald-Letnikov
        def grunwald_coeff(j, alpha):
            if j == 0:
                return 1.0
            return (-1)**j * np.math.gamma(alpha + 1) / (np.math.gamma(j + 1) * np.math.gamma(alpha - j + 1))
        
        # Construir matriz
        for i in range(N):
            for j in range(min(i + 1, N)):
                coeff = grunwald_coeff(j, alpha) / (dx**alpha)
                if self.params.boundary_type == BoundaryType.PERIODIC:
                    A[i, (i - j) % N] = coeff
                else:
                    if i - j >= 0:
                        A[i, i - j] = coeff
        
        return A

    def _setup_2d_operators(self):
        """Configura operadores 2D usando productos de Kronecker"""
        N1, N2 = self.params.N
        dx1, dx2 = self.dx
        
        # Matrices 1D
        L1 = self._create_laplacian_matrix_1d(N1, dx1)
        L2 = self._create_laplacian_matrix_1d(N2, dx2)
        I1 = np.eye(N1)
        I2 = np.eye(N2)
        
        # Laplaciano 2D: L1 ⊗ I2 + I1 ⊗ L2
        self.laplacian_matrix = np.kron(L1, I2) + np.kron(I1, L2)

    def _setup_3d_operators(self):
        """Configura operadores 3D usando productos de Kronecker"""
        N1, N2, N3 = self.params.N
        dx1, dx2, dx3 = self.dx
        
        # Matrices 1D
        L1 = self._create_laplacian_matrix_1d(N1, dx1)
        L2 = self._create_laplacian_matrix_1d(N2, dx2)
        L3 = self._create_laplacian_matrix_1d(N3, dx3)
        I1 = np.eye(N1)
        I2 = np.eye(N2)
        I3 = np.eye(N3)
        
        # Laplaciano 3D
        self.laplacian_matrix = (np.kron(np.kron(L1, I2), I3) + 
                                np.kron(np.kron(I1, L2), I3) + 
                                np.kron(np.kron(I1, I2), L3))

    def apply_laplacian(self, psi: np.ndarray) -> np.ndarray:
        """
        Aplica el operador Laplaciano a un campo
        
        Args:
            psi: Campo de entrada
            
        Returns:
            Resultado de aplicar el Laplaciano
        """
        if self.params.grid_type == GridType.SPECTRAL:
            return self._apply_spectral_laplacian(psi)
        else:
            return self._apply_finite_difference_laplacian(psi)

    def _apply_spectral_laplacian(self, psi: np.ndarray) -> np.ndarray:
        """Aplica Laplaciano usando métodos espectrales"""
        psi_hat = np.fft.fftn(psi)
        
        if self.dimensions == 1:
            k = self.k[0]
            laplacian_spectrum = -(k**2)
        else:
            K = np.meshgrid(*self.k, indexing='ij')
            laplacian_spectrum = -sum(ki**2 for ki in K)
        
        result_hat = laplacian_spectrum * psi_hat
        return np.real(np.fft.ifftn(result_hat))

    def _apply_finite_difference_laplacian(self, psi: np.ndarray) -> np.ndarray:
        """Aplica Laplaciano usando diferencias finitas"""
        psi_flat = psi.flatten()
        result_flat = self.laplacian_matrix @ psi_flat
        return result_flat.reshape(psi.shape)

    def apply_fractional_laplacian(self, psi: np.ndarray) -> np.ndarray:
        """
        Aplica el operador Laplaciano fraccionario (-Δ)^(α/2)
        
        Args:
            psi: Campo de entrada
            
        Returns:
            Resultado de aplicar el Laplaciano fraccionario
        """
        if self.params.grid_type == GridType.SPECTRAL:
            return self._apply_spectral_fractional_laplacian(psi)
        else:
            return self._apply_finite_difference_fractional_laplacian(psi)

    def _apply_spectral_fractional_laplacian(self, psi: np.ndarray) -> np.ndarray:
        """Aplica Laplaciano fraccionario usando métodos espectrales"""
        psi_hat = np.fft.fftn(psi)
        result_hat = self.fractional_laplacian_spectrum * psi_hat
        return np.real(np.fft.ifftn(result_hat))

    def _apply_finite_difference_fractional_laplacian(self, psi: np.ndarray) -> np.ndarray:
        """Aplica Laplaciano fraccionario usando diferencias finitas"""
        if self.fractional_laplacian_matrix is None:
            warnings.warn("Matriz Laplaciano fraccionario no disponible, usando aproximación")
            return self.apply_laplacian(psi)
        
        psi_flat = psi.flatten()
        result_flat = self.fractional_laplacian_matrix @ psi_flat
        return result_flat.reshape(psi.shape)

    def refine_grid(self, psi: np.ndarray, criterion: str = 'gradient') -> bool:
        """
        Refina la grilla adaptativa basándose en un criterio
        
        Args:
            psi: Campo actual
            criterion: Criterio de refinamiento ('gradient', 'curvature', 'error')
            
        Returns:
            True si se realizó refinamiento
        """
        if self.params.grid_type != GridType.ADAPTIVE:
            return False
        
        if criterion == 'gradient':
            return self._refine_by_gradient(psi)
        elif criterion == 'curvature':
            return self._refine_by_curvature(psi)
        elif criterion == 'error':
            return self._refine_by_error(psi)
        else:
            raise ValueError(f"Criterio de refinamiento no reconocido: {criterion}")

    def _refine_by_gradient(self, psi: np.ndarray) -> bool:
        """Refina basándose en el gradiente del campo"""
        if self.dimensions == 1:
            gradient = np.gradient(psi, self.dx[0])
            gradient_magnitude = np.abs(gradient)
        else:
            gradients = np.gradient(psi, *self.dx)
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Identificar regiones que necesitan refinamiento
        refine_mask = gradient_magnitude > self.params.refinement_threshold
        
        if np.any(refine_mask):
            logger.info(f"Refinando {np.sum(refine_mask)} puntos basándose en gradiente")
            # Aquí iría la lógica de refinamiento real
            return True
        
        return False

    def interpolate_to_grid(self, psi: np.ndarray, target_grid: 'GridManager') -> np.ndarray:
        """
        Interpola un campo a otra grilla
        
        Args:
            psi: Campo a interpolar
            target_grid: Grilla objetivo
            
        Returns:
            Campo interpolado
        """
        from scipy.interpolate import griddata
        
        if self.dimensions != target_grid.dimensions:
            raise ValueError("Las grillas deben tener las mismas dimensiones")
        
        # Crear puntos de origen y destino
        if self.dimensions == 1:
            points_src = self.x[0]
            points_dst = target_grid.x[0]
            return np.interp(points_dst, points_src, psi)
        else:
            # Para dimensiones múltiples, usar griddata
            points_src = np.column_stack([X.ravel() for X in self.X])
            points_dst = np.column_stack([X.ravel() for X in target_grid.X])
            
            psi_interpolated = griddata(points_src, psi.ravel(), points_dst, method='linear')
            return psi_interpolated.reshape(target_grid.get_shape())

    def get_shape(self) -> Tuple[int, ...]:
        """Retorna la forma de la grilla"""
        return tuple(self.params.N)

    def get_total_points(self) -> int:
        """Retorna el número total de puntos de grilla"""
        return np.prod(self.params.N)

    def get_memory_usage(self) -> float:
        """Estima el uso de memoria en MB"""
        total_points = self.get_total_points()
        # Estimación para arrays complejos y matrices
        memory_mb = total_points * 16 / (1024**2)  # 16 bytes por número complejo
        if hasattr(self, 'laplacian_matrix') and self.laplacian_matrix is not None:
            memory_mb += self.laplacian_matrix.nbytes / (1024**2)
        return memory_mb

    def export_grid_info(self) -> Dict[str, Any]:
        """Exporta información de la grilla"""
        info = {
            'dimensions': self.dimensions,
            'grid_type': self.params.grid_type.value,
            'boundary_type': self.params.boundary_type.value,
            'shape': self.get_shape(),
            'total_points': self.get_total_points(),
            'domain_lengths': self.params.L,
            'grid_spacings': self.dx,
            'memory_usage_mb': self.get_memory_usage()
        }
        
        if hasattr(self, 'fractional_laplacian_spectrum'):
            info['spectral_method'] = True
            info['alpha'] = self.params.alpha
        
        return info

    def __str__(self) -> str:
        """Representación string del GridManager"""
        return (f"GridManager({self.dimensions}D, {self.params.grid_type.value}, "
                f"shape={self.get_shape()}, boundary={self.params.boundary_type.value})")

    def __repr__(self) -> str:
        return self.__str__()


# Funciones de utilidad para crear grillas comunes
def create_uniform_grid_1d(L: float = 10.0, N: int = 128, 
                          boundary_type: BoundaryType = BoundaryType.PERIODIC) -> GridManager:
    """Crea una grilla uniforme 1D"""
    params = GridParameters(
        dimensions=1,
        L=L,
        N=N,
        grid_type=GridType.UNIFORM,
        boundary_type=boundary_type
    )
    return GridManager(params)

def create_spectral_grid_1d(L: float = 10.0, N: int = 128, alpha: float = 0.5) -> GridManager:
    """Crea una grilla espectral 1D para operadores fraccionarios"""
    params = GridParameters(
        dimensions=1,
        L=L,
        N=N,
        grid_type=GridType.SPECTRAL,
        boundary_type=BoundaryType.PERIODIC,
        alpha=alpha
    )
    return GridManager(params)

def create_uniform_grid_2d(L: List[float] = [10.0, 10.0], N: List[int] = [64, 64],
                          boundary_type: BoundaryType = BoundaryType.PERIODIC) -> GridManager:
    """Crea una grilla uniforme 2D"""
    params = GridParameters(
        dimensions=2,
        L=L,
        N=N,
        grid_type=GridType.UNIFORM,
        boundary_type=boundary_type
    )
    return GridManager(params)

def create_adaptive_grid_1d(L: float = 10.0, N: int = 128, 
                           refinement_threshold: float = 1e-3) -> GridManager:
    """Crea una grilla adaptativa 1D"""
    params = GridParameters(
        dimensions=1,
        L=L,
        N=N,
        grid_type=GridType.ADAPTIVE,
        boundary_type=BoundaryType.PERIODIC,
        refinement_threshold=refinement_threshold
    )
    return GridManager(params)


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear grilla espectral 1D para simulación MFSU
    grid = create_spectral_grid_1d(L=10.0, N=128, alpha=0.5)
    
    print("Información de la grilla:")
    print(grid.export_grid_info())
    
    # Crear campo de prueba
    x = grid.x[0]
    psi = np.exp(-(x - 5)**2)  # Gaussiana centrada
    
    # Aplicar operadores
    laplacian_psi = grid.apply_laplacian(psi)
    fractional_laplacian_psi = grid.apply_fractional_laplacian(psi)
    
    print(f"\nCampo original: max={np.max(psi):.4f}, min={np.min(psi):.4f}")
    print(f"Laplaciano: max={np.max(laplacian_psi):.4f}, min={np.min(laplacian_psi):.4f}")
    print(f"Laplaciano fraccionario: max={np.max(fractional_laplacian_psi):.4f}, min={np.min(fractional_laplacian_psi):.4f}")
