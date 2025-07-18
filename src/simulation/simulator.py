"""
Motor Principal de Simulación MFSU

Implementa el simulador principal para la ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Este módulo contiene las clases principales para ejecutar simulaciones
del Modelo Fractal Estocástico Unificado.
"""

import numpy as np
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Dict, Any, List
from pathlib import Path

from ..core.mfsu_equation import MFSUEquation
from ..core.fractional_operators import FractionalLaplacian
from ..core.stochastic_processes import FractionalNoise
from ..core.nonlinear_dynamics import NonlinearTerms
from ..core.numerical_methods import NumericalSolver
from .grid_manager import GridManager, GridType
from .time_evolution import TimeEvolution, AdaptiveTimeStep
from .boundary_conditions import BoundaryConditions
from ..utils.logger import setup_logger
from ..utils.data_io import DataExporter
from ..utils.parameter_validation import validate_mfsu_parameters

logger = logging.getLogger(__name__)


class SimulationStatus(Enum):
    """Estado de la simulación."""
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationConfig:
    """
    Configuración para la simulación MFSU.
    """
    # Parámetros físicos de la ecuación MFSU
    alpha: float = 0.5          # Parámetro de difusión fraccional
    beta: float = 0.1           # Intensidad del ruido fraccional
    gamma: float = 0.01         # Parámetro de no linealidad
    hurst: float = 0.7          # Exponente de Hurst
    
    # Parámetros numéricos
    dt: float = 0.01            # Paso temporal
    dx: float = 0.1             # Resolución espacial
    grid_size: int = 100        # Tamaño de grilla
    max_time: float = 10.0      # Tiempo máximo
    
    # Configuración de grilla
    grid_type: GridType = GridType.UNIFORM_1D
    spatial_dimensions: int = 1
    domain_bounds: tuple = (-5.0, 5.0)
    
    # Método numérico
    numerical_method: str = "rk4"
    adaptive_timestep: bool = False
    tolerance: float = 1e-6
    
    # Condiciones de frontera
    boundary_type: str = "periodic"
    boundary_value: float = 0.0
    
    # Configuración de salida
    save_interval: int = 10
    output_dir: str = "output"
    save_snapshots: bool = True
    save_statistics: bool = True
    
    # Configuración de ruido
    noise_seed: Optional[int] = None
    noise_correlation_length: float = 1.0
    
    # Término de forzamiento
    external_force: Optional[Callable] = None
    force_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Configuración de paralelización
    use_gpu: bool = False
    n_threads: int = 1
    
    def __post_init__(self):
        """Validación post-inicialización."""
        validate_mfsu_parameters(self.__dict__)


@dataclass
class SimulationResult:
    """
    Resultado de una simulación MFSU.
    """
    # Datos principales
    psi: np.ndarray                    # Campo simulado ψ(x,t)
    time_points: np.ndarray           # Puntos temporales
    spatial_grid: np.ndarray          # Grilla espacial
    
    # Estadísticas
    statistics: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Información de la simulación
    config: Optional[SimulationConfig] = None
    execution_time: float = 0.0
    status: SimulationStatus = SimulationStatus.COMPLETED
    
    # Datos adicionales
    energy: Optional[np.ndarray] = None
    fractal_dimension: Optional[np.ndarray] = None
    correlation_function: Optional[np.ndarray] = None
    
    # Metadatos
    metadata: Dict[str, Any] = field(default_factory=dict)


class MFSUSimulator:
    """
    Simulador principal para el Modelo Fractal Estocástico Unificado.
    
    Implementa la ecuación:
    ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Inicializa el simulador MFSU.
        
        Args:
            config: Configuración de la simulación
        """
        self.config = config
        self.status = SimulationStatus.READY
        
        # Configurar logging
        self.logger = setup_logger(f"MFSU_Simulator_{id(self)}")
        
        # Inicializar componentes
        self._initialize_components()
        
        # Estado de la simulación
        self.current_time = 0.0
        self.current_step = 0
        self.psi = None
        self.initial_condition = None
        
        # Resultados y estadísticas
        self.time_series = []
        self.psi_snapshots = []
        self.statistics = {}
        
        self.logger.info(f"Simulador MFSU inicializado con α={config.alpha}, "
                        f"β={config.beta}, γ={config.gamma}")
    
    def _initialize_components(self):
        """Inicializa los componentes del simulador."""
        # Gestor de grilla
        self.grid_manager = GridManager(
            grid_type=self.config.grid_type,
            size=self.config.grid_size,
            bounds=self.config.domain_bounds,
            dx=self.config.dx
        )
        
        # Operadores fraccionarios
        self.fractional_laplacian = FractionalLaplacian(
            alpha=self.config.alpha,
            grid=self.grid_manager.get_grid()
        )
        
        # Procesos estocásticos
        self.noise_generator = FractionalNoise(
            hurst=self.config.hurst,
            grid=self.grid_manager.get_grid(),
            correlation_length=self.config.noise_correlation_length,
            seed=self.config.noise_seed
        )
        
        # Términos no lineales
        self.nonlinear_terms = NonlinearTerms(gamma=self.config.gamma)
        
        # Ecuación MFSU
        self.mfsu_equation = MFSUEquation(
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            fractional_op=self.fractional_laplacian,
            noise_gen=self.noise_generator,
            nonlinear_terms=self.nonlinear_terms,
            external_force=self.config.external_force
        )
        
        # Solver numérico
        self.solver = NumericalSolver(
            method=self.config.numerical_method,
            dt=self.config.dt,
            adaptive=self.config.adaptive_timestep,
            tolerance=self.config.tolerance
        )
        
        # Condiciones de frontera
        self.boundary_conditions = BoundaryConditions(
            boundary_type=self.config.boundary_type,
            value=self.config.boundary_value,
            grid=self.grid_manager.get_grid()
        )
        
        # Evolución temporal
        if self.config.adaptive_timestep:
            self.time_evolution = AdaptiveTimeStep(
                dt_initial=self.config.dt,
                tolerance=self.config.tolerance
            )
        else:
            self.time_evolution = TimeEvolution(dt=self.config.dt)
        
        # Exportador de datos
        self.data_exporter = DataExporter(
            output_dir=self.config.output_dir,
            format="hdf5"
        )
    
    def set_initial_condition(self, initial_condition: Union[Callable, np.ndarray, str]):
        """
        Establece la condición inicial para la simulación.
        
        Args:
            initial_condition: Condición inicial como función, array o string
        """
        x = self.grid_manager.get_grid()
        
        if callable(initial_condition):
            self.psi = initial_condition(x)
        elif isinstance(initial_condition, np.ndarray):
            if initial_condition.shape != x.shape:
                raise ValueError("La condición inicial debe tener el mismo tamaño que la grilla")
            self.psi = initial_condition.copy()
        elif isinstance(initial_condition, str):
            # Condiciones predefinidas
            self.psi = self._get_predefined_initial_condition(initial_condition, x)
        else:
            raise ValueError("Tipo de condición inicial no soportado")
        
        self.initial_condition = self.psi.copy()
        self.logger.info(f"Condición inicial establecida: {type(initial_condition)}")
    
    def _get_predefined_initial_condition(self, name: str, x: np.ndarray) -> np.ndarray:
        """Obtiene condiciones iniciales predefinidas."""
        if name == "gaussian":
            return np.exp(-x**2 / 2)
        elif name == "soliton":
            return np.sech(x)
        elif name == "sine":
            return np.sin(2 * np.pi * x / (x[-1] - x[0]))
        elif name == "random":
            np.random.seed(self.config.noise_seed)
            return np.random.randn(*x.shape) * 0.1
        elif name == "zero":
            return np.zeros_like(x)
        else:
            raise ValueError(f"Condición inicial '{name}' no reconocida")
    
    def run(self) -> SimulationResult:
        """
        Ejecuta la simulación MFSU completa.
        
        Returns:
            SimulationResult: Resultado de la simulación
        """
        if self.psi is None:
            self.logger.info("No se estableció condición inicial, usando condición por defecto")
            self.set_initial_condition("gaussian")
        
        self.logger.info("Iniciando simulación MFSU")
        start_time = time.time()
        
        try:
            self.status = SimulationStatus.RUNNING
            self._run_simulation_loop()
            self.status = SimulationStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Error durante la simulación: {e}")
            self.status = SimulationStatus.FAILED
            raise
        
        execution_time = time.time() - start_time
        self.logger.info(f"Simulación completada en {execution_time:.2f} segundos")
        
        return self._create_result(execution_time)
    
    def _run_simulation_loop(self):
        """Loop principal de simulación."""
        n_steps = int(self.config.max_time / self.config.dt)
        save_every = max(1, self.config.save_interval)
        
        self.time_series = [0.0]
        if self.config.save_snapshots:
            self.psi_snapshots = [self.psi.copy()]
        
        for step in range(n_steps):
            # Calcular siguiente paso
            dt_actual = self._compute_timestep()
            
            # Evolucionar el sistema
            psi_new = self._evolve_step(dt_actual)
            
            # Aplicar condiciones de frontera
            psi_new = self.boundary_conditions.apply(psi_new)
            
            # Actualizar estado
            self.psi = psi_new
            self.current_time += dt_actual
            self.current_step = step + 1
            
            # Guardar datos si es necesario
            if (step + 1) % save_every == 0:
                self.time_series.append(self.current_time)
                if self.config.save_snapshots:
                    self.psi_snapshots.append(self.psi.copy())
                
                # Calcular estadísticas
                self._update_statistics()
            
            # Log progreso
            if (step + 1) % (n_steps // 10) == 0:
                progress = (step + 1) / n_steps * 100
                self.logger.info(f"Progreso: {progress:.1f}% - Tiempo: {self.current_time:.2f}")
    
    def _compute_timestep(self) -> float:
        """Computa el paso temporal (adaptativo o fijo)."""
        if self.config.adaptive_timestep:
            return self.time_evolution.get_adaptive_dt(
                self.psi, self.mfsu_equation, self.current_time
            )
        return self.config.dt
    
    def _evolve_step(self, dt: float) -> np.ndarray:
        """
        Evoluciona el sistema un paso temporal.
        
        Args:
            dt: Paso temporal
            
        Returns:
            np.ndarray: Nuevo estado del campo ψ
        """
        return self.solver.step(
            self.psi, 
            self.mfsu_equation, 
            self.current_time, 
            dt
        )
    
    def _update_statistics(self):
        """Actualiza estadísticas de la simulación."""
        # Energía
        energy = np.sum(np.abs(self.psi)**2) * self.config.dx
        if 'energy' not in self.statistics:
            self.statistics['energy'] = []
        self.statistics['energy'].append(energy)
        
        # Momento
        x = self.grid_manager.get_grid()
        momentum = np.sum(np.abs(self.psi)**2 * x) * self.config.dx
        if 'momentum' not in self.statistics:
            self.statistics['momentum'] = []
        self.statistics['momentum'].append(momentum)
        
        # Varianza
        variance = np.var(np.abs(self.psi)**2)
        if 'variance' not in self.statistics:
            self.statistics['variance'] = []
        self.statistics['variance'].append(variance)
    
    def _create_result(self, execution_time: float) -> SimulationResult:
        """Crea el objeto de resultado de la simulación."""
        # Convertir listas a arrays
        time_points = np.array(self.time_series)
        
        if self.config.save_snapshots and self.psi_snapshots:
            psi_array = np.array(self.psi_snapshots)
        else:
            psi_array = self.psi.reshape(1, -1)
        
        # Convertir estadísticas
        stats_arrays = {}
        for key, values in self.statistics.items():
            stats_arrays[key] = np.array(values)
        
        # Crear resultado
        result = SimulationResult(
            psi=psi_array,
            time_points=time_points,
            spatial_grid=self.grid_manager.get_grid(),
            statistics=stats_arrays,
            config=self.config,
            execution_time=execution_time,
            status=self.status,
            metadata={
                'grid_type': self.config.grid_type.value,
                'total_steps': self.current_step,
                'final_time': self.current_time,
                'numerical_method': self.config.numerical_method,
            }
        )
        
        # Guardar resultado si se requiere
        if self.config.output_dir:
            self._save_result(result)
        
        return result
    
    def _save_result(self, result: SimulationResult):
        """Guarda el resultado de la simulación."""
        try:
            filename = f"mfsu_simulation_{int(time.time())}.h5"
            filepath = Path(self.config.output_dir) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            self.data_exporter.save_simulation_result(result, filepath)
            self.logger.info(f"Resultado guardado en: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"No se pudo guardar el resultado: {e}")
    
    def pause(self):
        """Pausa la simulación."""
        if self.status == SimulationStatus.RUNNING:
            self.status = SimulationStatus.PAUSED
            self.logger.info("Simulación pausada")
    
    def resume(self):
        """Reanuda la simulación."""
        if self.status == SimulationStatus.PAUSED:
            self.status = SimulationStatus.RUNNING
            self.logger.info("Simulación reanudada")
    
    def cancel(self):
        """Cancela la simulación."""
        self.status = SimulationStatus.CANCELLED
        self.logger.info("Simulación cancelada")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual de la simulación.
        
        Returns:
            Dict con información del estado actual
        """
        return {
            'status': self.status.value,
            'current_time': self.current_time,
            'current_step': self.current_step,
            'progress': self.current_time / self.config.max_time * 100,
            'psi_current': self.psi.copy() if self.psi is not None else None,
            'statistics': {k: v[-1] if v else 0 for k, v in self.statistics.items()}
        }
    
    def reset(self):
        """Reinicia el simulador al estado inicial."""
        self.current_time = 0.0
        self.current_step = 0
        self.status = SimulationStatus.READY
        
        if self.initial_condition is not None:
            self.psi = self.initial_condition.copy()
        
        self.time_series = []
        self.psi_snapshots = []
        self.statistics = {}
        
        self.logger.info("Simulador reiniciado")


def create_simulator_from_config(config_path: str) -> MFSUSimulator:
    """
    Crea un simulador desde un archivo de configuración.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        MFSUSimulator: Simulador configurado
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extraer configuración de simulación
    sim_config = config_dict.get('simulation', {})
    config = SimulationConfig(**sim_config)
    
    return MFSUSimulator(config)
