"""
Aplicación de Superconductividad para el Simulador MFSU

Este módulo implementa la aplicación específica de la ecuación MFSU
para el estudio de fenómenos superconductivos, incluyendo:
- Transiciones de fase superconductora
- Flujo de vórtices
- Efectos de temperatura crítica
- Análisis de resistencia vs temperatura

Ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Autor: MFSU Development Team
Fecha: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy.interpolate import interp1d
from ..core.mfsu_equation import MFSUEquation
from ..core.stochastic_processes import StochasticProcess
from .base_application import BaseApplication
from ..utils.constants import BOLTZMANN_CONSTANT, PLANCK_CONSTANT


@dataclass
class SuperconductorMaterial:
    """Clase para definir propiedades de materiales superconductores"""
    name: str
    tc_critical: float  # Temperatura crítica (K)
    coherence_length: float  # Longitud de coherencia (m)
    penetration_depth: float  # Profundidad de penetración (m)
    gap_energy: float  # Energía del gap superconductor (eV)
    alpha_coefficient: float  # Coeficiente α específico del material
    beta_coefficient: float  # Coeficiente β específico del material
    gamma_coefficient: float  # Coeficiente γ específico del material


class SuperconductivityApplication(BaseApplication):
    """
    Aplicación especializada para simulación de fenómenos superconductivos
    usando la ecuación MFSU.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Parámetros específicos de superconductividad
        self.temperature_range = config.get('temperature_range', [1, 300])
        self.material_params = config.get('material_parameters', {})
        
        # Inicializar materiales superconductores predefinidos
        self.materials = self._initialize_materials()
        
        # Configurar ecuación MFSU para superconductividad
        self.mfsu_equation = None
        self.current_material = None
        
    def _initialize_materials(self) -> Dict[str, SuperconductorMaterial]:
        """Inicializa materiales superconductores predefinidos"""
        materials = {
            'YBCO': SuperconductorMaterial(
                name='YBa2Cu3O7-δ',
                tc_critical=93.0,
                coherence_length=1.5e-9,
                penetration_depth=150e-9,
                gap_energy=0.02,
                alpha_coefficient=0.8,
                beta_coefficient=0.15,
                gamma_coefficient=0.05
            ),
            'BSCCO': SuperconductorMaterial(
                name='Bi2Sr2CaCu2O8+δ',
                tc_critical=85.0,
                coherence_length=1.2e-9,
                penetration_depth=200e-9,
                gap_energy=0.018,
                alpha_coefficient=0.7,
                beta_coefficient=0.12,
                gamma_coefficient=0.04
            ),
            'MgB2': SuperconductorMaterial(
                name='MgB2',
                tc_critical=39.0,
                coherence_length=5.0e-9,
                penetration_depth=100e-9,
                gap_energy=0.007,
                alpha_coefficient=0.6,
                beta_coefficient=0.1,
                gamma_coefficient=0.03
            ),
            'NbTi': SuperconductorMaterial(
                name='NbTi',
                tc_critical=9.8,
                coherence_length=4.0e-9,
                penetration_depth=200e-9,
                gap_energy=0.0015,
                alpha_coefficient=0.5,
                beta_coefficient=0.08,
                gamma_coefficient=0.02
            )
        }
        return materials
    
    def setup_simulation(self, material_name: str, temperature: float, 
                        grid_params: Dict[str, Any]) -> None:
        """
        Configura la simulación para un material específico y temperatura
        
        Args:
            material_name: Nombre del material superconductor
            temperature: Temperatura del sistema (K)
            grid_params: Parámetros de la grilla espacial
        """
        if material_name not in self.materials:
            raise ValueError(f"Material {material_name} no encontrado")
        
        self.current_material = self.materials[material_name]
        self.current_temperature = temperature
        
        # Calcular parámetros dependientes de temperatura
        t_reduced = temperature / self.current_material.tc_critical
        
        # Ajustar coeficientes MFSU basados en temperatura
        alpha = self._calculate_alpha(t_reduced)
        beta = self._calculate_beta(t_reduced)
        gamma = self._calculate_gamma(t_reduced)
        hurst = self._calculate_hurst(t_reduced)
        
        # Configurar ecuación MFSU
        mfsu_params = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'hurst': hurst,
            'grid_size': grid_params.get('grid_size', 100),
            'dt': grid_params.get('dt', 0.01),
            'dx': grid_params.get('dx', 0.1)
        }
        
        self.mfsu_equation = MFSUEquation(mfsu_params)
        
        self.logger.info(f"Simulación configurada para {material_name} a {temperature}K")
        self.logger.info(f"Parámetros MFSU: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
    
    def _calculate_alpha(self, t_reduced: float) -> float:
        """Calcula coeficiente α basado en temperatura reducida"""
        base_alpha = self.current_material.alpha_coefficient
        
        if t_reduced < 1.0:
            # Región superconductora: α aumenta cerca de Tc
            return base_alpha * (1.0 + 0.5 * np.exp(-(1.0 - t_reduced) * 5))
        else:
            # Región normal: α disminuye
            return base_alpha * np.exp(-(t_reduced - 1.0) * 2)
    
    def _calculate_beta(self, t_reduced: float) -> float:
        """Calcula coeficiente β basado en temperatura reducida"""
        base_beta = self.current_material.beta_coefficient
        
        if t_reduced < 1.0:
            # Fluctuaciones térmicas menores en estado superconductor
            return base_beta * (1.0 - 0.8 * (1.0 - t_reduced))
        else:
            # Fluctuaciones térmicas mayores en estado normal
            return base_beta * (1.0 + 0.5 * (t_reduced - 1.0))
    
    def _calculate_gamma(self, t_reduced: float) -> float:
        """Calcula coeficiente γ basado en temperatura reducida"""
        base_gamma = self.current_material.gamma_coefficient
        
        if t_reduced < 1.0:
            # Término no lineal fuerte en estado superconductor
            return base_gamma * (1.0 + 2.0 * (1.0 - t_reduced))
        else:
            # Término no lineal débil en estado normal
            return base_gamma * 0.1
    
    def _calculate_hurst(self, t_reduced: float) -> float:
        """Calcula exponente de Hurst basado en temperatura"""
        if t_reduced < 1.0:
            # Correlaciones a largo plazo en estado superconductor
            return 0.8 - 0.2 * t_reduced
        else:
            # Correlaciones menores en estado normal
            return 0.5 + 0.1 * np.exp(-(t_reduced - 1.0))
    
    def calculate_order_parameter(self, psi: np.ndarray) -> float:
        """
        Calcula el parámetro de orden superconductor
        
        Args:
            psi: Campo complejo de la función de onda
            
        Returns:
            Magnitud del parámetro de orden
        """
        return np.mean(np.abs(psi))
    
    def calculate_resistance(self, psi: np.ndarray) -> float:
        """
        Calcula la resistencia basada en el parámetro de orden
        
        Args:
            psi: Campo complejo de la función de onda
            
        Returns:
            Resistencia normalizada
        """
        order_param = self.calculate_order_parameter(psi)
        
        # Modelo de resistencia basado en fluctuaciones del parámetro de orden
        if order_param > 0.1:
            # Estado superconductor: resistencia muy baja
            return 0.01 * np.exp(-order_param * 10)
        else:
            # Estado normal: resistencia alta
            return 1.0 - 0.9 * order_param
    
    def simulate_temperature_sweep(self, temperatures: np.ndarray, 
                                 time_steps: int = 1000) -> Dict[str, np.ndarray]:
        """
        Simula barrido de temperatura para obtener curva R vs T
        
        Args:
            temperatures: Array de temperaturas a simular
            time_steps: Número de pasos temporales por temperatura
            
        Returns:
            Diccionario con resultados del barrido
        """
        results = {
            'temperatures': temperatures,
            'resistance': np.zeros_like(temperatures),
            'order_parameter': np.zeros_like(temperatures),
            'susceptibility': np.zeros_like(temperatures)
        }
        
        grid_params = {
            'grid_size': 64,
            'dt': 0.01,
            'dx': 0.1
        }
        
        for i, temp in enumerate(temperatures):
            self.logger.info(f"Simulando temperatura {temp:.1f}K ({i+1}/{len(temperatures)})")
            
            # Configurar simulación para esta temperatura
            self.setup_simulation(list(self.materials.keys())[0], temp, grid_params)
            
            # Condición inicial: campo aleatorio pequeño
            psi_initial = 0.1 * (np.random.randn(grid_params['grid_size']) + 
                               1j * np.random.randn(grid_params['grid_size']))
            
            # Evolucionar el sistema
            psi_final = self._evolve_system(psi_initial, time_steps)
            
            # Calcular propiedades
            results['resistance'][i] = self.calculate_resistance(psi_final)
            results['order_parameter'][i] = self.calculate_order_parameter(psi_final)
            results['susceptibility'][i] = self._calculate_susceptibility(psi_final)
        
        return results
    
    def _evolve_system(self, psi_initial: np.ndarray, time_steps: int) -> np.ndarray:
        """
        Evoluciona el sistema durante un número específico de pasos temporales
        
        Args:
            psi_initial: Condición inicial
            time_steps: Número de pasos temporales
            
        Returns:
            Estado final del sistema
        """
        psi = psi_initial.copy()
        
        for step in range(time_steps):
            # Aplicar un paso de evolución temporal usando MFSU
            if self.mfsu_equation:
                dpsi_dt = self.mfsu_equation.evaluate(psi, step * self.mfsu_equation.dt)
                psi += dpsi_dt * self.mfsu_equation.dt
            
            # Aplicar condiciones de frontera periódicas
            psi = self._apply_boundary_conditions(psi)
        
        return psi
    
    def _apply_boundary_conditions(self, psi: np.ndarray) -> np.ndarray:
        """Aplica condiciones de frontera periódicas"""
        # Condiciones de frontera periódicas ya están implícitas en FFT
        return psi
    
    def _calculate_susceptibility(self, psi: np.ndarray) -> float:
        """Calcula la susceptibilidad magnética"""
        # Susceptibilidad basada en variaciones del parámetro de orden
        order_param = np.abs(psi)
        return np.var(order_param)
    
    def analyze_vortex_dynamics(self, external_field: float, 
                              simulation_time: float = 10.0) -> Dict[str, Any]:
        """
        Analiza la dinámica de vórtices bajo campo magnético externo
        
        Args:
            external_field: Intensidad del campo magnético externo
            simulation_time: Tiempo total de simulación
            
        Returns:
            Diccionario con resultados del análisis de vórtices
        """
        if not self.mfsu_equation:
            raise ValueError("Simulación no configurada. Llamar setup_simulation primero.")
        
        grid_size = self.mfsu_equation.grid_size
        time_steps = int(simulation_time / self.mfsu_equation.dt)
        
        # Condición inicial con vórtices
        psi_initial = self._create_vortex_initial_condition(grid_size, external_field)
        
        # Arrays para almacenar evolución temporal
        psi_history = []
        vortex_positions = []
        
        psi = psi_initial.copy()
        
        for step in range(time_steps):
            # Evolucionar sistema con término de campo magnético
            f_external = self._magnetic_field_term(external_field, step)
            dpsi_dt = self.mfsu_equation.evaluate(psi, step * self.mfsu_equation.dt, f_external)
            psi += dpsi_dt * self.mfsu_equation.dt
            
            # Guardar estado cada cierto número de pasos
            if step % 10 == 0:
                psi_history.append(psi.copy())
                vortex_pos = self._detect_vortices(psi)
                vortex_positions.append(vortex_pos)
        
        return {
            'psi_evolution': psi_history,
            'vortex_positions': vortex_positions,
            'final_state': psi,
            'external_field': external_field
        }
    
    def _create_vortex_initial_condition(self, grid_size: int, field_strength: float) -> np.ndarray:
        """Crea condición inicial con vórtices"""
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Crear vórtices dependiendo de la intensidad del campo
        num_vortices = max(1, int(field_strength * 2))
        psi = np.ones((grid_size, grid_size), dtype=complex)
        
        for i in range(num_vortices):
            # Posición aleatoria del vórtice
            vx = np.random.uniform(-2, 2)
            vy = np.random.uniform(-2, 2)
            
            # Crear estructura de vórtice
            r = np.sqrt((X - vx)**2 + (Y - vy)**2)
            theta = np.arctan2(Y - vy, X - vx)
            
            # Función de onda del vórtice
            vortex = np.tanh(r) * np.exp(1j * theta)
            psi *= vortex
        
        return psi.flatten()
    
    def _magnetic_field_term(self, field_strength: float, time_step: int) -> np.ndarray:
        """Calcula término de campo magnético externo"""
        grid_size = self.mfsu_equation.grid_size
        
        # Campo magnético uniforme en z
        # Simplificación: término proporcional al campo
        return field_strength * 0.1 * np.ones(grid_size, dtype=complex)
    
    def _detect_vortices(self, psi: np.ndarray) -> List[Tuple[float, float]]:
        """Detecta posiciones de vórtices en el campo"""
        # Simplificación: detectar mínimos locales en |psi|
        grid_size = int(np.sqrt(len(psi)))
        psi_2d = psi.reshape(grid_size, grid_size)
        magnitude = np.abs(psi_2d)
        
        # Encontrar mínimos locales (centros de vórtices)
        vortex_positions = []
        threshold = 0.1 * np.max(magnitude)
        
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                if (magnitude[i, j] < threshold and 
                    magnitude[i, j] < magnitude[i-1, j] and
                    magnitude[i, j] < magnitude[i+1, j] and
                    magnitude[i, j] < magnitude[i, j-1] and
                    magnitude[i, j] < magnitude[i, j+1]):
                    vortex_positions.append((i, j))
        
        return vortex_positions
    
    def export_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Exporta resultados de simulación a archivo
        
        Args:
            results: Diccionario con resultados
            filename: Nombre del archivo de salida
        """
        import json
        
        # Convertir arrays numpy a listas para JSON
        export_data = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    export_data[key] = {
                        'real': value.real.tolist(),
                        'imag': value.imag.tolist()
                    }
                else:
                    export_data[key] = value.tolist()
            else:
                export_data[key] = value
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Resultados exportados a {filename}")
    
    def get_material_info(self, material_name: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de un material
        
        Args:
            material_name: Nombre del material
            
        Returns:
            Diccionario con propiedades del material
        """
        if material_name not in self.materials:
            raise ValueError(f"Material {material_name} no encontrado")
        
        material = self.materials[material_name]
        
        return {
            'name': material.name,
            'tc_critical': material.tc_critical,
            'coherence_length': material.coherence_length,
            'penetration_depth': material.penetration_depth,
            'gap_energy': material.gap_energy,
            'alpha_coefficient': material.alpha_coefficient,
            'beta_coefficient': material.beta_coefficient,
            'gamma_coefficient': material.gamma_coefficient,
            'ginzburg_landau_parameter': material.penetration_depth / material.coherence_length
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Valida parámetros de simulación
        
        Args:
            params: Parámetros a validar
            
        Returns:
            True si los parámetros son válidos
        """
        required_params = ['material_name', 'temperature', 'grid_size']
        
        for param in required_params:
            if param not in params:
                self.logger.error(f"Parámetro requerido faltante: {param}")
                return False
        
        # Validar rangos
        if params['temperature'] < 0:
            self.logger.error("La temperatura no puede ser negativa")
            return False
        
        if params['grid_size'] < 10:
            self.logger.error("El tamaño de grilla debe ser al menos 10")
            return False
        
        if params['material_name'] not in self.materials:
            self.logger.error(f"Material no reconocido: {params['material_name']}")
            return False
        
        return True
