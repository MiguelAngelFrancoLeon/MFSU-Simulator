"""
Tests para el módulo de aplicaciones en superconductividad del simulador MFSU.

Este módulo contiene tests unitarios y de integración para verificar:
- Implementación correcta de la ecuación MFSU en superconductividad
- Cálculo de resistividad vs temperatura
- Transiciones de fase superconductoras
- Validación con datos experimentales
- Parámetros físicos específicos del material

Ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Importaciones del proyecto (ajustar según la estructura real)
try:
    from src.applications.superconductivity import (
        SuperconductivitySimulator,
        calculate_critical_temperature,
        resistance_vs_temperature,
        cooper_pair_density,
        london_penetration_depth
    )
    from src.core.mfsu_equation import MFSUEquation
    from src.core.fractional_operators import FractionalLaplacian
    from src.utils.constants import PHYSICAL_CONSTANTS
except ImportError as e:
    pytest.skip(f"Módulos del proyecto no disponibles: {e}", allow_module_level=True)


class TestSuperconductivitySimulator:
    """Tests para la clase principal SuperconductivitySimulator."""
    
    @pytest.fixture
    def simulator(self):
        """Fixture que crea una instancia del simulador."""
        params = {
            'alpha': 0.5,
            'beta': 0.1, 
            'gamma': 0.01,
            'hurst': 0.7,
            'temperature_range': (1, 300),
            'material_type': 'YBCO',
            'tc_target': 93.0  # K para YBCO
        }
        return SuperconductivitySimulator(params)
    
    @pytest.fixture
    def sample_data(self):
        """Datos de ejemplo para tests."""
        temperatures = np.linspace(1, 300, 100)
        # Simular curva R vs T típica de superconductor
        resistances = np.where(
            temperatures < 93,
            0.0,  # Resistencia cero por debajo de Tc
            0.1 * (temperatures - 93) + 0.001 * np.random.randn(len(temperatures))
        )
        return pd.DataFrame({
            'temperature': temperatures,
            'resistance': resistances
        })
    
    def test_simulator_initialization(self, simulator):
        """Test inicialización correcta del simulador."""
        assert simulator.alpha == 0.5
        assert simulator.beta == 0.1
        assert simulator.gamma == 0.01
        assert simulator.material_type == 'YBCO'
        assert simulator.tc_target == 93.0
        assert hasattr(simulator, 'mfsu_equation')
    
    def test_invalid_parameters(self):
        """Test manejo de parámetros inválidos."""
        with pytest.raises(ValueError, match="alpha debe estar entre 0 y 2"):
            SuperconductivitySimulator({'alpha': -0.1})
        
        with pytest.raises(ValueError, match="temperatura debe ser positiva"):
            SuperconductivitySimulator({'temperature_range': (-10, 100)})
    
    def test_critical_temperature_calculation(self, simulator):
        """Test cálculo de temperatura crítica."""
        tc_calculated = simulator.calculate_critical_temperature()
        
        # Verificar que está en rango razonable
        assert 50 < tc_calculated < 150  # K
        assert isinstance(tc_calculated, float)
    
    def test_resistance_vs_temperature_curve(self, simulator, sample_data):
        """Test cálculo de curva R vs T."""
        temperatures = sample_data['temperature'].values
        resistances = simulator.calculate_resistance_curve(temperatures)
        
        # Verificar propiedades físicas
        assert len(resistances) == len(temperatures)
        assert np.all(resistances >= 0)  # Resistencia no negativa
        
        # Verificar transición superconductora
        low_temp_resistance = np.mean(resistances[temperatures < 50])
        high_temp_resistance = np.mean(resistances[temperatures > 150])
        
        assert low_temp_resistance < high_temp_resistance  # Transición clara
    
    def test_cooper_pair_density(self, simulator):
        """Test cálculo de densidad de pares de Cooper."""
        temperatures = np.array([10, 50, 93, 150, 200])
        densities = simulator.calculate_cooper_pair_density(temperatures)
        
        assert len(densities) == len(temperatures)
        assert np.all(densities >= 0)
        
        # Densidad debe decrecer con temperatura
        assert densities[0] > densities[-1]
        
        # Por encima de Tc, densidad debe ser muy baja
        above_tc = temperatures > simulator.tc_target
        assert np.all(densities[above_tc] < 0.1)
    
    def test_london_penetration_depth(self, simulator):
        """Test cálculo de profundidad de penetración de London."""
        temperatures = np.array([4, 20, 77, 92])  # Por debajo de Tc
        lambda_l = simulator.calculate_london_depth(temperatures)
        
        assert len(lambda_l) == len(temperatures)
        assert np.all(lambda_l > 0)
        
        # λ_L debe aumentar al acercarse a Tc
        assert lambda_l[0] < lambda_l[-1]
    
    def test_mfsu_equation_integration(self, simulator):
        """Test integración con ecuación MFSU."""
        # Configurar condiciones iniciales
        x = np.linspace(0, 10, 100)
        psi_initial = np.exp(-(x - 5)**2)  # Paquete gaussiano
        
        # Evolucionar sistema
        t_final = 1.0
        dt = 0.01
        psi_final = simulator.evolve_order_parameter(psi_initial, t_final, dt)
        
        assert psi_final.shape == psi_initial.shape
        assert not np.array_equal(psi_initial, psi_final)  # Debe haber evolución
        
        # Verificar conservación aproximada de norma (para γ pequeño)
        norm_initial = np.trapz(np.abs(psi_initial)**2, x)
        norm_final = np.trapz(np.abs(psi_final)**2, x)
        assert abs(norm_final - norm_initial) / norm_initial < 0.1
    
    def test_fractal_dimension_analysis(self, simulator):
        """Test análisis de dimensión fractal de la transición."""
        temperatures = np.linspace(80, 100, 200)
        resistances = simulator.calculate_resistance_curve(temperatures)
        
        # Calcular dimensión fractal cerca de la transición
        fractal_dim = simulator.calculate_fractal_dimension(
            temperatures, resistances, 
            window_size=20
        )
        
        assert 1.0 < fractal_dim < 2.0  # Dimensión fractal física
    
    def test_stochastic_noise_effects(self, simulator):
        """Test efectos del ruido estocástico."""
        # Simular con y sin ruido
        temperatures = np.linspace(85, 95, 50)
        
        simulator.beta = 0.0  # Sin ruido
        resistance_no_noise = simulator.calculate_resistance_curve(temperatures)
        
        simulator.beta = 0.1  # Con ruido
        resistance_with_noise = simulator.calculate_resistance_curve(temperatures)
        
        # El ruido debe suavizar la transición
        grad_no_noise = np.gradient(resistance_no_noise)
        grad_with_noise = np.gradient(resistance_with_noise)
        
        assert np.max(np.abs(grad_no_noise)) > np.max(np.abs(grad_with_noise))


class TestDataValidation:
    """Tests para validación con datos experimentales."""
    
    @pytest.fixture
    def experimental_data(self):
        """Datos experimentales simulados."""
        # Simular datos típicos de YBCO
        temp = np.linspace(1, 300, 150)
        resistance = np.piecewise(
            temp,
            [temp < 93, temp >= 93],
            [0.0, lambda t: 0.001 * (t - 93)**1.5]
        )
        # Añadir ruido experimental
        resistance += 0.0001 * np.random.randn(len(temp))
        
        return pd.DataFrame({
            'temperature': temp,
            'resistance': np.maximum(resistance, 0)  # No resistencia negativa
        })
    
    def test_fit_to_experimental_data(self, experimental_data):
        """Test ajuste a datos experimentales."""
        simulator = SuperconductivitySimulator({
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'material_type': 'YBCO'
        })
        
        # Ajustar parámetros
        fitted_params = simulator.fit_experimental_data(
            experimental_data['temperature'],
            experimental_data['resistance']
        )
        
        assert 'alpha' in fitted_params
        assert 'beta' in fitted_params
        assert 'gamma' in fitted_params
        assert 'tc_fitted' in fitted_params
        
        # Verificar que Tc ajustada está cerca del valor esperado
        assert 90 < fitted_params['tc_fitted'] < 96
    
    def test_goodness_of_fit(self, experimental_data):
        """Test calidad del ajuste."""
        simulator = SuperconductivitySimulator({
            'alpha': 0.5,
            'beta': 0.1, 
            'gamma': 0.01
        })
        
        fitted_params = simulator.fit_experimental_data(
            experimental_data['temperature'],
            experimental_data['resistance']
        )
        
        # Calcular R²
        r_squared = simulator.calculate_r_squared(
            experimental_data['temperature'],
            experimental_data['resistance'],
            fitted_params
        )
        
        assert 0.8 < r_squared < 1.0  # Buen ajuste esperado
    
    @pytest.mark.parametrize("material,tc_expected", [
        ('YBCO', 93),
        ('BSCCO', 110),
        ('MgB2', 39),
        ('Hg-1223', 135)
    ])
    def test_different_materials(self, material, tc_expected):
        """Test simulación de diferentes materiales superconductores."""
        simulator = SuperconductivitySimulator({
            'material_type': material,
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01
        })
        
        tc_calc = simulator.calculate_critical_temperature()
        
        # Permitir 10% de error en Tc
        assert abs(tc_calc - tc_expected) / tc_expected < 0.1


class TestNumericalMethods:
    """Tests para métodos numéricos específicos."""
    
    def test_fractional_derivative_accuracy(self):
        """Test precisión del operador fraccionario."""
        from src.core.fractional_operators import FractionalLaplacian
        
        x = np.linspace(-5, 5, 256)
        dx = x[1] - x[0]
        
        # Función test: gaussiana
        psi = np.exp(-x**2)
        
        # Calcular derivada fraccionaria
        frac_op = FractionalLaplacian(alpha=1.5)
        d_frac = frac_op.apply(psi, dx)
        
        assert len(d_frac) == len(psi)
        assert not np.any(np.isnan(d_frac))
        assert not np.any(np.isinf(d_frac))
    
    def test_time_stepping_stability(self):
        """Test estabilidad del esquema temporal."""
        simulator = SuperconductivitySimulator({
            'alpha': 1.0,
            'beta': 0.05,
            'gamma': 0.01
        })
        
        x = np.linspace(0, 10, 128)
        psi_0 = np.exp(-(x - 5)**2)
        
        # Test diferentes pasos temporales
        dt_values = [0.001, 0.01, 0.1]
        
        for dt in dt_values:
            try:
                psi_final = simulator.evolve_order_parameter(psi_0, 1.0, dt)
                assert np.all(np.isfinite(psi_final))
                # La norma no debe explotar
                assert np.max(np.abs(psi_final)) < 10 * np.max(np.abs(psi_0))
            except Exception as e:
                if dt > 0.05:  # Pasos grandes pueden ser inestables
                    continue
                else:
                    raise e
    
    def test_boundary_conditions(self):
        """Test condiciones de frontera."""
        simulator = SuperconductivitySimulator({
            'boundary_conditions': 'periodic'
        })
        
        x = np.linspace(0, 2*np.pi, 64)
        psi_0 = np.sin(x) + 1j * np.cos(x)  # Condición periódica
        
        psi_evolved = simulator.evolve_order_parameter(psi_0, 0.1, 0.01)
        
        # Verificar periodicidad
        assert abs(psi_evolved[0] - psi_evolved[-1]) < 1e-10


class TestPerformance:
    """Tests de rendimiento y benchmarks."""
    
    def test_simulation_performance(self):
        """Test tiempo de ejecución de simulación."""
        import time
        
        simulator = SuperconductivitySimulator({
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01
        })
        
        x = np.linspace(0, 10, 256)
        psi_0 = np.exp(-(x - 5)**2)
        
        start_time = time.time()
        psi_final = simulator.evolve_order_parameter(psi_0, 1.0, 0.01)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Debe completarse en tiempo razonable
        assert execution_time < 30.0  # 30 segundos máximo
    
    def test_memory_usage(self):
        """Test uso de memoria."""
        import tracemalloc
        
        tracemalloc.start()
        
        simulator = SuperconductivitySimulator({
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01
        })
        
        # Simulación de tamaño medio
        x = np.linspace(0, 20, 512)
        psi_0 = np.random.randn(512) + 1j * np.random.randn(512)
        
        psi_final = simulator.evolve_order_parameter(psi_0, 2.0, 0.01)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Uso de memoria razonable (< 100 MB)
        assert peak < 100 * 1024 * 1024


class TestEdgeCases:
    """Tests para casos límite y situaciones especiales."""
    
    def test_zero_temperature(self):
        """Test comportamiento a temperatura cero."""
        simulator = SuperconductivitySimulator({
            'alpha': 0.5,
            'beta': 0.0,  # Sin ruido térmico
            'gamma': 0.01
        })
        
        resistance = simulator.calculate_resistance_curve(np.array([0.0]))
        assert resistance[0] == 0.0  # Resistencia cero perfecta
    
    def test_high_temperature_limit(self):
        """Test límite de alta temperatura."""
        simulator = SuperconductivitySimulator({
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'tc_target': 93.0
        })
        
        high_temps = np.array([300, 500, 1000])
        resistances = simulator.calculate_resistance_curve(high_temps)
        
        # Comportamiento lineal a altas temperaturas
        assert np.all(np.diff(resistances) > 0)  # Monotónicamente creciente
    
    def test_extreme_parameters(self):
        """Test parámetros extremos."""
        # Alpha muy pequeño (difusión lenta)
        simulator_slow = SuperconductivitySimulator({
            'alpha': 0.1,
            'beta': 0.01,
            'gamma': 0.001
        })
        
        # Alpha cerca de 2 (casi normal)
        simulator_fast = SuperconductivitySimulator({
            'alpha': 1.9,
            'beta': 0.01,
            'gamma': 0.001
        })
        
        x = np.linspace(0, 10, 64)
        psi_0 = np.exp(-(x - 5)**2)
        
        # Ambos deben funcionar sin fallar
        psi_slow = simulator_slow.evolve_order_parameter(psi_0, 0.1, 0.001)
        psi_fast = simulator_fast.evolve_order_parameter(psi_0, 0.1, 0.001)
        
        assert np.all(np.isfinite(psi_slow))
        assert np.all(np.isfinite(psi_fast))


# Tests de integración
class TestIntegration:
    """Tests de integración con otros módulos."""
    
    def test_analysis_module_integration(self):
        """Test integración con módulo de análisis."""
        try:
            from src.analysis.fractal_analysis import calculate_fractal_dimension
            from src.analysis.spectral_analysis import power_spectrum
            
            simulator = SuperconductivitySimulator({
                'alpha': 0.5,
                'beta': 0.1,
                'gamma': 0.01
            })
            
            temps = np.linspace(80, 100, 100)
            resistances = simulator.calculate_resistance_curve(temps)
            
            # Análisis fractal
            fractal_dim = calculate_fractal_dimension(resistances)
            assert 1.0 < fractal_dim < 2.0
            
            # Análisis espectral
            freqs, power = power_spectrum(resistances)
            assert len(freqs) == len(power)
            
        except ImportError:
            pytest.skip("Módulos de análisis no disponibles")
    
    def test_visualization_integration(self):
        """Test integración con herramientas de visualización."""
        try:
            from src.analysis.visualization import plot_resistance_curve
            
            simulator = SuperconductivitySimulator({
                'material_type': 'YBCO'
            })
            
            temps = np.linspace(70, 110, 50)
            resistances = simulator.calculate_resistance_curve(temps)
            
            # Generar plot (sin mostrar)
            fig = plot_resistance_curve(temps, resistances, show=False)
            assert fig is not None
            
        except ImportError:
            pytest.skip("Módulos de visualización no disponibles")


if __name__ == '__main__':
    # Configuración para ejecutar tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ])
