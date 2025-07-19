"""
Pruebas unitarias para el módulo de dinámica de gases del simulador MFSU.

Este módulo contiene pruebas para validar la implementación de la ecuación MFSU
en el contexto de dinámica de gases, incluyendo:
- Conservación de masa
- Conservación de momento
- Comportamiento de flujos turbulentos
- Validación con soluciones analíticas conocidas
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from applications.gas_dynamics import GasDynamicsSimulator
from core.mfsu_equation import MFSUEquation
from core.fractional_operators import FractionalLaplacian
from core.stochastic_processes import FractalNoise
from simulation.grid_manager import GridManager
from utils.constants import PhysicalConstants


class TestGasDynamicsSimulator:
    """Clase principal para pruebas del simulador de dinámica de gases."""
    
    @pytest.fixture
    def setup_simulator(self):
        """Configuración básica del simulador para las pruebas."""
        config = {
            'grid_size': 64,
            'domain_size': 1.0,
            'dt': 0.001,
            'total_time': 1.0,
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.7,
            'reynolds_number': 1000,
            'mach_number': 0.3,
            'viscosity': 1e-3,
            'density': 1.0
        }
        simulator = GasDynamicsSimulator(config)
        return simulator
    
    def test_simulator_initialization(self, setup_simulator):
        """Prueba la inicialización correcta del simulador."""
        simulator = setup_simulator
        
        assert simulator.config['grid_size'] == 64
        assert simulator.config['reynolds_number'] == 1000
        assert simulator.config['mach_number'] == 0.3
        assert hasattr(simulator, 'mfsu_solver')
        assert hasattr(simulator, 'grid_manager')
    
    def test_initial_conditions_gaussian_packet(self, setup_simulator):
        """Prueba la generación de condiciones iniciales tipo paquete gaussiano."""
        simulator = setup_simulator
        
        # Generar condiciones iniciales gaussianas
        initial_field = simulator.generate_gaussian_initial_conditions(
            center=(0.5, 0.5),
            width=0.1,
            amplitude=1.0
        )
        
        assert initial_field.shape == (64, 64)
        assert np.abs(np.max(initial_field) - 1.0) < 0.1  # Verificar amplitud aproximada
        assert np.sum(initial_field) > 0  # Campo no vacío
    
    def test_initial_conditions_turbulent_field(self, setup_simulator):
        """Prueba la generación de campo turbulento inicial."""
        simulator = setup_simulator
        
        # Generar campo turbulento
        turbulent_field = simulator.generate_turbulent_initial_conditions(
            energy_spectrum_type='kolmogorov',
            energy_level=1.0
        )
        
        assert turbulent_field.shape == (64, 64)
        assert np.std(turbulent_field) > 0  # Campo con variabilidad
        
        # Verificar espectro de energía (debería seguir ley de Kolmogorov k^(-5/3))
        k_spectrum, energy_spectrum = simulator.compute_energy_spectrum(turbulent_field)
        slope = simulator.fit_power_law(k_spectrum[1:10], energy_spectrum[1:10])
        assert -2.0 < slope < -1.0  # Aproximadamente -5/3
    
    def test_mass_conservation(self, setup_simulator):
        """Prueba la conservación de masa durante la simulación."""
        simulator = setup_simulator
        
        # Configurar condiciones iniciales
        initial_density = simulator.generate_gaussian_initial_conditions(
            center=(0.5, 0.5),
            width=0.2,
            amplitude=1.0
        )
        
        initial_mass = np.sum(initial_density) * simulator.grid_manager.dx**2
        
        # Ejecutar simulación por algunos pasos
        final_density = simulator.run_simulation(
            initial_density,
            steps=100,
            save_interval=10
        )
        
        final_mass = np.sum(final_density) * simulator.grid_manager.dx**2
        
        # Verificar conservación de masa (tolerancia del 1%)
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 0.01, f"Error de conservación de masa: {mass_error:.4f}"
    
    def test_momentum_conservation(self, setup_simulator):
        """Prueba la conservación de momento en ausencia de fuerzas externas."""
        simulator = setup_simulator
        
        # Configurar campo de velocidad inicial con momento neto
        velocity_field = simulator.generate_uniform_velocity_field(
            vx=0.1,
            vy=0.0
        )
        
        density_field = np.ones((64, 64))
        
        initial_momentum = simulator.compute_total_momentum(density_field, velocity_field)
        
        # Simulación sin fuerzas externas
        final_density, final_velocity = simulator.run_momentum_simulation(
            density_field,
            velocity_field,
            steps=50
        )
        
        final_momentum = simulator.compute_total_momentum(final_density, final_velocity)
        
        # Verificar conservación de momento
        momentum_error = np.linalg.norm(final_momentum - initial_momentum) / np.linalg.norm(initial_momentum)
        assert momentum_error < 0.05, f"Error de conservación de momento: {momentum_error:.4f}"
    
    def test_reynolds_number_scaling(self, setup_simulator):
        """Prueba el comportamiento correcto según el número de Reynolds."""
        simulator = setup_simulator
        
        # Probar diferentes números de Reynolds
        reynolds_numbers = [100, 1000, 10000]
        dissipation_rates = []
        
        for re_num in reynolds_numbers:
            simulator.config['reynolds_number'] = re_num
            simulator.update_viscosity()
            
            # Generar campo turbulento
            initial_field = simulator.generate_turbulent_initial_conditions()
            
            # Medir tasa de disipación después de algunos pasos
            final_field = simulator.run_simulation(initial_field, steps=20)
            
            energy_initial = np.sum(initial_field**2)
            energy_final = np.sum(final_field**2)
            dissipation_rate = (energy_initial - energy_final) / energy_initial
            
            dissipation_rates.append(dissipation_rate)
        
        # Verificar que mayor Re implica menor disipación
        assert dissipation_rates[0] > dissipation_rates[1] > dissipation_rates[2]
    
    def test_mach_number_compressibility(self, setup_simulator):
        """Prueba efectos de compresibilidad según el número de Mach."""
        simulator = setup_simulator
        
        # Configurar simulación con diferentes números de Mach
        mach_numbers = [0.1, 0.5, 0.8]  # Subsónico a transónico
        
        for mach in mach_numbers:
            simulator.config['mach_number'] = mach
            simulator.update_sound_speed()
            
            # Generar perturbación de presión
            pressure_field = simulator.generate_pressure_pulse(
                center=(0.5, 0.5),
                amplitude=0.1,
                width=0.1
            )
            
            # Simular propagación
            final_pressure = simulator.run_compressible_simulation(
                pressure_field,
                steps=30
            )
            
            # Verificar velocidad de propagación
            wave_speed = simulator.measure_wave_propagation_speed(
                pressure_field,
                final_pressure
            )
            
            expected_speed = mach * simulator.sound_speed
            speed_error = abs(wave_speed - expected_speed) / expected_speed
            
            assert speed_error < 0.1, f"Error en velocidad de onda para Mach {mach}: {speed_error:.3f}"
    
    def test_fractal_dimension_turbulence(self, setup_simulator):
        """Prueba que la turbulencia desarrollada tenga dimensión fractal apropiada."""
        simulator = setup_simulator
        
        # Generar turbulencia desarrollada
        initial_field = simulator.generate_turbulent_initial_conditions(
            energy_level=10.0
        )
        
        # Evolucionar para desarrollar estructura fractal
        evolved_field = simulator.run_simulation(initial_field, steps=100)
        
        # Calcular dimensión fractal
        fractal_dim = simulator.compute_fractal_dimension(evolved_field)
        
        # Para turbulencia 2D, esperamos dimensión entre 2.3 y 2.7
        assert 2.0 < fractal_dim < 3.0, f"Dimensión fractal fuera de rango: {fractal_dim:.3f}"
    
    def test_energy_cascade_scaling(self, setup_simulator):
        """Prueba la cascada de energía y escalamiento en turbulencia."""
        simulator = setup_simulator
        
        # Generar campo turbulento con inyección de energía en escalas grandes
        initial_field = simulator.generate_forced_turbulence(
            forcing_scale=8,  # Escala de inyección
            forcing_amplitude=1.0
        )
        
        # Evolucionar con forzamiento continuo
        final_field = simulator.run_forced_simulation(
            initial_field,
            steps=200,
            forcing_scale=8
        )
        
        # Analizar espectro de energía
        k_spectrum, energy_spectrum = simulator.compute_energy_spectrum(final_field)
        
        # Verificar escalamiento inercial (k^(-5/3) para turbulencia 3D, k^(-3) para 2D)
        inertial_range = (k_spectrum > 2) & (k_spectrum < 16)
        if np.sum(inertial_range) > 5:
            slope = simulator.fit_power_law(
                k_spectrum[inertial_range],
                energy_spectrum[inertial_range]
            )
            # Para turbulencia 2D esperamos slope ≈ -3
            assert -4.0 < slope < -2.0, f"Escalamiento inercial incorrecto: {slope:.3f}"
    
    def test_boundary_conditions_periodic(self, setup_simulator):
        """Prueba condiciones de frontera periódicas."""
        simulator = setup_simulator
        
        # Configurar campo que cruza fronteras
        field = np.zeros((64, 64))
        field[0, :] = 1.0  # Frontera superior
        field[-1, :] = 1.0  # Frontera inferior
        field[:, 0] = 1.0  # Frontera izquierda
        field[:, -1] = 1.0  # Frontera derecha
        
        # Aplicar un paso de evolución
        evolved_field = simulator.apply_single_time_step(field)
        
        # Verificar continuidad en fronteras
        top_bottom_diff = np.mean(abs(evolved_field[0, :] - evolved_field[-1, :]))
        left_right_diff = np.mean(abs(evolved_field[:, 0] - evolved_field[:, -1]))
        
        assert top_bottom_diff < 0.1, "Condiciones periódicas fallaron en dirección y"
        assert left_right_diff < 0.1, "Condiciones periódicas fallaron en dirección x"
    
    def test_stochastic_noise_effects(self, setup_simulator):
        """Prueba los efectos del ruido estocástico fractal."""
        simulator = setup_simulator
        
        # Simular con y sin ruido
        initial_field = simulator.generate_gaussian_initial_conditions()
        
        # Simulación determinística (β = 0)
        simulator.config['beta'] = 0.0
        final_deterministic = simulator.run_simulation(initial_field.copy(), steps=50)
        
        # Simulación estocástica (β > 0)
        simulator.config['beta'] = 0.1
        final_stochastic = simulator.run_simulation(initial_field.copy(), steps=50)
        
        # El ruido debería aumentar la variabilidad
        var_deterministic = np.var(final_deterministic)
        var_stochastic = np.var(final_stochastic)
        
        assert var_stochastic > var_deterministic, "El ruido no aumentó la variabilidad"
        
        # Verificar que el ruido tiene estructura fractal
        noise_field = final_stochastic - final_deterministic
        hurst_measured = simulator.estimate_hurst_exponent(noise_field)
        expected_hurst = simulator.config['hurst']
        
        assert abs(hurst_measured - expected_hurst) < 0.2, f"Exponente de Hurst incorrecto: {hurst_measured:.3f}"
    
    def test_nonlinear_term_stability(self, setup_simulator):
        """Prueba la estabilidad del término no lineal γψ³."""
        simulator = setup_simulator
        
        # Probar diferentes amplitudes iniciales
        amplitudes = [0.1, 1.0, 5.0, 10.0]
        final_amplitudes = []
        
        for amp in amplitudes:
            initial_field = simulator.generate_gaussian_initial_conditions(
                amplitude=amp
            )
            
            final_field = simulator.run_simulation(initial_field, steps=100)
            final_amp = np.max(np.abs(final_field))
            final_amplitudes.append(final_amp)
        
        # Para amplitudes grandes, el término no lineal debería limitar el crecimiento
        assert final_amplitudes[-1] < amplitudes[-1], "Término no lineal no está limitando amplitudes grandes"
    
    @pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 1.0, 1.5])
    def test_fractional_operator_orders(self, setup_simulator, alpha):
        """Prueba diferentes órdenes del operador fraccionario."""
        simulator = setup_simulator
        simulator.config['alpha'] = alpha
        
        # Generar condiciones iniciales suaves
        initial_field = simulator.generate_gaussian_initial_conditions(
            width=0.2
        )
        
        # Verificar que la simulación es estable
        try:
            final_field = simulator.run_simulation(initial_field, steps=20)
            assert not np.any(np.isnan(final_field)), f"NaN encontrados para α={alpha}"
            assert not np.any(np.isinf(final_field)), f"Inf encontrados para α={alpha}"
            assert np.max(np.abs(final_field)) < 1e6, f"Campo explota para α={alpha}"
        except Exception as e:
            pytest.fail(f"Simulación falló para α={alpha}: {str(e)}")
    
    def test_performance_benchmarks(self, setup_simulator):
        """Prueba de rendimiento y benchmarks."""
        import time
        
        simulator = setup_simulator
        initial_field = simulator.generate_turbulent_initial_conditions()
        
        # Medir tiempo de ejecución
        start_time = time.time()
        final_field = simulator.run_simulation(initial_field, steps=10)
        execution_time = time.time() - start_time
        
        # Verificar que la simulación no sea excesivamente lenta
        # (ajustar según hardware objetivo)
        assert execution_time < 30.0, f"Simulación demasiado lenta: {execution_time:.2f}s"
        
        # Verificar uso eficiente de memoria
        memory_usage = simulator.get_memory_usage()
        assert memory_usage < 1e9, f"Uso excesivo de memoria: {memory_usage/1e6:.1f} MB"
    
    def test_data_export_import(self, setup_simulator):
        """Prueba exportación e importación de datos de simulación."""
        import tempfile
        import os
        
        simulator = setup_simulator
        initial_field = simulator.generate_gaussian_initial_conditions()
        
        # Ejecutar simulación corta
        simulation_data = simulator.run_simulation_with_history(
            initial_field,
            steps=10,
            save_interval=2
        )
        
        # Exportar datos
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "test_simulation.h5")
            simulator.export_simulation_data(simulation_data, export_path)
            
            # Verificar que el archivo existe
            assert os.path.exists(export_path)
            
            # Importar datos
            imported_data = simulator.import_simulation_data(export_path)
            
            # Verificar integridad de datos
            assert len(imported_data['time_series']) == len(simulation_data['time_series'])
            np.testing.assert_array_almost_equal(
                imported_data['final_field'],
                simulation_data['final_field'],
                decimal=10
            )


class TestGasDynamicsValidation:
    """Pruebas de validación contra soluciones conocidas."""
    
    def test_linear_wave_propagation(self):
        """Validación contra propagación lineal de ondas."""
        # Configurar simulación lineal (γ = 0, β = 0)
        config = {
            'grid_size': 128,
            'domain_size': 2.0,
            'dt': 0.001,
            'alpha': 2.0,  # Laplaciano clásico
            'beta': 0.0,   # Sin ruido
            'gamma': 0.0,  # Sin no linealidad
            'mach_number': 0.3
        }
        
        simulator = GasDynamicsSimulator(config)
        
        # Condición inicial: onda sinusoidal
        x = np.linspace(0, 2, 128)
        y = np.linspace(0, 2, 128)
        X, Y = np.meshgrid(x, y)
        
        k = 2 * np.pi  # Número de onda
        initial_field = np.sin(k * X)
        
        # Solución analítica para tiempo t
        t_final = 0.1
        steps = int(t_final / config['dt'])
        
        final_field = simulator.run_simulation(initial_field, steps=steps)
        
        # Solución exacta para difusión lineal
        diffusion_coeff = config['alpha'] * (k)**(config['alpha'])
        analytical_solution = initial_field * np.exp(-diffusion_coeff * t_final)
        
        # Comparar con solución analítica
        error = np.mean((final_field - analytical_solution)**2)
        relative_error = error / np.mean(analytical_solution**2)
        
        assert relative_error < 0.01, f"Error relativo demasiado grande: {relative_error:.6f}"
    
    def test_soliton_propagation(self):
        """Validación de propagación de solitones."""
        # Esta prueba requeriría una configuración especial donde
        # la no linealidad balancea la dispersión
        pass  # Implementar según necesidades específicas
    
    def test_shock_wave_formation(self):
        """Validación de formación de ondas de choque."""
        # Configurar simulación no lineal fuerte
        config = {
            'grid_size': 256,
            'domain_size': 1.0,
            'dt': 0.0005,
            'alpha': 2.0,
            'beta': 0.0,
            'gamma': 1.0,  # No linealidad fuerte
            'mach_number': 2.0  # Supersónico
        }
        
        simulator = GasDynamicsSimulator(config)
        
        # Condición inicial: escalón
        initial_field = np.zeros((256, 256))
        initial_field[100:156, :] = 1.0
        
        # Evolucionar para formar shock
        final_field = simulator.run_simulation(initial_field, steps=200)
        
        # Verificar formación de gradientes fuertes
        gradient = np.gradient(final_field, axis=0)
        max_gradient = np.max(np.abs(gradient))
        
        assert max_gradient > 10.0, "No se formó gradiente fuerte (shock)"


# Funciones auxiliares para las pruebas
def generate_test_data():
    """Genera datos de prueba para validación."""
    # Esta función podría generar datos sintéticos
    # para comparar con resultados experimentales
    pass


def compare_with_reference_data():
    """Compara resultados con datos de referencia."""
    # Cargar datos de validación desde data/reference/
    pass


# Configuración de pytest
def pytest_configure(config):
    """Configuración personalizada de pytest."""
    config.addinivalue_line("markers", "slow: marca pruebas como lentas")
    config.addinivalue_line("markers", "integration: pruebas de integración")


# Ejecución si se llama directamente
if __name__ == "__main__":
    pytest.main([__file__])
