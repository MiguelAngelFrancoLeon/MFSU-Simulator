"""
Test suite para métodos numéricos del simulador MFSU.

Este módulo contiene tests para validar la correcta implementación de los
métodos numéricos especializados para resolver la ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Tests incluidos:
- Validación de estabilidad numérica
- Conservación de propiedades físicas
- Precisión de operadores fraccionarios
- Convergencia temporal y espacial
- Manejo de condiciones de frontera
"""

import unittest
import numpy as np
import pytest
from unittest.mock import Mock, patch
import warnings

# Imports del proyecto (ajustar según estructura real)
from src.core.numerical_methods import (
    FractionalDifferentialSolver,
    AdaptiveTimeStepSolver,
    SpectrumBasedSolver,
    StochasticIntegrator,
    NonlinearTermSolver
)
from src.core.fractional_operators import FractionalLaplacian
from src.core.stochastic_processes import FractalNoise
from src.utils.constants import NUMERICAL_CONSTANTS


class TestFractionalDifferentialSolver(unittest.TestCase):
    """Tests para el solver principal de la ecuación diferencial fraccionaria."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.grid_size = 64
        self.dt = 0.01
        self.dx = 0.1
        self.alpha = 0.5  # Parámetro fraccionario
        self.beta = 0.1   # Intensidad del ruido
        self.gamma = 0.01 # Parámetro no lineal
        
        # Crear grilla espacial
        self.x = np.linspace(-10, 10, self.grid_size)
        
        # Inicializar solver
        self.solver = FractionalDifferentialSolver(
            grid_size=self.grid_size,
            dx=self.dx,
            dt=self.dt,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma
        )
    
    def test_solver_initialization(self):
        """Test de inicialización correcta del solver."""
        self.assertEqual(self.solver.grid_size, self.grid_size)
        self.assertEqual(self.solver.dx, self.dx)
        self.assertEqual(self.solver.dt, self.dt)
        self.assertEqual(self.solver.alpha, self.alpha)
        
        # Verificar que las matrices de diferenciación se inicializaron
        self.assertIsNotNone(self.solver.fractional_diff_matrix)
        self.assertEqual(self.solver.fractional_diff_matrix.shape, 
                        (self.grid_size, self.grid_size))
    
    def test_stability_condition(self):
        """Test de condición de estabilidad CFL para esquemas fraccionarios."""
        # Para operadores fraccionarios, la condición de estabilidad es más restrictiva
        max_stable_dt = self.solver.compute_max_stable_timestep()
        
        # El timestep debe ser menor que el límite de estabilidad
        self.assertLess(self.dt, max_stable_dt)
        
        # Test con parámetros inestables
        unstable_solver = FractionalDifferentialSolver(
            grid_size=self.grid_size,
            dx=0.01,  # dx muy pequeño
            dt=1.0,   # dt muy grande
            alpha=1.8 # alpha cerca del límite
        )
        
        with self.assertWarns(UserWarning):
            unstable_solver.check_stability()
    
    def test_gaussian_packet_evolution(self):
        """Test de evolución de un paquete gaussiano."""
        # Condición inicial: paquete gaussiano
        psi_0 = np.exp(-(self.x**2) / 2.0) * np.exp(1j * self.x)
        
        # Evolucionar por algunos pasos
        n_steps = 10
        psi = psi_0.copy()
        
        for _ in range(n_steps):
            psi = self.solver.evolve_step(psi)
        
        # Verificaciones básicas
        self.assertEqual(len(psi), self.grid_size)
        self.assertTrue(np.all(np.isfinite(psi)))
        
        # La norma total debería conservarse (aproximadamente)
        initial_norm = np.sum(np.abs(psi_0)**2) * self.dx
        final_norm = np.sum(np.abs(psi)**2) * self.dx
        np.testing.assert_allclose(final_norm, initial_norm, rtol=1e-2)
    
    def test_soliton_stability(self):
        """Test de estabilidad de soluciones solitónicas."""
        # Perfil solitónico aproximado (para ecuación no lineal)
        A = 1.0
        k = 1.0
        psi_soliton = A / np.cosh(k * self.x) * np.exp(1j * k * self.x)
        
        # Evolucionar y verificar que mantiene forma aproximada
        psi = psi_soliton.copy()
        initial_peak = np.max(np.abs(psi))
        
        for _ in range(20):
            psi = self.solver.evolve_step(psi)
        
        final_peak = np.max(np.abs(psi))
        
        # El pico no debe cambiar dramáticamente
        self.assertGreater(final_peak / initial_peak, 0.8)
        self.assertLess(final_peak / initial_peak, 1.2)
    
    def test_energy_conservation(self):
        """Test de conservación de energía (hamiltoniano)."""
        # Condición inicial aleatoria suave
        np.random.seed(42)
        psi_0 = np.random.randn(self.grid_size) + 1j * np.random.randn(self.grid_size)
        psi_0 = psi_0 / np.linalg.norm(psi_0)
        
        # Calcular energía inicial
        initial_energy = self.solver.compute_energy(psi_0)
        
        # Evolucionar
        psi = psi_0.copy()
        for _ in range(50):
            psi = self.solver.evolve_step(psi)
        
        final_energy = self.solver.compute_energy(psi)
        
        # En presencia de ruido, la energía puede fluctuar, pero no debe diverger
        self.assertLess(abs(final_energy - initial_energy) / initial_energy, 0.5)


class TestSpectrumBasedSolver(unittest.TestCase):
    """Tests para el solver basado en transformada de Fourier."""
    
    def setUp(self):
        self.grid_size = 128  # Potencia de 2 para FFT eficiente
        self.dx = 0.1
        self.dt = 0.005
        self.alpha = 1.5
        
        self.solver = SpectrumBasedSolver(
            grid_size=self.grid_size,
            dx=self.dx,
            dt=self.dt,
            alpha=self.alpha
        )
    
    def test_fft_consistency(self):
        """Test de consistencia de las transformadas FFT."""
        # Función de prueba
        x = np.linspace(-5, 5, self.grid_size)
        f = np.exp(-x**2 / 2)
        
        # FFT ida y vuelta
        f_k = self.solver.fft(f)
        f_recovered = self.solver.ifft(f_k)
        
        np.testing.assert_allclose(f, f_recovered, rtol=1e-12)
    
    def test_fractional_laplacian_spectrum(self):
        """Test del operador laplaciano fraccionario en el espacio de Fourier."""
        # Crear una onda plana: e^(ikx)
        k_test = 2.0
        x = np.linspace(-np.pi, np.pi, self.grid_size)
        psi = np.exp(1j * k_test * x)
        
        # Aplicar laplaciano fraccionario
        laplacian_psi = self.solver.apply_fractional_laplacian(psi)
        
        # Para una onda plana, (-Δ)^(α/2) e^(ikx) = |k|^α e^(ikx)
        expected = (k_test**self.alpha) * psi
        
        np.testing.assert_allclose(
            laplacian_psi, expected, 
            rtol=1e-10, 
            err_msg="Laplaciano fraccionario incorrecto para onda plana"
        )
    
    def test_spectral_accuracy(self):
        """Test de precisión espectral para funciones suaves."""
        # Función suave conocida
        x = np.linspace(-2*np.pi, 2*np.pi, self.grid_size)
        f = np.sin(2*x) * np.exp(-x**2/8)
        
        # Derivada analítica
        df_analytical = 2*np.cos(2*x)*np.exp(-x**2/8) - x/4*np.sin(2*x)*np.exp(-x**2/8)
        
        # Derivada numérica espectral
        df_numerical = self.solver.spectral_derivative(f, order=1)
        
        # Comparar (excluyendo bordes donde puede haber errores)
        interior = slice(10, -10)
        np.testing.assert_allclose(
            df_numerical[interior], 
            df_analytical[interior], 
            rtol=1e-6
        )


class TestStochasticIntegrator(unittest.TestCase):
    """Tests para la integración de términos estocásticos."""
    
    def setUp(self):
        self.grid_size = 64
        self.dt = 0.01
        self.hurst = 0.7
        self.beta = 0.1
        
        self.integrator = StochasticIntegrator(
            grid_size=self.grid_size,
            dt=self.dt,
            hurst=self.hurst,
            beta=self.beta
        )
    
    def test_fractal_noise_properties(self):
        """Test de propiedades estadísticas del ruido fractal."""
        # Generar múltiples realizaciones
        n_realizations = 1000
        noise_samples = []
        
        for _ in range(n_realizations):
            noise = self.integrator.generate_fractal_noise()
            noise_samples.append(noise)
        
        noise_array = np.array(noise_samples)
        
        # Media debe ser aproximadamente cero
        mean_noise = np.mean(noise_array)
        self.assertLess(abs(mean_noise), 0.1)
        
        # Verificar que la varianza está en rango esperado
        var_noise = np.var(noise_array)
        self.assertGreater(var_noise, 0.5)
        self.assertLess(var_noise, 2.0)
    
    def test_stochastic_integration_convergence(self):
        """Test de convergencia del integrador estocástico."""
        # Función de prueba simple
        psi = np.ones(self.grid_size, dtype=complex)
        
        # Integrar múltiples veces con mismo seed
        np.random.seed(123)
        result1 = self.integrator.integrate_stochastic_term(psi)
        
        np.random.seed(123)
        result2 = self.integrator.integrate_stochastic_term(psi)
        
        # Con mismo seed, resultados deben ser idénticos
        np.testing.assert_array_equal(result1, result2)
    
    def test_multiplicative_noise_effect(self):
        """Test del efecto del ruido multiplicativo βξ_H(x,t)ψ."""
        # Estado inicial no nulo
        psi = np.exp(-(np.linspace(-3, 3, self.grid_size))**2)
        
        # Aplicar término estocástico
        stoch_term = self.integrator.apply_multiplicative_noise(psi)
        
        # Verificar que el término tiene la estructura correcta
        self.assertEqual(stoch_term.shape, psi.shape)
        self.assertTrue(np.all(np.isfinite(stoch_term)))
        
        # La amplitud debe ser proporcional a β
        max_amplitude = np.max(np.abs(stoch_term))
        self.assertLess(max_amplitude, 10 * self.beta)  # Factor de seguridad


class TestNonlinearTermSolver(unittest.TestCase):
    """Tests para el solver de términos no lineales."""
    
    def setUp(self):
        self.grid_size = 32
        self.dt = 0.01
        self.gamma = 0.05
        
        self.nl_solver = NonlinearTermSolver(
            grid_size=self.grid_size,
            dt=self.dt,
            gamma=self.gamma
        )
    
    def test_cubic_nonlinearity(self):
        """Test del término cúbico -γψ³."""
        # Estado de prueba
        psi = np.array([1+0j, 2+1j, 0.5-0.5j, -1+2j], dtype=complex)
        
        # Calcular término no lineal
        nl_term = self.nl_solver.compute_cubic_term(psi)
        
        # Verificar forma analítica: -γ|ψ|²ψ
        expected = -self.gamma * np.abs(psi)**2 * psi
        
        np.testing.assert_allclose(nl_term, expected, rtol=1e-12)
    
    def test_nonlinear_stability(self):
        """Test de estabilidad del solver no lineal."""
        # Condición inicial pequeña (régimen lineal)
        psi_small = 0.01 * np.ones(self.grid_size, dtype=complex)
        
        # Condición inicial grande (régimen no lineal)
        psi_large = 10.0 * np.ones(self.grid_size, dtype=complex)
        
        # Evolucionar ambos
        psi_small_evolved = self.nl_solver.evolve_nonlinear(psi_small, n_steps=100)
        psi_large_evolved = self.nl_solver.evolve_nonlinear(psi_large, n_steps=100)
        
        # Verificar que no hay explosiones numéricas
        self.assertTrue(np.all(np.isfinite(psi_small_evolved)))
        self.assertTrue(np.all(np.isfinite(psi_large_evolved)))
        
        # En régimen no lineal, la solución debe ser acotada
        self.assertLess(np.max(np.abs(psi_large_evolved)), 1e10)
    
    def test_phase_space_conservation(self):
        """Test de conservación en el espacio de fases."""
        # Condición inicial
        x = np.linspace(-5, 5, self.grid_size)
        psi_0 = np.exp(-x**2/2) * (1 + 0.1*np.sin(2*x))
        
        # Evolucionar solo con término no lineal
        psi = psi_0.copy()
        initial_phase_integral = np.sum(np.angle(psi))
        
        for _ in range(50):
            psi = self.nl_solver.evolve_nonlinear_step(psi)
        
        final_phase_integral = np.sum(np.angle(psi))
        
        # La integral de fase puede cambiar, pero no debe explotar
        self.assertLess(abs(final_phase_integral - initial_phase_integral), 100)


class TestAdaptiveTimeStepSolver(unittest.TestCase):
    """Tests para el solver con paso temporal adaptivo."""
    
    def setUp(self):
        self.grid_size = 64
        self.initial_dt = 0.01
        self.tolerance = 1e-6
        
        self.adaptive_solver = AdaptiveTimeStepSolver(
            grid_size=self.grid_size,
            initial_dt=self.initial_dt,
            tolerance=self.tolerance
        )
    
    def test_timestep_adaptation(self):
        """Test de adaptación automática del paso temporal."""
        # Condición inicial suave (debería permitir dt grandes)
        x = np.linspace(-3, 3, self.grid_size)
        psi_smooth = np.exp(-x**2/4)
        
        # Condición inicial rápidamente variante (debería requerir dt pequeños)
        psi_sharp = np.exp(-x**2/0.1) * np.sin(10*x)
        
        # Calcular pasos temporales óptimos
        dt_smooth = self.adaptive_solver.compute_optimal_timestep(psi_smooth)
        dt_sharp = self.adaptive_solver.compute_optimal_timestep(psi_sharp)
        
        # Para función suave, dt puede ser mayor
        self.assertGreater(dt_smooth, dt_sharp)
        self.assertGreater(dt_smooth, 0.005)  # Al menos algo razonable
        self.assertLess(dt_sharp, 0.01)       # Debe ser pequeño
    
    def test_error_estimation(self):
        """Test de estimación del error local."""
        # Estado de prueba
        psi = np.random.randn(self.grid_size) + 1j * np.random.randn(self.grid_size)
        psi = psi / np.linalg.norm(psi)
        
        # Estimar error para diferentes dt
        dt_values = [0.1, 0.01, 0.001]
        errors = []
        
        for dt in dt_values:
            error = self.adaptive_solver.estimate_local_error(psi, dt)
            errors.append(error)
        
        # El error debe decrecer con dt más pequeño
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
    
    def test_adaptive_convergence(self):
        """Test de convergencia con paso adaptivo."""
        # Problema de prueba con solución conocida
        x = np.linspace(-2, 2, self.grid_size)
        psi_exact = lambda t: np.exp(-1j*t) * np.exp(-x**2/2)
        
        # Condición inicial
        psi_0 = psi_exact(0)
        
        # Evolucionar con paso adaptivo
        t_final = 1.0
        psi_adaptive = self.adaptive_solver.solve_adaptive(psi_0, t_final)
        
        # Comparar con solución exacta
        psi_expected = psi_exact(t_final)
        
        # Error debe ser menor que la tolerancia especificada
        error = np.linalg.norm(psi_adaptive - psi_expected) / np.linalg.norm(psi_expected)
        self.assertLess(error, 10 * self.tolerance)


class TestNumericalMethodsIntegration(unittest.TestCase):
    """Tests de integración entre todos los métodos numéricos."""
    
    def test_full_mfsu_equation_solver(self):
        """Test de integración completa de la ecuación MFSU."""
        # Parámetros físicos
        grid_size = 64
        dx = 0.1
        dt = 0.005
        alpha = 0.8
        beta = 0.05
        gamma = 0.02
        hurst = 0.6
        
        # Condición inicial: paquete de ondas
        x = np.linspace(-5, 5, grid_size)
        psi_0 = np.exp(-(x-1)**2) * np.exp(1j * 2*x)
        
        # Crear solver completo (mock de la implementación real)
        from src.core.numerical_methods import CompleteMFSUSolver
        
        solver = CompleteMFSUSolver(
            grid_size=grid_size,
            dx=dx, dt=dt,
            alpha=alpha, beta=beta, gamma=gamma,
            hurst=hurst
        )
        
        # Evolucionar por tiempo corto
        n_steps = 100
        psi_final, diagnostics = solver.evolve(psi_0, n_steps)
        
        # Verificaciones de integridad
        self.assertEqual(psi_final.shape, psi_0.shape)
        self.assertTrue(np.all(np.isfinite(psi_final)))
        
        # Diagnósticos deben incluir información útil
        self.assertIn('energy_evolution', diagnostics)
        self.assertIn('mass_conservation', diagnostics)
        self.assertIn('timestep_history', diagnostics)
    
    def test_performance_scaling(self):
        """Test de escalabilidad con el tamaño de grilla."""
        import time
        
        grid_sizes = [32, 64, 128]
        times = []
        
        for N in grid_sizes:
            # Setup simple
            solver = FractionalDifferentialSolver(
                grid_size=N, dx=0.1, dt=0.01, 
                alpha=0.5, beta=0.1, gamma=0.01
            )
            
            x = np.linspace(-3, 3, N)
            psi = np.exp(-x**2)
            
            # Medir tiempo
            start_time = time.time()
            for _ in range(10):
                psi = solver.evolve_step(psi)
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
        
        # El tiempo debe escalar razonablemente (no exponencialmente)
        # Para FFT, esperamos O(N log N)
        scaling_factor = times[-1] / times[0]  # 128² vs 32²
        expected_max_scaling = (128/32)**2 * np.log(128)/np.log(32)  # Factor teórico
        
        self.assertLess(scaling_factor, 2 * expected_max_scaling)


# Utilities para tests
class TestUtilities(unittest.TestCase):
    """Tests para funciones auxiliares de los métodos numéricos."""
    
    def test_grid_generation(self):
        """Test de generación correcta de grillas."""
        from src.core.numerical_methods import generate_spatial_grid, generate_momentum_grid
        
        N = 64
        L = 10.0
        
        x_grid = generate_spatial_grid(N, L)
        k_grid = generate_momentum_grid(N, L)
        
        # Verificar propiedades
        self.assertEqual(len(x_grid), N)
        self.assertEqual(len(k_grid), N)
        self.assertAlmostEqual(x_grid[1] - x_grid[0], L/(N-1), places=10)
        
        # Grilla de momento debe ser simétrica alrededor de cero
        self.assertAlmostEqual(np.mean(k_grid), 0.0, places=10)
    
    def test_boundary_condition_handling(self):
        """Test de manejo de condiciones de frontera."""
        from src.core.numerical_methods import apply_boundary_conditions
        
        # Datos de prueba
        psi = np.random.randn(64) + 1j * np.random.randn(64)
        
        # Condiciones periódicas
        psi_periodic = apply_boundary_conditions(psi, boundary_type='periodic')
        np.testing.assert_array_equal(psi_periodic, psi)  # No debe cambiar
        
        # Condiciones de Dirichlet (ceros en bordes)
        psi_dirichlet = apply_boundary_conditions(psi, boundary_type='dirichlet')
        self.assertEqual(psi_dirichlet[0], 0.0)
        self.assertEqual(psi_dirichlet[-1], 0.0)
        
        # Condiciones de Neumann (derivada cero en bordes)
        psi_neumann = apply_boundary_conditions(psi, boundary_type='neumann')
        # Primera derivada en bordes debe ser aproximadamente cero
        self.assertLess(abs(psi_neumann[1] - psi_neumann[0]), 1e-10)
        self.assertLess(abs(psi_neumann[-1] - psi_neumann[-2]), 1e-10)


# Configuración de test suite
if __name__ == '__main__':
    # Configurar numpy para advertencias
    np.seterr(all='warn')
    
    # Ejecutar tests con verbosidad
    unittest.main(verbosity=2, buffer=True)
    
    # Opcional: ejecutar con pytest para mayor funcionalidad
    # pytest.main([__file__, '-v', '--tb=short'])
