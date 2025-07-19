"""
Tests para el módulo core.mfsu_equation
======================================

Pruebas exhaustivas para la implementación de la ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Este módulo verifica:
- Correcta implementación de cada término de la ecuación
- Conservación de propiedades físicas
- Convergencia numérica
- Casos límite y condiciones especiales
- Integración con operadores fraccionarios y procesos estocásticos

Autor: MFSU Development Team
Fecha: 2025
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch
import warnings
from typing import Dict, Any, Tuple, Optional

# Importar módulos del proyecto
from tests import (
    get_default_mfsu_params, 
    setup_test_data,
    assert_mfsu_solution_properties,
    run_convergence_test,
    TEST_CONFIG
)

try:
    from core.mfsu_equation import MFSUEquation
    from core.fractional_operators import FractionalOperator
    from core.stochastic_processes import StochasticProcess
    from core.numerical_methods import NumericalSolver
except ImportError as e:
    pytest.skip(f"No se pudieron importar módulos core: {e}", allow_module_level=True)


class TestMFSUEquation:
    """Pruebas principales para la clase MFSUEquation."""
    
    def setup_method(self):
        """Configuración para cada test."""
        self.params = get_default_mfsu_params()
        self.test_data = setup_test_data()
        self.mfsu = MFSUEquation(**self.params)
    
    def test_initialization(self):
        """Test de inicialización correcta de la ecuación MFSU."""
        # Test con parámetros por defecto
        mfsu = MFSUEquation()
        assert hasattr(mfsu, 'alpha')
        assert hasattr(mfsu, 'beta')
        assert hasattr(mfsu, 'gamma')
        assert hasattr(mfsu, 'hurst')
        
        # Test con parámetros personalizados
        custom_params = {
            'alpha': 1.5,
            'beta': 0.2,
            'gamma': 0.05,
            'hurst': 0.8
        }
        mfsu_custom = MFSUEquation(**custom_params)
        assert mfsu_custom.alpha == 1.5
        assert mfsu_custom.beta == 0.2
        assert mfsu_custom.gamma == 0.05
        assert mfsu_custom.hurst == 0.8
    
    def test_parameter_validation(self):
        """Test de validación de parámetros."""
        # Parámetros válidos
        valid_params = [
            {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.01, 'hurst': 0.7},
            {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'hurst': 0.5},
            {'alpha': 1.8, 'beta': 1.0, 'gamma': 1.0, 'hurst': 0.9}
        ]
        
        for params in valid_params:
            mfsu = MFSUEquation(**params)
            assert mfsu.validate_parameters()
        
        # Parámetros inválidos
        invalid_params = [
            {'alpha': -0.1},  # alpha negativo
            {'alpha': 2.1},   # alpha > 2
            {'beta': -0.1},   # beta negativo
            {'gamma': -0.1},  # gamma negativo
            {'hurst': 0.0},   # hurst en el límite inferior
            {'hurst': 1.0},   # hurst en el límite superior
        ]
        
        for invalid in invalid_params:
            params = get_default_mfsu_params()
            params.update(invalid)
            with pytest.raises(ValueError):
                MFSUEquation(**params)
    
    def test_fractional_laplacian_term(self):
        """Test del término fraccionario α(-Δ)^(α/2)ψ."""
        x = self.test_data['x']
        psi = self.test_data['psi0']
        
        # Test con función conocida: psi = exp(-x²)
        psi_gaussian = np.exp(-x**2)
        
        # El laplaciano fraccionario de una gaussiana debe ser proporcional a ella misma
        laplacian_term = self.mfsu.fractional_laplacian(psi_gaussian)
        
        # Verificar que el resultado tiene la forma correcta
        assert laplacian_term.shape == psi_gaussian.shape
        assert not np.any(np.isnan(laplacian_term))
        assert not np.any(np.isinf(laplacian_term))
        
        # Test de simetría para funciones pares
        psi_even = np.exp(-x**2) * np.cos(x**2)
        laplacian_even = self.mfsu.fractional_laplacian(psi_even)
        # La transformada debe preservar paridad
        npt.assert_array_almost_equal(
            laplacian_even, 
            laplacian_even[::-1], 
            decimal=10
        )
    
    def test_stochastic_term(self):
        """Test del término estocástico β ξ_H(x,t)ψ."""
        psi = self.test_data['psi0']
        xi = self.test_data['noise'][0]  # Tomar un slice temporal
        
        # Test básico del término estocástico
        stochastic_term = self.mfsu.stochastic_term(psi, xi)
        
        assert stochastic_term.shape == psi.shape
        assert not np.any(np.isnan(stochastic_term))
        
        # Test de linealidad: β(aψ)ξ = a(βψξ) para a escalar
        a = 2.0
        scaled_psi = a * psi
        scaled_term = self.mfsu.stochastic_term(scaled_psi, xi)
        expected = a * stochastic_term
        npt.assert_array_almost_equal(scaled_term, expected, decimal=12)
        
        # Test con ruido cero
        zero_noise = np.zeros_like(xi)
        zero_term = self.mfsu.stochastic_term(psi, zero_noise)
        npt.assert_array_almost_equal(zero_term, np.zeros_like(psi), decimal=15)
        
        # Test de escalamiento con β
        original_beta = self.mfsu.beta
        self.mfsu.beta = 2 * original_beta
        scaled_beta_term = self.mfsu.stochastic_term(psi, xi)
        expected_scaled = 2 * stochastic_term
        npt.assert_array_almost_equal(scaled_beta_term, expected_scaled, decimal=12)
        self.mfsu.beta = original_beta  # Restaurar
    
    def test_nonlinear_term(self):
        """Test del término no lineal -γψ³."""
        psi = self.test_data['psi0']
        
        # Test básico del término no lineal
        nonlinear_term = self.mfsu.nonlinear_term(psi)
        
        assert nonlinear_term.shape == psi.shape
        assert not np.any(np.isnan(nonlinear_term))
        
        # Test de escalamiento cúbico: γ(aψ)³ = a³γψ³
        a = 2.0
        scaled_psi = a * psi
        scaled_term = self.mfsu.nonlinear_term(scaled_psi)
        expected = (a**3) * nonlinear_term
        npt.assert_array_almost_equal(scaled_term, expected, decimal=12)
        
        # Test con psi real positivo
        psi_real = np.abs(psi)
        real_term = self.mfsu.nonlinear_term(psi_real)
        expected_real = -self.mfsu.gamma * psi_real**3
        npt.assert_array_almost_equal(real_term, expected_real, decimal=15)
        
        # Test con γ = 0 (término desactivado)
        original_gamma = self.mfsu.gamma
        self.mfsu.gamma = 0.0
        zero_gamma_term = self.mfsu.nonlinear_term(psi)
        npt.assert_array_almost_equal(zero_gamma_term, np.zeros_like(psi), decimal=15)
        self.mfsu.gamma = original_gamma  # Restaurar
    
    def test_forcing_term(self):
        """Test del término de forzamiento f(x,t)."""
        forcing = self.test_data['forcing'][0]  # Tomar un slice temporal
        
        # Test con forzamiento cero
        zero_forcing = np.zeros_like(forcing)
        zero_term = self.mfsu.forcing_term(zero_forcing)
        npt.assert_array_equal(zero_term, zero_forcing)
        
        # Test con forzamiento constante
        constant_forcing = np.ones_like(forcing)
        constant_term = self.mfsu.forcing_term(constant_forcing)
        npt.assert_array_equal(constant_term, constant_forcing)
        
        # Test con forzamiento complejo
        complex_forcing = forcing + 1j * forcing
        complex_term = self.mfsu.forcing_term(complex_forcing)
        npt.assert_array_equal(complex_term, complex_forcing)
    
    def test_complete_rhs(self):
        """Test de la evaluación completa del lado derecho de la ecuación."""
        psi = self.test_data['psi0']
        xi = self.test_data['noise'][0]
        f = self.test_data['forcing'][0]
        
        # Evaluar el lado derecho completo
        rhs = self.mfsu.evaluate_rhs(psi, xi, f)
        
        # Verificar dimensiones
        assert rhs.shape == psi.shape
        assert not np.any(np.isnan(rhs))
        assert not np.any(np.isinf(rhs))
        
        # Verificar que es la suma de todos los términos
        frac_term = self.mfsu.fractional_laplacian(psi)
        stoch_term = self.mfsu.stochastic_term(psi, xi)
        nonlin_term = self.mfsu.nonlinear_term(psi)
        force_term = self.mfsu.forcing_term(f)
        
        expected_rhs = frac_term + stoch_term + nonlin_term + force_term
        npt.assert_array_almost_equal(rhs, expected_rhs, decimal=12)
    
    def test_energy_conservation_linear_case(self):
        """Test de conservación de energía en el caso lineal (γ=0, β=0, f=0)."""
        # Configurar caso lineal
        linear_params = self.params.copy()
        linear_params['gamma'] = 0.0
        linear_params['beta'] = 0.0
        mfsu_linear = MFSUEquation(**linear_params)
        
        psi = self.test_data['psi0']
        xi = np.zeros_like(self.test_data['noise'][0])
        f = np.zeros_like(self.test_data['forcing'][0])
        
        # La energía inicial
        energy_initial = np.sum(np.abs(psi)**2)
        
        # Evolucionar un paso pequeño
        dt = 1e-4
        rhs = mfsu_linear.evaluate_rhs(psi, xi, f)
        psi_next = psi + dt * rhs
        energy_next = np.sum(np.abs(psi_next)**2)
        
        # En el caso lineal sin ruido ni forzamiento, la energía debe conservarse
        relative_change = abs(energy_next - energy_initial) / energy_initial
        assert relative_change < 1e-6, f"Energía no conservada: cambio relativo {relative_change}"
    
    @pytest.mark.slow
    def test_convergence_with_grid_refinement(self):
        """Test de convergencia al refinar la grilla espacial."""
        def solve_mfsu(params):
            """Función auxiliar para resolver MFSU con parámetros dados."""
            mfsu = MFSUEquation(**params)
            test_data = setup_test_data(params['grid_size'])
            
            # Simular evolución temporal simple (Euler explícito)
            psi = test_data['psi0'].copy()
            xi = test_data['noise'][0]
            f = test_data['forcing'][0]
            
            dt = params.get('dt', 1e-4)
            n_steps = 10
            
            for _ in range(n_steps):
                rhs = mfsu.evaluate_rhs(psi, xi, f)
                psi = psi + dt * rhs
            
            return psi
        
        # Ejecutar test de convergencia
        convergence_results = run_convergence_test(
            solve_mfsu, 
            self.params,
            grid_sizes=[32, 64, 128]
        )
        
        # Verificar que hay convergencia (tasa positiva)
        assert convergence_results['convergence_rate'] > 0.5, \
            f"Convergencia insuficiente: tasa = {convergence_results['convergence_rate']}"
    
    def test_symmetry_properties(self):
        """Test de propiedades de simetría de la ecuación."""
        x = self.test_data['x']
        
        # Test con función par
        psi_even = np.exp(-x**2)
        xi_even = np.random.randn(len(x))
        xi_even = (xi_even + xi_even[::-1]) / 2  # Hacer par
        f_even = np.zeros_like(x)
        
        rhs_even = self.mfsu.evaluate_rhs(psi_even, xi_even, f_even)
        
        # El lado derecho debe preservar paridad para funciones pares
        npt.assert_array_almost_equal(
            rhs_even[:len(x)//2], 
            rhs_even[len(x)//2:][::-1], 
            decimal=8,
            err_msg="Simetría par no preservada"
        )
    
    def test_dimensional_analysis(self):
        """Test de análisis dimensional de la ecuación."""
        # Verificar que los parámetros tienen las unidades correctas
        # (Este test es conceptual, las unidades dependen de la implementación específica)
        
        psi = self.test_data['psi0']
        xi = self.test_data['noise'][0]
        f = self.test_data['forcing'][0]
        
        rhs = self.mfsu.evaluate_rhs(psi, xi, f)
        
        # El lado derecho debe tener las mismas unidades que ∂ψ/∂t
        # Esto implica que rhs y psi/dt deben ser dimensionalmente consistentes
        assert rhs.shape == psi.shape
        assert rhs.dtype == psi.dtype or np.iscomplexobj(rhs)
    
    def test_boundary_conditions(self):
        """Test del comportamiento en los bordes del dominio."""
        x = self.test_data['x']
        
        # Test con condición de borde cero
        psi_zero_boundary = self.test_data['psi0'].copy()
        psi_zero_boundary[0] = 0.0
        psi_zero_boundary[-1] = 0.0
        
        xi = self.test_data['noise'][0]
        f = self.test_data['forcing'][0]
        
        rhs = self.mfsu.evaluate_rhs(psi_zero_boundary, xi, f)
        
        # Verificar que el resultado no produce NaN en los bordes
        assert not np.isnan(rhs[0])
        assert not np.isnan(rhs[-1])
    
    def test_special_cases(self):
        """Test de casos especiales de la ecuación."""
        psi = self.test_data['psi0']
        xi = self.test_data['noise'][0]
        f = self.test_data['forcing'][0]
        
        # Caso 1: Solo término fraccionario (α≠0, β=γ=0, f=0)
        params_frac = self.params.copy()
        params_frac.update({'beta': 0.0, 'gamma': 0.0})
        mfsu_frac = MFSUEquation(**params_frac)
        
        rhs_frac = mfsu_frac.evaluate_rhs(psi, np.zeros_like(xi), np.zeros_like(f))
        expected_frac = mfsu_frac.fractional_laplacian(psi)
        npt.assert_array_almost_equal(rhs_frac, expected_frac, decimal=12)
        
        # Caso 2: Solo término estocástico (β≠0, α=γ=0, f=0)
        params_stoch = self.params.copy()
        params_stoch.update({'alpha': 0.0, 'gamma': 0.0})
        mfsu_stoch = MFSUEquation(**params_stoch)
        
        rhs_stoch = mfsu_stoch.evaluate_rhs(psi, xi, np.zeros_like(f))
        expected_stoch = mfsu_stoch.stochastic_term(psi, xi)
        npt.assert_array_almost_equal(rhs_stoch, expected_stoch, decimal=12)
        
        # Caso 3: Solo término no lineal (γ≠0, α=β=0, f=0)
        params_nonlin = self.params.copy()
        params_nonlin.update({'alpha': 0.0, 'beta': 0.0})
        mfsu_nonlin = MFSUEquation(**params_nonlin)
        
        rhs_nonlin = mfsu_nonlin.evaluate_rhs(psi, np.zeros_like(xi), np.zeros_like(f))
        expected_nonlin = mfsu_nonlin.nonlinear_term(psi)
        npt.assert_array_almost_equal(rhs_nonlin, expected_nonlin, decimal=12)
    
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 1.8])
    def test_fractional_order_variations(self, alpha):
        """Test con diferentes órdenes fraccionarios."""
        params = self.params.copy()
        params['alpha'] = alpha
        mfsu = MFSUEquation(**params)
        
        psi = self.test_data['psi0']
        xi = self.test_data['noise'][0]
        f = self.test_data['forcing'][0]
        
        rhs = mfsu.evaluate_rhs(psi, xi, f)
        
        # Verificar que no hay problemas numéricos
        assert not np.any(np.isnan(rhs))
        assert not np.any(np.isinf(rhs))
        assert rhs.shape == psi.shape
    
    def test_error_handling(self):
        """Test de manejo de errores y casos límite."""
        # Test con arrays de tamaños incompatibles
        psi = self.test_data['psi0']
        xi_wrong_size = np.random.randn(len(psi) + 5)
        f = self.test_data['forcing'][0]
        
        with pytest.raises((ValueError, IndexError)):
            self.mfsu.evaluate_rhs(psi, xi_wrong_size, f)
        
        # Test con valores NaN en la entrada
        psi_nan = psi.copy()
        psi_nan[len(psi)//2] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rhs_nan = self.mfsu.evaluate_rhs(psi_nan, self.test_data['noise'][0], f)
            # Debe detectar y manejar NaN apropiadamente
            assert np.any(np.isnan(rhs_nan)) or np.allclose(rhs_nan, 0, equal_nan=True)
    
    def test_performance_benchmarks(self):
        """Test básico de rendimiento."""
        import time
        
        psi = self.test_data['psi0']
        xi = self.test_data['noise'][0]
        f = self.test_data['forcing'][0]
        
        # Benchmark de evaluación del lado derecho
        n_iterations = 100
        start_time = time.time()
        
        for _ in range(n_iterations):
            rhs = self.mfsu.evaluate_rhs(psi, xi, f)
        
        elapsed_time = time.time() - start_time
        time_per_eval = elapsed_time / n_iterations
        
        # Verificar que el tiempo es razonable (< 1 segundo por evaluación)
        assert time_per_eval < 1.0, f"Evaluación muy lenta: {time_per_eval:.3f} s"
        
        # Log del benchmark
        print(f"Benchmark: {time_per_eval*1000:.2f} ms por evaluación del RHS")


class TestMFSUEquationIntegration:
    """Pruebas de integración con otros módulos."""
    
    def setup_method(self):
        """Configuración para pruebas de integración."""
        self.params = get_default_mfsu_params()
        self.mfsu = MFSUEquation(**self.params)
    
    @pytest.mark.integration
    def test_integration_with_numerical_solver(self):
        """Test de integración con el solver numérico."""
        try:
            from core.numerical_methods import NumericalSolver
        except ImportError:
            pytest.skip("Módulo numerical_methods no disponible")
        
        solver = NumericalSolver(self.mfsu)
        test_data = setup_test_data()
        
        # Resolver por algunos pasos temporales
        solution = solver.solve(
            test_data['psi0'], 
            test_data['t'][:10],
            noise=test_data['noise'][:10],
            forcing=test_data['forcing'][:10]
        )
        
        assert_mfsu_solution_properties(solution, self.params)
    
    @pytest.mark.integration
    def test_integration_with_stochastic_process(self):
        """Test de integración con procesos estocásticos."""
        try:
            from core.stochastic_processes import StochasticProcess
        except ImportError:
            pytest.skip("Módulo stochastic_processes no disponible")
        
        stoch_process = StochasticProcess(hurst=self.params['hurst'])
        test_data = setup_test_data()
        
        # Generar ruido fractal
        fractal_noise = stoch_process.generate_fbm(
            test_data['x'], 
            test_data['t'][0]
        )
        
        # Evaluar la ecuación con ruido fractal
        rhs = self.mfsu.evaluate_rhs(
            test_data['psi0'], 
            fractal_noise, 
            test_data['forcing'][0]
        )
        
        assert not np.any(np.isnan(rhs))
        assert rhs.shape == test_data['psi0'].shape


# Marks adicionales para organizar las pruebas
pytestmark = pytest.mark.core
