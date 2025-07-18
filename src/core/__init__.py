"""
MFSU Core Module - Componentes fundamentales del simulador

Este módulo contiene las implementaciones centrales de la ecuación MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Módulos incluidos:
- mfsu_equation: Implementación principal de la ecuación
- fractional_operators: Operadores fraccionarios (-Δ)^(α/2)
- stochastic_processes: Procesos estocásticos y ruido fractal ξ_H
- nonlinear_dynamics: Términos no lineales γψ³
- numerical_methods: Métodos numéricos especializados
"""

from .mfsu_equation import MFSUEquation
from .fractional_operators import FractionalOperator, FractionalLaplacian
from .stochastic_processes import StochasticProcess, FractalNoise, HurstProcess
from .nonlinear_dynamics import NonlinearTerm, CubicNonlinearity
from .numerical_methods import NumericalSolver, FiniteDifferenceMethod, SpectralMethod

__all__ = [
    # Ecuación principal
    'MFSUEquation',
    
    # Operadores fraccionarios
    'FractionalOperator',
    'FractionalLaplacian',
    
    # Procesos estocásticos
    'StochasticProcess',
    'FractalNoise',
    'HurstProcess',
    
    # Dinámicas no lineales
    'NonlinearTerm',
    'CubicNonlinearity',
    
    # Métodos numéricos
    'NumericalSolver',
    'FiniteDifferenceMethod',
    'SpectralMethod',
]
