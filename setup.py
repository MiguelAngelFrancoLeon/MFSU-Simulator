from setuptools import setup, find_packages

setup(
    name="mfsu-simulator",
    version="1.0.0",
    author="MFSU Development Team",
    description="Unified Stochastic Fractal Model Simulator",
    packages=find_packages(),
    install_requires=[...],
    entry_points={
        'console_scripts': [
            'mfsu-sim=src.simulation.simulator:main',
            'mfsu-gui=gui.main_window:main',
        ],
    },
)
