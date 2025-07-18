# MFSU Simulator - Unified Stochastic Fractal Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15828185.svg)](https://doi.org/10.5281/zenodo.15828185)

## Descripción

El **MFSU Simulator** es una implementación computacional del Modelo Estocástico Fractal Unificado (MFSU), que integra geometría fractal, procesos estocásticos y teoría de campos cuánticos para modelar sistemas complejos en física y cosmología.

### Ecuación Fundamental

```
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
```

Donde:
- **α**: Coeficiente de difusión fractal
- **β**: Intensidad estocástica
- **γ**: Parámetro de amortiguamiento no lineal
- **ξ_H(x,t)**: Ruido fraccionario con exponente de Hurst H
- **f(x,t)**: Función de fuerza externa

## Características Principales

### 🔬 Aplicaciones Físicas
- **Superconductividad**: Modelado de pares de Cooper y transiciones de fase
- **Dinámica de Gases**: Simulación de flujos turbulentos y cascadas de energía
- **Cosmología**: Análisis de materia oscura y fluctuaciones del universo temprano

### 📊 Capacidades de Análisis
- Análisis de dimensión fractal en tiempo real
- Visualización espectral y estadística
- Exportación de datos en múltiples formatos
- Benchmarking con soluciones analíticas

### 🖥️ Interfaces
- **GUI Desktop**: Interfaz gráfica completa con Qt/Tkinter
- **Web Interface**: Dashboard interactivo con Plotly/Dash
- **API REST**: Integración con otros sistemas
- **Jupyter Notebooks**: Análisis interactivo

## Instalación

### Instalación Rápida
```bash
pip install mfsu-simulator
```

### Instalación desde Código Fuente
```bash
git clone https://github.com/username/mfsu-simulator.git
cd mfsu-simulator
pip install -e .
```

### Dependencias
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Plotly >= 5.0.0
- Numba >= 0.54.0

## Uso Rápido

### Línea de Comandos
```bash
# Simulación básica
mfsu-sim --application superconductivity --time 10.0

# Con parámetros personalizados
mfsu-sim --alpha 0.7 --beta 0.15 --gamma 0.02 --output results.csv

# Análisis en lotes
mfsu-batch --parameter-sweep alpha 0.1:2.0:0.1
```

### Interfaz Gráfica
```bash
mfsu-gui
```

### Python API
```python
from mfsu import MFSUSimulator, SuperconductivityApp

# Crear simulador
sim = MFSUSimulator(
    alpha=0.5,
    beta=0.1,
    gamma=0.01,
    hurst=0.7,
    grid_size=100
)

# Configurar aplicación
app = SuperconductivityApp(temperature=77)
sim.set_application(app)

# Ejecutar simulación
results = sim.run(time_steps=1000)

# Analizar resultados
fractal_dim = sim.analyze_fractal_dimension()
spectrum = sim.compute_power_spectrum()
```

## Ejemplos de Uso

### 1. Superconductividad
```python
# Modelar transición superconductora
from mfsu.applications import SuperconductivityApp

app = SuperconductivityApp(
    material="YBCO",
    temperature=77,  # K
    magnetic_field=0.1  # T
)

sim = MFSUSimulator(alpha=0.6, beta=0.2, gamma=0.015)
sim.set_application(app)

# Simular enfriamiento
temperatures = np.linspace(300, 4, 100)
critical_temp = app.find_critical_temperature(temperatures)
```

### 2. Dinámica de Gases
```python
# Simular turbulencia
from mfsu.applications import GasDynamicsApp

app = GasDynamicsApp(
    reynolds_number=1000,
    mach_number=0.3,
    viscosity=1e-5
)

sim = MFSUSimulator(alpha=1.2, beta=0.05, gamma=0.001)
energy_cascade = sim.analyze_energy_spectrum()
```

### 3. Cosmología
```python
# Evolución del universo temprano
from mfsu.applications import CosmologyApp

app = CosmologyApp(
    hubble_constant=70,
    omega_matter=0.3,
    omega_lambda=0.7
)

sim = MFSUSimulator(alpha=0.8, beta=0.3, gamma=0.005)
density_fluctuations = sim.compute_density_field()
```

## Configuración

### Archivo `config.yaml`
```yaml
simulation:
  default_parameters:
    alpha: 0.5
    beta: 0.1
    gamma: 0.01
    hurst: 0.7
  
  numerical:
    dt: 0.01
    dx: 0.1
    grid_size: 100
    max_time: 10.0
  
  output:
    format: "hdf5"
    compression: true
    precision: "float64"
```

### Variables de Entorno
```bash
export MFSU_CONFIG_PATH="/path/to/config.yaml"
export MFSU_DATA_DIR="/path/to/data"
export MFSU_PARALLEL_WORKERS=4
```

## Análisis y Visualización

### Jupyter Notebooks Incluidos
- `01_Introduction_to_MFSU.ipynb`: Introducción teórica
- `02_Superconductivity_Analysis.ipynb`: Análisis de superconductividad
- `03_Gas_Dynamics_Simulation.ipynb`: Simulación de fluidos
- `04_Cosmological_Applications.ipynb`: Aplicaciones cosmológicas
- `05_Fractal_Analysis.ipynb`: Análisis fractal avanzado
- `06_Parameter_Sensitivity.ipynb`: Sensibilidad de parámetros

### Herramientas de Análisis
```python
# Análisis fractal
fractal_dim = sim.analyze_fractal_dimension()
hurst_exponent = sim.estimate_hurst_exponent()

# Análisis espectral
power_spectrum = sim.compute_power_spectrum()
correlation_function = sim.compute_correlation()

# Análisis estadístico
moments = sim.compute_statistical_moments()
distribution = sim.analyze_probability_distribution()
```

## Benchmarking y Validación

### Soluciones Analíticas
El simulador incluye comparaciones con:
- Soluciones solitónicas exactas
- Modelos de turbulencia de Kolmogorov
- Fluctuaciones del fondo cósmico de microondas

### Tests de Rendimiento
```bash
# Ejecutar benchmarks
python scripts/benchmark.py

# Resultados típicos:
# - Grid 100x100: ~0.1s por paso temporal
# - Grid 1000x1000: ~10s por paso temporal
# - Precisión numérica: 1e-12
```

## Desarrollo y Contribución

### Estructura del Proyecto
```
mfsu-simulator/
├── src/core/           # Núcleo matemático
├── src/applications/   # Aplicaciones específicas
├── src/analysis/       # Herramientas de análisis
├── gui/               # Interfaz gráfica
├── web/               # Interfaz web
├── tests/             # Tests unitarios
├── docs/              # Documentación
└── notebooks/         # Jupyter notebooks
```

### Contribuir
1. Fork el repositorio
2. Crear rama: `git checkout -b feature/nueva-caracteristica`
3. Commit: `git commit -m "Agregar nueva característica"`
4. Push: `git push origin feature/nueva-caracteristica`
5. Pull Request

### Tests
```bash
# Ejecutar todos los tests
pytest tests/

# Tests específicos
pytest tests/test_core/test_mfsu_equation.py
pytest tests/test_applications/test_superconductivity.py

# Cobertura
pytest --cov=src tests/
```

## Documentación

### Documentación Completa
- **Online**: https://mfsu-simulator.readthedocs.io
- **PDF**: [Descargar documentación](docs/build/latex/MFSU.pdf)

### Generar Documentación Localmente
```bash
cd docs/
make html
open build/html/index.html
```

## Datasets Incluidos

### Datos Experimentales
- `data/experimental/RvT_300K.csv`: Datos de resistividad vs temperatura
- `data/experimental/turbulence_data.csv`: Mediciones de turbulencia
- `data/experimental/cosmic_background.csv`: Datos del CMB

### Datos de Referencia
- `data/reference/benchmark_solutions.json`: Soluciones analíticas
- `data/reference/validation_data.json`: Datos de validación

## Citas y Referencias

### Citar este Software
```bibtex
@software{mfsu_simulator_2025,
  author = {Franco León, Miguel Ángel},
  title = {MFSU Simulator: Unified Stochastic Fractal Model},
  url = {https://github.com/username/mfsu-simulator},
  version = {1.0.0},
  year = {2025},
  doi = {10.5281/zenodo.15828185}
}
```

### Publicaciones Relacionadas
- Franco León, M. A. (2025). "Unified Stochastic Fractal Model: A Framework for Complex Systems". *Zenodo*. DOI: 10.5281/zenodo.15828185

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto y Soporte

- **Issues**: [GitHub Issues](https://github.com/username/mfsu-simulator/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/username/mfsu-simulator/discussions)
- **Email**: support@mfsu-simulator.org
- **Website**: https://mfsu-simulator.org

## Agradecimientos

- Comunidad científica por el desarrollo teórico del MFSU
- Contribuidores del código abierto
- Instituciones de investigación que proporcionaron datos experimentales

---

**Nota**: Este simulador es una herramienta de investigación. Los resultados deben ser validados con datos experimentales apropiados para aplicaciones científicas serias.
