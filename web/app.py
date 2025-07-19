<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MFSU Simulator - Documentación</title>
    <link rel="stylesheet" href="../static/css/styles.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #667eea;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.2em;
            margin: 10px 0 0 0;
        }
        
        .nav-menu {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(90deg, #f8f9fa, #e9ecef);
            border-radius: 10px;
        }
        
        .nav-menu a {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .nav-menu a:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .section {
            margin: 40px 0;
            padding: 30px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin-top: 0;
        }
        
        .formula-box {
            background: linear-gradient(135deg, #f8f9ff, #e3f2fd);
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
        }
        
        .parameter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .parameter-card {
            background: linear-gradient(135deg, #fff, #f8f9fa);
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .parameter-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        
        .parameter-card h4 {
            color: #667eea;
            margin-top: 0;
        }
        
        .code-block {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 15px 0;
            overflow-x: auto;
        }
        
        .highlight {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .applications-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }
        
        .application-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px;
            padding: 25px;
            transition: all 0.3s ease;
        }
        
        .application-card:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
        }
        
        .back-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .back-to-top:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 MFSU Simulator</h1>
            <p>Modelo Fractal Estocástico Unificado - Documentación Técnica</p>
        </div>

        <div class="nav-menu">
            <a href="#introduction">Introducción</a>
            <a href="#theory">Teoría</a>
            <a href="#parameters">Parámetros</a>
            <a href="#applications">Aplicaciones</a>
            <a href="#numerical">Métodos Numéricos</a>
            <a href="#api">API Reference</a>
            <a href="#examples">Ejemplos</a>
        </div>

        <section id="introduction" class="section">
            <h2>🚀 Introducción al MFSU</h2>
            <p>El <strong>Modelo Fractal Estocástico Unificado (MFSU)</strong> representa un avance significativo en la modelización de sistemas complejos que exhiben comportamientos fractales y estocásticos. Este simulador implementa una ecuación diferencial parcial que unifica múltiples fenómenos físicos bajo un marco matemático común.</p>
            
            <div class="highlight">
                <strong>🎯 Características Principales:</strong>
                <ul>
                    <li>Operadores fraccionarios para modelar memorias no locales</li>
                    <li>Procesos estocásticos con ruido fractal</li>
                    <li>Dinámicas no lineales para sistemas complejos</li>
                    <li>Aplicaciones en superconductividad, dinámica de fluidos y cosmología</li>
                </ul>
            </div>
        </section>

        <section id="theory" class="section">
            <h2>🧮 Formulación Teórica</h2>
            
            <h3>Ecuación MFSU</h3>
            <p>La ecuación fundamental del modelo MFSU está dada por:</p>
            
            <div class="formula-box">
                $$\frac{\partial \psi}{\partial t} = \alpha(-\Delta)^{\alpha/2}\psi + \beta \xi_H(x,t)\psi - \gamma\psi^3 + f(x,t)$$
            </div>

            <h3>Descripción de Términos</h3>
            <div class="parameter-grid">
                <div class="parameter-card">
                    <h4>$\alpha(-\Delta)^{\alpha/2}\psi$</h4>
                    <p><strong>Operador Fraccionario:</strong> Modela difusión anómala y efectos de memoria no local. El parámetro $\alpha$ controla el orden fraccionario.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>$\beta \xi_H(x,t)\psi$</h4>
                    <p><strong>Ruido Fractal:</strong> $\xi_H(x,t)$ es un proceso estocástico con exponente de Hurst $H$, representando fluctuaciones correlacionadas.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>$-\gamma\psi^3$</h4>
                    <p><strong>Término No Lineal:</strong> Introduce dinámicas no lineales y efectos de saturación en el sistema.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>$f(x,t)$</h4>
                    <p><strong>Forzamiento Externo:</strong> Término de fuente que puede representar condiciones de contorno o fuerzas externas.</p>
                </div>
            </div>
        </section>

        <section id="parameters" class="section">
            <h2>⚙️ Parámetros del Modelo</h2>
            
            <h3>Parámetros Principales</h3>
            <div class="parameter-grid">
                <div class="parameter-card">
                    <h4>$\alpha$ - Orden Fraccionario</h4>
                    <p><strong>Rango:</strong> $(0, 2]$</p>
                    <p><strong>Defecto:</strong> 0.5</p>
                    <p><strong>Descripción:</strong> Controla el carácter fractal del operador de difusión. Valores menores implican mayor memoria no local.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>$\beta$ - Intensidad Estocástica</h4>
                    <p><strong>Rango:</strong> $[0, \infty)$</p>
                    <p><strong>Defecto:</strong> 0.1</p>
                    <p><strong>Descripción:</strong> Controla la intensidad del ruido fractal en el sistema.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>$\gamma$ - Parámetro No Lineal</h4>
                    <p><strong>Rango:</strong> $[0, \infty)$</p>
                    <p><strong>Defecto:</strong> 0.01</p>
                    <p><strong>Descripción:</strong> Intensidad de los efectos no lineales y de saturación.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>$H$ - Exponente de Hurst</h4>
                    <p><strong>Rango:</strong> $(0, 1)$</p>
                    <p><strong>Defecto:</strong> 0.7</p>
                    <p><strong>Descripción:</strong> Caracteriza las correlaciones a largo plazo del ruido fractal.</p>
                </div>
            </div>

            <h3>Parámetros Numéricos</h3>
            <div class="code-block">
# Configuración por defecto
numerical:
  dt: 0.01          # Paso temporal
  dx: 0.1           # Paso espacial
  grid_size: 100    # Tamaño de grilla
  max_time: 10.0    # Tiempo máximo de simulación
            </div>
        </section>

        <section id="applications" class="section">
            <h2>🔬 Aplicaciones</h2>
            
            <div class="applications-grid">
                <div class="application-card">
                    <h3>🔌 Superconductividad</h3>
                    <p>Modelado de transiciones de fase y comportamiento crítico en superconductores de alta temperatura. El MFSU captura las fluctuaciones cuánticas y los efectos de desorden.</p>
                    <ul>
                        <li>Análisis de resistividad vs temperatura</li>
                        <li>Predicción de temperaturas críticas</li>
                        <li>Efectos de desorden cuántico</li>
                    </ul>
                </div>
                
                <div class="application-card">
                    <h3>🌪️ Dinámica de Gases</h3>
                    <p>Simulación de turbulencia y flujos complejos con memoria fractal. Aplicable a meteorología, ingeniería aeroespacial y estudios ambientales.</p>
                    <ul>
                        <li>Modelado de turbulencia atmosférica</li>
                        <li>Análisis de cascadas energéticas</li>
                        <li>Predicción de patrones de flujo</li>
                    </ul>
                </div>
                
                <div class="application-card">
                    <h3>🌌 Cosmología</h3>
                    <p>Estudio de estructuras cósmicas y evolución del universo. El modelo captura las correlaciones a gran escala y la formación de estructuras.</p>
                    <ul>
                        <li>Formación de estructura cósmica</li>
                        <li>Análisis de radiación de fondo</li>
                        <li>Evolución de densidades</li>
                    </ul>
                </div>
            </div>
        </section>

        <section id="numerical" class="section">
            <h2>🧮 Métodos Numéricos</h2>
            
            <h3>Esquemas de Integración</h3>
            <p>El simulador implementa múltiples métodos numéricos especializados para manejar los operadores fraccionarios y procesos estocásticos:</p>
            
            <div class="parameter-grid">
                <div class="parameter-card">
                    <h4>Operadores Fraccionarios</h4>
                    <p>Implementación vía transformadas de Fourier y diferencias finitas fraccionarias para el término $(-\Delta)^{\alpha/2}$.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>Procesos Estocásticos</h4>
                    <p>Generación de ruido fractal mediante síntesis de Fourier con correlaciones de largo alcance.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>Integración Temporal</h4>
                    <p>Esquemas adaptativos Runge-Kutta con control automático de paso temporal.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>Condiciones de Frontera</h4>
                    <p>Soporte para condiciones periódicas, absorventes y reflectivas.</p>
                </div>
            </div>

            <h3>Optimizaciones</h3>
            <div class="highlight">
                <strong>🚀 Aceleración Computacional:</strong>
                <ul>
                    <li><strong>NumPy/SciPy:</strong> Operaciones vectorizadas</li>
                    <li><strong>Numba:</strong> Compilación JIT para bucles críticos</li>
                    <li><strong>FFTW:</strong> Transformadas de Fourier optimizadas</li>
                    <li><strong>Paralelización:</strong> OpenMP para cálculos multi-hilo</li>
                </ul>
            </div>
        </section>

        <section id="api" class="section">
            <h2>📚 Referencia de API</h2>
            
            <h3>Clase Principal: MFSUSimulator</h3>
            <div class="code-block">
from src.simulation.simulator import MFSUSimulator

# Inicialización
sim = MFSUSimulator(
    alpha=0.5,       # Orden fraccionario
    beta=0.1,        # Intensidad estocástica  
    gamma=0.01,      # Parámetro no lineal
    hurst=0.7        # Exponente de Hurst
)

# Configurar grilla
sim.setup_grid(nx=100, ny=100, dx=0.1, dy=0.1)

# Ejecutar simulación
results = sim.run(
    initial_condition='gaussian',
    max_time=10.0,
    dt=0.01
)
            </div>

            <h3>Métodos Principales</h3>
            <div class="parameter-grid">
                <div class="parameter-card">
                    <h4>setup_grid()</h4>
                    <p>Configura la grilla espacial y las condiciones de frontera.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>set_initial_condition()</h4>
                    <p>Define el estado inicial del sistema (Gaussiano, solitón, etc.).</p>
                </div>
                
                <div class="parameter-card">
                    <h4>run()</h4>
                    <p>Ejecuta la simulación completa y retorna los resultados.</p>
                </div>
                
                <div class="parameter-card">
                    <h4>analyze_results()</h4>
                    <p>Realiza análisis fractal, espectral y estadístico de los resultados.</p>
                </div>
            </div>
        </section>

        <section id="examples" class="section">
            <h2>💡 Ejemplos de Uso</h2>
            
            <h3>Ejemplo 1: Simulación Básica</h3>
            <div class="code-block">
import numpy as np
from src.simulation.simulator import MFSUSimulator

# Crear simulador
sim = MFSUSimulator(alpha=0.8, beta=0.2, gamma=0.05)

# Configurar
sim.setup_grid(nx=128, ny=128, dx=0.1, dy=0.1)
sim.set_initial_condition('gaussian', amplitude=1.0, width=2.0)

# Simular
results = sim.run(max_time=5.0, save_interval=0.1)

# Visualizar
sim.plot_evolution(results)
            </div>

            <h3>Ejemplo 2: Análisis de Superconductividad</h3>
            <div class="code-block">
from src.applications.superconductivity import SuperconductivityMFSU

# Configuración específica para superconductores
sc_sim = SuperconductivityMFSU(
    material='YBCO',
    temperature=77,  # Kelvin
    alpha=0.6,
    beta=0.15
)

# Simular transición superconductora
tc_data = sc_sim.critical_temperature_analysis(
    temp_range=(50, 150),
    steps=100
)

# Graficar R vs T
sc_sim.plot_resistance_vs_temperature(tc_data)
            </div>

            <h3>Ejemplo 3: Barrido de Parámetros</h3>
            <div class="code-block">
from scripts.parameter_sweep import ParameterSweep

# Configurar barrido
sweep = ParameterSweep()
sweep.add_parameter('alpha', np.linspace(0.1, 1.9, 20))
sweep.add_parameter('beta', np.logspace(-2, 0, 10))

# Ejecutar en paralelo
results = sweep.run_parallel(
    application='gas_dynamics',
    metric='fractal_dimension',
    n_cores=8
)

# Analizar resultados
sweep.plot_parameter_space(results)
            </div>
        </section>

        <div class="highlight" style="margin-top: 40px; text-align: center;">
            <h3>🔗 Enlaces Útiles</h3>
            <p>
                <a href="https://zenodo.org/records/15828185" target="_blank">📄 Publicación Original</a> | 
                <a href="/simulator">🖥️ Interfaz Web</a> | 
                <a href="https://github.com/mfsu-simulator" target="_blank">💻 GitHub Repository</a>
            </p>
        </div>
    </div>

    <button class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
        ↑
    </button>

    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('.nav-menu a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Show/hide back-to-top button
        window.addEventListener('scroll', () => {
            const backToTop = document.querySelector('.back-to-top');
            if (window.pageYOffset > 300) {
                backToTop.style.display = 'block';
            } else {
                backToTop.style.display = 'none';
            }
        });

        // Initialize MathJax rendering
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            },
            startup: {
                ready: () => {
                    console.log('MathJax is loaded and ready!');
                    MathJax.startup.defaultReady();
                }
            }
        };
    </script>
</body>
</html>
