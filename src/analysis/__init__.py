"""
Módulo de Análisis para el Simulador MFSU
=========================================

Este módulo contiene herramientas para el análisis de datos generados por el
simulador MFSU, incluyendo análisis fractal, espectral, estadístico y 
herramientas de visualización.

La ecuación MFSU que se analiza es:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Donde:
- α: parámetro del operador fraccionario
- β: intensidad del ruido fractal
- γ: parámetro del término no lineal
- ξ_H: ruido fractal con exponente de Hurst H
- f(x,t): término de forzamiento externo

Módulos disponibles:
- fractal_analysis: Análisis de dimensión fractal y propiedades fractales
- spectral_analysis: Análisis espectral y transformadas de Fourier
- statistical_analysis: Análisis estadístico y distribuciones
- visualization: Herramientas de visualización y gráficos

Autor: MFSU Development Team
Versión: 1.0.0
"""

# Importaciones principales
try:
    from .fractal_analysis import (
        FractalAnalyzer,
        calculate_fractal_dimension,
        hurst_exponent,
        lacunarity_analysis,
        multifractal_spectrum,
        detrended_fluctuation_analysis
    )
except ImportError as e:
    print(f"Warning: No se pudo importar fractal_analysis: {e}")
    FractalAnalyzer = None

try:
    from .spectral_analysis import (
        SpectralAnalyzer,
        power_spectrum,
        cross_spectrum,
        coherence_analysis,
        wavelet_analysis,
        fourier_transform_2d,
        spectral_entropy
    )
except ImportError as e:
    print(f"Warning: No se pudo importar spectral_analysis: {e}")
    SpectralAnalyzer = None

try:
    from .statistical_analysis import (
        StatisticalAnalyzer,
        moments_analysis,
        distribution_fitting,
        correlation_analysis,
        entropy_measures,
        complexity_measures,
        anomaly_detection
    )
except ImportError as e:
    print(f"Warning: No se pudo importar statistical_analysis: {e}")
    StatisticalAnalyzer = None

try:
    from .visualization import (
        MFSUVisualizer,
        plot_field_evolution,
        plot_fractal_spectrum,
        plot_power_spectrum,
        plot_statistical_distributions,
        create_interactive_plots,
        export_visualization
    )
except ImportError as e:
    print(f"Warning: No se pudo importar visualization: {e}")
    MFSUVisualizer = None

# Versión del módulo
__version__ = "1.0.0"

# Información del módulo
__author__ = "MFSU Development Team"
__email__ = "mfsu@research.org"
__status__ = "Development"

# Lista de exportaciones públicas
__all__ = [
    # Clases principales
    'FractalAnalyzer',
    'SpectralAnalyzer', 
    'StatisticalAnalyzer',
    'MFSUVisualizer',
    
    # Funciones de análisis fractal
    'calculate_fractal_dimension',
    'hurst_exponent',
    'lacunarity_analysis',
    'multifractal_spectrum',
    'detrended_fluctuation_analysis',
    
    # Funciones de análisis espectral
    'power_spectrum',
    'cross_spectrum',
    'coherence_analysis',
    'wavelet_analysis',
    'fourier_transform_2d',
    'spectral_entropy',
    
    # Funciones de análisis estadístico
    'moments_analysis',
    'distribution_fitting',
    'correlation_analysis',
    'entropy_measures',
    'complexity_measures',
    'anomaly_detection',
    
    # Funciones de visualización
    'plot_field_evolution',
    'plot_fractal_spectrum',
    'plot_power_spectrum',
    'plot_statistical_distributions',
    'create_interactive_plots',
    'export_visualization',
    
    # Funciones de conveniencia
    'analyze_mfsu_field',
    'complete_analysis_pipeline',
    'generate_analysis_report'
]

# Funciones de conveniencia para análisis completo
def analyze_mfsu_field(field_data, parameters=None, analysis_type='complete'):
    """
    Función de conveniencia para analizar un campo MFSU.
    
    Parameters:
    -----------
    field_data : numpy.ndarray
        Datos del campo ψ(x,t) a analizar
    parameters : dict, optional
        Parámetros de la simulación MFSU (α, β, γ, etc.)
    analysis_type : str, default='complete'
        Tipo de análisis: 'fractal', 'spectral', 'statistical', 'complete'
        
    Returns:
    --------
    dict
        Diccionario con los resultados del análisis
    """
    results = {}
    
    if analysis_type in ['fractal', 'complete']:
        if FractalAnalyzer is not None:
            analyzer = FractalAnalyzer()
            results['fractal'] = analyzer.analyze(field_data, parameters)
        else:
            results['fractal'] = {'error': 'FractalAnalyzer no disponible'}
    
    if analysis_type in ['spectral', 'complete']:
        if SpectralAnalyzer is not None:
            analyzer = SpectralAnalyzer()
            results['spectral'] = analyzer.analyze(field_data, parameters)
        else:
            results['spectral'] = {'error': 'SpectralAnalyzer no disponible'}
    
    if analysis_type in ['statistical', 'complete']:
        if StatisticalAnalyzer is not None:
            analyzer = StatisticalAnalyzer()
            results['statistical'] = analyzer.analyze(field_data, parameters)
        else:
            results['statistical'] = {'error': 'StatisticalAnalyzer no disponible'}
    
    return results

def complete_analysis_pipeline(field_data, parameters=None, save_results=True, 
                             output_dir='./analysis_output'):
    """
    Pipeline completo de análisis para datos MFSU.
    
    Parameters:
    -----------
    field_data : numpy.ndarray
        Datos del campo ψ(x,t) a analizar
    parameters : dict, optional
        Parámetros de la simulación MFSU
    save_results : bool, default=True
        Si guardar los resultados en archivos
    output_dir : str, default='./analysis_output'
        Directorio donde guardar los resultados
        
    Returns:
    --------
    dict
        Diccionario completo con todos los análisis
    """
    import os
    import json
    from datetime import datetime
    
    # Realizar análisis completo
    results = analyze_mfsu_field(field_data, parameters, 'complete')
    
    # Agregar metadatos
    results['metadata'] = {
        'analysis_date': datetime.now().isoformat(),
        'field_shape': field_data.shape,
        'parameters': parameters,
        'version': __version__
    }
    
    # Guardar resultados si se solicita
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar resultados en JSON
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generar visualizaciones si está disponible
        if MFSUVisualizer is not None:
            viz = MFSUVisualizer()
            viz.save_all_plots(field_data, results, output_dir)
    
    return results

def generate_analysis_report(results, output_file='mfsu_analysis_report.html'):
    """
    Genera un reporte HTML con los resultados del análisis.
    
    Parameters:
    -----------
    results : dict
        Resultados del análisis completo
    output_file : str, default='mfsu_analysis_report.html'
        Nombre del archivo de reporte
        
    Returns:
    --------
    str
        Ruta al archivo de reporte generado
    """
    try:
        from jinja2 import Template
        import os
        
        # Template básico para el reporte
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MFSU Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin: 30px 0; }
                .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>MFSU Analysis Report</h1>
            <div class="section">
                <h2>Metadata</h2>
                <p>Analysis Date: {{ metadata.analysis_date }}</p>
                <p>Field Shape: {{ metadata.field_shape }}</p>
                <p>MFSU Version: {{ metadata.version }}</p>
            </div>
            
            <div class="section">
                <h2>Fractal Analysis</h2>
                {% if fractal.error %}
                    <p class="error">{{ fractal.error }}</p>
                {% else %}
                    <div class="metric">Fractal Dimension: {{ fractal.dimension }}</div>
                    <div class="metric">Hurst Exponent: {{ fractal.hurst }}</div>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Spectral Analysis</h2>
                {% if spectral.error %}
                    <p class="error">{{ spectral.error }}</p>
                {% else %}
                    <div class="metric">Spectral Entropy: {{ spectral.entropy }}</div>
                    <div class="metric">Dominant Frequency: {{ spectral.dominant_freq }}</div>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                {% if statistical.error %}
                    <p class="error">{{ statistical.error }}</p>
                {% else %}
                    <div class="metric">Mean: {{ statistical.mean }}</div>
                    <div class="metric">Variance: {{ statistical.variance }}</div>
                    <div class="metric">Skewness: {{ statistical.skewness }}</div>
                    <div class="metric">Kurtosis: {{ statistical.kurtosis }}</div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        html_content = template.render(**results)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return os.path.abspath(output_file)
        
    except ImportError:
        print("Warning: jinja2 no está disponible. Generando reporte simple.")
        # Generar reporte simple en texto
        with open(output_file.replace('.html', '.txt'), 'w') as f:
            f.write("MFSU Analysis Report\n")
            f.write("===================\n\n")
            f.write(f"Analysis Date: {results.get('metadata', {}).get('analysis_date', 'N/A')}\n")
            f.write(f"Field Shape: {results.get('metadata', {}).get('field_shape', 'N/A')}\n")
            f.write(f"Version: {results.get('metadata', {}).get('version', 'N/A')}\n\n")
            
            for section, data in results.items():
                if section != 'metadata':
                    f.write(f"{section.upper()} ANALYSIS:\n")
                    f.write(f"{'-' * (len(section) + 10)}\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
        
        return os.path.abspath(output_file.replace('.html', '.txt'))

# Configuración del logging para el módulo
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
