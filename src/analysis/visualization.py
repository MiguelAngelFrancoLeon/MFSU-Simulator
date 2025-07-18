"""
Herramientas de Visualización para el Simulador MFSU
===================================================

Este módulo contiene herramientas avanzadas de visualización para analizar
los resultados de las simulaciones del Modelo Fractal Estocástico Unificado.

Autor: MFSU Development Team
Versión: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import signal
from scipy.stats import gaussian_kde
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MFSUVisualizer:
    """
    Clase principal para visualización de resultados MFSU
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Inicializar el visualizador
        
        Args:
            figsize: Tamaño de las figuras por defecto
            dpi: Resolución de las figuras
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colormap = 'viridis'
        
        # Configurar colores personalizados para MFSU
        self.mfsu_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        
        # Crear colormap personalizado
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'mfsu', ['#0d1b2a', '#1b263b', '#415a77', '#778da9', '#e0e1dd']
        )
    
    def plot_solution_evolution(self, 
                              t: np.ndarray,
                              x: np.ndarray,
                              psi: np.ndarray,
                              title: str = "Evolución de la Solución MFSU",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualizar la evolución temporal de la solución ψ(x,t)
        
        Args:
            t: Array de tiempo
            x: Array espacial
            psi: Solución ψ(x,t) con shape (len(t), len(x))
            title: Título del gráfico
            save_path: Ruta para guardar la figura
            
        Returns:
            Figure de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Gráfico 1: Mapa de calor de la evolución
        im1 = axes[0, 0].imshow(np.abs(psi), aspect='auto', cmap=self.custom_cmap,
                               extent=[x[0], x[-1], t[0], t[-1]], origin='lower')
        axes[0, 0].set_xlabel('Posición x')
        axes[0, 0].set_ylabel('Tiempo t')
        axes[0, 0].set_title('|ψ(x,t)| - Mapa de Calor')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Gráfico 2: Perfiles en diferentes tiempos
        time_indices = np.linspace(0, len(t)-1, 5, dtype=int)
        for i, idx in enumerate(time_indices):
            axes[0, 1].plot(x, np.abs(psi[idx]), 
                           label=f't = {t[idx]:.2f}', 
                           color=plt.cm.viridis(i/4))
        axes[0, 1].set_xlabel('Posición x')
        axes[0, 1].set_ylabel('|ψ(x,t)|')
        axes[0, 1].set_title('Perfiles Temporales')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Fase de la solución
        phase = np.angle(psi)
        im2 = axes[1, 0].imshow(phase, aspect='auto', cmap='hsv',
                               extent=[x[0], x[-1], t[0], t[-1]], origin='lower')
        axes[1, 0].set_xlabel('Posición x')
        axes[1, 0].set_ylabel('Tiempo t')
        axes[1, 0].set_title('Fase de ψ(x,t)')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Gráfico 4: Energía total vs tiempo
        energy = np.sum(np.abs(psi)**2, axis=1) * (x[1] - x[0])
        axes[1, 1].plot(t, energy, color=self.mfsu_colors['primary'], linewidth=2)
        axes[1, 1].set_xlabel('Tiempo t')
        axes[1, 1].set_ylabel('Energía Total')
        axes[1, 1].set_title('Conservación de Energía')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_fractal_analysis(self,
                            data: np.ndarray,
                            scales: np.ndarray,
                            dimensions: np.ndarray,
                            hurst_exponent: float,
                            title: str = "Análisis Fractal MFSU") -> plt.Figure:
        """
        Visualizar el análisis de dimensión fractal
        
        Args:
            data: Datos para análisis fractal
            scales: Escalas utilizadas
            dimensions: Dimensiones fractales calculadas
            hurst_exponent: Exponente de Hurst
            title: Título del gráfico
            
        Returns:
            Figure de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Gráfico 1: Datos originales
        axes[0, 0].plot(data, color=self.mfsu_colors['primary'], alpha=0.7)
        axes[0, 0].set_title('Serie Temporal Original')
        axes[0, 0].set_xlabel('Tiempo')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Dimensión fractal vs escala
        axes[0, 1].loglog(scales, dimensions, 'o-', 
                         color=self.mfsu_colors['secondary'], markersize=6)
        axes[0, 1].set_xlabel('Escala')
        axes[0, 1].set_ylabel('Dimensión Fractal')
        axes[0, 1].set_title('Análisis de Dimensión Fractal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Análisis R/S (Hurst)
        n = len(data)
        lags = np.arange(10, n//4, 10)
        rs_values = []
        
        for lag in lags:
            chunks = [data[i:i+lag] for i in range(0, len(data)-lag, lag)]
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) == lag:
                    mean_chunk = np.mean(chunk)
                    cumdev = np.cumsum(chunk - mean_chunk)
                    R = np.max(cumdev) - np.min(cumdev)
                    S = np.std(chunk)
                    if S > 0:
                        rs_chunk.append(R/S)
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if rs_values:
            axes[1, 0].loglog(lags[:len(rs_values)], rs_values, 'o-',
                             color=self.mfsu_colors['accent'], markersize=6)
            axes[1, 0].set_xlabel('Lag')
            axes[1, 0].set_ylabel('R/S')
            axes[1, 0].set_title(f'Análisis R/S (H = {hurst_exponent:.3f})')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Distribución de valores
        axes[1, 1].hist(data, bins=50, density=True, alpha=0.7,
                       color=self.mfsu_colors['info'], edgecolor='black')
        
        # Ajuste gaussiano para comparación
        mu, sigma = np.mean(data), np.std(data)
        x_gauss = np.linspace(np.min(data), np.max(data), 100)
        y_gauss = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x_gauss-mu)/sigma)**2)
        axes[1, 1].plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Gaussiana')
        
        axes[1, 1].set_xlabel('Valor')
        axes[1, 1].set_ylabel('Densidad')
        axes[1, 1].set_title('Distribución de Valores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        return fig
    
    def plot_spectral_analysis(self,
                             t: np.ndarray,
                             data: np.ndarray,
                             title: str = "Análisis Espectral MFSU") -> plt.Figure:
        """
        Análisis espectral de la solución
        
        Args:
            t: Array de tiempo
            data: Datos para análisis espectral
            title: Título del gráfico
            
        Returns:
            Figure de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Calcular FFT
        dt = t[1] - t[0]
        freqs = np.fft.fftfreq(len(data), dt)
        fft_data = np.fft.fft(data)
        power_spectrum = np.abs(fft_data)**2
        
        # Gráfico 1: Serie temporal
        axes[0, 0].plot(t, np.real(data), color=self.mfsu_colors['primary'], 
                       label='Parte Real')
        axes[0, 0].plot(t, np.imag(data), color=self.mfsu_colors['secondary'], 
                       label='Parte Imaginaria')
        axes[0, 0].set_xlabel('Tiempo')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].set_title('Serie Temporal')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Espectro de potencia
        positive_freqs = freqs[freqs > 0]
        positive_power = power_spectrum[freqs > 0]
        
        axes[0, 1].loglog(positive_freqs, positive_power, 
                         color=self.mfsu_colors['accent'], linewidth=2)
        axes[0, 1].set_xlabel('Frecuencia')
        axes[0, 1].set_ylabel('Potencia')
        axes[0, 1].set_title('Espectro de Potencia')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Espectrograma
        f, t_spec, Sxx = signal.spectrogram(data, fs=1/dt, nperseg=min(256, len(data)//4))
        im = axes[1, 0].pcolormesh(t_spec, f, 10*np.log10(Sxx), cmap=self.custom_cmap)
        axes[1, 0].set_xlabel('Tiempo')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Espectrograma')
        plt.colorbar(im, ax=axes[1, 0], label='Potencia (dB)')
        
        # Gráfico 4: Coherencia y fase
        if len(data) > 100:
            # Calcular la coherencia consigo mismo desplazado
            lag_data = np.roll(data, len(data)//10)
            f_coh, Cxy = signal.coherence(data, lag_data, fs=1/dt, nperseg=min(64, len(data)//8))
            axes[1, 1].plot(f_coh, Cxy, color=self.mfsu_colors['warning'], linewidth=2)
            axes[1, 1].set_xlabel('Frecuencia')
            axes[1, 1].set_ylabel('Coherencia')
            axes[1, 1].set_title('Coherencia Temporal')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        return fig
    
    def plot_parameter_sensitivity(self,
                                 parameters: Dict[str, np.ndarray],
                                 results: Dict[str, np.ndarray],
                                 title: str = "Análisis de Sensibilidad de Parámetros") -> plt.Figure:
        """
        Análisis de sensibilidad de parámetros
        
        Args:
            parameters: Diccionario con arrays de parámetros
            results: Diccionario con arrays de resultados
            title: Título del gráfico
            
        Returns:
            Figure de matplotlib
        """
        n_params = len(parameters)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 8), dpi=self.dpi)
        if n_params == 1:
            axes = [axes]
        axes = axes.flatten()
        
        colors = [self.mfsu_colors['primary'], self.mfsu_colors['secondary'], 
                 self.mfsu_colors['accent'], self.mfsu_colors['warning']]
        
        for i, (param_name, param_values) in enumerate(parameters.items()):
            if i < len(axes):
                for j, (result_name, result_values) in enumerate(results.items()):
                    axes[i].plot(param_values, result_values, 'o-', 
                               color=colors[j % len(colors)], 
                               label=result_name, markersize=6)
                
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('Resultado')
                axes[i].set_title(f'Sensibilidad a {param_name}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar ejes no utilizados
        for i in range(len(parameters), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        return fig
    
    def create_interactive_plot(self,
                              t: np.ndarray,
                              x: np.ndarray,
                              psi: np.ndarray,
                              title: str = "Simulación MFSU Interactiva") -> go.Figure:
        """
        Crear un gráfico interactivo con Plotly
        
        Args:
            t: Array de tiempo
            x: Array espacial
            psi: Solución ψ(x,t)
            title: Título del gráfico
            
        Returns:
            Figure de plotly
        """
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mapa de Calor |ψ(x,t)|', 'Perfiles Temporales',
                           'Fase de ψ(x,t)', 'Energía Total'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Mapa de calor
        fig.add_trace(
            go.Heatmap(
                z=np.abs(psi),
                x=x,
                y=t,
                colorscale='Viridis',
                name='|ψ(x,t)|'
            ),
            row=1, col=1
        )
        
        # Perfiles en diferentes tiempos
        time_indices = np.linspace(0, len(t)-1, 5, dtype=int)
        for i, idx in enumerate(time_indices):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.abs(psi[idx]),
                    mode='lines',
                    name=f't = {t[idx]:.2f}',
                    line=dict(color=px.colors.qualitative.Set1[i])
                ),
                row=1, col=2
            )
        
        # Fase
        fig.add_trace(
            go.Heatmap(
                z=np.angle(psi),
                x=x,
                y=t,
                colorscale='HSV',
                name='Fase'
            ),
            row=2, col=1
        )
        
        # Energía
        energy = np.sum(np.abs(psi)**2, axis=1) * (x[1] - x[0])
        fig.add_trace(
            go.Scatter(
                x=t,
                y=energy,
                mode='lines',
                name='Energía Total',
                line=dict(color='blue', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_animation(self,
                        t: np.ndarray,
                        x: np.ndarray,
                        psi: np.ndarray,
                        title: str = "Animación MFSU",
                        interval: int = 100) -> animation.FuncAnimation:
        """
        Crear animación de la evolución temporal
        
        Args:
            t: Array de tiempo
            x: Array espacial
            psi: Solución ψ(x,t)
            title: Título de la animación
            interval: Intervalo entre frames (ms)
            
        Returns:
            FuncAnimation object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Configurar límites
        ax1.set_xlim(x[0], x[-1])
        ax1.set_ylim(0, np.max(np.abs(psi)) * 1.1)
        ax1.set_xlabel('Posición x')
        ax1.set_ylabel('|ψ(x,t)|')
        ax1.set_title('Amplitud')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(x[0], x[-1])
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_xlabel('Posición x')
        ax2.set_ylabel('Fase')
        ax2.set_title('Fase')
        ax2.grid(True, alpha=0.3)
        
        line1, = ax1.plot([], [], color=self.mfsu_colors['primary'], linewidth=2)
        line2, = ax2.plot([], [], color=self.mfsu_colors['secondary'], linewidth=2)
        
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def animate(frame):
            line1.set_data(x, np.abs(psi[frame]))
            line2.set_data(x, np.angle(psi[frame]))
            time_text.set_text(f'Tiempo = {t[frame]:.3f}')
            return line1, line2, time_text
        
        anim = animation.FuncAnimation(fig, animate, frames=len(t), 
                                     interval=interval, blit=True, repeat=True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return anim
    
    def export_results(self,
                      results: Dict[str, np.ndarray],
                      filename: str,
                      format: str = 'png') -> None:
        """
        Exportar resultados de visualización
        
        Args:
            results: Diccionario con resultados
            filename: Nombre del archivo
            format: Formato de exportación ('png', 'pdf', 'svg', 'html')
        """
        if format in ['png', 'pdf', 'svg']:
            # Crear figura resumen
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
            
            # Aquí puedes agregar visualizaciones específicas según los resultados
            for i, (key, data) in enumerate(results.items()):
                if i < 4:  # Solo primeros 4 resultados
                    ax = axes[i//2, i%2]
                    if data.ndim == 1:
                        ax.plot(data, color=self.mfsu_colors['primary'])
                    elif data.ndim == 2:
                        im = ax.imshow(data, cmap=self.custom_cmap, aspect='auto')
                        plt.colorbar(im, ax=ax)
                    ax.set_title(key)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(filename, format=format, dpi=150, bbox_inches='tight')
            plt.close()
        
        elif format == 'html':
            # Crear reporte HTML interactivo
            # Implementar exportación HTML con plotly
            pass
    
    def set_style(self, style: str = 'default') -> None:
        """
        Configurar estilo de visualización
        
        Args:
            style: Estilo a aplicar ('default', 'dark', 'minimal', 'scientific')
        """
        if style == 'dark':
            plt.style.use('dark_background')
            self.mfsu_colors = {
                'primary': '#00d4ff',
                'secondary': '#ff6b35',
                'accent': '#32ff32',
                'warning': '#ff3232',
                'info': '#ff32ff'
            }
        elif style == 'minimal':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.mfsu_colors = {
                'primary': '#2c3e50',
                'secondary': '#e74c3c',
                'accent': '#27ae60',
                'warning': '#f39c12',
                'info': '#9b59b6'
            }
        elif style == 'scientific':
            plt.style.use('seaborn-v0_8-paper')
            self.mfsu_colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent': '#2ca02c',
                'warning': '#d62728',
                'info': '#9467bd'
            }
        else:  # default
            plt.style.use('seaborn-v0_8')
            self.mfsu_colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent': '#2ca02c',
                'warning': '#d62728',
                'info': '#9467bd'
            }


def create_comparison_plot(results_dict: Dict[str, Dict[str, np.ndarray]],
                          parameter_name: str,
                          title: str = "Comparación de Resultados") -> plt.Figure:
    """
    Crear gráfico de comparación entre diferentes configuraciones
    
    Args:
        results_dict: Diccionario con resultados de diferentes configuraciones
        parameter_name: Nombre del parámetro que varía
        title: Título del gráfico
        
    Returns:
        Figure de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for i, (config_name, results) in enumerate(results_dict.items()):
        color = colors[i]
        
        # Ejemplo de comparación - adaptar según necesidades específicas
        if 'energy' in results:
            axes[0, 0].plot(results['energy'], color=color, label=config_name, linewidth=2)
        if 'amplitude' in results:
            axes[0, 1].plot(results['amplitude'], color=color, label=config_name, linewidth=2)
        if 'phase' in results:
            axes[1, 0].plot(results['phase'], color=color, label=config_name, linewidth=2)
        if 'spectrum' in results:
            axes[1, 1].loglog(results['spectrum'], color=color, label=config_name, linewidth=2)
    
    axes[0, 0].set_title('Evolución de Energía')
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('Energía')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Amplitud Máxima')
    axes[0, 1].set_xlabel('Tiempo')
    axes[0, 1].set_ylabel('Amplitud')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Fase Promedio')
    axes[1, 0].set_xlabel('Tiempo')
    axes[1, 0].set_ylabel('Fase')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Espectro de Potencia')
    axes[1, 1].set_xlabel('Frecuencia')
    axes[1, 1].set_ylabel('Potencia')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'{title} - Variación de {parameter_name}', fontsize=16, y=1.02)
    
    return fig


# Funciones auxiliares para visualización específica de aplicaciones

def plot_superconductivity_results(temperature: np.ndarray,
                                  resistance: np.ndarray,
                                  critical_temp: float,
                                  title: str = "Análisis de Superconductividad") -> plt.Figure:
    """
    Visualización específica para aplicaciones de superconductividad
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # Gráfico R vs T
    ax1.plot(temperature, resistance, 'o-', color='#1f77b4', linewidth=2, markersize=4)
    ax1.axvline(critical_temp, color='red', linestyle='--', linewidth=2, 
                label=f'Tc = {critical_temp:.2f} K')
    ax1.set_xlabel('Temperatura (K)')
    ax1.set_ylabel('Resistencia (Ω)')
    ax1.set_title('Transición Superconductora')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico log-log cerca de la transición
    transition_mask = (temperature >= critical_temp * 0.8) & (temperature <= critical_temp * 1.2)
    if np.any(transition_mask):
        temp_trans = temperature[transition_mask]
        res_trans = resistance[transition_mask]
        ax2.loglog(temp_trans, res_trans, 'o-', color='#ff7f0e', linewidth=2, markersize=4)
        ax2.set_xlabel('Temperatura (K)')
        ax2.set_ylabel('Resistencia (Ω)')
        ax2.set_title('Región de Transición (log-log)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig


def plot_cosmology_results(redshift: np.ndarray,
                          density_perturbation: np.ndarray,
                          title: str = "Análisis Cosmológico") -> plt.Figure:
    """
    Visualización específica para aplicaciones cosmológicas
    
    Args:
        redshift: Array de corrimiento al rojo
        density_perturbation: Perturbaciones de densidad
        title: Título del gráfico
        
    Returns:
        Figure de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # Gráfico 1: Evolución de perturbaciones de densidad
    axes[0, 0].plot(redshift, density_perturbation, 'o-', color='#1f77b4', 
                   linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Corrimiento al rojo (z)')
    axes[0, 0].set_ylabel('δ(z)')
    axes[0, 0].set_title('Evolución de Perturbaciones de Densidad')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_xaxis()  # z decrece con el tiempo
    
    # Gráfico 2: Espectro de potencia
    k = np.linspace(0.01, 10, len(density_perturbation))
    power_spectrum = np.abs(np.fft.fft(density_perturbation))**2
    axes[0, 1].loglog(k, power_spectrum[:len(k)], color='#ff7f0e', linewidth=2)
    axes[0, 1].set_xlabel('Número de onda k (h/Mpc)')
    axes[0, 1].set_ylabel('P(k)')
    axes[0, 1].set_title('Espectro de Potencia')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Factor de crecimiento
    growth_factor = density_perturbation / density_perturbation[0]
    axes[1, 0].plot(redshift, growth_factor, 's-', color='#2ca02c', 
                   linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Corrimiento al rojo (z)')
    axes[1, 0].set_ylabel('D(z)/D(0)')
    axes[1, 0].set_title('Factor de Crecimiento')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_xaxis()
    
    # Gráfico 4: Función de correlación
    correlation = np.correlate(density_perturbation, density_perturbation, mode='full')
    correlation = correlation[correlation.size // 2:]
    correlation = correlation / correlation[0]
    r = np.arange(len(correlation))
    axes[1, 1].plot(r, correlation, color='#d62728', linewidth=2)
    axes[1, 1].set_xlabel('Separación r')
    axes[1, 1].set_ylabel('ξ(r)')
    axes[1, 1].set_title('Función de Correlación')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig


def plot_gas_dynamics_results(x: np.ndarray,
                             velocity: np.ndarray,
                             pressure: np.ndarray,
                             density: np.ndarray,
                             title: str = "Análisis de Dinámica de Gases") -> plt.Figure:
    """
    Visualización específica para aplicaciones de dinámica de gases
    
    Args:
        x: Array espacial
        velocity: Campo de velocidades
        pressure: Campo de presión
        density: Campo de densidad
        title: Título del gráfico
        
    Returns:
        Figure de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # Gráfico 1: Velocidad
    axes[0, 0].plot(x, velocity, color='#1f77b4', linewidth=2)
    axes[0, 0].set_xlabel('Posición x')
    axes[0, 0].set_ylabel('Velocidad u(x)')
    axes[0, 0].set_title('Campo de Velocidades')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Presión
    axes[0, 1].plot(x, pressure, color='#ff7f0e', linewidth=2)
    axes[0, 1].set_xlabel('Posición x')
    axes[0, 1].set_ylabel('Presión p(x)')
    axes[0, 1].set_title('Campo de Presión')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Densidad
    axes[1, 0].plot(x, density, color='#2ca02c', linewidth=2)
    axes[1, 0].set_xlabel('Posición x')
    axes[1, 0].set_ylabel('Densidad ρ(x)')
    axes[1, 0].set_title('Campo de Densidad')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Número de Mach
    sound_speed = np.sqrt(1.4 * pressure / density)  # Asumiendo γ = 1.4
    mach_number = velocity / sound_speed
    axes[1, 1].plot(x, mach_number, color='#d62728', linewidth=2)
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Mach = 1')
    axes[1, 1].set_xlabel('Posición x')
    axes[1, 1].set_ylabel('Número de Mach')
    axes[1, 1].set_title('Número de Mach')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig


class MFSUDashboard:
    """
    Dashboard interactivo para análisis completo de simulaciones MFSU
    """
    
    def __init__(self):
        self.visualizer = MFSUVisualizer()
        self.current_data = {}
        
    def load_simulation_data(self, data_path: str) -> None:
        """
        Cargar datos de simulación
        
        Args:
            data_path: Ruta a los datos de simulación
        """
        # Implementar carga de datos desde archivos
        pass
    
    def create_comprehensive_report(self,
                                  simulation_results: Dict[str, np.ndarray],
                                  parameters: Dict[str, float],
                                  save_path: str = "mfsu_report.html") -> None:
        """
        Crear reporte completo de simulación
        
        Args:
            simulation_results: Resultados de la simulación
            parameters: Parámetros utilizados
            save_path: Ruta donde guardar el reporte
        """
        # Crear página HTML completa con todos los análisis
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Simulación MFSU</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .section {{
                    background-color: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .parameter-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    margin: 20px 0;
                }}
                .parameter-card {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte de Simulación MFSU</h1>
                <p>Modelo Fractal Estocástico Unificado</p>
            </div>
            
            <div class="section">
                <h2>Parámetros de Simulación</h2>
                <div class="parameter-grid">
        """
        
        # Agregar parámetros
        for param, value in parameters.items():
            html_content += f"""
                    <div class="parameter-card">
                        <h3>{param}</h3>
                        <p>{value}</p>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Resultados de Simulación</h2>
                <p>Los gráficos y análisis se insertarían aquí...</p>
            </div>
            
            <div class="section">
                <h2>Análisis Fractal</h2>
                <p>Análisis de dimensión fractal y exponente de Hurst...</p>
            </div>
            
            <div class="section">
                <h2>Análisis Espectral</h2>
                <p>Espectros de potencia y análisis frecuencial...</p>
            </div>
            
            <div class="section">
                <h2>Conclusiones</h2>
                <p>Resumen de resultados principales...</p>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Reporte guardado en: {save_path}")


# Funciones de utilidad para análisis específicos

def calculate_fractal_dimension(data: np.ndarray, max_scale: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcular dimensión fractal usando el método de conteo de cajas
    
    Args:
        data: Datos para análisis
        max_scale: Escala máxima para análisis
        
    Returns:
        Tuple con escalas y dimensiones
    """
    if max_scale is None:
        max_scale = len(data) // 4
    
    scales = np.logspace(0, np.log10(max_scale), 20).astype(int)
    scales = np.unique(scales)
    
    dimensions = []
    
    for scale in scales:
        # Implementar conteo de cajas simplificado
        n_boxes = len(data) // scale
        if n_boxes > 0:
            boxes = np.array_split(data, n_boxes)
            non_empty_boxes = sum(1 for box in boxes if np.any(np.abs(box) > 1e-10))
            if non_empty_boxes > 0:
                dimension = np.log(non_empty_boxes) / np.log(1.0 / scale)
                dimensions.append(dimension)
            else:
                dimensions.append(0)
        else:
            dimensions.append(0)
    
    return scales[:len(dimensions)], np.array(dimensions)


def calculate_hurst_exponent(data: np.ndarray) -> float:
    """
    Calcular exponente de Hurst usando análisis R/S
    
    Args:
        data: Serie temporal
        
    Returns:
        Exponente de Hurst
    """
    n = len(data)
    lags = np.arange(10, n//4, 10)
    rs_values = []
    
    for lag in lags:
        chunks = [data[i:i+lag] for i in range(0, len(data)-lag, lag)]
        rs_chunk = []
        
        for chunk in chunks:
            if len(chunk) == lag:
                mean_chunk = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean_chunk)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(chunk)
                if S > 0:
                    rs_chunk.append(R/S)
        
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
    
    if len(rs_values) > 1:
        # Ajuste lineal en escala log-log
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        return hurst
    else:
        return 0.5  # Valor por defecto


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo para pruebas
    t = np.linspace(0, 10, 1000)
    x = np.linspace(-10, 10, 200)
    
    # Simular solución MFSU de ejemplo
    X, T = np.meshgrid(x, t)
    psi = np.exp(-0.5 * (X**2 + 0.1*T**2)) * np.exp(1j * (X + 0.1*T**2))
    
    # Crear visualizador
    visualizer = MFSUVisualizer()
    
    # Generar visualizaciones
    fig1 = visualizer.plot_solution_evolution(t, x, psi)
    plt.show()
    
    # Análisis fractal de ejemplo
    data_1d = np.sum(np.abs(psi)**2, axis=1)
    scales, dimensions = calculate_fractal_dimension(data_1d)
    hurst = calculate_hurst_exponent(data_1d)
    
    fig2 = visualizer.plot_fractal_analysis(data_1d, scales, dimensions, hurst)
    plt.show()
    
    # Análisis espectral
    fig3 = visualizer.plot_spectral_analysis(t, data_1d)
    plt.show()
    
    # Crear gráfico interactivo
    interactive_fig = visualizer.create_interactive_plot(t, x, psi)
    # interactive_fig.show()  # Descomenta para mostrar en notebook
    
    print("Visualizaciones generadas exitosamente!")
    print(f"Exponente de Hurst calculado: {hurst:.3f}")
    print(f"Dimensión fractal promedio: {np.mean(dimensions):.3f}")
    
    # Crear dashboard
    dashboard = MFSUDashboard()
    
    # Generar reporte
    simulation_results = {
        'energy': np.sum(np.abs(psi)**2, axis=1),
        'amplitude': np.max(np.abs(psi), axis=1),
        'phase': np.mean(np.angle(psi), axis=1)
    }
    
    parameters = {
        'alpha': 0.5,
        'beta': 0.1,
        'gamma': 0.01,
        'hurst': hurst,
        'dt': t[1] - t[0],
        'dx': x[1] - x[0]
    }
    
    dashboard.create_comprehensive_report(simulation_results, parameters)
    
    print("¡Archivo visualization.py creado exitosamente!")
    print("Incluye:")
    print("- Visualización de evolución temporal")
    print("- Análisis fractal y exponente de Hurst")
    print("- Análisis espectral completo")
    print("- Gráficos interactivos con Plotly")
    print("- Animaciones de evolución temporal")
    print("- Visualizaciones específicas para aplicaciones")
    print("- Dashboard para análisis completo")
    print("- Exportación de resultados en múltiples formatos")
