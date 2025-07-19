"""
Panel de Visualización para el Simulador MFSU
============================================

Este módulo implementa el panel de visualización principal para mostrar
los resultados de las simulaciones del Modelo Fractal Estocástico Unificado (MFSU).

Autor: MFSU Development Team
Versión: 1.0.0
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                            QPushButton, QComboBox, QLabel, QSlider, QCheckBox,
                            QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
                            QSplitter, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QPalette, QColor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import fftpack
from scipy.stats import gaussian_kde


class PlotCanvas(FigureCanvas):
    """Canvas personalizado para matplotlib con funcionalidades específicas para MFSU"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Configuración del estilo
        plt.style.use('seaborn-v0_8')
        self.fig.patch.set_facecolor('white')
        
        # Almacenamiento de datos
        self.current_data = None
        self.time_series = []
        self.animation = None
        
    def clear_plot(self):
        """Limpia todos los subplots"""
        self.fig.clear()
        self.draw()


class VisualizationPanel(QWidget):
    """
    Panel principal de visualización para el simulador MFSU.
    
    Características:
    - Visualización 2D/3D de campos ψ
    - Análisis espectral en tiempo real
    - Gráficos de dimensión fractal
    - Análisis estadístico de procesos estocásticos
    - Animaciones temporales
    """
    
    # Señales
    plot_updated = pyqtSignal()
    animation_started = pyqtSignal()
    animation_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation_data = None
        self.current_time_step = 0
        self.is_animating = False
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation_frame)
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        main_layout = QHBoxLayout(self)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel de control izquierdo
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)
        
        # Panel de visualización central con tabs
        viz_panel = self.create_visualization_tabs()
        main_splitter.addWidget(viz_panel)
        
        # Configurar proporciones del splitter
        main_splitter.setSizes([300, 1000])
        main_layout.addWidget(main_splitter)
        
    def create_control_panel(self):
        """Crea el panel de control lateral"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Grupo de tipo de visualización
        viz_group = QGroupBox("Tipo de Visualización")
        viz_layout = QVBoxLayout(viz_group)
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Campo ψ (2D)", "Campo ψ (3D)", "Evolución Temporal",
            "Análisis Espectral", "Dimensión Fractal", 
            "Estadísticas", "Comparación Multi-parámetros"
        ])
        viz_layout.addWidget(self.plot_type_combo)
        
        # Checkbox para componentes
        self.show_real = QCheckBox("Parte Real")
        self.show_real.setChecked(True)
        self.show_imag = QCheckBox("Parte Imaginaria")
        self.show_magnitude = QCheckBox("Magnitud |ψ|")
        self.show_magnitude.setChecked(True)
        self.show_phase = QCheckBox("Fase arg(ψ)")
        
        viz_layout.addWidget(self.show_real)
        viz_layout.addWidget(self.show_imag)
        viz_layout.addWidget(self.show_magnitude)
        viz_layout.addWidget(self.show_phase)
        
        layout.addWidget(viz_group)
        
        # Grupo de control temporal
        time_group = QGroupBox("Control Temporal")
        time_layout = QVBoxLayout(time_group)
        
        # Slider de tiempo
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_label = QLabel("Tiempo: 0.00")
        
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)
        
        # Controles de animación
        anim_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ Play")
        self.pause_button = QPushButton("⏸ Pause")
        self.stop_button = QPushButton("⏹ Stop")
        
        anim_layout.addWidget(self.play_button)
        anim_layout.addWidget(self.pause_button)
        anim_layout.addWidget(self.stop_button)
        time_layout.addLayout(anim_layout)
        
        # Velocidad de animación
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Velocidad:"))
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(1, 100)
        self.speed_spinbox.setValue(10)
        self.speed_spinbox.setSuffix(" fps")
        speed_layout.addWidget(self.speed_spinbox)
        time_layout.addLayout(speed_layout)
        
        layout.addWidget(time_group)
        
        # Grupo de parámetros de visualización
        params_group = QGroupBox("Parámetros de Visualización")
        params_layout = QGridLayout(params_group)
        
        # Colormap
        params_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "cividis",
            "hot", "cool", "jet", "seismic", "RdBu"
        ])
        params_layout.addWidget(self.colormap_combo, 0, 1)
        
        # Rango de colores
        params_layout.addWidget(QLabel("Rango Z:"), 1, 0)
        self.auto_range = QCheckBox("Auto")
        self.auto_range.setChecked(True)
        params_layout.addWidget(self.auto_range, 1, 1)
        
        self.z_min_spinbox = QDoubleSpinBox()
        self.z_min_spinbox.setRange(-1000, 1000)
        self.z_min_spinbox.setDecimals(3)
        params_layout.addWidget(QLabel("Z min:"), 2, 0)
        params_layout.addWidget(self.z_min_spinbox, 2, 1)
        
        self.z_max_spinbox = QDoubleSpinBox()
        self.z_max_spinbox.setRange(-1000, 1000)
        self.z_max_spinbox.setValue(1.0)
        self.z_max_spinbox.setDecimals(3)
        params_layout.addWidget(QLabel("Z max:"), 3, 0)
        params_layout.addWidget(self.z_max_spinbox, 3, 1)
        
        layout.addWidget(params_group)
        
        # Grupo de análisis
        analysis_group = QGroupBox("Análisis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.calc_spectrum_btn = QPushButton("Calcular Espectro")
        self.calc_fractal_btn = QPushButton("Dimensión Fractal")
        self.calc_stats_btn = QPushButton("Estadísticas")
        self.export_data_btn = QPushButton("Exportar Datos")
        
        analysis_layout.addWidget(self.calc_spectrum_btn)
        analysis_layout.addWidget(self.calc_fractal_btn)
        analysis_layout.addWidget(self.calc_stats_btn)
        analysis_layout.addWidget(self.export_data_btn)
        
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return panel
    
    def create_visualization_tabs(self):
        """Crea el widget de tabs para diferentes visualizaciones"""
        tab_widget = QTabWidget()
        
        # Tab 1: Visualización principal
        main_viz_tab = QWidget()
        main_layout = QVBoxLayout(main_viz_tab)
        
        self.main_canvas = PlotCanvas(width=10, height=8)
        main_layout.addWidget(self.main_canvas)
        
        tab_widget.addTab(main_viz_tab, "Visualización Principal")
        
        # Tab 2: Análisis espectral
        spectral_tab = QWidget()
        spectral_layout = QVBoxLayout(spectral_tab)
        
        self.spectral_canvas = PlotCanvas(width=10, height=6)
        spectral_layout.addWidget(self.spectral_canvas)
        
        tab_widget.addTab(spectral_tab, "Análisis Espectral")
        
        # Tab 3: Estadísticas
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_canvas = PlotCanvas(width=10, height=6)
        stats_layout.addWidget(self.stats_canvas)
        
        tab_widget.addTab(stats_tab, "Estadísticas")
        
        # Tab 4: Comparación
        compare_tab = QWidget()
        compare_layout = QVBoxLayout(compare_tab)
        
        self.compare_canvas = PlotCanvas(width=10, height=6)
        compare_layout.addWidget(self.compare_canvas)
        
        tab_widget.addTab(compare_tab, "Comparación")
        
        return tab_widget
    
    def setup_connections(self):
        """Configura las conexiones de señales"""
        # Controles de visualización
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        self.colormap_combo.currentTextChanged.connect(self.update_plot)
        self.auto_range.toggled.connect(self.toggle_range_controls)
        
        # Controles temporales
        self.time_slider.valueChanged.connect(self.time_slider_changed)
        self.play_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        
        # Checkboxes de componentes
        self.show_real.toggled.connect(self.update_plot)
        self.show_imag.toggled.connect(self.update_plot)
        self.show_magnitude.toggled.connect(self.update_plot)
        self.show_phase.toggled.connect(self.update_plot)
        
        # Botones de análisis
        self.calc_spectrum_btn.clicked.connect(self.calculate_spectrum)
        self.calc_fractal_btn.clicked.connect(self.calculate_fractal_dimension)
        self.calc_stats_btn.clicked.connect(self.calculate_statistics)
        self.export_data_btn.clicked.connect(self.export_data)
    
    def set_simulation_data(self, data):
        """Establece los datos de simulación"""
        self.simulation_data = data
        if data is not None:
            # Actualizar slider de tiempo
            max_time_steps = len(data.get('time_series', [0]))
            self.time_slider.setMaximum(max_time_steps - 1)
            self.time_slider.setValue(0)
            self.current_time_step = 0
            self.update_time_label()
            self.update_plot()
    
    def update_time_label(self):
        """Actualiza la etiqueta de tiempo"""
        if self.simulation_data:
            time_series = self.simulation_data.get('time_series', [])
            if time_series and self.current_time_step < len(time_series):
                current_time = time_series[self.current_time_step]
                self.time_label.setText(f"Tiempo: {current_time:.3f}")
    
    def time_slider_changed(self, value):
        """Maneja el cambio del slider de tiempo"""
        self.current_time_step = value
        self.update_time_label()
        if not self.is_animating:
            self.update_plot()
    
    def toggle_range_controls(self, auto):
        """Habilita/deshabilita controles de rango manual"""
        self.z_min_spinbox.setEnabled(not auto)
        self.z_max_spinbox.setEnabled(not auto)
        if not auto:
            self.update_plot()
    
    def start_animation(self):
        """Inicia la animación temporal"""
        if self.simulation_data:
            self.is_animating = True
            fps = self.speed_spinbox.value()
            self.animation_timer.start(1000 // fps)
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.animation_started.emit()
    
    def pause_animation(self):
        """Pausa la animación"""
        self.animation_timer.stop()
        self.is_animating = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
    
    def stop_animation(self):
        """Detiene la animación y regresa al inicio"""
        self.animation_timer.stop()
        self.is_animating = False
        self.current_time_step = 0
        self.time_slider.setValue(0)
        self.update_time_label()
        self.update_plot()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.animation_stopped.emit()
    
    def update_animation_frame(self):
        """Actualiza un frame de la animación"""
        if self.simulation_data:
            max_steps = len(self.simulation_data.get('time_series', []))
            if self.current_time_step < max_steps - 1:
                self.current_time_step += 1
                self.time_slider.setValue(self.current_time_step)
                self.update_time_label()
                self.update_plot()
            else:
                # Terminar animación o hacer loop
                self.stop_animation()
    
    def update_plot(self):
        """Actualiza el gráfico principal según el tipo seleccionado"""
        if not self.simulation_data:
            return
            
        plot_type = self.plot_type_combo.currentText()
        
        self.main_canvas.clear_plot()
        
        try:
            if plot_type == "Campo ψ (2D)":
                self.plot_field_2d()
            elif plot_type == "Campo ψ (3D)":
                self.plot_field_3d()
            elif plot_type == "Evolución Temporal":
                self.plot_temporal_evolution()
            elif plot_type == "Análisis Espectral":
                self.plot_spectral_analysis()
            elif plot_type == "Dimensión Fractal":
                self.plot_fractal_dimension()
            elif plot_type == "Estadísticas":
                self.plot_statistics()
            elif plot_type == "Comparación Multi-parámetros":
                self.plot_multiparameter_comparison()
                
            self.main_canvas.draw()
            self.plot_updated.emit()
            
        except Exception as e:
            print(f"Error actualizando gráfico: {e}")
    
    def plot_field_2d(self):
        """Grafica el campo ψ en 2D"""
        if 'psi_field' not in self.simulation_data:
            return
            
        psi_data = self.simulation_data['psi_field'][self.current_time_step]
        x_grid = self.simulation_data.get('x_grid', np.arange(psi_data.shape[1]))
        y_grid = self.simulation_data.get('y_grid', np.arange(psi_data.shape[0]))
        
        colormap = self.colormap_combo.currentText()
        
        # Crear subplots según las componentes seleccionadas
        components_to_show = []
        if self.show_real.isChecked():
            components_to_show.append(('Real', np.real(psi_data)))
        if self.show_imag.isChecked():
            components_to_show.append(('Imaginaria', np.imag(psi_data)))
        if self.show_magnitude.isChecked():
            components_to_show.append(('Magnitud', np.abs(psi_data)))
        if self.show_phase.isChecked():
            components_to_show.append(('Fase', np.angle(psi_data)))
        
        n_plots = len(components_to_show)
        if n_plots == 0:
            return
            
        # Organizar subplots
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 1, 2
        elif n_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
            
        for i, (name, data) in enumerate(components_to_show):
            ax = self.main_canvas.fig.add_subplot(rows, cols, i+1)
            
            # Determinar rango de colores
            if self.auto_range.isChecked():
                vmin, vmax = data.min(), data.max()
            else:
                vmin = self.z_min_spinbox.value()
                vmax = self.z_max_spinbox.value()
            
            im = ax.imshow(data, extent=[x_grid.min(), x_grid.max(), 
                                       y_grid.min(), y_grid.max()],
                          cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
            
            ax.set_title(f'{name} de ψ')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Colorbar
            self.main_canvas.fig.colorbar(im, ax=ax, shrink=0.8)
        
        self.main_canvas.fig.suptitle(f'Campo ψ - t = {self.simulation_data["time_series"][self.current_time_step]:.3f}')
        self.main_canvas.fig.tight_layout()
    
    def plot_field_3d(self):
        """Grafica el campo ψ en 3D (superficie)"""
        if 'psi_field' not in self.simulation_data:
            return
            
        psi_data = self.simulation_data['psi_field'][self.current_time_step]
        x_grid = self.simulation_data.get('x_grid', np.arange(psi_data.shape[1]))
        y_grid = self.simulation_data.get('y_grid', np.arange(psi_data.shape[0]))
        
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Solo mostrar magnitud en 3D por simplicidad
        Z = np.abs(psi_data)
        
        ax = self.main_canvas.fig.add_subplot(111, projection='3d')
        
        colormap = self.colormap_combo.currentText()
        surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.9,
                              linewidth=0, antialiased=True)
        
        ax.set_title(f'|ψ| en 3D - t = {self.simulation_data["time_series"][self.current_time_step]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('|ψ|')
        
        self.main_canvas.fig.colorbar(surf, ax=ax, shrink=0.5)
    
    def plot_temporal_evolution(self):
        """Grafica la evolución temporal en puntos específicos"""
        if 'temporal_evolution' not in self.simulation_data:
            return
            
        time_series = self.simulation_data['time_series']
        evolution_data = self.simulation_data['temporal_evolution']
        
        ax = self.main_canvas.fig.add_subplot(111)
        
        # Graficar evolución en varios puntos del espacio
        for i, point_data in enumerate(evolution_data):
            label = f'Punto {i+1}'
            if self.show_magnitude.isChecked():
                ax.plot(time_series, np.abs(point_data), 
                       label=f'|ψ| {label}', linewidth=2)
            if self.show_real.isChecked():
                ax.plot(time_series, np.real(point_data), 
                       label=f'Re(ψ) {label}', linestyle='--')
            if self.show_imag.isChecked():
                ax.plot(time_series, np.imag(point_data), 
                       label=f'Im(ψ) {label}', linestyle=':')
        
        # Línea vertical para tiempo actual
        current_time = time_series[self.current_time_step]
        ax.axvline(current_time, color='red', linestyle='-', alpha=0.7, 
                  label='Tiempo Actual')
        
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('ψ')
        ax.set_title('Evolución Temporal del Campo ψ')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def calculate_spectrum(self):
        """Calcula y muestra el análisis espectral"""
        if not self.simulation_data or 'psi_field' not in self.simulation_data:
            return
            
        psi_data = self.simulation_data['psi_field'][self.current_time_step]
        
        # FFT 2D
        fft_data = fftpack.fft2(psi_data)
        power_spectrum = np.abs(fft_data)**2
        
        # Frecuencias
        dx = self.simulation_data.get('dx', 1.0)
        dy = self.simulation_data.get('dy', 1.0)
        
        freqs_x = fftpack.fftfreq(psi_data.shape[1], dx)
        freqs_y = fftpack.fftfreq(psi_data.shape[0], dy)
        
        self.spectral_canvas.clear_plot()
        
        # Subplot 1: Espectro de potencia 2D
        ax1 = self.spectral_canvas.fig.add_subplot(221)
        im1 = ax1.imshow(np.log10(power_spectrum + 1e-10), 
                        extent=[freqs_x.min(), freqs_x.max(), 
                               freqs_y.min(), freqs_y.max()],
                        cmap='viridis', origin='lower')
        ax1.set_title('Espectro de Potencia 2D (log)')
        ax1.set_xlabel('kx')
        ax1.set_ylabel('ky')
        self.spectral_canvas.fig.colorbar(im1, ax=ax1)
        
        # Subplot 2: Espectro radial promedio
        ax2 = self.spectral_canvas.fig.add_subplot(222)
        
        # Calcular espectro radial
        center_x, center_y = psi_data.shape[1]//2, psi_data.shape[0]//2
        y, x = np.ogrid[:psi_data.shape[0], :psi_data.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        radial_spectrum = []
        radii = []
        for radius in range(1, min(center_x, center_y)):
            mask = (r == radius)
            if np.any(mask):
                radial_spectrum.append(np.mean(power_spectrum[mask]))
                radii.append(radius)
        
        ax2.loglog(radii, radial_spectrum, 'b-', linewidth=2)
        ax2.set_xlabel('k (número de onda radial)')
        ax2.set_ylabel('Potencia')
        ax2.set_title('Espectro Radial')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Fase del campo
        ax3 = self.spectral_canvas.fig.add_subplot(223)
        phase_data = np.angle(psi_data)
        im3 = ax3.imshow(phase_data, cmap='hsv', origin='lower')
        ax3.set_title('Fase del Campo ψ')
        self.spectral_canvas.fig.colorbar(im3, ax=ax3)
        
        # Subplot 4: Histograma de amplitudes
        ax4 = self.spectral_canvas.fig.add_subplot(224)
        amplitudes = np.abs(psi_data).flatten()
        ax4.hist(amplitudes, bins=50, density=True, alpha=0.7, 
                edgecolor='black')
        ax4.set_xlabel('|ψ|')
        ax4.set_ylabel('Densidad de Probabilidad')
        ax4.set_title('Distribución de Amplitudes')
        ax4.grid(True, alpha=0.3)
        
        self.spectral_canvas.fig.tight_layout()
        self.spectral_canvas.draw()
    
    def calculate_fractal_dimension(self):
        """Calcula la dimensión fractal del campo"""
        if not self.simulation_data:
            return
            
        # Implementación simplificada de dimensión fractal
        # En una implementación completa, esto sería más sofisticado
        
        psi_data = self.simulation_data['psi_field'][self.current_time_step]
        magnitude_data = np.abs(psi_data)
        
        # Método box-counting simplificado
        def box_count_dimension(data, max_box_size=None):
            if max_box_size is None:
                max_box_size = min(data.shape) // 4
                
            sizes = []
            counts = []
            
            for box_size in range(2, max_box_size, 2):
                # Contar cajas no vacías
                boxes_x = data.shape[1] // box_size
                boxes_y = data.shape[0] // box_size
                
                count = 0
                threshold = np.mean(magnitude_data)
                
                for i in range(boxes_y):
                    for j in range(boxes_x):
                        box_data = data[i*box_size:(i+1)*box_size, 
                                      j*box_size:(j+1)*box_size]
                        if np.any(box_data > threshold):
                            count += 1
                
                if count > 0:
                    sizes.append(1.0/box_size)
                    counts.append(count)
            
            return sizes, counts
        
        sizes, counts = box_count_dimension(magnitude_data)
        
        self.main_canvas.clear_plot()
        ax = self.main_canvas.fig.add_subplot(111)
        
        if len(sizes) > 2:
            log_sizes = np.log(sizes)
            log_counts = np.log(counts)
            
            # Ajuste lineal
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = coeffs[0]
            
            ax.loglog(sizes, counts, 'bo-', markersize=6, linewidth=2, 
                     label='Datos')
            ax.loglog(sizes, np.exp(coeffs[1]) * np.array(sizes)**coeffs[0], 
                     'r--', linewidth=2, 
                     label=f'Ajuste (D ≈ {fractal_dim:.3f})')
            
            ax.set_xlabel('1/Tamaño de Caja')
            ax.set_ylabel('Número de Cajas')
            ax.set_title(f'Dimensión Fractal - Box Counting\nD ≈ {fractal_dim:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Datos insuficientes para cálculo fractal', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=14)
        
        self.main_canvas.draw()
    
    def calculate_statistics(self):
        """Calcula y muestra estadísticas del campo"""
        if not self.simulation_data:
            return
            
        psi_data = self.simulation_data['psi_field'][self.current_time_step]
        time_series = self.simulation_data['time_series']
        
        # Estadísticas básicas
        magnitude = np.abs(psi_data)
        real_part = np.real(psi_data)
        imag_part = np.imag(psi_data)
        
        stats = {
            'Magnitud': {
                'Media': np.mean(magnitude),
                'Std': np.std(magnitude),
                'Max': np.max(magnitude),
                'Min': np.min(magnitude),
                'Skewness': self._calculate_skewness(magnitude),
                'Kurtosis': self._calculate_kurtosis(magnitude)
            },
            'Parte Real': {
                'Media': np.mean(real_part),
                'Std': np.std(real_part),
                'Max': np.max(real_part),
                'Min': np.min(real_part),
                'Skewness': self._calculate_skewness(real_part),
                'Kurtosis': self._calculate_kurtosis(real_part)
            },
            'Parte Imaginaria': {
                'Media': np.mean(imag_part),
                'Std': np.std(imag_part),
                'Max': np.max(imag_part),
                'Min': np.min(imag_part),
                'Skewness': self._calculate_skewness(imag_part),
                'Kurtosis': self._calculate_kurtosis(imag_part)
            }
        }
        
        self.stats_canvas.clear_plot()
        
        # Crear visualizaciones estadísticas
        fig = self.stats_canvas.fig
        
        # Subplot 1: Histogramas
        ax1 = fig.add_subplot(221)
        ax1.hist(magnitude.flatten(), bins=50, alpha=0.7, label='|ψ|', 
                density=True, color='blue', edgecolor='black')
        ax1.hist(real_part.flatten(), bins=50, alpha=0.7, label='Re(ψ)', 
                density=True, color='red', edgecolor='black')
        ax1.hist(imag_part.flatten(), bins=50, alpha=0.7, label='Im(ψ)', 
                density=True, color='green', edgecolor='black')
        ax1.set_xlabel('Valor')
        ax1.set_ylabel('Densidad')
        ax1.set_title('Distribuciones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Q-Q plot (comparación con distribución normal)
        ax2 = fig.add_subplot(222)
        from scipy import stats as scipy_stats
        
        # Q-Q plot para la magnitud
        sorted_data = np.sort(magnitude.flatten())
        n = len(sorted_data)
        theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
        sample_quantiles = np.quantile(sorted_data, np.linspace(0.01, 0.99, n))
        
        ax2.plot(theoretical_quantiles, sample_quantiles, 'bo', alpha=0.6)
        ax2.plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
                [theoretical_quantiles.min(), theoretical_quantiles.max()], 
                'r--', linewidth=2)
        ax2.set_xlabel('Cuantiles Teóricos (Normal)')
        ax2.set_ylabel('Cuantiles de |ψ|')
        ax2.set_title('Q-Q Plot vs Normal')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Función de autocorrelación espacial
        ax3 = fig.add_subplot(223)
        
        # Calcular autocorrelación 1D (promedio sobre una dirección)
        central_row = magnitude[magnitude.shape[0]//2, :]
        autocorr = np.correlate(central_row, central_row, mode='full')
        autocorr = autocorr[autocorr.size//2:]
        autocorr /= autocorr[0]  # Normalizar
        
        lags = np.arange(len(autocorr))
        ax3.plot(lags[:len(lags)//4], autocorr[:len(lags)//4], 'b-', linewidth=2)
        ax3.set_xlabel('Lag espacial')
        ax3.set_ylabel('Autocorrelación')
        ax3.set_title('Autocorrelación Espacial')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Tabla de estadísticas
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Crear tabla de texto con estadísticas
        table_text = []
        for component, component_stats in stats.items():
            table_text.append([component, '', ''])
            for stat_name, value in component_stats.items():
                if isinstance(value, (int, float)):
                    table_text.append(['', stat_name, f'{value:.4f}'])
                else:
                    table_text.append(['', stat_name, str(value)])
            table_text.append(['', '', ''])  # Separador
        
        # Mostrar tabla
        table = ax4.table(cellText=table_text,
                         colLabels=['Componente', 'Estadística', 'Valor'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Colorear encabezados
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        fig.suptitle(f'Estadísticas del Campo - t = {time_series[self.current_time_step]:.3f}')
        fig.tight_layout()
        self.stats_canvas.draw()
    
    def _calculate_skewness(self, data):
        """Calcula la asimetría (skewness)"""
        data_flat = data.flatten()
        mean = np.mean(data_flat)
        std = np.std(data_flat)
        if std == 0:
            return 0
        normalized = (data_flat - mean) / std
        return np.mean(normalized**3)
    
    def _calculate_kurtosis(self, data):
        """Calcula la curtosis"""
        data_flat = data.flatten()
        mean = np.mean(data_flat)
        std = np.std(data_flat)
        if std == 0:
            return 0
        normalized = (data_flat - mean) / std
        return np.mean(normalized**4) - 3  # Excess kurtosis
    
    def plot_multiparameter_comparison(self):
        """Compara resultados con diferentes parámetros"""
        if not hasattr(self.simulation_data, 'parameter_sweep'):
            # Si no hay datos de barrido de parámetros, mostrar mensaje
            self.main_canvas.clear_plot()
            ax = self.main_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay datos de barrido de parámetros disponibles\n\n'
                              'Ejecute una simulación con múltiples parámetros\n'
                              'para habilitar esta visualización',
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.main_canvas.draw()
            return
        
        # Aquí iría el código para comparación multiparámetro
        # cuando esté disponible en los datos de simulación
        
    def plot_spectral_analysis(self):
        """Análisis espectral avanzado"""
        if not self.simulation_data:
            return
            
        # Reutilizar el método de cálculo de espectro
        self.calculate_spectrum()
    
    def export_data(self):
        """Exporta los datos actuales"""
        if not self.simulation_data:
            return
            
        try:
            # Crear diccionario con datos para exportar
            export_data = {
                'time_step': self.current_time_step,
                'current_time': self.simulation_data['time_series'][self.current_time_step],
                'psi_field_real': np.real(self.simulation_data['psi_field'][self.current_time_step]).tolist(),
                'psi_field_imag': np.imag(self.simulation_data['psi_field'][self.current_time_step]).tolist(),
                'psi_field_magnitude': np.abs(self.simulation_data['psi_field'][self.current_time_step]).tolist(),
                'x_grid': self.simulation_data.get('x_grid', []).tolist(),
                'y_grid': self.simulation_data.get('y_grid', []).tolist(),
                'parameters': self.simulation_data.get('parameters', {})
            }
            
            # En una implementación completa, aquí se abriría un diálogo
            # para seleccionar el archivo y formato de exportación
            import json
            filename = f"mfsu_export_t_{self.current_time_step:04d}.json"
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Datos exportados a: {filename}")
            
        except Exception as e:
            print(f"Error exportando datos: {e}")
    
    def get_current_plot_data(self):
        """Retorna los datos del gráfico actual"""
        if not self.simulation_data:
            return None
            
        return {
            'time_step': self.current_time_step,
            'plot_type': self.plot_type_combo.currentText(),
            'psi_field': self.simulation_data['psi_field'][self.current_time_step],
            'time': self.simulation_data['time_series'][self.current_time_step]
        }
    
    def save_current_plot(self, filename=None):
        """Guarda el gráfico actual"""
        if filename is None:
            plot_type = self.plot_type_combo.currentText().replace(' ', '_')
            filename = f"mfsu_{plot_type}_t{self.current_time_step:04d}.png"
        
        try:
            self.main_canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado como: {filename}")
        except Exception as e:
            print(f"Error guardando gráfico: {e}")
    
    def reset_view(self):
        """Resetea la vista a los valores por defecto"""
        self.current_time_step = 0
        self.time_slider.setValue(0)
        self.plot_type_combo.setCurrentIndex(0)
        self.colormap_combo.setCurrentIndex(0)
        self.auto_range.setChecked(True)
        self.show_real.setChecked(True)
        self.show_imag.setChecked(False)
        self.show_magnitude.setChecked(True)
        self.show_phase.setChecked(False)
        self.stop_animation()
        self.update_plot()


# Clase adicional para manejar visualizaciones interactivas con Plotly
class InteractivePlotWidget(QWidget):
    """Widget para gráficos interactivos usando Plotly"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz para Plotly"""
        layout = QVBoxLayout(self)
        
        # En una implementación completa, aquí se integraría
        # un widget de Plotly para visualizaciones interactivas
        label = QLabel("Visualización Interactiva con Plotly\n(Requiere integración web)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: gray; font-size: 14px;")
        
        layout.addWidget(label)
    
    def create_interactive_plot(self, data, plot_type='surface'):
        """Crea un gráfico interactivo con Plotly"""
        # Aquí se implementaría la lógica para crear gráficos
        # interactivos con Plotly
        pass


if __name__ == "__main__":
    # Código de prueba
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Crear datos de prueba
    test_data = {
        'time_series': np.linspace(0, 10, 100),
        'psi_field': [],
        'x_grid': np.linspace(-5, 5, 50),
        'y_grid': np.linspace(-5, 5, 50),
        'parameters': {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.01}
    }
    
    # Generar campo de prueba
    X, Y = np.meshgrid(test_data['x_grid'], test_data['y_grid'])
    for t in test_data['time_series']:
        # Función gaussiana compleja que evoluciona en el tiempo
        psi = np.exp(-(X**2 + Y**2)/(2 + 0.1*t)) * np.exp(1j*(X*Y*t*0.1))
        psi += 0.1 * (np.random.random(X.shape) + 1j*np.random.random(X.shape))
        test_data['psi_field'].append(psi)
    
    # Crear y mostrar el panel de visualización
    viz_panel = VisualizationPanel()
    viz_panel.set_simulation_data(test_data)
    viz_panel.show()
    
    sys.exit(app.exec_())
