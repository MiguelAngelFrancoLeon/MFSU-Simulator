#!/usr/bin/env python3
"""
Ventana Principal del Simulador MFSU
====================================

Interfaz gráfica principal para el Modelo Fractal Estocástico Unificado (MFSU).
Implementa la ecuación: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Autor: MFSU Development Team
Versión: 1.0.0
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Importar módulos del simulador
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.mfsu_equation import MFSUEquation
    from core.fractional_operators import FractionalOperator
    from core.stochastic_processes import StochasticProcess
    from simulation.simulator import MFSUSimulator
    from analysis.visualization import MFSUVisualizer
    from utils.parameter_validation import ParameterValidator
    from utils.logger import MFSULogger
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

# Importar paneles de GUI
from parameter_panel import ParameterPanel
from visualization_panel import VisualizationPanel
from control_panel import ControlPanel
from dialogs.settings_dialog import SettingsDialog
from dialogs.export_dialog import ExportDialog
from dialogs.about_dialog import AboutDialog


class MFSUMainWindow:
    """
    Ventana principal del simulador MFSU
    
    Características:
    - Interfaz intuitiva para configurar parámetros MFSU
    - Visualización en tiempo real de simulaciones
    - Control de simulación (inicio/pausa/stop)
    - Exportación de resultados
    - Configuración de aplicaciones específicas
    """
    
    def __init__(self, root):
        self.root = root
        self.setup_main_window()
        self.initialize_components()
        self.create_menu_bar()
        self.create_main_layout()
        self.setup_status_bar()
        self.load_default_config()
        
        # Estado de la simulación
        self.simulation_running = False
        self.current_simulation = None
        self.simulation_thread = None
        
        # Logger
        self.logger = MFSULogger("MFSU_GUI")
        
    def setup_main_window(self):
        """Configuración inicial de la ventana principal"""
        self.root.title("MFSU Simulator - Modelo Fractal Estocástico Unificado v1.0.0")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores personalizados
        style.configure('MFSU.TFrame', background='#f0f0f0')
        style.configure('MFSU.TLabel', background='#f0f0f0', font=('Arial', 9))
        style.configure('MFSU.TButton', font=('Arial', 9))
        
        # Icono de la aplicación (si existe)
        try:
            self.root.iconbitmap(os.path.join('web', 'static', 'assets', 'images', 'mfsu_icon.ico'))
        except:
            pass  # Continuar si no hay icono
            
    def initialize_components(self):
        """Inicializar componentes principales"""
        self.validator = ParameterValidator()
        self.visualizer = MFSUVisualizer()
        
        # Parámetros por defecto de la ecuación MFSU
        self.default_params = {
            'alpha': 0.5,      # Exponente del operador fraccionario
            'beta': 0.1,       # Intensidad del proceso estocástico
            'gamma': 0.01,     # Coeficiente no lineal
            'hurst': 0.7,      # Parámetro de Hurst para ξ_H
            'dt': 0.01,        # Paso temporal
            'dx': 0.1,         # Paso espacial
            'grid_size': 100,  # Tamaño de la grilla
            'max_time': 10.0,  # Tiempo máximo de simulación
            'boundary': 'periodic'  # Condiciones de frontera
        }
        
    def create_menu_bar(self):
        """Crear barra de menú"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Nuevo", command=self.new_simulation, accelerator="Ctrl+N")
        file_menu.add_command(label="Abrir...", command=self.load_simulation, accelerator="Ctrl+O")
        file_menu.add_command(label="Guardar", command=self.save_simulation, accelerator="Ctrl+S")
        file_menu.add_command(label="Guardar como...", command=self.save_simulation_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exportar resultados...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.on_closing)
        
        # Menú Simulación
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulación", menu=sim_menu)
        sim_menu.add_command(label="Iniciar", command=self.start_simulation, accelerator="F5")
        sim_menu.add_command(label="Pausar", command=self.pause_simulation, accelerator="F6")
        sim_menu.add_command(label="Detener", command=self.stop_simulation, accelerator="F7")
        sim_menu.add_separator()
        sim_menu.add_command(label="Reiniciar parámetros", command=self.reset_parameters)
        
        # Menú Análisis
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Análisis", menu=analysis_menu)
        analysis_menu.add_command(label="Análisis fractal", command=self.fractal_analysis)
        analysis_menu.add_command(label="Análisis espectral", command=self.spectral_analysis)
        analysis_menu.add_command(label="Análisis estadístico", command=self.statistical_analysis)
        
        # Menú Aplicaciones
        app_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aplicaciones", menu=app_menu)
        app_menu.add_command(label="Superconductividad", command=lambda: self.load_application("superconductivity"))
        app_menu.add_command(label="Dinámica de gases", command=lambda: self.load_application("gas_dynamics"))
        app_menu.add_command(label="Cosmología", command=lambda: self.load_application("cosmology"))
        
        # Menú Herramientas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Herramientas", menu=tools_menu)
        tools_menu.add_command(label="Configuración...", command=self.open_settings)
        tools_menu.add_command(label="Validar parámetros", command=self.validate_parameters)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Documentación", command=self.open_documentation)
        help_menu.add_command(label="Tutorial", command=self.open_tutorial)
        help_menu.add_separator()
        help_menu.add_command(label="Acerca de...", command=self.show_about)
        
        # Atajos de teclado
        self.root.bind('<Control-n>', lambda e: self.new_simulation())
        self.root.bind('<Control-o>', lambda e: self.load_simulation())
        self.root.bind('<Control-s>', lambda e: self.save_simulation())
        self.root.bind('<F5>', lambda e: self.start_simulation())
        self.root.bind('<F6>', lambda e: self.pause_simulation())
        self.root.bind('<F7>', lambda e: self.stop_simulation())
        
    def create_main_layout(self):
        """Crear diseño principal de la interfaz"""
        # Crear paneles principales
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo - Controles
        left_frame = ttk.Frame(main_paned, style='MFSU.TFrame')
        main_paned.add(left_frame, weight=1)
        
        # Panel derecho - Visualización
        right_frame = ttk.Frame(main_paned, style='MFSU.TFrame')
        main_paned.add(right_frame, weight=2)
        
        # Configurar panel izquierdo
        self.setup_left_panel(left_frame)
        
        # Configurar panel derecho
        self.setup_right_panel(right_frame)
        
    def setup_left_panel(self, parent):
        """Configurar panel izquierdo con controles"""
        # Notebook para organizar pestañas
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de parámetros MFSU
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parámetros MFSU")
        self.parameter_panel = ParameterPanel(params_frame, self.default_params)
        
        # Pestaña de control de simulación
        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="Control")
        self.control_panel = ControlPanel(control_frame, self)
        
        # Pestaña de aplicaciones específicas
        app_frame = ttk.Frame(notebook)
        notebook.add(app_frame, text="Aplicaciones")
        self.setup_applications_panel(app_frame)
        
    def setup_right_panel(self, parent):
        """Configurar panel derecho con visualización"""
        # Panel de visualización
        self.visualization_panel = VisualizationPanel(parent, self)
        
    def setup_applications_panel(self, parent):
        """Configurar panel de aplicaciones específicas"""
        # Selector de aplicación
        app_frame = ttk.LabelFrame(parent, text="Aplicación Específica")
        app_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.app_var = tk.StringVar(value="general")
        apps = [
            ("General", "general"),
            ("Superconductividad", "superconductivity"),
            ("Dinámica de Gases", "gas_dynamics"),
            ("Cosmología", "cosmology")
        ]
        
        for text, value in apps:
            ttk.Radiobutton(app_frame, text=text, variable=self.app_var, 
                          value=value, command=self.on_application_change).pack(anchor=tk.W)
        
        # Parámetros específicos de aplicación
        self.app_params_frame = ttk.LabelFrame(parent, text="Parámetros Específicos")
        self.app_params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def setup_status_bar(self):
        """Configurar barra de estado"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Listo")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        
    def load_default_config(self):
        """Cargar configuración por defecto"""
        try:
            config_path = os.path.join('config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'simulation' in config and 'default_parameters' in config['simulation']:
                        self.default_params.update(config['simulation']['default_parameters'])
                        self.parameter_panel.update_parameters(self.default_params)
        except Exception as e:
            self.logger.warning(f"No se pudo cargar configuración por defecto: {e}")
            
    # Métodos de simulación
    def start_simulation(self):
        """Iniciar simulación"""
        if self.simulation_running:
            messagebox.showwarning("Advertencia", "Ya hay una simulación en ejecución")
            return
            
        try:
            # Validar parámetros
            params = self.parameter_panel.get_parameters()
            if not self.validator.validate_mfsu_parameters(params):
                messagebox.showerror("Error", "Parámetros inválidos")
                return
                
            # Crear simulador
            self.current_simulation = MFSUSimulator(params)
            
            # Iniciar simulación en hilo separado
            self.simulation_running = True
            self.simulation_thread = threading.Thread(target=self._run_simulation_thread)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            self.status_var.set("Simulación iniciada...")
            self.control_panel.update_button_states(running=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar simulación: {e}")
            self.logger.error(f"Error iniciando simulación: {e}")
            
    def _run_simulation_thread(self):
        """Ejecutar simulación en hilo separado"""
        try:
            # Callback para actualizar progreso
            def progress_callback(step, total_steps, data):
                progress = (step / total_steps) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_var.set(f"Simulación: paso {step}/{total_steps}"))
                
                # Actualizar visualización
                if step % 10 == 0:  # Actualizar cada 10 pasos
                    self.root.after(0, lambda: self.visualization_panel.update_plot(data))
                    
            # Ejecutar simulación
            results = self.current_simulation.run(progress_callback=progress_callback)
            
            # Finalizar
            self.root.after(0, lambda: self._simulation_completed(results))
            
        except Exception as e:
            self.root.after(0, lambda: self._simulation_error(e))
            
    def _simulation_completed(self, results):
        """Callback cuando la simulación se completa"""
        self.simulation_running = False
        self.status_var.set("Simulación completada")
        self.progress_var.set(100)
        self.control_panel.update_button_states(running=False)
        
        # Mostrar resultados finales
        self.visualization_panel.show_final_results(results)
        
        messagebox.showinfo("Completado", "Simulación completada exitosamente")
        
    def _simulation_error(self, error):
        """Callback cuando hay error en la simulación"""
        self.simulation_running = False
        self.status_var.set("Error en simulación")
        self.progress_var.set(0)
        self.control_panel.update_button_states(running=False)
        
        messagebox.showerror("Error", f"Error durante la simulación: {error}")
        self.logger.error(f"Error en simulación: {error}")
        
    def pause_simulation(self):
        """Pausar simulación"""
        if self.current_simulation:
            self.current_simulation.pause()
            self.status_var.set("Simulación pausada")
            
    def stop_simulation(self):
        """Detener simulación"""
        if self.current_simulation:
            self.current_simulation.stop()
            self.simulation_running = False
            self.status_var.set("Simulación detenida")
            self.progress_var.set(0)
            self.control_panel.update_button_states(running=False)
            
    # Métodos de archivo
    def new_simulation(self):
        """Nueva simulación"""
        if self.simulation_running:
            if not messagebox.askyesno("Confirmación", 
                                     "Hay una simulación en ejecución. ¿Detenerla?"):
                return
            self.stop_simulation()
            
        self.parameter_panel.reset_to_defaults()
        self.visualization_panel.clear()
        self.status_var.set("Nueva simulación")
        
    def load_simulation(self):
        """Cargar configuración de simulación"""
        filename = filedialog.askopenfilename(
            title="Cargar configuración",
            filetypes=[("Archivos YAML", "*.yaml"), ("Archivos JSON", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    if filename.endswith('.yaml'):
                        config = yaml.safe_load(f)
                    else:
                        import json
                        config = json.load(f)
                        
                self.parameter_panel.load_from_config(config)
                self.status_var.set(f"Configuración cargada: {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando configuración: {e}")
                
    def save_simulation(self):
        """Guardar configuración actual"""
        # Implementar lógica de guardado
        pass
        
    def save_simulation_as(self):
        """Guardar configuración como..."""
        # Implementar lógica de guardado
        pass
        
    def export_results(self):
        """Exportar resultados"""
        if self.current_simulation and hasattr(self.current_simulation, 'results'):
            dialog = ExportDialog(self.root, self.current_simulation.results)
            dialog.show()
        else:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            
    # Métodos de aplicaciones
    def load_application(self, app_name):
        """Cargar aplicación específica"""
        self.app_var.set(app_name)
        self.on_application_change()
        
    def on_application_change(self):
        """Callback cuando cambia la aplicación seleccionada"""
        app_name = self.app_var.get()
        
        # Limpiar panel de parámetros específicos
        for widget in self.app_params_frame.winfo_children():
            widget.destroy()
            
        # Cargar parámetros específicos según la aplicación
        if app_name == "superconductivity":
            self.load_superconductivity_params()
        elif app_name == "gas_dynamics":
            self.load_gas_dynamics_params()
        elif app_name == "cosmology":
            self.load_cosmology_params()
            
    def load_superconductivity_params(self):
        """Cargar parámetros para superconductividad"""
        ttk.Label(self.app_params_frame, text="Temperatura (K):").pack(anchor=tk.W)
        self.temp_var = tk.DoubleVar(value=77.0)
        ttk.Entry(self.app_params_frame, textvariable=self.temp_var).pack(fill=tk.X, padx=5)
        
        ttk.Label(self.app_params_frame, text="Campo Magnético (T):").pack(anchor=tk.W)
        self.field_var = tk.DoubleVar(value=0.1)
        ttk.Entry(self.app_params_frame, textvariable=self.field_var).pack(fill=tk.X, padx=5)
        
    def load_gas_dynamics_params(self):
        """Cargar parámetros para dinámica de gases"""
        ttk.Label(self.app_params_frame, text="Número de Reynolds:").pack(anchor=tk.W)
        self.reynolds_var = tk.DoubleVar(value=1000.0)
        ttk.Entry(self.app_params_frame, textvariable=self.reynolds_var).pack(fill=tk.X, padx=5)
        
        ttk.Label(self.app_params_frame, text="Número de Mach:").pack(anchor=tk.W)
        self.mach_var = tk.DoubleVar(value=0.3)
        ttk.Entry(self.app_params_frame, textvariable=self.mach_var).pack(fill=tk.X, padx=5)
        
    def load_cosmology_params(self):
        """Cargar parámetros para cosmología"""
        ttk.Label(self.app_params_frame, text="Constante de Hubble:").pack(anchor=tk.W)
        self.hubble_var = tk.DoubleVar(value=70.0)
        ttk.Entry(self.app_params_frame, textvariable=self.hubble_var).pack(fill=tk.X, padx=5)
        
        ttk.Label(self.app_params_frame, text="Ω_materia:").pack(anchor=tk.W)
        self.omega_var = tk.DoubleVar(value=0.3)
        ttk.Entry(self.app_params_frame, textvariable=self.omega_var).pack(fill=tk.X, padx=5)
        
    # Métodos de análisis
    def fractal_analysis(self):
        """Realizar análisis fractal"""
        if self.current_simulation and hasattr(self.current_simulation, 'results'):
            # Implementar análisis fractal
            messagebox.showinfo("Info", "Análisis fractal - Por implementar")
        else:
            messagebox.showwarning("Advertencia", "No hay datos para analizar")
            
    def spectral_analysis(self):
        """Realizar análisis espectral"""
        if self.current_simulation and hasattr(self.current_simulation, 'results'):
            # Implementar análisis espectral
            messagebox.showinfo("Info", "Análisis espectral - Por implementar")
        else:
            messagebox.showwarning("Advertencia", "No hay datos para analizar")
            
    def statistical_analysis(self):
        """Realizar análisis estadístico"""
        if self.current_simulation and hasattr(self.current_simulation, 'results'):
            # Implementar análisis estadístico
            messagebox.showinfo("Info", "Análisis estadístico - Por implementar")
        else:
            messagebox.showwarning("Advertencia", "No hay datos para analizar")
            
    # Métodos auxiliares
    def reset_parameters(self):
        """Reiniciar parámetros a valores por defecto"""
        self.parameter_panel.reset_to_defaults()
        self.status_var.set("Parámetros reiniciados")
        
    def validate_parameters(self):
        """Validar parámetros actuales"""
        params = self.parameter_panel.get_parameters()
        if self.validator.validate_mfsu_parameters(params):
            messagebox.showinfo("Validación", "Todos los parámetros son válidos")
        else:
            messagebox.showerror("Error", "Parámetros inválidos detectados")
            
    def open_settings(self):
        """Abrir diálogo de configuración"""
        dialog = SettingsDialog(self.root)
        dialog.show()
        
    def open_documentation(self):
        """Abrir documentación"""
        import webbrowser
        doc_path = os.path.join('docs', 'build', 'html', 'index.html')
        if os.path.exists(doc_path):
            webbrowser.open(f'file://{os.path.abspath(doc_path)}')
        else:
            messagebox.showwarning("Advertencia", "Documentación no encontrada")
            
    def open_tutorial(self):
        """Abrir tutorial"""
        messagebox.showinfo("Tutorial", "Tutorial - Por implementar")
        
    def show_about(self):
        """Mostrar diálogo Acerca de"""
        dialog = AboutDialog(self.root)
        dialog.show()
        
    def on_closing(self):
        """Callback para cerrar la aplicación"""
        if self.simulation_running:
            if messagebox.askyesno("Confirmación", 
                                 "Hay una simulación en ejecución. ¿Salir de todas formas?"):
                self.stop_simulation()
            else:
                return
                
        self.root.quit()
        self.root.destroy()


def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = MFSUMainWindow(root)
    
    # Configurar evento de cierre
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Iniciar bucle principal
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()


if __name__ == "__main__":
    main()
