"""
Diálogo de configuración para el simulador MFSU
Permite ajustar parámetros de la ecuación, configuración numérica y aplicaciones específicas
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yaml
import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SettingsDialog:
    """
    Diálogo de configuración avanzada para el simulador MFSU
    Maneja parámetros de la ecuación: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
    """
    
    def __init__(self, parent, config_manager):
        self.parent = parent
        self.config_manager = config_manager
        self.current_config = config_manager.get_config().copy()
        self.result = None
        
        # Crear ventana principal
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configuración del Simulador MFSU")
        self.dialog.geometry("800x700")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centrar la ventana
        self._center_window()
        
        # Crear la interfaz
        self._create_widgets()
        self._load_current_config()
        
        # Configurar evento de cierre
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _center_window(self):
        """Centra la ventana en la pantalla"""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"800x700+{x}+{y}")
    
    def _create_widgets(self):
        """Crea todos los widgets del diálogo"""
        # Frame principal con scroll
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas y scrollbar para hacer scrolleable
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Crear pestañas
        self._create_equation_tab()
        self._create_numerical_tab()
        self._create_applications_tab()
        self._create_visualization_tab()
        self._create_advanced_tab()
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Botones inferiores
        self._create_buttons()
        
        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_equation_tab(self):
        """Crea la pestaña de parámetros de la ecuación MFSU"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Ecuación MFSU")
        
        # Frame para parámetros principales
        main_params = ttk.LabelFrame(tab, text="Parámetros Principales de la Ecuación")
        main_params.pack(fill=tk.X, padx=5, pady=5)
        
        # α (alpha) - Coeficiente del operador fraccionario
        ttk.Label(main_params, text="α (Coeficiente fraccionario):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.alpha_var = tk.DoubleVar()
        self.alpha_spin = ttk.Spinbox(main_params, from_=0.1, to=2.0, increment=0.1, 
                                     textvariable=self.alpha_var, width=15)
        self.alpha_spin.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(main_params, text="Controla la difusión fraccionaria").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # β (beta) - Coeficiente del ruido estocástico
        ttk.Label(main_params, text="β (Coeficiente estocástico):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.beta_var = tk.DoubleVar()
        self.beta_spin = ttk.Spinbox(main_params, from_=0.0, to=1.0, increment=0.01, 
                                    textvariable=self.beta_var, width=15)
        self.beta_spin.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(main_params, text="Intensidad del ruido ξ_H(x,t)").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # γ (gamma) - Coeficiente no lineal
        ttk.Label(main_params, text="γ (Coeficiente no lineal):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.gamma_var = tk.DoubleVar()
        self.gamma_spin = ttk.Spinbox(main_params, from_=0.0, to=1.0, increment=0.001, 
                                     textvariable=self.gamma_var, width=15)
        self.gamma_spin.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(main_params, text="Término cúbico no lineal").grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # H (Hurst) - Exponente de Hurst para el ruido fractal
        ttk.Label(main_params, text="H (Exponente de Hurst):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.hurst_var = tk.DoubleVar()
        self.hurst_spin = ttk.Spinbox(main_params, from_=0.1, to=0.9, increment=0.05, 
                                     textvariable=self.hurst_var, width=15)
        self.hurst_spin.grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(main_params, text="Correlación del ruido fractal").grid(row=3, column=2, sticky=tk.W, padx=5)
        
        # Frame para función de fuerza externa
        force_frame = ttk.LabelFrame(tab, text="Función de Fuerza Externa f(x,t)")
        force_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(force_frame, text="Tipo de fuerza:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.force_type_var = tk.StringVar()
        force_combo = ttk.Combobox(force_frame, textvariable=self.force_type_var, 
                                  values=["None", "Gaussian", "Sinusoidal", "Custom"], width=15)
        force_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Parámetros de la fuerza
        ttk.Label(force_frame, text="Amplitud:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.force_amplitude_var = tk.DoubleVar()
        ttk.Spinbox(force_frame, from_=0.0, to=10.0, increment=0.1, 
                   textvariable=self.force_amplitude_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(force_frame, text="Frecuencia:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.force_frequency_var = tk.DoubleVar()
        ttk.Spinbox(force_frame, from_=0.1, to=100.0, increment=0.1, 
                   textvariable=self.force_frequency_var, width=15).grid(row=2, column=1, padx=5, pady=2)
    
    def _create_numerical_tab(self):
        """Crea la pestaña de configuración numérica"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Configuración Numérica")
        
        # Frame para parámetros de discretización
        discretization_frame = ttk.LabelFrame(tab, text="Discretización")
        discretization_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Paso temporal
        ttk.Label(discretization_frame, text="Paso temporal (dt):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.dt_var = tk.DoubleVar()
        ttk.Spinbox(discretization_frame, from_=0.0001, to=0.1, increment=0.0001, 
                   textvariable=self.dt_var, width=15, format="%.4f").grid(row=0, column=1, padx=5, pady=2)
        
        # Paso espacial
        ttk.Label(discretization_frame, text="Paso espacial (dx):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.dx_var = tk.DoubleVar()
        ttk.Spinbox(discretization_frame, from_=0.01, to=1.0, increment=0.01, 
                   textvariable=self.dx_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        # Tamaño de grilla
        ttk.Label(discretization_frame, text="Puntos de grilla:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.grid_size_var = tk.IntVar()
        ttk.Spinbox(discretization_frame, from_=50, to=1000, increment=10, 
                   textvariable=self.grid_size_var, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        # Tiempo máximo
        ttk.Label(discretization_frame, text="Tiempo máximo:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_time_var = tk.DoubleVar()
        ttk.Spinbox(discretization_frame, from_=1.0, to=100.0, increment=1.0, 
                   textvariable=self.max_time_var, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        # Frame para métodos numéricos
        methods_frame = ttk.LabelFrame(tab, text="Métodos Numéricos")
        methods_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(methods_frame, text="Método de integración:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.integration_method_var = tk.StringVar()
        method_combo = ttk.Combobox(methods_frame, textvariable=self.integration_method_var,
                                   values=["Euler", "Runge-Kutta-4", "Adams-Bashforth", "Implicit"], width=15)
        method_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(methods_frame, text="Tolerancia:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tolerance_var = tk.DoubleVar()
        ttk.Spinbox(methods_frame, from_=1e-8, to=1e-3, increment=1e-8, 
                   textvariable=self.tolerance_var, width=15, format="%.2e").grid(row=1, column=1, padx=5, pady=2)
        
        # Frame para condiciones de frontera
        boundary_frame = ttk.LabelFrame(tab, text="Condiciones de Frontera")
        boundary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(boundary_frame, text="Tipo de frontera:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.boundary_type_var = tk.StringVar()
        boundary_combo = ttk.Combobox(boundary_frame, textvariable=self.boundary_type_var,
                                     values=["Periodic", "Dirichlet", "Neumann", "Absorbing"], width=15)
        boundary_combo.grid(row=0, column=1, padx=5, pady=2)
    
    def _create_applications_tab(self):
        """Crea la pestaña de aplicaciones específicas"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Aplicaciones")
        
        # Superconductividad
        supercon_frame = ttk.LabelFrame(tab, text="Superconductividad")
        supercon_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(supercon_frame, text="Rango de temperatura (K):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        temp_frame = ttk.Frame(supercon_frame)
        temp_frame.grid(row=0, column=1, padx=5, pady=2)
        
        self.temp_min_var = tk.DoubleVar()
        self.temp_max_var = tk.DoubleVar()
        ttk.Spinbox(temp_frame, from_=1, to=400, textvariable=self.temp_min_var, width=8).pack(side=tk.LEFT)
        ttk.Label(temp_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Spinbox(temp_frame, from_=1, to=400, textvariable=self.temp_max_var, width=8).pack(side=tk.LEFT)
        
        # Dinámica de gases
        gas_frame = ttk.LabelFrame(tab, text="Dinámica de Gases")
        gas_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(gas_frame, text="Número de Reynolds:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.reynolds_var = tk.DoubleVar()
        ttk.Spinbox(gas_frame, from_=10, to=10000, textvariable=self.reynolds_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(gas_frame, text="Número de Mach:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.mach_var = tk.DoubleVar()
        ttk.Spinbox(gas_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.mach_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        # Cosmología
        cosmo_frame = ttk.LabelFrame(tab, text="Aplicaciones Cosmológicas")
        cosmo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cosmo_frame, text="Constante de Hubble:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.hubble_var = tk.DoubleVar()
        ttk.Spinbox(cosmo_frame, from_=50, to=100, textvariable=self.hubble_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(cosmo_frame, text="Omega materia:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.omega_matter_var = tk.DoubleVar()
        ttk.Spinbox(cosmo_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.omega_matter_var, width=15).grid(row=1, column=1, padx=5, pady=2)
    
    def _create_visualization_tab(self):
        """Crea la pestaña de configuración de visualización"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Visualización")
        
        # Frame para parámetros de gráficos
        plot_frame = ttk.LabelFrame(tab, text="Configuración de Gráficos")
        plot_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(plot_frame, text="Intervalo de actualización (ms):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.update_interval_var = tk.IntVar()
        ttk.Spinbox(plot_frame, from_=50, to=5000, increment=50, textvariable=self.update_interval_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(plot_frame, text="Resolución de exportación (DPI):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.dpi_var = tk.IntVar()
        ttk.Spinbox(plot_frame, from_=72, to=600, increment=72, textvariable=self.dpi_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        # Checkboxes para opciones de visualización
        self.show_grid_var = tk.BooleanVar()
        ttk.Checkbutton(plot_frame, text="Mostrar grilla", variable=self.show_grid_var).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.show_colorbar_var = tk.BooleanVar()
        ttk.Checkbutton(plot_frame, text="Mostrar barra de colores", variable=self.show_colorbar_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.animate_var = tk.BooleanVar()
        ttk.Checkbutton(plot_frame, text="Animación en tiempo real", variable=self.animate_var).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
    
    def _create_advanced_tab(self):
        """Crea la pestaña de configuraciones avanzadas"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Avanzado")
        
        # Frame para configuración de rendimiento
        performance_frame = ttk.LabelFrame(tab, text="Rendimiento")
        performance_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(performance_frame, text="Número de hilos:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_threads_var = tk.IntVar()
        ttk.Spinbox(performance_frame, from_=1, to=16, textvariable=self.num_threads_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        self.use_gpu_var = tk.BooleanVar()
        ttk.Checkbutton(performance_frame, text="Usar GPU (si está disponible)", variable=self.use_gpu_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Frame para logging
        logging_frame = ttk.LabelFrame(tab, text="Logging")
        logging_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(logging_frame, text="Nivel de log:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.log_level_var = tk.StringVar()
        log_combo = ttk.Combobox(logging_frame, textvariable=self.log_level_var,
                               values=["DEBUG", "INFO", "WARNING", "ERROR"], width=15)
        log_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Frame para importar/exportar configuración
        io_frame = ttk.LabelFrame(tab, text="Importar/Exportar Configuración")
        io_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(io_frame, text="Importar desde archivo", command=self._import_config).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(io_frame, text="Exportar a archivo", command=self._export_config).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(io_frame, text="Restaurar valores por defecto", command=self._reset_defaults).grid(row=0, column=2, padx=5, pady=5)
    
    def _create_buttons(self):
        """Crea los botones de acción del diálogo"""
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Aceptar", command=self._on_accept).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Aplicar", command=self._on_apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Validar", command=self._validate_config).pack(side=tk.LEFT, padx=5)
    
    def _load_current_config(self):
        """Carga la configuración actual en los widgets"""
        try:
            # Parámetros principales
            self.alpha_var.set(self.current_config.get('simulation', {}).get('default_parameters', {}).get('alpha', 0.5))
            self.beta_var.set(self.current_config.get('simulation', {}).get('default_parameters', {}).get('beta', 0.1))
            self.gamma_var.set(self.current_config.get('simulation', {}).get('default_parameters', {}).get('gamma', 0.01))
            self.hurst_var.set(self.current_config.get('simulation', {}).get('default_parameters', {}).get('hurst', 0.7))
            
            # Parámetros numéricos
            numerical = self.current_config.get('simulation', {}).get('numerical', {})
            self.dt_var.set(numerical.get('dt', 0.01))
            self.dx_var.set(numerical.get('dx', 0.1))
            self.grid_size_var.set(numerical.get('grid_size', 100))
            self.max_time_var.set(numerical.get('max_time', 10.0))
            
            # Métodos numéricos
            self.integration_method_var.set(numerical.get('integration_method', 'Runge-Kutta-4'))
            self.tolerance_var.set(numerical.get('tolerance', 1e-6))
            self.boundary_type_var.set(numerical.get('boundary_type', 'Periodic'))
            
            # Aplicaciones
            apps = self.current_config.get('applications', {})
            supercon = apps.get('superconductivity', {})
            temp_range = supercon.get('temperature_range', [1, 300])
            self.temp_min_var.set(temp_range[0])
            self.temp_max_var.set(temp_range[1])
            
            gas = apps.get('gas_dynamics', {})
            self.reynolds_var.set(gas.get('reynolds_number', 1000))
            self.mach_var.set(gas.get('mach_number', 0.3))
            
            cosmo = apps.get('cosmology', {})
            self.hubble_var.set(cosmo.get('hubble_constant', 70))
            self.omega_matter_var.set(cosmo.get('omega_matter', 0.3))
            
            # Fuerza externa
            force = self.current_config.get('simulation', {}).get('force', {})
            self.force_type_var.set(force.get('type', 'None'))
            self.force_amplitude_var.set(force.get('amplitude', 1.0))
            self.force_frequency_var.set(force.get('frequency', 1.0))
            
            # Visualización
            viz = self.current_config.get('visualization', {})
            self.update_interval_var.set(viz.get('update_interval', 100))
            self.dpi_var.set(viz.get('dpi', 150))
            self.show_grid_var.set(viz.get('show_grid', True))
            self.show_colorbar_var.set(viz.get('show_colorbar', True))
            self.animate_var.set(viz.get('animate', False))
            
            # Configuraciones avanzadas
            advanced = self.current_config.get('advanced', {})
            self.num_threads_var.set(advanced.get('num_threads', 4))
            self.use_gpu_var.set(advanced.get('use_gpu', False))
            self.log_level_var.set(advanced.get('log_level', 'INFO'))
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            messagebox.showerror("Error", f"Error al cargar la configuración: {e}")
    
    def _save_config(self):
        """Guarda la configuración actual desde los widgets"""
        try:
            # Actualizar configuración
            if 'simulation' not in self.current_config:
                self.current_config['simulation'] = {}
            if 'default_parameters' not in self.current_config['simulation']:
                self.current_config['simulation']['default_parameters'] = {}
            if 'numerical' not in self.current_config['simulation']:
                self.current_config['simulation']['numerical'] = {}
            
            # Parámetros principales
            params = self.current_config['simulation']['default_parameters']
            params['alpha'] = self.alpha_var.get()
            params['beta'] = self.beta_var.get()
            params['gamma'] = self.gamma_var.get()
            params['hurst'] = self.hurst_var.get()
            
            # Parámetros numéricos
            numerical = self.current_config['simulation']['numerical']
            numerical['dt'] = self.dt_var.get()
            numerical['dx'] = self.dx_var.get()
            numerical['grid_size'] = self.grid_size_var.get()
            numerical['max_time'] = self.max_time_var.get()
            numerical['integration_method'] = self.integration_method_var.get()
            numerical['tolerance'] = self.tolerance_var.get()
            numerical['boundary_type'] = self.boundary_type_var.get()
            
            # Fuerza externa
            if 'force' not in self.current_config['simulation']:
                self.current_config['simulation']['force'] = {}
            force = self.current_config['simulation']['force']
            force['type'] = self.force_type_var.get()
            force['amplitude'] = self.force_amplitude_var.get()
            force['frequency'] = self.force_frequency_var.get()
            
            # Aplicaciones
            if 'applications' not in self.current_config:
                self.current_config['applications'] = {}
            apps = self.current_config['applications']
            
            if 'superconductivity' not in apps:
                apps['superconductivity'] = {}
            apps['superconductivity']['temperature_range'] = [self.temp_min_var.get(), self.temp_max_var.get()]
            
            if 'gas_dynamics' not in apps:
                apps['gas_dynamics'] = {}
            apps['gas_dynamics']['reynolds_number'] = self.reynolds_var.get()
            apps['gas_dynamics']['mach_number'] = self.mach_var.get()
            
            if 'cosmology' not in apps:
                apps['cosmology'] = {}
            apps['cosmology']['hubble_constant'] = self.hubble_var.get()
            apps['cosmology']['omega_matter'] = self.omega_matter_var.get()
            
            # Visualización
            if 'visualization' not in self.current_config:
                self.current_config['visualization'] = {}
            viz = self.current_config['visualization']
            viz['update_interval'] = self.update_interval_var.get()
            viz['dpi'] = self.dpi_var.get()
            viz['show_grid'] = self.show_grid_var.get()
            viz['show_colorbar'] = self.show_colorbar_var.get()
            viz['animate'] = self.animate_var.get()
            
            # Configuraciones avanzadas
            if 'advanced' not in self.current_config:
                self.current_config['advanced'] = {}
            advanced = self.current_config['advanced']
            advanced['num_threads'] = self.num_threads_var.get()
            advanced['use_gpu'] = self.use_gpu_var.get()
            advanced['log_level'] = self.log_level_var.get()
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            raise
    
    def _validate_config(self):
        """Valida la configuración actual"""
        try:
            errors = []
            warnings = []
            
            # Validar parámetros principales
            alpha = self.alpha_var.get()
            if alpha <= 0 or alpha > 2:
                errors.append("α debe estar entre 0 y 2")
            
            beta = self.beta_var.get()
            if beta < 0 or beta > 1:
                errors.append("β debe estar entre 0 y 1")
            
            gamma = self.gamma_var.get()
            if gamma < 0:
                errors.append("γ debe ser no negativo")
            
            hurst = self.hurst_var.get()
            if hurst <= 0 or hurst >= 1:
                errors.append("H debe estar entre 0 y 1")
            
            # Validar parámetros numéricos
            dt = self.dt_var.get()
            dx = self.dx_var.get()
            if dt <= 0:
                errors.append("dt debe ser positivo")
            if dx <= 0:
                errors.append("dx debe ser positivo")
            
            # Validar estabilidad CFL
            cfl = dt / (dx**alpha)
            if cfl > 0.1:
                warnings.append(f"Condición CFL posiblemente inestable: {cfl:.4f}")
            
            # Validar aplicaciones
            temp_min = self.temp_min_var.get()
            temp_max = self.temp_max_var.get()
            if temp_min >= temp_max:
                errors.append("Temperatura mínima debe ser menor que la máxima")
            
            reynolds = self.reynolds_var.get()
            if reynolds <= 0:
                errors.append("Número de Reynolds debe ser positivo")
            
            # Mostrar resultados
            if errors:
                messagebox.showerror("Errores de Validación", "\n".join(errors))
                return False
            elif warnings:
                result = messagebox.askquestion("Advertencias", 
                                              f"Se encontraron advertencias:\n" + "\n".join(warnings) + 
                                              "\n\n¿Continuar de todas formas?")
                return result == 'yes'
            else:
                messagebox.showinfo("Validación", "Configuración válida")
                return True
                
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la validación: {e}")
            return False
    
    def _import_config(self):
        """Importa configuración desde un archivo"""
        try:
            filename = filedialog.askopenfilename(
                title="Importar Configuración",
                filetypes=[("YAML files", "*.yaml *.yml"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    if filename.lower().endswith(('.yaml', '.yml')):
                        imported_config = yaml.safe_load(f)
                    else:
                        imported_config = json.load(f)
                
                self.current_config.update(imported_config)
                self._load_current_config()
                messagebox.showinfo("Éxito", "Configuración importada correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al importar configuración: {e}")
    
    def _export_config(self):
        """Exporta la configuración actual a un archivo"""
        try:
            self._save_config()
            
            filename = filedialog.asksaveasfilename(
                title="Exportar Configuración",
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    if filename.lower().endswith('.json'):
                        json.dump(self.current_config, f, indent=2, ensure_ascii=False)
                    else:
                        yaml.dump(self.current_config, f, default_flow_style=False, 
                                allow_unicode=True, indent=2)
                
                messagebox.showinfo("Éxito", "Configuración exportada correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar configuración: {e}")
    
    def _reset_defaults(self):
        """Restaura los valores por defecto"""
        result = messagebox.askquestion("Confirmar", 
                                       "¿Está seguro de que desea restaurar los valores por defecto?")
        if result == 'yes':
            try:
                # Configuración por defecto
                default_config = {
                    'simulation': {
                        'default_parameters': {
                            'alpha': 0.5,
                            'beta': 0.1,
                            'gamma': 0.01,
                            'hurst': 0.7
                        },
                        'numerical': {
                            'dt': 0.01,
                            'dx': 0.1,
                            'grid_size': 100,
                            'max_time': 10.0,
                            'integration_method': 'Runge-Kutta-4',
                            'tolerance': 1e-6,
                            'boundary_type': 'Periodic'
                        },
                        'force': {
                            'type': 'None',
                            'amplitude': 1.0,
                            'frequency': 1.0
                        }
                    },
                    'applications': {
                        'superconductivity': {
                            'temperature_range': [1, 300]
                        },
                        'gas_dynamics': {
                            'reynolds_number': 1000,
                            'mach_number': 0.3
                        },
                        'cosmology': {
                            'hubble_constant': 70,
                            'omega_matter': 0.3
                        }
                    },
                    'visualization': {
                        'update_interval': 100,
                        'dpi': 150,
                        'show_grid': True,
                        'show_colorbar': True,
                        'animate': False
                    },
                    'advanced': {
                        'num_threads': 4,
                        'use_gpu': False,
                        'log_level': 'INFO'
                    }
                }
                
                self.current_config = default_config
                self._load_current_config()
                messagebox.showinfo("Éxito", "Valores por defecto restaurados")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al restaurar valores por defecto: {e}")
    
    def _on_accept(self):
        """Maneja el botón Aceptar"""
        if self._validate_config():
            try:
                self._save_config()
                self.result = self.current_config
                self.dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar configuración: {e}")
    
    def _on_apply(self):
        """Maneja el botón Aplicar"""
        if self._validate_config():
            try:
                self._save_config()
                self.config_manager.update_config(self.current_config)
                messagebox.showinfo("Éxito", "Configuración aplicada correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error al aplicar configuración: {e}")
    
    def _on_cancel(self):
        """Maneja el botón Cancelar"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Muestra el diálogo y retorna la configuración si se acepta"""
        self.dialog.wait_window()
        return self.result


# Clase auxiliar para manejar la configuración
class ConfigManager:
    """Maneja la carga y guardado de configuraciones"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_default_config()
    
    def _load_default_config(self):
        """Carga la configuración por defecto o desde archivo"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Error cargando configuración desde {self.config_path}: {e}")
        
        # Configuración por defecto si no se puede cargar
        return {
            'simulation': {
                'default_parameters': {
                    'alpha': 0.5,
                    'beta': 0.1,
                    'gamma': 0.01,
                    'hurst': 0.7
                },
                'numerical': {
                    'dt': 0.01,
                    'dx': 0.1,
                    'grid_size': 100,
                    'max_time': 10.0,
                    'integration_method': 'Runge-Kutta-4',
                    'tolerance': 1e-6,
                    'boundary_type': 'Periodic'
                }
            }
        }
    
    def get_config(self):
        """Retorna la configuración actual"""
        return self.config.copy()
    
    def update_config(self, new_config):
        """Actualiza la configuración"""
        self.config = new_config.copy()
        self.save_config()
    
    def save_config(self):
        """Guarda la configuración actual"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            raise


# Función de prueba
def main():
    """Función de prueba para el diálogo"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    
    config_manager = ConfigManager()
    dialog = SettingsDialog(root, config_manager)
    result = dialog.show()
    
    if result:
        print("Configuración aceptada:")
        print(yaml.dump(result, default_flow_style=False))
    else:
        print("Configuración cancelada")
    
    root.destroy()


if __name__ == "__main__":
    main()
