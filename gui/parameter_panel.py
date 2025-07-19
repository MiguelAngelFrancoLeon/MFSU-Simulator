"""
Panel de parámetros para el simulador MFSU
Permite configurar todos los parámetros de la ecuación:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
"""

import tkinter as tk
from tkinter import ttk
import yaml
import json
from typing import Dict, Any, Callable, Optional
from pathlib import Path


class ParameterPanel:
    """Panel de parámetros para configurar la simulación MFSU"""
    
    def __init__(self, parent, config_path: Optional[str] = None, 
                 callback: Optional[Callable] = None):
        """
        Inicializa el panel de parámetros
        
        Args:
            parent: Widget padre
            config_path: Ruta al archivo de configuración
            callback: Función callback para cambios de parámetros
        """
        self.parent = parent
        self.callback = callback
        self.config_path = config_path or "config.yaml"
        
        # Valores por defecto de los parámetros MFSU
        self.default_params = {
            # Parámetros principales de la ecuación MFSU
            'alpha': 0.5,           # Coeficiente del operador fractal
            'beta': 0.1,            # Intensidad del ruido estocástico
            'gamma': 0.01,          # Coeficiente no lineal
            'hurst': 0.7,           # Parámetro de Hurst para ξ_H
            
            # Parámetros numéricos
            'dt': 0.01,             # Paso temporal
            'dx': 0.1,              # Paso espacial
            'grid_size': 100,       # Tamaño de la grilla
            'max_time': 10.0,       # Tiempo máximo de simulación
            
            # Parámetros del término de forzado f(x,t)
            'forcing_amplitude': 0.0,   # Amplitud del forzado
            'forcing_frequency': 1.0,   # Frecuencia del forzado
            'forcing_type': 'none',     # Tipo de forzado
            
            # Condiciones iniciales
            'initial_condition': 'gaussian',  # Tipo de condición inicial
            'ic_amplitude': 1.0,             # Amplitud inicial
            'ic_width': 1.0,                 # Ancho inicial
            
            # Condiciones de frontera
            'boundary_condition': 'periodic',  # Tipo de frontera
            
            # Aplicación específica
            'application': 'general'  # Aplicación (general, superconductivity, gas_dynamics, cosmology)
        }
        
        self.current_params = self.default_params.copy()
        self.param_vars = {}
        
        self._create_panel()
        self._load_config()
    
    def _create_panel(self):
        """Crea el panel de parámetros con pestañas organizadas"""
        
        # Frame principal
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook para organizar parámetros en pestañas
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Crear pestañas
        self._create_equation_tab()
        self._create_numerical_tab()
        self._create_forcing_tab()
        self._create_initial_conditions_tab()
        self._create_boundary_tab()
        self._create_application_tab()
        
        # Botones de control
        self._create_control_buttons()
    
    def _create_equation_tab(self):
        """Pestaña de parámetros principales de la ecuación MFSU"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Ecuación MFSU")
        
        # Frame con scroll
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parámetro α (coeficiente fractal)
        self._create_parameter_control(
            scrollable_frame, "alpha", "α (Coeficiente Fractal)", 
            0.1, 2.0, 0.01, 
            "Controla la intensidad del operador fractal (-Δ)^(α/2)"
        )
        
        # Parámetro β (intensidad estocástica)
        self._create_parameter_control(
            scrollable_frame, "beta", "β (Intensidad Estocástica)", 
            0.0, 1.0, 0.001,
            "Controla la intensidad del ruido fractal ξ_H(x,t)"
        )
        
        # Parámetro γ (término no lineal)
        self._create_parameter_control(
            scrollable_frame, "gamma", "γ (No Linealidad)", 
            0.0, 0.1, 0.0001,
            "Coeficiente del término no lineal -γψ³"
        )
        
        # Parámetro de Hurst
        self._create_parameter_control(
            scrollable_frame, "hurst", "H (Parámetro de Hurst)", 
            0.1, 0.9, 0.01,
            "Controla las propiedades de correlación del ruido fractal"
        )
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_numerical_tab(self):
        """Pestaña de parámetros numéricos"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Numérico")
        
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Paso temporal
        self._create_parameter_control(
            scrollable_frame, "dt", "Δt (Paso Temporal)", 
            0.001, 0.1, 0.0001,
            "Resolución temporal de la simulación"
        )
        
        # Paso espacial
        self._create_parameter_control(
            scrollable_frame, "dx", "Δx (Paso Espacial)", 
            0.01, 1.0, 0.001,
            "Resolución espacial de la grilla"
        )
        
        # Tamaño de grilla
        self._create_parameter_control(
            scrollable_frame, "grid_size", "Tamaño de Grilla", 
            50, 1000, 1,
            "Número de puntos en la grilla espacial"
        )
        
        # Tiempo máximo
        self._create_parameter_control(
            scrollable_frame, "max_time", "Tiempo Máximo", 
            1.0, 100.0, 0.1,
            "Duración total de la simulación"
        )
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_forcing_tab(self):
        """Pestaña de parámetros del término de forzado f(x,t)"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Forzado f(x,t)")
        
        canvas = tk.Canvas(tab)
        scrollable_frame = ttk.Frame(canvas)
        
        # Tipo de forzado
        frame = ttk.LabelFrame(scrollable_frame, text="Tipo de Forzado", padding=5)
        frame.pack(fill=tk.X, pady=5)
        
        self.param_vars["forcing_type"] = tk.StringVar(value=self.current_params["forcing_type"])
        
        forcing_types = [
            ("Ninguno", "none"),
            ("Sinusoidal", "sinusoidal"),
            ("Gaussiano", "gaussian"),
            ("Ruido Blanco", "white_noise"),
            ("Pulso", "pulse")
        ]
        
        for text, value in forcing_types:
            rb = ttk.Radiobutton(frame, text=text, variable=self.param_vars["forcing_type"], 
                               value=value, command=self._on_parameter_change)
            rb.pack(anchor=tk.W)
        
        # Amplitud del forzado
        self._create_parameter_control(
            scrollable_frame, "forcing_amplitude", "Amplitud del Forzado", 
            0.0, 10.0, 0.01,
            "Intensidad del término de forzado"
        )
        
        # Frecuencia del forzado
        self._create_parameter_control(
            scrollable_frame, "forcing_frequency", "Frecuencia del Forzado", 
            0.1, 10.0, 0.01,
            "Frecuencia temporal del forzado (cuando aplicable)"
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.pack(fill="both", expand=True)
    
    def _create_initial_conditions_tab(self):
        """Pestaña de condiciones iniciales"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Condiciones Iniciales")
        
        canvas = tk.Canvas(tab)
        scrollable_frame = ttk.Frame(canvas)
        
        # Tipo de condición inicial
        frame = ttk.LabelFrame(scrollable_frame, text="Tipo de Condición Inicial", padding=5)
        frame.pack(fill=tk.X, pady=5)
        
        self.param_vars["initial_condition"] = tk.StringVar(value=self.current_params["initial_condition"])
        
        ic_types = [
            ("Gaussiano", "gaussian"),
            ("Solitón", "soliton"),
            ("Campo Aleatorio", "random"),
            ("Pulso Cuadrado", "square_pulse"),
            ("Senoidal", "sinusoidal")
        ]
        
        for text, value in ic_types:
            rb = ttk.Radiobutton(frame, text=text, variable=self.param_vars["initial_condition"], 
                               value=value, command=self._on_parameter_change)
            rb.pack(anchor=tk.W)
        
        # Amplitud inicial
        self._create_parameter_control(
            scrollable_frame, "ic_amplitude", "Amplitud Inicial", 
            0.1, 10.0, 0.01,
            "Amplitud de la condición inicial"
        )
        
        # Ancho inicial
        self._create_parameter_control(
            scrollable_frame, "ic_width", "Ancho Inicial", 
            0.1, 10.0, 0.01,
            "Ancho característico de la condición inicial"
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.pack(fill="both", expand=True)
    
    def _create_boundary_tab(self):
        """Pestaña de condiciones de frontera"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Fronteras")
        
        frame = ttk.LabelFrame(tab, text="Condiciones de Frontera", padding=10)
        frame.pack(fill=tk.X, pady=10)
        
        self.param_vars["boundary_condition"] = tk.StringVar(value=self.current_params["boundary_condition"])
        
        boundary_types = [
            ("Periódicas", "periodic"),
            ("Dirichlet (ψ=0)", "dirichlet"),
            ("Neumann (∂ψ/∂n=0)", "neumann"),
            ("Absorbentes", "absorbing")
        ]
        
        for text, value in boundary_types:
            rb = ttk.Radiobutton(frame, text=text, variable=self.param_vars["boundary_condition"], 
                               value=value, command=self._on_parameter_change)
            rb.pack(anchor=tk.W, pady=2)
        
        # Información sobre las condiciones de frontera
        info_text = """
        • Periódicas: Ideales para sistemas sin fronteras
        • Dirichlet: Campo se anula en las fronteras
        • Neumann: Derivada normal se anula en las fronteras
        • Absorbentes: Minimizan reflexiones en las fronteras
        """
        
        info_label = ttk.Label(frame, text=info_text, font=("Arial", 8), foreground="gray")
        info_label.pack(anchor=tk.W, pady=5)
    
    def _create_application_tab(self):
        """Pestaña de aplicaciones específicas"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Aplicación")
        
        frame = ttk.LabelFrame(tab, text="Aplicación Específica", padding=10)
        frame.pack(fill=tk.X, pady=10)
        
        self.param_vars["application"] = tk.StringVar(value=self.current_params["application"])
        
        applications = [
            ("General", "general"),
            ("Superconductividad", "superconductivity"),
            ("Dinámica de Gases", "gas_dynamics"),
            ("Cosmología", "cosmology")
        ]
        
        for text, value in applications:
            rb = ttk.Radiobutton(frame, text=text, variable=self.param_vars["application"], 
                               value=value, command=self._on_application_change)
            rb.pack(anchor=tk.W, pady=2)
        
        # Frame para parámetros específicos de aplicación
        self.app_specific_frame = ttk.LabelFrame(tab, text="Parámetros Específicos", padding=10)
        self.app_specific_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def _create_parameter_control(self, parent, param_name: str, label: str, 
                                min_val: float, max_val: float, step: float, 
                                tooltip: str = ""):
        """Crea un control para un parámetro específico"""
        
        frame = ttk.LabelFrame(parent, text=label, padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        # Variable para el parámetro
        if param_name not in self.param_vars:
            self.param_vars[param_name] = tk.DoubleVar(value=self.current_params[param_name])
        
        var = self.param_vars[param_name]
        
        # Control principal (Scale)
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X)
        
        scale = ttk.Scale(control_frame, from_=min_val, to=max_val, 
                         variable=var, orient=tk.HORIZONTAL,
                         command=lambda x: self._on_parameter_change())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Entry para valor exacto
        entry = ttk.Entry(control_frame, textvariable=var, width=10)
        entry.pack(side=tk.RIGHT, padx=(5, 0))
        entry.bind('<Return>', lambda e: self._on_parameter_change())
        entry.bind('<FocusOut>', lambda e: self._on_parameter_change())
        
        # Label con valor actual
        value_label = ttk.Label(control_frame, text=f"{var.get():.4f}")
        value_label.pack(side=tk.RIGHT, padx=5)
        
        # Actualizar label cuando cambie el valor
        def update_label():
            value_label.config(text=f"{var.get():.4f}")
            
        var.trace('w', lambda *args: update_label())
        
        # Tooltip
        if tooltip:
            tooltip_label = ttk.Label(frame, text=tooltip, font=("Arial", 8), 
                                    foreground="gray")
            tooltip_label.pack(anchor=tk.W)
    
    def _create_control_buttons(self):
        """Crea los botones de control del panel"""
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Botón Reset
        reset_btn = ttk.Button(button_frame, text="Valores por Defecto", 
                              command=self._reset_parameters)
        reset_btn.pack(side=tk.LEFT, padx=2)
        
        # Botón Cargar
        load_btn = ttk.Button(button_frame, text="Cargar Config", 
                             command=self._load_config)
        load_btn.pack(side=tk.LEFT, padx=2)
        
        # Botón Guardar
        save_btn = ttk.Button(button_frame, text="Guardar Config", 
                             command=self._save_config)
        save_btn.pack(side=tk.LEFT, padx=2)
        
        # Botón Aplicar
        apply_btn = ttk.Button(button_frame, text="Aplicar Cambios", 
                              command=self._apply_parameters,
                              style="Accent.TButton")
        apply_btn.pack(side=tk.RIGHT, padx=2)
    
    def _on_parameter_change(self):
        """Callback cuando cambia un parámetro"""
        # Actualizar parámetros actuales
        for param_name, var in self.param_vars.items():
            try:
                if isinstance(var, tk.StringVar):
                    self.current_params[param_name] = var.get()
                else:
                    self.current_params[param_name] = var.get()
            except:
                pass
        
        # Llamar callback si existe
        if self.callback:
            self.callback(self.current_params)
    
    def _on_application_change(self):
        """Callback cuando cambia la aplicación"""
        self._on_parameter_change()
        self._update_application_specific_params()
    
    def _update_application_specific_params(self):
        """Actualiza parámetros específicos según la aplicación"""
        # Limpiar frame actual
        for widget in self.app_specific_frame.winfo_children():
            widget.destroy()
        
        app = self.current_params["application"]
        
        if app == "superconductivity":
            # Parámetros específicos de superconductividad
            ttk.Label(self.app_specific_frame, 
                     text="Configuración optimizada para superconductividad").pack(pady=5)
            
        elif app == "gas_dynamics":
            # Parámetros específicos de dinámica de gases
            ttk.Label(self.app_specific_frame, 
                     text="Configuración optimizada para dinámica de gases").pack(pady=5)
            
        elif app == "cosmology":
            # Parámetros específicos de cosmología
            ttk.Label(self.app_specific_frame, 
                     text="Configuración optimizada para aplicaciones cosmológicas").pack(pady=5)
    
    def _reset_parameters(self):
        """Restaura los parámetros por defecto"""
        self.current_params = self.default_params.copy()
        
        # Actualizar todas las variables
        for param_name, value in self.current_params.items():
            if param_name in self.param_vars:
                self.param_vars[param_name].set(value)
        
        self._update_application_specific_params()
        self._on_parameter_change()
    
    def _load_config(self):
        """Carga configuración desde archivo"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Actualizar parámetros desde config
                if 'simulation' in config and 'default_parameters' in config['simulation']:
                    params = config['simulation']['default_parameters']
                    
                    for param_name, value in params.items():
                        if param_name in self.current_params:
                            self.current_params[param_name] = value
                            if param_name in self.param_vars:
                                self.param_vars[param_name].set(value)
                
                self._update_application_specific_params()
                self._on_parameter_change()
                
        except Exception as e:
            print(f"Error cargando configuración: {e}")
    
    def _save_config(self):
        """Guarda configuración actual"""
        try:
            config = {
                'simulation': {
                    'default_parameters': self.current_params.copy()
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
                
            print(f"Configuración guardada en {self.config_path}")
            
        except Exception as e:
            print(f"Error guardando configuración: {e}")
    
    def _apply_parameters(self):
        """Aplica los parámetros actuales"""
        if self.callback:
            self.callback(self.current_params)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Retorna los parámetros actuales"""
        return self.current_params.copy()
    
    def set_parameters(self, params: Dict[str, Any]):
        """Establece nuevos parámetros"""
        for param_name, value in params.items():
            if param_name in self.current_params:
                self.current_params[param_name] = value
                if param_name in self.param_vars:
                    self.param_vars[param_name].set(value)
        
        self._update_application_specific_params()
        self._on_parameter_change()


# Función para testing standalone
def main():
    """Función principal para testing del panel"""
    
    def parameter_callback(params):
        print("Parámetros actualizados:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    root = tk.Tk()
    root.title("Panel de Parámetros MFSU - Test")
    root.geometry("800x600")
    
    # Aplicar tema moderno
    style = ttk.Style()
    style.theme_use('clam')
    
    panel = ParameterPanel(root, callback=parameter_callback)
    
    root.mainloop()


if __name__ == "__main__":
    main()
