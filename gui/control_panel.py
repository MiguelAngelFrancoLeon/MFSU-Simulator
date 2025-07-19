"""
Panel de Control para el Simulador MFSU
Permite controlar todos los parámetros de la simulación y ejecutar análisis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import json
from typing import Dict, Any, Callable
import numpy as np

class ControlPanel(ttk.Frame):
    """Panel de control principal para el simulador MFSU"""
    
    def __init__(self, parent, simulation_callback: Callable = None, 
                 analysis_callback: Callable = None):
        super().__init__(parent)
        self.parent = parent
        self.simulation_callback = simulation_callback
        self.analysis_callback = analysis_callback
        
        # Parámetros por defecto de la ecuación MFSU
        self.parameters = {
            # Parámetros de la ecuación principal
            'alpha': 0.5,           # Coeficiente del operador fractal
            'beta': 0.1,            # Intensidad del proceso estocástico
            'gamma': 0.01,          # Coeficiente no lineal
            'hurst': 0.7,           # Exponente de Hurst para ξ_H
            
            # Parámetros numéricos
            'dt': 0.01,             # Paso temporal
            'dx': 0.1,              # Paso espacial
            'grid_size': 100,       # Tamaño de la grilla
            'max_time': 10.0,       # Tiempo máximo de simulación
            
            # Condiciones iniciales
            'initial_condition': 'gaussian',  # gaussian, soliton, random
            'amplitude': 1.0,       # Amplitud inicial
            'width': 2.0,           # Ancho del pulso inicial
            
            # Fuerza externa f(x,t)
            'external_force': False,
            'force_amplitude': 0.1,
            'force_frequency': 1.0,
            
            # Aplicación específica
            'application': 'general',  # general, superconductivity, gas_dynamics, cosmology
        }
        
        self.is_running = False
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        
        # Frame principal con scrollbar
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parámetros de la Ecuación MFSU
        equation_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros de la Ecuación MFSU")
        equation_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_parameter_controls(equation_frame)
        
        # Parámetros Numéricos
        numerical_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros Numéricos")
        numerical_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_numerical_controls(numerical_frame)
        
        # Condiciones Iniciales
        initial_frame = ttk.LabelFrame(scrollable_frame, text="Condiciones Iniciales")
        initial_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_initial_controls(initial_frame)
        
        # Fuerza Externa
        force_frame = ttk.LabelFrame(scrollable_frame, text="Fuerza Externa f(x,t)")
        force_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_force_controls(force_frame)
        
        # Aplicaciones Específicas
        app_frame = ttk.LabelFrame(scrollable_frame, text="Aplicaciones")
        app_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_application_controls(app_frame)
        
        # Controles de Simulación
        control_frame = ttk.LabelFrame(scrollable_frame, text="Control de Simulación")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_simulation_controls(control_frame)
        
        # Configurar el canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_parameter_controls(self, parent):
        """Crea controles para los parámetros de la ecuación MFSU"""
        
        # Alpha (operador fractal)
        alpha_frame = ttk.Frame(parent)
        alpha_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(alpha_frame, text="α (Coef. Fractal):").pack(side="left")
        self.alpha_var = tk.DoubleVar(value=self.parameters['alpha'])
        alpha_scale = ttk.Scale(alpha_frame, from_=0.1, to=2.0, 
                               variable=self.alpha_var, orient="horizontal")
        alpha_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        alpha_entry = ttk.Entry(alpha_frame, textvariable=self.alpha_var, width=8)
        alpha_entry.pack(side="right")
        
        # Beta (intensidad estocástica)
        beta_frame = ttk.Frame(parent)
        beta_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(beta_frame, text="β (Intensidad ξ_H):").pack(side="left")
        self.beta_var = tk.DoubleVar(value=self.parameters['beta'])
        beta_scale = ttk.Scale(beta_frame, from_=0.0, to=1.0, 
                              variable=self.beta_var, orient="horizontal")
        beta_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        beta_entry = ttk.Entry(beta_frame, textvariable=self.beta_var, width=8)
        beta_entry.pack(side="right")
        
        # Gamma (término no lineal)
        gamma_frame = ttk.Frame(parent)
        gamma_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(gamma_frame, text="γ (No Lineal):").pack(side="left")
        self.gamma_var = tk.DoubleVar(value=self.parameters['gamma'])
        gamma_scale = ttk.Scale(gamma_frame, from_=0.0, to=0.1, 
                               variable=self.gamma_var, orient="horizontal")
        gamma_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        gamma_entry = ttk.Entry(gamma_frame, textvariable=self.gamma_var, width=8)
        gamma_entry.pack(side="right")
        
        # Hurst (exponente de Hurst)
        hurst_frame = ttk.Frame(parent)
        hurst_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(hurst_frame, text="H (Hurst):").pack(side="left")
        self.hurst_var = tk.DoubleVar(value=self.parameters['hurst'])
        hurst_scale = ttk.Scale(hurst_frame, from_=0.1, to=0.9, 
                               variable=self.hurst_var, orient="horizontal")
        hurst_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        hurst_entry = ttk.Entry(hurst_frame, textvariable=self.hurst_var, width=8)
        hurst_entry.pack(side="right")
        
    def create_numerical_controls(self, parent):
        """Crea controles para parámetros numéricos"""
        
        # Paso temporal
        dt_frame = ttk.Frame(parent)
        dt_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(dt_frame, text="dt (Paso temporal):").pack(side="left")
        self.dt_var = tk.DoubleVar(value=self.parameters['dt'])
        dt_entry = ttk.Entry(dt_frame, textvariable=self.dt_var, width=10)
        dt_entry.pack(side="right")
        
        # Paso espacial
        dx_frame = ttk.Frame(parent)
        dx_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(dx_frame, text="dx (Paso espacial):").pack(side="left")
        self.dx_var = tk.DoubleVar(value=self.parameters['dx'])
        dx_entry = ttk.Entry(dx_frame, textvariable=self.dx_var, width=10)
        dx_entry.pack(side="right")
        
        # Tamaño de grilla
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(grid_frame, text="Tamaño de grilla:").pack(side="left")
        self.grid_var = tk.IntVar(value=self.parameters['grid_size'])
        grid_entry = ttk.Entry(grid_frame, textvariable=self.grid_var, width=10)
        grid_entry.pack(side="right")
        
        # Tiempo máximo
        time_frame = ttk.Frame(parent)
        time_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(time_frame, text="Tiempo máximo:").pack(side="left")
        self.max_time_var = tk.DoubleVar(value=self.parameters['max_time'])
        time_entry = ttk.Entry(time_frame, textvariable=self.max_time_var, width=10)
        time_entry.pack(side="right")
        
    def create_initial_controls(self, parent):
        """Crea controles para condiciones iniciales"""
        
        # Tipo de condición inicial
        type_frame = ttk.Frame(parent)
        type_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(type_frame, text="Tipo:").pack(side="left")
        self.initial_type_var = tk.StringVar(value=self.parameters['initial_condition'])
        type_combo = ttk.Combobox(type_frame, textvariable=self.initial_type_var,
                                 values=['gaussian', 'soliton', 'random', 'custom'])
        type_combo.pack(side="right")
        
        # Amplitud
        amp_frame = ttk.Frame(parent)
        amp_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(amp_frame, text="Amplitud:").pack(side="left")
        self.amplitude_var = tk.DoubleVar(value=self.parameters['amplitude'])
        amp_entry = ttk.Entry(amp_frame, textvariable=self.amplitude_var, width=10)
        amp_entry.pack(side="right")
        
        # Ancho
        width_frame = ttk.Frame(parent)
        width_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(width_frame, text="Ancho:").pack(side="left")
        self.width_var = tk.DoubleVar(value=self.parameters['width'])
        width_entry = ttk.Entry(width_frame, textvariable=self.width_var, width=10)
        width_entry.pack(side="right")
        
    def create_force_controls(self, parent):
        """Crea controles para la fuerza externa"""
        
        # Activar fuerza externa
        self.force_enabled_var = tk.BooleanVar(value=self.parameters['external_force'])
        force_check = ttk.Checkbutton(parent, text="Activar fuerza externa f(x,t)",
                                     variable=self.force_enabled_var,
                                     command=self.toggle_force_controls)
        force_check.pack(fill="x", padx=5, pady=2)
        
        # Frame para controles de fuerza
        self.force_controls_frame = ttk.Frame(parent)
        self.force_controls_frame.pack(fill="x", padx=20, pady=2)
        
        # Amplitud de fuerza
        force_amp_frame = ttk.Frame(self.force_controls_frame)
        force_amp_frame.pack(fill="x", pady=1)
        
        ttk.Label(force_amp_frame, text="Amplitud:").pack(side="left")
        self.force_amplitude_var = tk.DoubleVar(value=self.parameters['force_amplitude'])
        force_amp_entry = ttk.Entry(force_amp_frame, textvariable=self.force_amplitude_var, width=10)
        force_amp_entry.pack(side="right")
        
        # Frecuencia de fuerza
        force_freq_frame = ttk.Frame(self.force_controls_frame)
        force_freq_frame.pack(fill="x", pady=1)
        
        ttk.Label(force_freq_frame, text="Frecuencia:").pack(side="left")
        self.force_frequency_var = tk.DoubleVar(value=self.parameters['force_frequency'])
        force_freq_entry = ttk.Entry(force_freq_frame, textvariable=self.force_frequency_var, width=10)
        force_freq_entry.pack(side="right")
        
        self.toggle_force_controls()
        
    def create_application_controls(self, parent):
        """Crea controles para aplicaciones específicas"""
        
        app_frame = ttk.Frame(parent)
        app_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(app_frame, text="Aplicación:").pack(side="left")
        self.application_var = tk.StringVar(value=self.parameters['application'])
        app_combo = ttk.Combobox(app_frame, textvariable=self.application_var,
                                values=['general', 'superconductivity', 'gas_dynamics', 'cosmology'])
        app_combo.pack(side="right")
        app_combo.bind('<<ComboboxSelected>>', self.on_application_change)
        
        # Frame para parámetros específicos de aplicación
        self.app_params_frame = ttk.Frame(parent)
        self.app_params_frame.pack(fill="x", padx=20, pady=5)
        
    def create_simulation_controls(self, parent):
        """Crea controles de simulación"""
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # Botón de inicio/parada
        self.run_button = ttk.Button(button_frame, text="Iniciar Simulación",
                                    command=self.toggle_simulation)
        self.run_button.pack(side="left", padx=2)
        
        # Botón de reseteo
        reset_button = ttk.Button(button_frame, text="Reset",
                                 command=self.reset_parameters)
        reset_button.pack(side="left", padx=2)
        
        # Botón de análisis
        analysis_button = ttk.Button(button_frame, text="Análisis",
                                    command=self.run_analysis)
        analysis_button.pack(side="left", padx=2)
        
        # Botones de configuración
        config_frame = ttk.Frame(parent)
        config_frame.pack(fill="x", padx=5, pady=2)
        
        load_button = ttk.Button(config_frame, text="Cargar Config",
                                command=self.load_configuration)
        load_button.pack(side="left", padx=2)
        
        save_button = ttk.Button(config_frame, text="Guardar Config",
                                command=self.save_configuration)
        save_button.pack(side="left", padx=2)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        
        # Etiqueta de estado
        self.status_var = tk.StringVar(value="Listo")
        status_label = ttk.Label(parent, textvariable=self.status_var)
        status_label.pack(fill="x", padx=5, pady=2)
        
    def toggle_force_controls(self):
        """Activa/desactiva controles de fuerza externa"""
        if self.force_enabled_var.get():
            self.force_controls_frame.pack(fill="x", padx=20, pady=2)
        else:
            self.force_controls_frame.pack_forget()
            
    def on_application_change(self, event=None):
        """Maneja el cambio de aplicación"""
        app = self.application_var.get()
        
        # Limpiar parámetros específicos anteriores
        for widget in self.app_params_frame.winfo_children():
            widget.destroy()
            
        # Agregar parámetros específicos según la aplicación
        if app == 'superconductivity':
            self.create_superconductivity_params()
        elif app == 'gas_dynamics':
            self.create_gas_dynamics_params()
        elif app == 'cosmology':
            self.create_cosmology_params()
            
    def create_superconductivity_params(self):
        """Parámetros específicos para superconductividad"""
        temp_frame = ttk.Frame(self.app_params_frame)
        temp_frame.pack(fill="x", pady=1)
        
        ttk.Label(temp_frame, text="Temperatura (K):").pack(side="left")
        self.temperature_var = tk.DoubleVar(value=77)
        temp_entry = ttk.Entry(temp_frame, textvariable=self.temperature_var, width=10)
        temp_entry.pack(side="right")
        
    def create_gas_dynamics_params(self):
        """Parámetros específicos para dinámica de gases"""
        re_frame = ttk.Frame(self.app_params_frame)
        re_frame.pack(fill="x", pady=1)
        
        ttk.Label(re_frame, text="Reynolds:").pack(side="left")
        self.reynolds_var = tk.DoubleVar(value=1000)
        re_entry = ttk.Entry(re_frame, textvariable=self.reynolds_var, width=10)
        re_entry.pack(side="right")
        
    def create_cosmology_params(self):
        """Parámetros específicos para cosmología"""
        h_frame = ttk.Frame(self.app_params_frame)
        h_frame.pack(fill="x", pady=1)
        
        ttk.Label(h_frame, text="H₀ (km/s/Mpc):").pack(side="left")
        self.hubble_var = tk.DoubleVar(value=70)
        h_entry = ttk.Entry(h_frame, textvariable=self.hubble_var, width=10)
        h_entry.pack(side="right")
        
    def get_parameters(self) -> Dict[str, Any]:
        """Obtiene todos los parámetros actuales"""
        params = {
            'alpha': self.alpha_var.get(),
            'beta': self.beta_var.get(),
            'gamma': self.gamma_var.get(),
            'hurst': self.hurst_var.get(),
            'dt': self.dt_var.get(),
            'dx': self.dx_var.get(),
            'grid_size': self.grid_var.get(),
            'max_time': self.max_time_var.get(),
            'initial_condition': self.initial_type_var.get(),
            'amplitude': self.amplitude_var.get(),
            'width': self.width_var.get(),
            'external_force': self.force_enabled_var.get(),
            'force_amplitude': self.force_amplitude_var.get(),
            'force_frequency': self.force_frequency_var.get(),
            'application': self.application_var.get(),
        }
        
        # Agregar parámetros específicos de aplicación
        if hasattr(self, 'temperature_var'):
            params['temperature'] = self.temperature_var.get()
        if hasattr(self, 'reynolds_var'):
            params['reynolds'] = self.reynolds_var.get()
        if hasattr(self, 'hubble_var'):
            params['hubble_constant'] = self.hubble_var.get()
            
        return params
        
    def set_parameters(self, params: Dict[str, Any]):
        """Establece parámetros desde un diccionario"""
        for key, value in params.items():
            if hasattr(self, f"{key}_var"):
                getattr(self, f"{key}_var").set(value)
                
    def toggle_simulation(self):
        """Inicia o detiene la simulación"""
        if not self.is_running:
            self.start_simulation()
        else:
            self.stop_simulation()
            
    def start_simulation(self):
        """Inicia la simulación"""
        try:
            params = self.get_parameters()
            
            # Validar parámetros
            if not self.validate_parameters(params):
                return
                
            self.is_running = True
            self.run_button.config(text="Detener Simulación")
            self.status_var.set("Ejecutando simulación...")
            
            # Llamar callback de simulación
            if self.simulation_callback:
                self.simulation_callback(params)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar simulación: {str(e)}")
            self.stop_simulation()
            
    def stop_simulation(self):
        """Detiene la simulación"""
        self.is_running = False
        self.run_button.config(text="Iniciar Simulación")
        self.status_var.set("Simulación detenida")
        self.progress_var.set(0)
        
    def run_analysis(self):
        """Ejecuta análisis de los resultados"""
        if self.analysis_callback:
            params = self.get_parameters()
            self.status_var.set("Ejecutando análisis...")
            self.analysis_callback(params)
            self.status_var.set("Análisis completado")
            
    def reset_parameters(self):
        """Resetea parámetros a valores por defecto"""
        self.set_parameters(self.parameters)
        self.status_var.set("Parámetros reseteados")
        
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Valida los parámetros de simulación"""
        try:
            # Validaciones básicas
            if params['alpha'] <= 0 or params['alpha'] > 2:
                raise ValueError("α debe estar entre 0 y 2")
                
            if params['beta'] < 0:
                raise ValueError("β debe ser no negativo")
                
            if params['gamma'] < 0:
                raise ValueError("γ debe ser no negativo")
                
            if params['hurst'] <= 0 or params['hurst'] >= 1:
                raise ValueError("H debe estar entre 0 y 1")
                
            if params['dt'] <= 0:
                raise ValueError("dt debe ser positivo")
                
            if params['dx'] <= 0:
                raise ValueError("dx debe ser positivo")
                
            if params['grid_size'] < 10:
                raise ValueError("Tamaño de grilla debe ser al menos 10")
                
            if params['max_time'] <= 0:
                raise ValueError("Tiempo máximo debe ser positivo")
                
            # Validación de estabilidad numérica
            cfl = params['dt'] / (params['dx'] ** params['alpha'])
            if cfl > 0.5:
                messagebox.showwarning("Advertencia", 
                    "Condición CFL puede ser inestable. Considere reducir dt o aumentar dx.")
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Error de Validación", str(e))
            return False
            
    def load_configuration(self):
        """Carga configuración desde archivo"""
        try:
            filename = filedialog.askopenfilename(
                title="Cargar Configuración",
                filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json")]
            )
            
            if filename:
                if filename.endswith('.yaml'):
                    with open(filename, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    with open(filename, 'r') as f:
                        config = json.load(f)
                        
                self.set_parameters(config.get('simulation', {}))
                self.status_var.set(f"Configuración cargada: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar configuración: {str(e)}")
            
    def save_configuration(self):
        """Guarda configuración actual"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Guardar Configuración",
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json")]
            )
            
            if filename:
                config = {
                    'simulation': self.get_parameters(),
                    'metadata': {
                        'version': '1.0.0',
                        'description': 'Configuración del Simulador MFSU'
                    }
                }
                
                if filename.endswith('.yaml'):
                    with open(filename, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                else:
                    with open(filename, 'w') as f:
                        json.dump(config, f, indent=2)
                        
                self.status_var.set(f"Configuración guardada: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar configuración: {str(e)}")
            
    def update_progress(self, value: float):
        """Actualiza la barra de progreso"""
        self.progress_var.set(value)
        self.parent.update_idletasks()
        
    def update_status(self, message: str):
        """Actualiza el mensaje de estado"""
        self.status_var.set(message)
        self.parent.update_idletasks()


# Ejemplo de uso
if __name__ == "__main__":
    def simulation_callback(params):
        print(f"Simulación iniciada con parámetros: {params}")
        
    def analysis_callback(params):
        print(f"Análisis ejecutado con parámetros: {params}")
    
    root = tk.Tk()
    root.title("Panel de Control MFSU")
    root.geometry("600x800")
    
    control_panel = ControlPanel(root, simulation_callback, analysis_callback)
    control_panel.pack(fill="both", expand=True)
    
    root.mainloop()
