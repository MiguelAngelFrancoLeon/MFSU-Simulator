"""
Diálogo "Acerca de" para el Simulador MFSU
Unified Stochastic Fractal Model Simulator

Este módulo contiene la ventana de información sobre el proyecto MFSU,
incluyendo detalles técnicos, créditos y referencias.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from tkinter import font as tkFont


class AboutDialog:
    """
    Diálogo que muestra información sobre el simulador MFSU,
    incluyendo la fórmula principal, aplicaciones y créditos.
    """
    
    def __init__(self, parent):
        """
        Inicializa el diálogo "Acerca de".
        
        Args:
            parent: Ventana padre
        """
        self.parent = parent
        self.dialog = None
        
    def show(self):
        """Muestra el diálogo "Acerca de"."""
        if self.dialog is not None:
            self.dialog.lift()
            return
            
        # Crear ventana del diálogo
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Acerca de MFSU Simulator")
        self.dialog.geometry("800x700")
        self.dialog.resizable(True, True)
        
        # Centrar la ventana
        self._center_window()
        
        # Configurar el estilo
        self._setup_styles()
        
        # Crear el contenido
        self._create_content()
        
        # Configurar eventos
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
    def _center_window(self):
        """Centra la ventana en la pantalla."""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
    def _setup_styles(self):
        """Configura los estilos de fuente."""
        self.title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        self.subtitle_font = tkFont.Font(family="Arial", size=12, weight="bold")
        self.normal_font = tkFont.Font(family="Arial", size=10)
        self.code_font = tkFont.Font(family="Consolas", size=9)
        
    def _create_content(self):
        """Crea el contenido principal del diálogo."""
        # Crear notebook para pestañas
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Pestaña principal
        self._create_main_tab(notebook)
        
        # Pestaña de fórmula
        self._create_formula_tab(notebook)
        
        # Pestaña de aplicaciones
        self._create_applications_tab(notebook)
        
        # Pestaña de créditos
        self._create_credits_tab(notebook)
        
        # Botón de cerrar
        btn_frame = tk.Frame(self.dialog)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        close_btn = ttk.Button(btn_frame, text="Cerrar", command=self._on_close)
        close_btn.pack(side="right")
        
    def _create_main_tab(self, notebook):
        """Crea la pestaña principal con información general."""
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="General")
        
        # Crear canvas con scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título principal
        title_label = tk.Label(
            scrollable_frame, 
            text="MFSU Simulator", 
            font=self.title_font,
            fg="#2C3E50"
        )
        title_label.pack(pady=(20, 5))
        
        subtitle_label = tk.Label(
            scrollable_frame, 
            text="Unified Stochastic Fractal Model Simulator", 
            font=self.subtitle_font,
            fg="#34495E"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Información del proyecto
        info_text = """
MFSU Simulator es una herramienta avanzada de simulación que implementa el
Modelo Fractal Estocástico Unificado para el análisis de sistemas complejos
en múltiples dominios científicos.

Versión: 1.0.0
Desarrollado en Python con interfaces gráfica y web

CARACTERÍSTICAS PRINCIPALES:
• Implementación completa de la ecuación MFSU
• Operadores fraccionarios de alta precisión
• Procesos estocásticos con ruido fractal
• Análisis de dimensión fractal
• Visualizaciones interactivas en tiempo real
• Múltiples aplicaciones científicas

CAPACIDADES TÉCNICAS:
• Simulación numérica de alta performance con Numba
• Análisis espectral y estadístico avanzado
• Exportación de datos en múltiples formatos
• Interfaz web responsiva con Plotly
• Documentación completa con Sphinx
• Suite completa de pruebas automatizadas
        """
        
        info_label = tk.Label(
            scrollable_frame,
            text=info_text.strip(),
            font=self.normal_font,
            justify="left",
            wraplength=750
        )
        info_label.pack(pady=10, padx=20)
        
        # Link al repositorio
        link_frame = tk.Frame(scrollable_frame)
        link_frame.pack(pady=20)
        
        link_label = tk.Label(
            link_frame,
            text="🔗 Repositorio en Zenodo:",
            font=self.normal_font
        )
        link_label.pack()
        
        zenodo_btn = tk.Button(
            link_frame,
            text="https://zenodo.org/records/15828185",
            fg="blue",
            cursor="hand2",
            relief="flat",
            command=lambda: webbrowser.open("https://zenodo.org/records/15828185")
        )
        zenodo_btn.pack(pady=5)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _create_formula_tab(self, notebook):
        """Crea la pestaña con la fórmula MFSU."""
        formula_frame = ttk.Frame(notebook)
        notebook.add(formula_frame, text="Fórmula MFSU")
        
        # Canvas con scrollbar
        canvas = tk.Canvas(formula_frame)
        scrollbar = ttk.Scrollbar(formula_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_label = tk.Label(
            scrollable_frame,
            text="Ecuación del Modelo Fractal Estocástico Unificado (MFSU)",
            font=self.subtitle_font,
            fg="#2C3E50"
        )
        title_label.pack(pady=20)
        
        # Fórmula principal
        formula_text = "∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)"
        
        formula_frame_inner = tk.Frame(scrollable_frame, bg="#F8F9FA", relief="solid", bd=1)
        formula_frame_inner.pack(pady=20, padx=20, fill="x")
        
        formula_label = tk.Label(
            formula_frame_inner,
            text=formula_text,
            font=("Times New Roman", 14),
            bg="#F8F9FA",
            fg="#2C3E50"
        )
        formula_label.pack(pady=15)
        
        # Explicación de términos
        explanation_text = """
COMPONENTES DE LA ECUACIÓN:

• ∂ψ/∂t: Derivada temporal del campo ψ
• α(-Δ)^(α/2)ψ: Término de difusión fraccionaria
  - α: Parámetro de orden fraccionario (0 < α ≤ 2)
  - (-Δ)^(α/2): Operador Laplaciano fraccionario
• β ξ_H(x,t)ψ: Término estocástico
  - β: Intensidad del ruido
  - ξ_H(x,t): Ruido fractal con parámetro de Hurst H
• -γψ³: Término no lineal (tipo Ginzburg-Landau)
  - γ: Parámetro de no linealidad
• f(x,t): Término de forzamiento externo

CARACTERÍSTICAS FÍSICAS:

• Combina difusión fraccionaria y procesos estocásticos
• Incluye no linealidades que permiten formación de estructuras
• Permite forzamiento externo para simulaciones específicas
• Describe fenómenos de memoria larga y correlaciones espaciales
• Aplicable a sistemas con propiedades fractales
        """
        
        explanation_label = tk.Label(
            scrollable_frame,
            text=explanation_text.strip(),
            font=self.normal_font,
            justify="left",
            wraplength=750
        )
        explanation_label.pack(pady=10, padx=20)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _create_applications_tab(self, notebook):
        """Crea la pestaña de aplicaciones."""
        apps_frame = ttk.Frame(notebook)
        notebook.add(apps_frame, text="Aplicaciones")
        
        # Canvas con scrollbar
        canvas = tk.Canvas(apps_frame)
        scrollbar = ttk.Scrollbar(apps_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_label = tk.Label(
            scrollable_frame,
            text="Aplicaciones del Modelo MFSU",
            font=self.subtitle_font,
            fg="#2C3E50"
        )
        title_label.pack(pady=20)
        
        applications_text = """
El simulador MFSU incluye implementaciones especializadas para diversos
dominios científicos:

🔬 SUPERCONDUCTIVIDAD
• Análisis de transiciones de fase en superconductores
• Modelado de redes de vórtices en superconductores tipo II
• Simulación de propiedades críticas y temperaturas de transición
• Aplicaciones en materiales de alta temperatura crítica

⚡ DINÁMICA DE GASES Y FLUIDOS
• Turbulencia fractal en fluidos complejos
• Análisis de cascadas de energía en turbulencia
• Modelado de flujos con memoria larga
• Simulación de procesos de mezcla anómalos

🌌 COSMOLOGÍA Y ASTROFÍSICA
• Formación de estructuras a gran escala del universo
• Análisis de fluctuaciones en el fondo cósmico de microondas
• Modelado de materia oscura con propiedades fractales
• Simulación de procesos de acreción en agujeros negros

🧬 BIOFÍSICA Y SISTEMAS BIOLÓGICOS
• Modelado de difusión anómala en membranas celulares
• Análisis de patrones fractales en crecimiento biológico
• Simulación de procesos de señalización celular
• Modelado de redes neuronales con conectividad fractal

💰 FINANZAS Y ECONOFÍSICA
• Análisis de series temporales financieras con memoria larga
• Modelado de volatilidad con procesos fractales
• Simulación de burbujas especulativas y crashes
• Análisis de correlaciones en mercados complejos

🌍 CIENCIAS DE LA TIERRA
• Modelado de procesos sísmicos con estadísticas fractales
• Análisis de patrones climáticos con memoria larga
• Simulación de procesos de erosión y sedimentación
• Modelado de redes hidrológicas fractales

VENTAJAS DEL ENFOQUE UNIFICADO:
• Marco teórico consistente para múltiples dominios
• Parámetros físicamente interpretables
• Flexibilidad para diferentes condiciones iniciales
• Capacidad de análisis comparativo entre aplicaciones
• Validación cruzada entre diferentes fenómenos
        """
        
        apps_label = tk.Label(
            scrollable_frame,
            text=applications_text.strip(),
            font=self.normal_font,
            justify="left",
            wraplength=750
        )
        apps_label.pack(pady=10, padx=20)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _create_credits_tab(self, notebook):
        """Crea la pestaña de créditos."""
        credits_frame = ttk.Frame(notebook)
        notebook.add(credits_frame, text="Créditos")
        
        # Canvas con scrollbar
        canvas = tk.Canvas(credits_frame)
        scrollbar = ttk.Scrollbar(credits_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_label = tk.Label(
            scrollable_frame,
            text="Equipo de Desarrollo y Agradecimientos",
            font=self.subtitle_font,
            fg="#2C3E50"
        )
        title_label.pack(pady=20)
        
        credits_text = """
EQUIPO DE DESARROLLO MFSU:
• Desarrollo del modelo teórico
• Implementación numérica y algoritmos
• Interfaces gráfica y web
• Documentación y validación

TECNOLOGÍAS UTILIZADAS:
• Python 3.8+ - Lenguaje principal
• NumPy & SciPy - Computación numérica
• Matplotlib & Plotly - Visualización
• Numba - Aceleración de código
• tkinter - Interfaz gráfica nativa
• Flask - Aplicación web
• Jupyter - Notebooks interactivos
• Sphinx - Documentación

BIBLIOTECAS ESPECIALIZADAS:
• PyFFTW - Transformadas de Fourier optimizadas
• scikit-learn - Análisis estadístico
• H5PY - Manejo de datasets grandes
• Pandas - Manipulación de datos
• YAML - Configuración flexible

AGRADECIMIENTOS ESPECIALES:
• Comunidad científica internacional por el feedback
• Desarrolladores de software libre y código abierto
• Investigadores en cálculo fraccionario y procesos estocásticos
• Beta testers y usuarios pioneros del simulador

LICENCIA:
Este proyecto se distribuye bajo licencia de código abierto.
Consulte el archivo LICENSE para más detalles.

CONTACTO Y SOPORTE:
• Documentación: docs/build/html/index.html
• Issues y reportes: Repositorio en Zenodo
• Comunidad: Foros de discusión científica

CÓMO CITAR:
Si utiliza MFSU Simulator en su investigación, por favor cite:
"MFSU Simulator: Unified Stochastic Fractal Model Simulator"
Zenodo: https://zenodo.org/records/15828185

VERSION HISTORY:
• v1.0.0 (2025): Lanzamiento inicial
  - Implementación completa de la ecuación MFSU
  - Interfaces gráfica y web funcionales
  - Aplicaciones en superconductividad, dinámica de gases y cosmología
  - Suite completa de análisis fractales y estadísticos
        """
        
        credits_label = tk.Label(
            scrollable_frame,
            text=credits_text.strip(),
            font=self.normal_font,
            justify="left",
            wraplength=750
        )
        credits_label.pack(pady=10, padx=20)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _on_close(self):
        """Cierra el diálogo."""
        if self.dialog:
            self.dialog.grab_release()
            self.dialog.destroy()
            self.dialog = None


# Función de conveniencia para mostrar el diálogo
def show_about_dialog(parent):
    """
    Muestra el diálogo "Acerca de" del simulador MFSU.
    
    Args:
        parent: Ventana padre
    """
    about = AboutDialog(parent)
    about.show()


if __name__ == "__main__":
    # Prueba independiente del diálogo
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal para la prueba
    
    about = AboutDialog(root)
    about.show()
    
    root.mainloop()
