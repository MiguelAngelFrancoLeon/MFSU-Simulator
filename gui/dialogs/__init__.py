"""
MFSU Simulator - Dialogs Module

Este módulo contiene los diálogos de la interfaz gráfica del simulador MFSU.
Incluye diálogos para configuración, exportación y información sobre la aplicación.

Modelo MFSU:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Autor: MFSU Development Team
Versión: 1.0.0
"""

# Importaciones para hacer disponibles los diálogos principales
try:
    from .settings_dialog import SettingsDialog
    from .export_dialog import ExportDialog
    from .about_dialog import AboutDialog
    
    # Lista de clases disponibles para importación
    __all__ = [
        'SettingsDialog',
        'ExportDialog', 
        'AboutDialog'
    ]
    
except ImportError as e:
    # Manejo de errores de importación durante desarrollo
    import warnings
    warnings.warn(f"Error importing dialogs: {e}", ImportWarning)
    
    # Clases dummy para evitar errores durante desarrollo
    class SettingsDialog:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SettingsDialog not implemented yet")
    
    class ExportDialog:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ExportDialog not implemented yet")
    
    class AboutDialog:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AboutDialog not implemented yet")
    
    __all__ = [
        'SettingsDialog',
        'ExportDialog',
        'AboutDialog'
    ]

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "MFSU Development Team"
__description__ = "MFSU Simulator Dialog Components"

# Configuración de logging para el módulo de diálogos
import logging
logger = logging.getLogger(__name__)
logger.info("MFSU Dialogs module initialized")

# Funciones de utilidad para los diálogos
def get_dialog_style():
    """
    Retorna el estilo común para todos los diálogos del simulador MFSU.
    
    Returns:
        dict: Diccionario con estilos CSS/Qt para los diálogos
    """
    return {
        'background_color': '#f0f0f0',
        'font_family': 'Arial, sans-serif',
        'font_size': '10pt',
        'button_style': 'QPushButton { padding: 8px; border-radius: 4px; }',
        'title_style': 'QLabel { font-weight: bold; font-size: 12pt; }'
    }

def validate_dialog_parameters(params):
    """
    Valida los parámetros comunes de los diálogos.
    
    Args:
        params (dict): Parámetros a validar
        
    Returns:
        bool: True si los parámetros son válidos
        
    Raises:
        ValueError: Si algún parámetro no es válido
    """
    required_keys = ['title', 'parent']
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Parámetro requerido faltante: {key}")
    
    if not isinstance(params['title'], str):
        raise ValueError("El título debe ser una cadena de texto")
    
    return True

# Constantes para los diálogos
DIALOG_TITLES = {
    'settings': 'Configuración del Simulador MFSU',
    'export': 'Exportar Resultados de Simulación',
    'about': 'Acerca del Simulador MFSU'
}

DIALOG_SIZES = {
    'settings': (600, 500),
    'export': (500, 400),
    'about': (400, 300)
}

# Mensajes de error comunes
ERROR_MESSAGES = {
    'invalid_parameter': 'Parámetro inválido en el diálogo',
    'file_not_found': 'Archivo no encontrado',
    'export_failed': 'Error al exportar los datos',
    'settings_not_saved': 'No se pudieron guardar las configuraciones'
}
