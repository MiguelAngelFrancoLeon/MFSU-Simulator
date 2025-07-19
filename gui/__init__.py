"""
GUI Package for MFSU Simulator
==============================

This package provides the graphical user interface components for the
Modified Fractal Stochastic Unified (MFSU) equation simulator.

The MFSU equation implemented is:
∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)

Where:
- α: Fractional diffusion coefficient
- β: Stochastic coupling strength
- γ: Non-linear interaction parameter
- ξ_H: Fractional Brownian motion with Hurst parameter H
- f(x,t): External forcing term

Modules:
--------
- main_window: Main application window with menu and layout management
- parameter_panel: Parameter input and validation controls
- visualization_panel: Real-time plotting and 3D visualization
- control_panel: Simulation control buttons and status display
- dialogs: Modal dialogs for settings, export, and about information

Classes:
--------
- MFSUMainWindow: Primary application window
- ParameterPanel: Parameter configuration interface
- VisualizationPanel: Real-time data visualization
- ControlPanel: Simulation execution controls

Usage:
------
    from gui import MFSUMainWindow
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MFSUMainWindow()
    window.show()
    sys.exit(app.exec_())

Dependencies:
-------------
- PyQt5 or PySide2 for GUI framework
- matplotlib for plotting
- numpy for numerical operations
- plotly for interactive visualizations

Author: MFSU Development Team
Version: 1.0.0
License: MIT
"""

import logging
import sys
from typing import Optional, Dict, Any, List, Tuple
import warnings

# Version information
__version__ = "1.0.0"
__author__ = "MFSU Development Team"
__email__ = "mfsu@example.com"
__license__ = "MIT"

# Configure logging for GUI package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# GUI Framework detection and import
GUI_FRAMEWORK = None
QT_AVAILABLE = False

try:
    # Try PyQt5 first
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QObject, pyqtSignal as Signal
    from PyQt5.QtGui import QIcon, QFont
    GUI_FRAMEWORK = "PyQt5"
    QT_AVAILABLE = True
    logger.info("Using PyQt5 as GUI framework")
except ImportError:
    try:
        # Fallback to PySide2
        from PySide2.QtWidgets import QApplication
        from PySide2.QtCore import QObject, Signal
        from PySide2.QtGui import QIcon, QFont
        GUI_FRAMEWORK = "PySide2"
        QT_AVAILABLE = True
        logger.info("Using PySide2 as GUI framework")
    except ImportError:
        logger.warning("No Qt framework found. GUI functionality disabled.")
        QT_AVAILABLE = False

# Import matplotlib and configure for Qt backend
try:
    import matplotlib
    if QT_AVAILABLE:
        if GUI_FRAMEWORK == "PyQt5":
            matplotlib.use('Qt5Agg')
        elif GUI_FRAMEWORK == "PySide2":
            matplotlib.use('Qt5Agg')  # PySide2 also uses Qt5Agg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib configured for Qt backend")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Plotting functionality disabled.")

# Check for additional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy is required but not found!")

try:
    import plotly
    PLOTLY_AVAILABLE = True
    logger.info("Plotly available for interactive plotting")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("Plotly not available. Interactive plotting disabled.")

# GUI Constants and Configuration
GUI_CONFIG = {
    'window_title': 'MFSU Simulator',
    'window_width': 1200,
    'window_height': 800,
    'min_window_width': 800,
    'min_window_height': 600,
    'default_font_family': 'Arial',
    'default_font_size': 10,
    'icon_path': 'assets/images/mfsu_icon.png',
    'stylesheet_path': 'static/css/styles.css',
    'update_interval': 100,  # milliseconds for real-time updates
    'max_plot_points': 10000,  # maximum points in real-time plots
}

# Color themes for the GUI
COLOR_THEMES = {
    'default': {
        'background': '#f0f0f0',
        'panel': '#ffffff',
        'border': '#c0c0c0',
        'text': '#000000',
        'accent': '#0066cc',
        'error': '#cc0000',
        'success': '#00cc00',
        'warning': '#ff9900',
    },
    'dark': {
        'background': '#2d2d2d',
        'panel': '#3d3d3d',
        'border': '#5d5d5d',
        'text': '#ffffff',
        'accent': '#66aaff',
        'error': '#ff6666',
        'success': '#66ff66',
        'warning': '#ffaa66',
    }
}

# Default MFSU parameters for GUI initialization
DEFAULT_MFSU_PARAMETERS = {
    'alpha': 0.5,      # Fractional diffusion coefficient
    'beta': 0.1,       # Stochastic coupling strength
    'gamma': 0.01,     # Non-linear interaction parameter
    'hurst': 0.7,      # Hurst parameter for fractional Brownian motion
    'dt': 0.01,        # Time step
    'dx': 0.1,         # Spatial step
    'grid_size': 100,  # Number of grid points
    'max_time': 10.0,  # Maximum simulation time
    'boundary_type': 'periodic',  # Boundary condition type
    'initial_condition': 'gaussian',  # Initial condition type
}

# Application domains for MFSU equation
APPLICATION_DOMAINS = {
    'superconductivity': {
        'name': 'Superconductivity',
        'description': 'Cooper pair dynamics and phase transitions',
        'typical_params': {
            'alpha': 0.8,
            'beta': 0.05,
            'gamma': 0.1,
            'hurst': 0.6
        }
    },
    'gas_dynamics': {
        'name': 'Gas Dynamics',
        'description': 'Turbulent flow and fractal structures',
        'typical_params': {
            'alpha': 0.3,
            'beta': 0.2,
            'gamma': 0.05,
            'hurst': 0.8
        }
    },
    'cosmology': {
        'name': 'Cosmology',
        'description': 'Large-scale structure formation',
        'typical_params': {
            'alpha': 0.6,
            'beta': 0.15,
            'gamma': 0.02,
            'hurst': 0.9
        }
    }
}

def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of all GUI dependencies.
    
    Returns:
        Dict[str, bool]: Dictionary with dependency status
    """
    dependencies = {
        'qt_framework': QT_AVAILABLE,
        'matplotlib': MATPLOTLIB_AVAILABLE,
        'numpy': NUMPY_AVAILABLE,
        'plotly': PLOTLY_AVAILABLE,
    }
    
    missing = [name for name, available in dependencies.items() if not available]
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
    
    return dependencies

def get_gui_info() -> Dict[str, Any]:
    """
    Get information about the GUI configuration.
    
    Returns:
        Dict[str, Any]: GUI configuration information
    """
    return {
        'version': __version__,
        'framework': GUI_FRAMEWORK,
        'qt_available': QT_AVAILABLE,
        'matplotlib_available': MATPLOTLIB_AVAILABLE,
        'dependencies': check_dependencies(),
        'config': GUI_CONFIG.copy(),
    }

def validate_mfsu_parameters(params: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate MFSU equation parameters.
    
    Args:
        params: Dictionary of MFSU parameters
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required parameters
    required_params = ['alpha', 'beta', 'gamma', 'hurst', 'dt', 'dx', 'grid_size', 'max_time']
    for param in required_params:
        if param not in params:
            errors.append(f"Missing required parameter: {param}")
    
    if errors:
        return False, errors
    
    # Validate parameter ranges
    if not (0.0 < params['alpha'] <= 2.0):
        errors.append("Alpha must be in range (0, 2]")
    
    if params['beta'] < 0:
        errors.append("Beta must be non-negative")
    
    if params['gamma'] < 0:
        errors.append("Gamma must be non-negative")
    
    if not (0.0 < params['hurst'] < 1.0):
        errors.append("Hurst parameter must be in range (0, 1)")
    
    if params['dt'] <= 0:
        errors.append("Time step (dt) must be positive")
    
    if params['dx'] <= 0:
        errors.append("Spatial step (dx) must be positive")
    
    if params['grid_size'] <= 0 or not isinstance(params['grid_size'], int):
        errors.append("Grid size must be a positive integer")
    
    if params['max_time'] <= 0:
        errors.append("Maximum time must be positive")
    
    # Stability condition for numerical scheme
    cfl_condition = params['dt'] / (params['dx'] ** params['alpha'])
    if cfl_condition > 0.1:  # Conservative stability limit
        errors.append(f"CFL condition violated: dt/dx^α = {cfl_condition:.4f} > 0.1")
    
    return len(errors) == 0, errors

def create_application(argv: Optional[List[str]] = None) -> Optional[QApplication]:
    """
    Create and configure the QApplication instance.
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
        
    Returns:
        QApplication instance or None if Qt not available
    """
    if not QT_AVAILABLE:
        logger.error("Cannot create application: Qt framework not available")
        return None
    
    if argv is None:
        argv = sys.argv
    
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
        
        # Configure application properties
        app.setApplicationName(GUI_CONFIG['window_title'])
        app.setApplicationVersion(__version__)
        app.setOrganizationName("MFSU Development Team")
        
        # Set default font
        font = QFont(GUI_CONFIG['default_font_family'], GUI_CONFIG['default_font_size'])
        app.setFont(font)
        
        logger.info("QApplication created and configured")
    
    return app

# Import GUI modules (only if Qt is available)
if QT_AVAILABLE:
    try:
        from .main_window import MFSUMainWindow
        from .parameter_panel import ParameterPanel
        from .visualization_panel import VisualizationPanel
        from .control_panel import ControlPanel
        
        # Import dialogs
        from .dialogs import SettingsDialog, ExportDialog, AboutDialog
        
        logger.info("All GUI modules imported successfully")
        
        # Define public API
        __all__ = [
            'MFSUMainWindow',
            'ParameterPanel', 
            'VisualizationPanel',
            'ControlPanel',
            'SettingsDialog',
            'ExportDialog',
            'AboutDialog',
            'create_application',
            'check_dependencies',
            'get_gui_info',
            'validate_mfsu_parameters',
            'GUI_CONFIG',
            'COLOR_THEMES',
            'DEFAULT_MFSU_PARAMETERS',
            'APPLICATION_DOMAINS',
        ]
        
    except ImportError as e:
        logger.error(f"Failed to import GUI modules: {e}")
        # Fallback: define minimal API
        __all__ = [
            'create_application',
            'check_dependencies', 
            'get_gui_info',
            'validate_mfsu_parameters',
            'GUI_CONFIG',
            'DEFAULT_MFSU_PARAMETERS',
        ]
else:
    logger.warning("Qt framework not available. GUI functionality disabled.")
    # Define minimal API for non-GUI environments
    __all__ = [
        'check_dependencies',
        'get_gui_info', 
        'validate_mfsu_parameters',
        'DEFAULT_MFSU_PARAMETERS',
    ]

# Module initialization
logger.info(f"MFSU GUI package initialized (version {__version__})")
if QT_AVAILABLE:
    logger.info(f"Using {GUI_FRAMEWORK} framework")
else:
    logger.warning("GUI functionality disabled - install PyQt5 or PySide2")
