"""
Sistema de logging para el simulador MFSU (Modelo Fractal Estocástico Unificado)

Este módulo proporciona un sistema de logging robusto y configurable para
todas las operaciones del simulador, incluyendo simulaciones, análisis,
y operaciones de I/O.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json

class MFSULogger:
    """
    Sistema de logging especializado para el simulador MFSU.
    
    Características:
    - Múltiples handlers (consola, archivo, archivo rotativo)
    - Formateo personalizado con información de contexto
    - Niveles de logging específicos para diferentes componentes
    - Logging de métricas de rendimiento
    - Integración con configuración YAML
    """
    
    _instances: Dict[str, 'MFSULogger'] = {}
    _configured = False
    
    def __new__(cls, name: str = "MFSU"):
        """Singleton pattern para cada nombre de logger."""
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str = "MFSU"):
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        self._initialized = True
        
        if not MFSULogger._configured:
            self._configure_logging()
            MFSULogger._configured = True
    
    def _configure_logging(self):
        """Configuración inicial del sistema de logging."""
        # Crear directorio de logs si no existe
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Configurar nivel base
        self.logger.setLevel(logging.DEBUG)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # Handler para archivo general
        file_handler = logging.FileHandler(
            log_dir / f"mfsu_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Handler rotativo para logs de simulación
        rotating_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mfsu_simulation.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.setFormatter(file_formatter)
        
        # Agregar handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(rotating_handler)
        
        # Evitar propagación a loggers padre
        self.logger.propagate = False
    
    def load_config(self, config_path: str = "config.yaml"):
        """
        Cargar configuración de logging desde archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                logging_config = config.get('logging', {})
                
                # Actualizar nivel de logging
                level = logging_config.get('level', 'INFO').upper()
                self.logger.setLevel(getattr(logging, level))
                
                # Configurar handlers específicos
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        handler.setLevel(getattr(logging, logging_config.get('console_level', 'INFO').upper()))
                    elif isinstance(handler, logging.FileHandler):
                        handler.setLevel(getattr(logging, logging_config.get('file_level', 'DEBUG').upper()))
                
                self.info(f"Configuración de logging cargada desde {config_path}")
            else:
                self.warning(f"Archivo de configuración {config_path} no encontrado, usando configuración por defecto")
                
        except Exception as e:
            self.error(f"Error cargando configuración de logging: {e}")
    
    def debug(self, message: str, **kwargs):
        """Log mensaje de debug con contexto adicional."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log mensaje informativo con contexto adicional."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log mensaje de advertencia con contexto adicional."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log mensaje de error con contexto adicional."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log mensaje crítico con contexto adicional."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """
        Log mensaje con contexto adicional.
        
        Args:
            level: Nivel de logging
            message: Mensaje principal
            **kwargs: Información de contexto adicional
        """
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            full_message = f"{message} | {context_str}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
    
    def log_simulation_start(self, parameters: Dict[str, Any]):
        """
        Log inicio de simulación con parámetros.
        
        Args:
            parameters: Diccionario con parámetros de simulación
        """
        self.info("=== INICIO DE SIMULACIÓN MFSU ===")
        self.info("Parámetros de simulación:", **parameters)
    
    def log_simulation_step(self, step: int, time: float, metrics: Dict[str, float]):
        """
        Log paso de simulación con métricas.
        
        Args:
            step: Número de paso
            time: Tiempo de simulación
            metrics: Métricas del paso actual
        """
        self.debug(f"Paso {step}", time=f"{time:.4f}", **metrics)
    
    def log_simulation_end(self, total_time: float, final_metrics: Dict[str, Any]):
        """
        Log fin de simulación con métricas finales.
        
        Args:
            total_time: Tiempo total de simulación
            final_metrics: Métricas finales
        """
        self.info("=== FIN DE SIMULACIÓN MFSU ===")
        self.info(f"Tiempo total: {total_time:.2f}s", **final_metrics)
    
    def log_equation_evaluation(self, component: str, value: float, **params):
        """
        Log evaluación de componentes de la ecuación MFSU.
        
        Args:
            component: Componente de la ecuación (fractal, stochastic, nonlinear, etc.)
            value: Valor calculado
            **params: Parámetros relevantes
        """
        self.debug(f"Ecuación MFSU - {component}", value=f"{value:.6f}", **params)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """
        Log métricas de rendimiento.
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            **metrics: Métricas adicionales
        """
        self.info(f"Performance - {operation}", 
                 duration=f"{duration:.4f}s", 
                 **metrics)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """
        Log error con contexto detallado.
        
        Args:
            error: Excepción capturada
            context: Contexto donde ocurrió el error
        """
        self.error(f"Error: {type(error).__name__}: {str(error)}", **context)
    
    def log_analysis_result(self, analysis_type: str, results: Dict[str, Any]):
        """
        Log resultados de análisis.
        
        Args:
            analysis_type: Tipo de análisis (fractal, spectral, statistical)
            results: Resultados del análisis
        """
        self.info(f"Análisis {analysis_type} completado", **results)
    
    def log_data_operation(self, operation: str, filename: str, size: int = None):
        """
        Log operaciones de I/O de datos.
        
        Args:
            operation: Tipo de operación (load, save, export)
            filename: Nombre del archivo
            size: Tamaño de los datos (opcional)
        """
        if size:
            self.info(f"Datos {operation}", filename=filename, size=f"{size} bytes")
        else:
            self.info(f"Datos {operation}", filename=filename)
    
    def create_context_manager(self, operation: str):
        """
        Crear context manager para logging automático de operaciones.
        
        Args:
            operation: Nombre de la operación
            
        Returns:
            Context manager para timing y logging automático
        """
        return LoggingContext(self, operation)


class LoggingContext:
    """Context manager para logging automático con timing."""
    
    def __init__(self, logger: MFSULogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Iniciando {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completado {self.operation}", duration=f"{duration:.4f}s")
        else:
            self.logger.error(f"Error en {self.operation}: {exc_val}", 
                            duration=f"{duration:.4f}s",
                            exception=exc_type.__name__)
        return False


# Instancia global del logger principal
main_logger = MFSULogger("MFSU")

# Funciones de conveniencia para el logger principal
def debug(message: str, **kwargs):
    """Log debug en el logger principal."""
    main_logger.debug(message, **kwargs)

def info(message: str, **kwargs):
    """Log info en el logger principal."""
    main_logger.info(message, **kwargs)

def warning(message: str, **kwargs):
    """Log warning en el logger principal."""
    main_logger.warning(message, **kwargs)

def error(message: str, **kwargs):
    """Log error en el logger principal."""
    main_logger.error(message, **kwargs)

def critical(message: str, **kwargs):
    """Log critical en el logger principal."""
    main_logger.critical(message, **kwargs)

def get_logger(name: str) -> MFSULogger:
    """
    Obtener logger específico por nombre.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Instancia de MFSULogger
    """
    return MFSULogger(name)

def configure_logging(config_path: str = "config.yaml"):
    """
    Configurar el sistema de logging global.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    main_logger.load_config(config_path)


# Ejemplo de uso:
if __name__ == "__main__":
    # Configurar logging
    configure_logging()
    
    # Ejemplo de logging básico
    info("Sistema de logging MFSU inicializado")
    debug("Mensaje de debug", parameter="valor")
    warning("Mensaje de advertencia", context="prueba")
    
    # Ejemplo con context manager
    with main_logger.create_context_manager("operación de prueba"):
        import time
        time.sleep(1)  # Simular trabajo
        info("Trabajo completado dentro del context manager")
    
    # Ejemplo de logging específico de simulación
    params = {"alpha": 0.5, "beta": 0.1, "gamma": 0.01}
    main_logger.log_simulation_start(params)
    
    # Simular algunos pasos de simulación
    for step in range(3):
        metrics = {"energy": 1.0 - 0.1*step, "entropy": 0.5 + 0.1*step}
        main_logger.log_simulation_step(step, step * 0.01, metrics)
    
    main_logger.log_simulation_end(0.03, {"final_energy": 0.7})
