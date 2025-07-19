"""
Módulo de entrada/salida de datos para el Simulador MFSU
Unified Stochastic Fractal Model (MFSU)

Soporta múltiples formatos para datos experimentales, resultados de simulación,
condiciones iniciales y configuraciones del sistema.
"""

import os
import json
import yaml
import h5py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import logging

# Configurar logger
logger = logging.getLogger(__name__)

class DataIOManager:
    """
    Gestor principal para operaciones de entrada/salida de datos
    """
    
    def __init__(self, base_path: str = "data/"):
        """
        Inicializar el gestor de datos
        
        Args:
            base_path: Directorio base para datos
        """
        self.base_path = Path(base_path)
        self.input_path = self.base_path / "input"
        self.output_path = self.base_path / "output"
        self.reference_path = self.base_path / "reference"
        
        # Crear directorios si no existen
        self._create_directories()
        
    def _create_directories(self):
        """Crear estructura de directorios necesaria"""
        directories = [
            self.input_path / "initial_conditions",
            self.input_path / "experimental",
            self.output_path / "simulations",
            self.output_path / "analysis",
            self.output_path / "exports",
            self.reference_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    # ==================== CONDICIONES INICIALES ====================
    
    def save_initial_condition(self, 
                             data: Union[np.ndarray, Dict], 
                             filename: str, 
                             metadata: Optional[Dict] = None) -> bool:
        """
        Guardar condiciones iniciales para el campo ψ
        
        Args:
            data: Campo inicial ψ(x,0) o diccionario con múltiples campos
            filename: Nombre del archivo
            metadata: Metadatos adicionales
        """
        try:
            filepath = self.input_path / "initial_conditions" / filename
            
            if filename.endswith('.json'):
                return self._save_initial_condition_json(data, filepath, metadata)
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                return self._save_initial_condition_hdf5(data, filepath, metadata)
            elif filename.endswith('.npy'):
                return self._save_initial_condition_numpy(data, filepath, metadata)
            else:
                raise ValueError(f"Formato no soportado: {filename}")
                
        except Exception as e:
            logger.error(f"Error guardando condición inicial: {e}")
            return False
    
    def load_initial_condition(self, filename: str) -> Dict[str, Any]:
        """
        Cargar condiciones iniciales
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Diccionario con datos y metadatos
        """
        try:
            filepath = self.input_path / "initial_conditions" / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
                
            if filename.endswith('.json'):
                return self._load_initial_condition_json(filepath)
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                return self._load_initial_condition_hdf5(filepath)
            elif filename.endswith('.npy'):
                return self._load_initial_condition_numpy(filepath)
            else:
                raise ValueError(f"Formato no soportado: {filename}")
                
        except Exception as e:
            logger.error(f"Error cargando condición inicial: {e}")
            return {}
    
    def _save_initial_condition_json(self, data, filepath, metadata):
        """Guardar en formato JSON"""
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'data': {}
        }
        
        if isinstance(data, np.ndarray):
            save_data['data']['psi'] = data.tolist()
            save_data['data']['shape'] = data.shape
            save_data['data']['dtype'] = str(data.dtype)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    save_data['data'][key] = {
                        'values': value.tolist(),
                        'shape': value.shape,
                        'dtype': str(value.dtype)
                    }
                else:
                    save_data['data'][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        return True
    
    def _load_initial_condition_json(self, filepath):
        """Cargar desde formato JSON"""
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        result = {
            'metadata': loaded_data.get('metadata', {}),
            'timestamp': loaded_data.get('timestamp'),
            'data': {}
        }
        
        for key, value in loaded_data['data'].items():
            if isinstance(value, dict) and 'values' in value:
                arr = np.array(value['values'])
                result['data'][key] = arr.reshape(value['shape']).astype(value['dtype'])
            else:
                result['data'][key] = value
                
        return result
    
    def _save_initial_condition_hdf5(self, data, filepath, metadata):
        """Guardar en formato HDF5"""
        with h5py.File(filepath, 'w') as f:
            # Metadatos
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['timestamp'] = datetime.now().isoformat()
            
            if metadata:
                for key, value in metadata.items():
                    meta_grp.attrs[key] = value
            
            # Datos
            data_grp = f.create_group('data')
            if isinstance(data, np.ndarray):
                data_grp.create_dataset('psi', data=data, compression='gzip')
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        data_grp.create_dataset(key, data=value, compression='gzip')
                    else:
                        data_grp.attrs[key] = value
        return True
    
    def _load_initial_condition_hdf5(self, filepath):
        """Cargar desde formato HDF5"""
        result = {'metadata': {}, 'data': {}}
        
        with h5py.File(filepath, 'r') as f:
            # Metadatos
            if 'metadata' in f:
                for key, value in f['metadata'].attrs.items():
                    result['metadata'][key] = value
            
            # Datos
            if 'data' in f:
                data_grp = f['data']
                for key in data_grp.keys():
                    result['data'][key] = np.array(data_grp[key])
                
                # Atributos adicionales
                for key, value in data_grp.attrs.items():
                    result['data'][key] = value
                    
        return result
    
    def _save_initial_condition_numpy(self, data, filepath, metadata):
        """Guardar en formato NumPy"""
        if isinstance(data, np.ndarray):
            np.save(filepath, data)
        else:
            # Para múltiples arrays, usar npz
            filepath = filepath.with_suffix('.npz')
            np.savez_compressed(filepath, **data)
        
        # Guardar metadatos por separado
        if metadata:
            meta_file = filepath.with_suffix('.meta.json')
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        return True
    
    def _load_initial_condition_numpy(self, filepath):
        """Cargar desde formato NumPy"""
        result = {'metadata': {}, 'data': {}}
        
        # Cargar datos
        if filepath.suffix == '.npy':
            result['data']['psi'] = np.load(filepath)
        else:  # .npz
            loaded = np.load(filepath)
            for key in loaded.files:
                result['data'][key] = loaded[key]
        
        # Cargar metadatos si existen
        meta_file = filepath.with_suffix('.meta.json')
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                result['metadata'] = json.load(f)
                
        return result
    
    # ==================== DATOS EXPERIMENTALES ====================
    
    def load_experimental_data(self, filename: str) -> pd.DataFrame:
        """
        Cargar datos experimentales (CSV, Excel, HDF5)
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            DataFrame con los datos experimentales
        """
        try:
            filepath = self.input_path / "experimental" / filename
            
            if filename.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                return pd.read_excel(filepath)
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                return pd.read_hdf(filepath, key='data')
            else:
                raise ValueError(f"Formato experimental no soportado: {filename}")
                
        except Exception as e:
            logger.error(f"Error cargando datos experimentales: {e}")
            return pd.DataFrame()
    
    def save_experimental_data(self, 
                             data: pd.DataFrame, 
                             filename: str, 
                             metadata: Optional[Dict] = None) -> bool:
        """
        Guardar datos experimentales procesados
        """
        try:
            filepath = self.input_path / "experimental" / filename
            
            if filename.endswith('.csv'):
                data.to_csv(filepath, index=False)
            elif filename.endswith(('.xlsx', '.xls')):
                data.to_excel(filepath, index=False)
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                data.to_hdf(filepath, key='data', mode='w')
                
                # Agregar metadatos si es HDF5
                if metadata:
                    with h5py.File(filepath, 'a') as f:
                        meta_grp = f.require_group('metadata')
                        for key, value in metadata.items():
                            meta_grp.attrs[key] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando datos experimentales: {e}")
            return False
    
    # ==================== RESULTADOS DE SIMULACIÓN ====================
    
    def save_simulation_results(self, 
                              results: Dict[str, Any], 
                              simulation_id: str,
                              format_type: str = 'hdf5') -> bool:
        """
        Guardar resultados completos de simulación MFSU
        
        Args:
            results: Diccionario con resultados de simulación
            simulation_id: ID único de la simulación
            format_type: Formato de archivo ('hdf5', 'pickle', 'json')
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mfsu_sim_{simulation_id}_{timestamp}.{format_type}"
            filepath = self.output_path / "simulations" / filename
            
            if format_type == 'hdf5':
                return self._save_simulation_hdf5(results, filepath)
            elif format_type == 'pickle':
                return self._save_simulation_pickle(results, filepath)
            elif format_type == 'json':
                return self._save_simulation_json(results, filepath)
            else:
                raise ValueError(f"Formato no soportado: {format_type}")
                
        except Exception as e:
            logger.error(f"Error guardando simulación: {e}")
            return False
    
    def load_simulation_results(self, filename: str) -> Dict[str, Any]:
        """Cargar resultados de simulación"""
        try:
            filepath = self.output_path / "simulations" / filename
            
            if filename.endswith('.h5') or filename.endswith('.hdf5'):
                return self._load_simulation_hdf5(filepath)
            elif filename.endswith('.pkl') or filename.endswith('.pickle'):
                return self._load_simulation_pickle(filepath)
            elif filename.endswith('.json'):
                return self._load_simulation_json(filepath)
            else:
                raise ValueError(f"Formato no soportado: {filename}")
                
        except Exception as e:
            logger.error(f"Error cargando simulación: {e}")
            return {}
    
    def _save_simulation_hdf5(self, results, filepath):
        """Guardar simulación en HDF5"""
        with h5py.File(filepath, 'w') as f:
            # Metadatos de simulación
            meta_grp = f.create_group('simulation_metadata')
            meta_grp.attrs['timestamp'] = datetime.now().isoformat()
            meta_grp.attrs['mfsu_version'] = '1.0.0'
            
            # Parámetros MFSU
            if 'parameters' in results:
                param_grp = f.create_group('parameters')
                for key, value in results['parameters'].items():
                    param_grp.attrs[key] = value
            
            # Series temporales del campo ψ
            if 'psi_evolution' in results:
                f.create_dataset('psi_evolution', 
                               data=results['psi_evolution'], 
                               compression='gzip')
            
            # Grid espacial y temporal
            if 'x_grid' in results:
                f.create_dataset('x_grid', data=results['x_grid'])
            if 't_grid' in results:
                f.create_dataset('t_grid', data=results['t_grid'])
            
            # Análisis adicional
            if 'analysis' in results:
                analysis_grp = f.create_group('analysis')
                for key, value in results['analysis'].items():
                    if isinstance(value, np.ndarray):
                        analysis_grp.create_dataset(key, data=value, compression='gzip')
                    else:
                        analysis_grp.attrs[key] = value
                        
        return True
    
    def _load_simulation_hdf5(self, filepath):
        """Cargar simulación desde HDF5"""
        results = {}
        
        with h5py.File(filepath, 'r') as f:
            # Metadatos
            if 'simulation_metadata' in f:
                results['metadata'] = dict(f['simulation_metadata'].attrs)
            
            # Parámetros
            if 'parameters' in f:
                results['parameters'] = dict(f['parameters'].attrs)
            
            # Datos principales
            for key in ['psi_evolution', 'x_grid', 't_grid']:
                if key in f:
                    results[key] = np.array(f[key])
            
            # Análisis
            if 'analysis' in f:
                results['analysis'] = {}
                analysis_grp = f['analysis']
                for key in analysis_grp.keys():
                    results['analysis'][key] = np.array(analysis_grp[key])
                for key, value in analysis_grp.attrs.items():
                    results['analysis'][key] = value
                    
        return results
    
    def _save_simulation_pickle(self, results, filepath):
        """Guardar simulación en Pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    
    def _load_simulation_pickle(self, filepath):
        """Cargar simulación desde Pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _save_simulation_json(self, results, filepath):
        """Guardar simulación en JSON (solo metadatos y parámetros)"""
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': results.get('metadata', {}),
            'parameters': results.get('parameters', {}),
            'summary': results.get('summary', {})
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        return True
    
    def _load_simulation_json(self, filepath):
        """Cargar simulación desde JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # ==================== CONFIGURACIÓN ====================
    
    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """Cargar configuración del sistema"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                config_path = self.base_path.parent / config_file
            
            with open(config_path, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self._default_config()
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.yaml") -> bool:
        """Guardar configuración"""
        try:
            config_path = self.base_path.parent / filename
            
            with open(config_path, 'w') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif filename.endswith('.json'):
                    json.dump(config, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            return False
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del MFSU"""
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
                    'max_time': 10.0
                }
            },
            'applications': {
                'superconductivity': {
                    'temperature_range': [1, 300],
                    'material_parameters': {}
                },
                'gas_dynamics': {
                    'reynolds_number': 1000,
                    'mach_number': 0.3
                },
                'cosmology': {
                    'hubble_constant': 70,
                    'omega_matter': 0.3
                }
            }
        }
    
    # ==================== UTILIDADES ====================
    
    def list_files(self, category: str) -> List[str]:
        """
        Listar archivos disponibles en una categoría
        
        Args:
            category: 'initial_conditions', 'experimental', 'simulations', etc.
        """
        try:
            if category == 'initial_conditions':
                path = self.input_path / "initial_conditions"
            elif category == 'experimental':
                path = self.input_path / "experimental"
            elif category == 'simulations':
                path = self.output_path / "simulations"
            elif category == 'analysis':
                path = self.output_path / "analysis"
            else:
                raise ValueError(f"Categoría no válida: {category}")
            
            if path.exists():
                return [f.name for f in path.iterdir() if f.is_file()]
            return []
            
        except Exception as e:
            logger.error(f"Error listando archivos: {e}")
            return []
    
    def export_data(self, 
                   data: Union[np.ndarray, pd.DataFrame, Dict], 
                   filename: str,
                   export_format: str = 'csv') -> bool:
        """
        Exportar datos en formato específico para análisis externo
        
        Args:
            data: Datos a exportar
            filename: Nombre del archivo
            export_format: Formato ('csv', 'excel', 'json', 'hdf5')
        """
        try:
            export_path = self.output_path / "exports"
            export_path.mkdir(exist_ok=True)
            
            filepath = export_path / f"{filename}.{export_format}"
            
            if export_format == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            elif export_format == 'excel' and isinstance(data, pd.DataFrame):
                data.to_excel(filepath, index=False)
            elif export_format == 'json':
                with open(filepath, 'w') as f:
                    if isinstance(data, dict):
                        json.dump(data, f, indent=2, default=str)
                    else:
                        json.dump(data.tolist(), f, indent=2)
            elif export_format == 'hdf5':
                with h5py.File(filepath, 'w') as f:
                    if isinstance(data, np.ndarray):
                        f.create_dataset('data', data=data, compression='gzip')
                    elif isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, np.ndarray):
                                f.create_dataset(key, data=value, compression='gzip')
                            else:
                                f.attrs[key] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error exportando datos: {e}")
            return False
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """
        Limpiar archivos antiguos de simulaciones
        
        Args:
            days: Días de antigüedad para considerar archivo como antiguo
            
        Returns:
            Número de archivos eliminados
        """
        try:
            import time
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            deleted_count = 0
            
            sim_path = self.output_path / "simulations"
            for file_path in sim_path.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    
            logger.info(f"Eliminados {deleted_count} archivos antiguos")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error en limpieza de archivos: {e}")
            return 0

# Funciones de conveniencia
def create_gaussian_initial_condition(x: np.ndarray, 
                                    x0: float = 0.0, 
                                    sigma: float = 1.0,
                                    amplitude: float = 1.0) -> np.ndarray:
    """
    Crear condición inicial gaussiana para ψ(x,0)
    
    Args:
        x: Grid espacial
        x0: Centro del paquete gaussiano
        sigma: Ancho del paquete
        amplitude: Amplitud máxima
        
    Returns:
        Campo inicial ψ(x,0)
    """
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def create_soliton_initial_condition(x: np.ndarray,
                                   x0: float = 0.0,
                                   width: float = 1.0,
                                   amplitude: float = 1.0) -> np.ndarray:
    """
    Crear condición inicial tipo solitón
    
    Args:
        x: Grid espacial
        x0: Centro del solitón
        width: Ancho característico
        amplitude: Amplitud máxima
        
    Returns:
        Campo inicial ψ(x,0)
    """
    return amplitude / np.cosh((x - x0) / width)

# Instancia global del gestor
data_manager = DataIOManager()
