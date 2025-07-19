"""
Pruebas para el módulo data_io.py del simulador MFSU
Testea funcionalidades de entrada/salida de datos, configuraciones, 
condiciones iniciales y datos experimentales.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
import h5py
import yaml
from unittest.mock import patch, mock_open

# Importar el módulo a testear (ajustar según la estructura real)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from utils.data_io import (
    load_initial_conditions, save_simulation_data, load_experimental_data,
    export_results, load_config, save_config, validate_data_format,
    create_simulation_metadata, load_benchmark_data
)

class TestInitialConditions:
    """Pruebas para carga de condiciones iniciales"""
    
    @pytest.fixture
    def temp_dir(self):
        """Directorio temporal para pruebas"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def gaussian_packet_data(self):
        """Datos de prueba para paquete gaussiano"""
        return {
            "type": "gaussian_packet",
            "parameters": {
                "amplitude": 1.0,
                "width": 2.0,
                "center": [0.0, 0.0],
                "momentum": [1.0, 0.0]
            },
            "grid": {
                "size": [100, 100],
                "spacing": 0.1,
                "domain": [[-5, 5], [-5, 5]]
            }
        }
    
    @pytest.fixture
    def soliton_profile_data(self):
        """Datos de prueba para perfil de solitón"""
        return {
            "type": "soliton_profile",
            "parameters": {
                "amplitude": 2.0,
                "velocity": 0.5,
                "width": 1.0,
                "position": 0.0
            },
            "mfsu_params": {
                "alpha": 0.5,
                "beta": 0.1,
                "gamma": 0.01,
                "hurst": 0.7
            }
        }
    
    def test_load_gaussian_packet(self, temp_dir, gaussian_packet_data):
        """Test carga de paquete gaussiano"""
        # Crear archivo temporal
        json_file = temp_dir / "gaussian_packet.json"
        with open(json_file, 'w') as f:
            json.dump(gaussian_packet_data, f)
        
        # Cargar datos
        loaded_data = load_initial_conditions(str(json_file))
        
        assert loaded_data["type"] == "gaussian_packet"
        assert loaded_data["parameters"]["amplitude"] == 1.0
        assert loaded_data["parameters"]["width"] == 2.0
        assert len(loaded_data["parameters"]["center"]) == 2
    
    def test_load_soliton_profile(self, temp_dir, soliton_profile_data):
        """Test carga de perfil de solitón"""
        json_file = temp_dir / "soliton_profile.json"
        with open(json_file, 'w') as f:
            json.dump(soliton_profile_data, f)
        
        loaded_data = load_initial_conditions(str(json_file))
        
        assert loaded_data["type"] == "soliton_profile"
        assert loaded_data["parameters"]["amplitude"] == 2.0
        assert "mfsu_params" in loaded_data
        assert loaded_data["mfsu_params"]["alpha"] == 0.5
    
    def test_load_nonexistent_file(self):
        """Test manejo de archivo inexistente"""
        with pytest.raises(FileNotFoundError):
            load_initial_conditions("nonexistent_file.json")
    
    def test_invalid_json_format(self, temp_dir):
        """Test manejo de JSON inválido"""
        invalid_json = temp_dir / "invalid.json"
        with open(invalid_json, 'w') as f:
            f.write("{invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            load_initial_conditions(str(invalid_json))

class TestSimulationData:
    """Pruebas para guardado de datos de simulación"""
    
    @pytest.fixture
    def sample_simulation_data(self):
        """Datos de simulación de prueba"""
        return {
            "psi": np.random.complex128((50, 100, 100)),  # [time, x, y]
            "time": np.linspace(0, 5, 50),
            "grid_x": np.linspace(-5, 5, 100),
            "grid_y": np.linspace(-5, 5, 100),
            "parameters": {
                "alpha": 0.5,
                "beta": 0.1,
                "gamma": 0.01,
                "hurst": 0.7
            },
            "metadata": {
                "simulation_id": "test_sim_001",
                "timestamp": "2025-07-19T10:30:00",
                "version": "1.0.0"
            }
        }
    
    def test_save_simulation_hdf5(self, tmp_path, sample_simulation_data):
        """Test guardado en formato HDF5"""
        output_file = tmp_path / "simulation_output.h5"
        
        save_simulation_data(sample_simulation_data, str(output_file))
        
        # Verificar que el archivo existe
        assert output_file.exists()
        
        # Verificar contenido
        with h5py.File(output_file, 'r') as f:
            assert 'psi' in f
            assert 'time' in f
            assert 'parameters' in f.attrs
            assert f['psi'].shape == (50, 100, 100)
    
    def test_save_simulation_numpy(self, tmp_path, sample_simulation_data):
        """Test guardado en formato NumPy"""
        output_file = tmp_path / "simulation_output.npz"
        
        save_simulation_data(sample_simulation_data, str(output_file), format='npz')
        
        assert output_file.exists()
        
        # Verificar contenido
        loaded_data = np.load(output_file, allow_pickle=True)
        assert 'psi' in loaded_data
        assert 'time' in loaded_data
        assert loaded_data['psi'].shape == (50, 100, 100)
    
    def test_invalid_save_format(self, tmp_path, sample_simulation_data):
        """Test formato de guardado inválido"""
        output_file = tmp_path / "output.invalid"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            save_simulation_data(sample_simulation_data, str(output_file))

class TestExperimentalData:
    """Pruebas para carga de datos experimentales"""
    
    @pytest.fixture
    def superconductivity_data(self):
        """Datos de superconductividad de prueba"""
        return pd.DataFrame({
            'Temperature': np.linspace(1, 300, 100),
            'Resistance': np.random.exponential(scale=10, size=100),
            'Critical_Current': np.random.normal(loc=50, scale=5, size=100)
        })
    
    @pytest.fixture
    def turbulence_data(self):
        """Datos de turbulencia de prueba"""
        return pd.DataFrame({
            'time': np.linspace(0, 10, 1000),
            'velocity_x': np.random.normal(size=1000),
            'velocity_y': np.random.normal(size=1000),
            'pressure': np.random.normal(loc=101325, scale=1000, size=1000)
        })
    
    def test_load_superconductivity_csv(self, tmp_path, superconductivity_data):
        """Test carga de datos de superconductividad"""
        csv_file = tmp_path / "RvT_300K.csv"
        superconductivity_data.to_csv(csv_file, index=False)
        
        loaded_data = load_experimental_data(str(csv_file), data_type="superconductivity")
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert 'Temperature' in loaded_data.columns
        assert 'Resistance' in loaded_data.columns
        assert len(loaded_data) == 100
    
    def test_load_turbulence_csv(self, tmp_path, turbulence_data):
        """Test carga de datos de turbulencia"""
        csv_file = tmp_path / "turbulence_data.csv"
        turbulence_data.to_csv(csv_file, index=False)
        
        loaded_data = load_experimental_data(str(csv_file), data_type="turbulence")
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert 'velocity_x' in loaded_data.columns
        assert 'velocity_y' in loaded_data.columns
        assert len(loaded_data) == 1000
    
    def test_unsupported_data_type(self, tmp_path):
        """Test tipo de datos no soportado"""
        dummy_file = tmp_path / "dummy.csv"
        pd.DataFrame({'col1': [1, 2, 3]}).to_csv(dummy_file, index=False)
        
        with pytest.raises(ValueError, match="Unsupported data type"):
            load_experimental_data(str(dummy_file), data_type="unknown")

class TestConfigurationFiles:
    """Pruebas para archivos de configuración"""
    
    @pytest.fixture
    def sample_config(self):
        """Configuración de prueba"""
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
                    'material_parameters': {
                        'coherence_length': 1e-6,
                        'penetration_depth': 1e-7
                    }
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
    
    def test_load_yaml_config(self, tmp_path, sample_config):
        """Test carga de configuración YAML"""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config['simulation']['default_parameters']['alpha'] == 0.5
        assert loaded_config['applications']['superconductivity']['temperature_range'] == [1, 300]
        assert loaded_config['applications']['cosmology']['hubble_constant'] == 70
    
    def test_save_yaml_config(self, tmp_path, sample_config):
        """Test guardado de configuración YAML"""
        config_file = tmp_path / "saved_config.yaml"
        
        save_config(sample_config, str(config_file))
        
        assert config_file.exists()
        
        # Verificar contenido guardado
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['simulation']['numerical']['dt'] == 0.01
        assert loaded_config['applications']['gas_dynamics']['reynolds_number'] == 1000
    
    def test_invalid_yaml_config(self, tmp_path):
        """Test configuración YAML inválida"""
        invalid_yaml = tmp_path / "invalid_config.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(invalid_yaml))

class TestDataValidation:
    """Pruebas para validación de datos"""
    
    def test_validate_mfsu_parameters(self):
        """Test validación de parámetros MFSU"""
        valid_params = {
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.7
        }
        
        assert validate_data_format(valid_params, 'mfsu_parameters') == True
    
    def test_invalid_mfsu_parameters(self):
        """Test parámetros MFSU inválidos"""
        # Alpha fuera de rango
        invalid_params1 = {
            'alpha': 2.5,  # Debe estar en (0, 2]
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.7
        }
        
        assert validate_data_format(invalid_params1, 'mfsu_parameters') == False
        
        # Hurst fuera de rango
        invalid_params2 = {
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 1.2  # Debe estar en (0, 1)
        }
        
        assert validate_data_format(invalid_params2, 'mfsu_parameters') == False
    
    def test_validate_simulation_grid(self):
        """Test validación de grilla de simulación"""
        valid_grid = {
            'size': [100, 100],
            'spacing': 0.1,
            'domain': [[-5, 5], [-5, 5]]
        }
        
        assert validate_data_format(valid_grid, 'simulation_grid') == True
    
    def test_invalid_simulation_grid(self):
        """Test grilla de simulación inválida"""
        invalid_grid = {
            'size': [100],  # Debe ser 2D o 3D
            'spacing': 0.1,
            'domain': [[-5, 5]]
        }
        
        assert validate_data_format(invalid_grid, 'simulation_grid') == False

class TestMetadataAndBenchmarks:
    """Pruebas para metadatos y datos de benchmark"""
    
    def test_create_simulation_metadata(self):
        """Test creación de metadatos de simulación"""
        params = {
            'alpha': 0.5,
            'beta': 0.1,
            'gamma': 0.01,
            'hurst': 0.7
        }
        
        metadata = create_simulation_metadata(params, simulation_type="superconductivity")
        
        assert 'simulation_id' in metadata
        assert 'timestamp' in metadata
        assert 'parameters' in metadata
        assert metadata['simulation_type'] == "superconductivity"
        assert metadata['mfsu_version'] is not None
    
    def test_load_benchmark_data(self, tmp_path):
        """Test carga de datos de benchmark"""
        benchmark_data = {
            "gaussian_evolution": {
                "initial_width": 1.0,
                "final_width": 1.41,
                "evolution_time": 1.0,
                "analytical_solution": [1.0, 0.95, 0.85, 0.7]
            },
            "soliton_stability": {
                "amplitude": 1.0,
                "velocity": 0.5,
                "stability_time": 10.0,
                "max_deviation": 0.01
            }
        }
        
        benchmark_file = tmp_path / "benchmark_solutions.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f)
        
        loaded_benchmarks = load_benchmark_data(str(benchmark_file))
        
        assert 'gaussian_evolution' in loaded_benchmarks
        assert 'soliton_stability' in loaded_benchmarks
        assert loaded_benchmarks['gaussian_evolution']['initial_width'] == 1.0

class TestExportResults:
    """Pruebas para exportación de resultados"""
    
    @pytest.fixture
    def analysis_results(self):
        """Resultados de análisis de prueba"""
        return {
            'fractal_dimension': 1.67,
            'spectral_exponent': -1.8,
            'statistical_moments': {
                'mean': 0.05,
                'std': 0.23,
                'skewness': 0.12,
                'kurtosis': 2.98
            },
            'correlation_length': 2.34,
            'energy_evolution': np.linspace(1, 0.1, 100)
        }
    
    def test_export_to_json(self, tmp_path, analysis_results):
        """Test exportación a JSON"""
        output_file = tmp_path / "results.json"
        
        export_results(analysis_results, str(output_file), format='json')
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['fractal_dimension'] == 1.67
        assert 'statistical_moments' in loaded_results
    
    def test_export_to_csv(self, tmp_path):
        """Test exportación a CSV"""
        tabular_data = pd.DataFrame({
            'time': np.linspace(0, 10, 50),
            'energy': np.random.exponential(size=50),
            'entropy': np.random.normal(size=50)
        })
        
        output_file = tmp_path / "timeseries.csv"
        
        export_results(tabular_data, str(output_file), format='csv')
        
        assert output_file.exists()
        
        loaded_df = pd.read_csv(output_file)
        assert len(loaded_df) == 50
        assert 'time' in loaded_df.columns

# Test de integración
class TestIntegrationDataIO:
    """Pruebas de integración para el módulo completo"""
    
    def test_full_workflow(self, tmp_path):
        """Test del flujo completo de datos"""
        # 1. Cargar configuración
        config_data = {
            'simulation': {
                'default_parameters': {
                    'alpha': 0.5, 'beta': 0.1, 'gamma': 0.01, 'hurst': 0.7
                }
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        assert config is not None
        
        # 2. Crear condiciones iniciales
        initial_conditions = {
            "type": "test_condition",
            "parameters": config['simulation']['default_parameters']
        }
        
        ic_file = tmp_path / "initial.json"
        with open(ic_file, 'w') as f:
            json.dump(initial_conditions, f)
        
        loaded_ic = load_initial_conditions(str(ic_file))
        assert loaded_ic['parameters']['alpha'] == 0.5
        
        # 3. Simular guardado de resultados
        sim_data = {
            'psi': np.random.complex128((10, 20, 20)),
            'time': np.linspace(0, 1, 10),
            'parameters': config['simulation']['default_parameters']
        }
        
        output_file = tmp_path / "simulation.h5"
        save_simulation_data(sim_data, str(output_file))
        
        assert output_file.exists()
        
        # 4. Verificar integridad de datos guardados
        with h5py.File(output_file, 'r') as f:
            assert f['psi'].shape == (10, 20, 20)


if __name__ == "__main__":
    pytest.main([__file__])
