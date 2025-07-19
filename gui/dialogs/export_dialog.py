import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QLineEdit, QPushButton, QComboBox, 
                                QCheckBox, QTextEdit, QFileDialog, QMessageBox,
                                QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar,
                                QTabWidget, QWidget, QFormLayout, QListWidget,
                                QListWidgetItem, QSplitter)
    from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
    from PyQt5.QtGui import QFont, QIcon
except ImportError:
    from PySide2.QtWidgets import *
    from PySide2.QtCore import Qt, Signal as pyqtSignal, QThread, QTimer
    from PySide2.QtGui import QFont, QIcon

class ExportWorkerThread(QThread):
    """Thread para realizar exportaciones pesadas sin bloquear la GUI"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    export_completed = pyqtSignal(bool, str)
    
    def __init__(self, export_config, simulation_data):
        super().__init__()
        self.export_config = export_config
        self.simulation_data = simulation_data
        
    def run(self):
        try:
            self.perform_export()
            self.export_completed.emit(True, "Exportación completada exitosamente")
        except Exception as e:
            self.export_completed.emit(False, f"Error en exportación: {str(e)}")
    
    def perform_export(self):
        """Ejecuta la exportación según la configuración"""
        total_steps = len(self.export_config['formats']) * 2
        current_step = 0
        
        for format_type in self.export_config['formats']:
            self.status_updated.emit(f"Exportando en formato {format_type}...")
            
            if format_type == 'csv':
                self.export_csv()
            elif format_type == 'json':
                self.export_json()
            elif format_type == 'hdf5':
                self.export_hdf5()
            elif format_type == 'matlab':
                self.export_matlab()
            elif format_type == 'pdf_report':
                self.export_pdf_report()
            elif format_type == 'numpy':
                self.export_numpy()
                
            current_step += 1
            self.progress_updated.emit(int((current_step / total_steps) * 100))
        
        self.status_updated.emit("Finalizando exportación...")
        
    def export_csv(self):
        """Exporta datos a formato CSV"""
        output_path = self.export_config['output_path']
        
        # Exportar datos de la simulación principal
        if 'psi_field' in self.simulation_data:
            psi_data = self.simulation_data['psi_field']
            df = pd.DataFrame({
                'time': self.simulation_data.get('time', []),
                'real_part': np.real(psi_data).flatten(),
                'imag_part': np.imag(psi_data).flatten(),
                'magnitude': np.abs(psi_data).flatten()
            })
            df.to_csv(f"{output_path}/mfsu_field_data.csv", index=False)
        
        # Exportar parámetros
        if 'parameters' in self.simulation_data:
            params_df = pd.DataFrame([self.simulation_data['parameters']])
            params_df.to_csv(f"{output_path}/simulation_parameters.csv", index=False)
            
    def export_json(self):
        """Exporta datos a formato JSON"""
        output_path = self.export_config['output_path']
        
        # Convertir datos complejos a formato exportable
        export_data = {}
        for key, value in self.simulation_data.items():
            if isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    export_data[key] = {
                        'real': value.real.tolist(),
                        'imag': value.imag.tolist(),
                        'shape': value.shape
                    }
                else:
                    export_data[key] = value.tolist()
            else:
                export_data[key] = value
        
        with open(f"{output_path}/mfsu_simulation.json", 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_hdf5(self):
        """Exporta datos a formato HDF5"""
        output_path = self.export_config['output_path']
        
        with h5py.File(f"{output_path}/mfsu_simulation.h5", 'w') as f:
            # Crear grupos principales
            sim_group = f.create_group('simulation')
            params_group = f.create_group('parameters')
            analysis_group = f.create_group('analysis')
            
            # Guardar datos de simulación
            for key, value in self.simulation_data.items():
                if isinstance(value, np.ndarray):
                    sim_group.create_dataset(key, data=value)
                elif isinstance(value, dict) and key == 'parameters':
                    for param_key, param_value in value.items():
                        params_group.attrs[param_key] = param_value
                        
            # Metadatos
            f.attrs['export_date'] = datetime.now().isoformat()
            f.attrs['mfsu_version'] = '1.0.0'
            f.attrs['equation'] = '∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)'
    
    def export_matlab(self):
        """Exporta datos a formato MATLAB (.mat)"""
        from scipy.io import savemat
        output_path = self.export_config['output_path']
        
        # Preparar datos para MATLAB
        matlab_data = {}
        for key, value in self.simulation_data.items():
            if isinstance(value, np.ndarray):
                matlab_data[key] = value
            elif isinstance(value, (int, float, complex)):
                matlab_data[key] = value
                
        savemat(f"{output_path}/mfsu_simulation.mat", matlab_data)
    
    def export_pdf_report(self):
        """Genera un reporte PDF completo"""
        output_path = self.export_config['output_path']
        
        with PdfPages(f"{output_path}/mfsu_report.pdf") as pdf:
            # Página 1: Información general
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.text(0.1, 0.9, 'MFSU Simulation Report', fontsize=20, fontweight='bold')
            ax.text(0.1, 0.8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12)
            ax.text(0.1, 0.7, 'Equation: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)', fontsize=10)
            
            # Mostrar parámetros
            if 'parameters' in self.simulation_data:
                y_pos = 0.6
                for key, value in self.simulation_data['parameters'].items():
                    ax.text(0.1, y_pos, f'{key}: {value}', fontsize=10)
                    y_pos -= 0.05
                    
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Páginas adicionales con gráficos
            if 'psi_field' in self.simulation_data:
                self.add_field_plots_to_pdf(pdf)
    
    def add_field_plots_to_pdf(self, pdf):
        """Añade gráficos del campo al PDF"""
        psi_field = self.simulation_data['psi_field']
        
        # Gráfico de magnitud
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if len(psi_field.shape) == 2:  # 1D + tiempo
            im1 = ax1.imshow(np.abs(psi_field).T, aspect='auto', origin='lower')
            ax1.set_title('|ψ(x,t)|')
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Posición')
            plt.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(np.angle(psi_field).T, aspect='auto', origin='lower')
            ax2.set_title('arg(ψ(x,t))')
            ax2.set_xlabel('Tiempo')
            ax2.set_ylabel('Posición')
            plt.colorbar(im2, ax=ax2)
            
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def export_numpy(self):
        """Exporta arrays de NumPy"""
        output_path = self.export_config['output_path']
        
        for key, value in self.simulation_data.items():
            if isinstance(value, np.ndarray):
                np.save(f"{output_path}/{key}.npy", value)


class ExportDialog(QDialog):
    """Diálogo para exportar resultados de simulaciones MFSU"""
    
    def __init__(self, simulation_data=None, parent=None):
        super().__init__(parent)
        self.simulation_data = simulation_data or {}
        self.export_worker = None
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        self.setWindowTitle("Exportar Resultados MFSU")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        # Layout principal
        main_layout = QVBoxLayout(self)
        
        # Pestañas
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Pestaña de formatos
        self.setup_formats_tab()
        
        # Pestaña de configuración
        self.setup_config_tab()
        
        # Pestaña de contenido
        self.setup_content_tab()
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Label de estado
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Vista Previa")
        self.export_button = QPushButton("Exportar")
        self.cancel_button = QPushButton("Cancelar")
        
        buttons_layout.addWidget(self.preview_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(buttons_layout)
        
    def setup_formats_tab(self):
        """Configura la pestaña de formatos de exportación"""
        formats_tab = QWidget()
        layout = QVBoxLayout(formats_tab)
        
        # Grupo de formatos
        formats_group = QGroupBox("Formatos de Exportación")
        formats_layout = QGridLayout(formats_group)
        
        self.format_checkboxes = {}
        formats = [
            ('csv', 'CSV (Comma Separated Values)', 'Datos tabulares, compatible con Excel'),
            ('json', 'JSON (JavaScript Object Notation)', 'Datos estructurados, fácil de leer'),
            ('hdf5', 'HDF5 (Hierarchical Data Format)', 'Formato científico, alta eficiencia'),
            ('matlab', 'MATLAB (.mat)', 'Compatible con MATLAB/Octave'),
            ('pdf_report', 'Reporte PDF', 'Documento completo con gráficos'),
            ('numpy', 'Arrays NumPy (.npy)', 'Arrays nativos de Python/NumPy')
        ]
        
        for i, (fmt, name, description) in enumerate(formats):
            checkbox = QCheckBox(name)
            checkbox.setToolTip(description)
            if fmt in ['csv', 'json']:  # Formatos por defecto
                checkbox.setChecked(True)
            self.format_checkboxes[fmt] = checkbox
            
            formats_layout.addWidget(checkbox, i // 2, i % 2)
            
        layout.addWidget(formats_group)
        
        # Configuración de salida
        output_group = QGroupBox("Directorio de Salida")
        output_layout = QFormLayout(output_group)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(os.path.join(os.getcwd(), "data/output/exports"))
        
        self.browse_button = QPushButton("Examinar...")
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(self.browse_button)
        
        output_layout.addRow("Directorio:", path_layout)
        
        # Nombre base de archivos
        self.filename_prefix_edit = QLineEdit()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename_prefix_edit.setText(f"mfsu_export_{timestamp}")
        output_layout.addRow("Prefijo de archivo:", self.filename_prefix_edit)
        
        layout.addWidget(output_group)
        
        self.tab_widget.addTab(formats_tab, "Formatos")
        
    def setup_config_tab(self):
        """Configura la pestaña de configuración avanzada"""
        config_tab = QWidget()
        layout = QVBoxLayout(config_tab)
        
        # Configuración de precisión
        precision_group = QGroupBox("Configuración de Precisión")
        precision_layout = QFormLayout(precision_group)
        
        self.float_precision_spin = QSpinBox()
        self.float_precision_spin.setRange(1, 16)
        self.float_precision_spin.setValue(8)
        precision_layout.addRow("Precisión decimal:", self.float_precision_spin)
        
        self.scientific_notation_check = QCheckBox("Usar notación científica")
        precision_layout.addRow("", self.scientific_notation_check)
        
        layout.addWidget(precision_group)
        
        # Configuración de compresión
        compression_group = QGroupBox("Compresión")
        compression_layout = QFormLayout(compression_group)
        
        self.compress_data_check = QCheckBox("Comprimir datos")
        self.compress_data_check.setChecked(True)
        compression_layout.addRow("", self.compress_data_check)
        
        self.compression_level_spin = QSpinBox()
        self.compression_level_spin.setRange(1, 9)
        self.compression_level_spin.setValue(6)
        compression_layout.addRow("Nivel de compresión:", self.compression_level_spin)
        
        layout.addWidget(compression_group)
        
        # Metadatos
        metadata_group = QGroupBox("Metadatos")
        metadata_layout = QFormLayout(metadata_group)
        
        self.include_metadata_check = QCheckBox("Incluir metadatos")
        self.include_metadata_check.setChecked(True)
        metadata_layout.addRow("", self.include_metadata_check)
        
        self.author_edit = QLineEdit()
        self.author_edit.setText(os.getenv('USER', 'Usuario'))
        metadata_layout.addRow("Autor:", self.author_edit)
        
        self.description_text = QTextEdit()
        self.description_text.setMaximumHeight(80)
        self.description_text.setPlainText("Exportación de resultados de simulación MFSU")
        metadata_layout.addRow("Descripción:", self.description_text)
        
        layout.addWidget(metadata_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(config_tab, "Configuración")
        
    def setup_content_tab(self):
        """Configura la pestaña de selección de contenido"""
        content_tab = QWidget()
        layout = QVBoxLayout(content_tab)
        
        # Lista de datos disponibles
        available_group = QGroupBox("Datos Disponibles")
        available_layout = QVBoxLayout(available_group)
        
        self.content_list = QListWidget()
        self.populate_content_list()
        available_layout.addWidget(self.content_list)
        
        # Botones de selección
        select_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Seleccionar Todo")
        self.select_none_button = QPushButton("Deseleccionar Todo")
        
        select_layout.addWidget(self.select_all_button)
        select_layout.addWidget(self.select_none_button)
        select_layout.addStretch()
        
        available_layout.addLayout(select_layout)
        layout.addWidget(available_group)
        
        # Información del contenido
        info_group = QGroupBox("Información del Contenido Seleccionado")
        info_layout = QVBoxLayout(info_group)
        
        self.content_info_text = QTextEdit()
        self.content_info_text.setReadOnly(True)
        self.content_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.content_info_text)
        
        layout.addWidget(info_group)
        
        self.tab_widget.addTab(content_tab, "Contenido")
        
    def populate_content_list(self):
        """Llena la lista con el contenido disponible para exportar"""
        self.content_list.clear()
        
        if not self.simulation_data:
            item = QListWidgetItem("No hay datos de simulación disponibles")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.content_list.addItem(item)
            return
            
        for key, value in self.simulation_data.items():
            item = QListWidgetItem(key)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, key)
            
            # Información adicional sobre el tipo de dato
            if isinstance(value, np.ndarray):
                shape_str = "x".join(map(str, value.shape))
                dtype_str = str(value.dtype)
                item.setText(f"{key} (Array {shape_str}, {dtype_str})")
            elif isinstance(value, dict):
                item.setText(f"{key} (Diccionario, {len(value)} elementos)")
            elif isinstance(value, list):
                item.setText(f"{key} (Lista, {len(value)} elementos)")
            else:
                item.setText(f"{key} ({type(value).__name__})")
                
            self.content_list.addItem(item)
            
    def connect_signals(self):
        """Conecta las señales de la interfaz"""
        self.browse_button.clicked.connect(self.browse_output_path)
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button.clicked.connect(self.reject)
        self.preview_button.clicked.connect(self.show_preview)
        
        self.select_all_button.clicked.connect(self.select_all_content)
        self.select_none_button.clicked.connect(self.select_no_content)
        
        self.content_list.itemChanged.connect(self.update_content_info)
        
        # Validación en tiempo real
        self.output_path_edit.textChanged.connect(self.validate_export_config)
        
    def browse_output_path(self):
        """Abre el diálogo para seleccionar directorio de salida"""
        path = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar Directorio de Salida",
            self.output_path_edit.text()
        )
        if path:
            self.output_path_edit.setText(path)
            
    def select_all_content(self):
        """Selecciona todo el contenido"""
        for i in range(self.content_list.count()):
            item = self.content_list.item(i)
            if item.flags() & Qt.ItemIsEnabled:
                item.setCheckState(Qt.Checked)
                
    def select_no_content(self):
        """Deselecciona todo el contenido"""
        for i in range(self.content_list.count()):
            item = self.content_list.item(i)
            if item.flags() & Qt.ItemIsEnabled:
                item.setCheckState(Qt.Unchecked)
                
    def update_content_info(self):
        """Actualiza la información del contenido seleccionado"""
        selected_items = []
        total_size = 0
        
        for i in range(self.content_list.count()):
            item = self.content_list.item(i)
            if item.checkState() == Qt.Checked:
                key = item.data(Qt.UserRole)
                if key and key in self.simulation_data:
                    selected_items.append(key)
                    value = self.simulation_data[key]
                    if isinstance(value, np.ndarray):
                        total_size += value.nbytes
                        
        info_text = f"Elementos seleccionados: {len(selected_items)}\n"
        info_text += f"Tamaño estimado: {total_size / (1024*1024):.2f} MB\n"
        info_text += f"Items: {', '.join(selected_items)}"
        
        self.content_info_text.setPlainText(info_text)
        
    def validate_export_config(self):
        """Valida la configuración de exportación"""
        path = self.output_path_edit.text()
        is_valid = True
        status_msg = ""
        
        if not path:
            is_valid = False
            status_msg = "Debe especificar un directorio de salida"
        elif not os.path.exists(os.path.dirname(path)):
            is_valid = False
            status_msg = "El directorio padre no existe"
            
        # Verificar formatos seleccionados
        selected_formats = [fmt for fmt, checkbox in self.format_checkboxes.items() 
                          if checkbox.isChecked()]
        if not selected_formats:
            is_valid = False
            status_msg = "Debe seleccionar al menos un formato de exportación"
            
        self.export_button.setEnabled(is_valid)
        self.status_label.setText(status_msg)
        
    def get_export_config(self):
        """Obtiene la configuración actual de exportación"""
        # Crear directorio si no existe
        output_path = self.output_path_edit.text()
        os.makedirs(output_path, exist_ok=True)
        
        # Formatos seleccionados
        selected_formats = [fmt for fmt, checkbox in self.format_checkboxes.items() 
                          if checkbox.isChecked()]
        
        # Contenido seleccionado
        selected_content = {}
        for i in range(self.content_list.count()):
            item = self.content_list.item(i)
            if item.checkState() == Qt.Checked:
                key = item.data(Qt.UserRole)
                if key and key in self.simulation_data:
                    selected_content[key] = self.simulation_data[key]
                    
        return {
            'output_path': output_path,
            'filename_prefix': self.filename_prefix_edit.text(),
            'formats': selected_formats,
            'precision': self.float_precision_spin.value(),
            'scientific_notation': self.scientific_notation_check.isChecked(),
            'compress': self.compress_data_check.isChecked(),
            'compression_level': self.compression_level_spin.value(),
            'include_metadata': self.include_metadata_check.isChecked(),
            'author': self.author_edit.text(),
            'description': self.description_text.toPlainText(),
            'content': selected_content
        }
        
    def show_preview(self):
        """Muestra una vista previa de la exportación"""
        config = self.get_export_config()
        
        preview_text = "VISTA PREVIA DE EXPORTACIÓN\n"
        preview_text += "=" * 50 + "\n\n"
        preview_text += f"Directorio de salida: {config['output_path']}\n"
        preview_text += f"Prefijo de archivos: {config['filename_prefix']}\n"
        preview_text += f"Formatos: {', '.join(config['formats'])}\n\n"
        
        preview_text += "CONTENIDO A EXPORTAR:\n"
        preview_text += "-" * 30 + "\n"
        for key, value in config['content'].items():
            if isinstance(value, np.ndarray):
                preview_text += f"• {key}: Array {value.shape} ({value.dtype})\n"
            else:
                preview_text += f"• {key}: {type(value).__name__}\n"
                
        # Mostrar en diálogo
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Vista Previa de Exportación")
        preview_dialog.setModal(True)
        preview_dialog.resize(500, 400)
        
        layout = QVBoxLayout(preview_dialog)
        text_edit = QTextEdit()
        text_edit.setPlainText(preview_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(preview_dialog.accept)
        layout.addWidget(close_button)
        
        preview_dialog.exec_()
        
    def start_export(self):
        """Inicia el proceso de exportación"""
        config = self.get_export_config()
        
        if not config['content']:
            QMessageBox.warning(self, "Advertencia", 
                              "No hay contenido seleccionado para exportar.")
            return
            
        # Configurar UI para exportación
        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Crear y configurar el worker thread
        self.export_worker = ExportWorkerThread(config, config['content'])
        self.export_worker.progress_updated.connect(self.progress_bar.setValue)
        self.export_worker.status_updated.connect(self.status_label.setText)
        self.export_worker.export_completed.connect(self.on_export_completed)
        
        # Iniciar exportación
        self.export_worker.start()
        
    def on_export_completed(self, success, message):
        """Maneja la finalización de la exportación"""
        self.progress_bar.setVisible(False)
        self.export_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Exportación Exitosa", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Error de Exportación", message)
            self.status_label.setText("Error en exportación")
            
        if self.export_worker:
            self.export_worker.deleteLater()
            self.export_worker = None


# Función de utilidad para usar desde la línea de comandos
def export_simulation_data(simulation_data, output_path, formats=['csv', 'json']):
    """
    Función de utilidad para exportar datos sin GUI
    
    Args:
        simulation_data: Diccionario con los datos de la simulación
        output_path: Directorio de salida
        formats: Lista de formatos a exportar
    """
    config = {
        'output_path': output_path,
        'filename_prefix': f'mfsu_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'formats': formats,
        'content': simulation_data
    }
    
    worker = ExportWorkerThread(config, simulation_data)
    worker.perform_export()
    

if __name__ == "__main__":
    # Ejemplo de uso independiente
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Datos de ejemplo
    sample_data = {
        'psi_field': np.random.complex128((100, 50)),
        'parameters': {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.01},
        'time': np.linspace(0, 10, 100),
        'spatial_grid': np.linspace(-10, 10, 50)
    }
    
    dialog = ExportDialog(sample_data)
    dialog.show()
    
    sys.exit(app.exec_())
