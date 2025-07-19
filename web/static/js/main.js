/**
 * MFSU Simulator - Main JavaScript Controller
 * Unified Stochastic Fractal Model Simulator
 * 
 * Ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
 */

class MFSUSimulator {
    constructor() {
        this.isRunning = false;
        this.simulationData = null;
        this.currentStep = 0;
        this.maxSteps = 1000;
        this.animationId = null;
        
        // Parámetros por defecto del MFSU
        this.parameters = {
            alpha: 0.5,      // Coeficiente operador fraccionario
            beta: 0.1,       // Intensidad del ruido fractal
            gamma: 0.01,     // Coeficiente no lineal
            hurst: 0.7,      // Exponente de Hurst
            dt: 0.01,        // Paso temporal
            dx: 0.1,         // Paso espacial
            gridSize: 100,   // Tamaño de la grilla
            maxTime: 10.0,   // Tiempo máximo de simulación
            application: 'superconductivity' // Aplicación seleccionada
        };
        
        // Configuraciones por aplicación
        this.applicationConfigs = {
            superconductivity: {
                temperatureRange: [1, 300],
                criticalTemp: 93,
                materialType: 'YBCO',
                defaultParams: { alpha: 0.5, beta: 0.1, gamma: 0.01 }
            },
            gas_dynamics: {
                reynoldsNumber: 1000,
                machNumber: 0.3,
                viscosity: 1e-5,
                defaultParams: { alpha: 1.0, beta: 0.05, gamma: 0.02 }
            },
            cosmology: {
                hubbleConstant: 70,
                omegaMatter: 0.3,
                omegaLambda: 0.7,
                defaultParams: { alpha: 0.8, beta: 0.15, gamma: 0.005 }
            }
        };
        
        this.initializeEventListeners();
        this.setupUI();
        this.initializePlots();
    }

    initializeEventListeners() {
        // Controles de simulación
        document.getElementById('startBtn')?.addEventListener('click', () => this.startSimulation());
        document.getElementById('pauseBtn')?.addEventListener('click', () => this.pauseSimulation());
        document.getElementById('resetBtn')?.addEventListener('click', () => this.resetSimulation());
        document.getElementById('exportBtn')?.addEventListener('click', () => this.exportData());
        
        // Controles de parámetros
        document.getElementById('alphaSlider')?.addEventListener('input', (e) => this.updateParameter('alpha', parseFloat(e.target.value)));
        document.getElementById('betaSlider')?.addEventListener('input', (e) => this.updateParameter('beta', parseFloat(e.target.value)));
        document.getElementById('gammaSlider')?.addEventListener('input', (e) => this.updateParameter('gamma', parseFloat(e.target.value)));
        document.getElementById('hurstSlider')?.addEventListener('input', (e) => this.updateParameter('hurst', parseFloat(e.target.value)));
        
        // Selector de aplicación
        document.getElementById('applicationSelect')?.addEventListener('change', (e) => this.changeApplication(e.target.value));
        
        // Condiciones iniciales
        document.getElementById('initialConditionSelect')?.addEventListener('change', (e) => this.setInitialCondition(e.target.value));
        
        // Análisis en tiempo real
        document.getElementById('enableRealTimeAnalysis')?.addEventListener('change', (e) => this.toggleRealTimeAnalysis(e.target.checked));
        
        // Configuración avanzada
        document.getElementById('advancedConfigBtn')?.addEventListener('click', () => this.showAdvancedConfig());
    }

    setupUI() {
        // Actualizar valores mostrados de los parámetros
        this.updateParameterDisplays();
        
        // Configurar opciones de aplicación
        const appSelect = document.getElementById('applicationSelect');
        if (appSelect) {
            Object.keys(this.applicationConfigs).forEach(app => {
                const option = document.createElement('option');
                option.value = app;
                option.textContent = app.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                appSelect.appendChild(option);
            });
        }
        
        // Mostrar fórmula MFSU
        this.displayMFSUEquation();
    }

    displayMFSUEquation() {
        const equationDiv = document.getElementById('mfsuEquation');
        if (equationDiv) {
            equationDiv.innerHTML = `
                <div class="equation-display">
                    <h3>Ecuación MFSU</h3>
                    <div class="math-formula">
                        ∂ψ/∂t = α(-Δ)<sup>α/2</sup>ψ + β ξ<sub>H</sub>(x,t)ψ - γψ³ + f(x,t)
                    </div>
                    <div class="parameter-legend">
                        <p><strong>α:</strong> Coeficiente del operador fraccionario</p>
                        <p><strong>β:</strong> Intensidad del ruido fractal</p>
                        <p><strong>γ:</strong> Coeficiente no lineal</p>
                        <p><strong>ξ<sub>H</sub>:</strong> Ruido gaussiano fraccionario (Hurst = ${this.parameters.hurst})</p>
                    </div>
                </div>
            `;
        }
    }

    updateParameterDisplays() {
        Object.keys(this.parameters).forEach(param => {
            const slider = document.getElementById(`${param}Slider`);
            const display = document.getElementById(`${param}Value`);
            
            if (slider && display) {
                slider.value = this.parameters[param];
                display.textContent = this.parameters[param].toFixed(3);
            }
        });
    }

    updateParameter(paramName, value) {
        this.parameters[paramName] = value;
        const display = document.getElementById(`${paramName}Value`);
        if (display) {
            display.textContent = value.toFixed(3);
        }
        
        // Actualizar ecuación si es necesario
        if (paramName === 'hurst') {
            this.displayMFSUEquation();
        }
        
        // Recalcular si la simulación está corriendo
        if (this.isRunning) {
            this.updateSimulationParameters();
        }
    }

    changeApplication(appName) {
        this.parameters.application = appName;
        const config = this.applicationConfigs[appName];
        
        if (config && config.defaultParams) {
            Object.assign(this.parameters, config.defaultParams);
            this.updateParameterDisplays();
        }
        
        // Mostrar configuración específica de la aplicación
        this.displayApplicationConfig(appName);
        
        // Resetear simulación con nueva configuración
        this.resetSimulation();
    }

    displayApplicationConfig(appName) {
        const configDiv = document.getElementById('applicationConfig');
        if (!configDiv) return;
        
        const config = this.applicationConfigs[appName];
        let configHTML = `<h4>Configuración: ${appName.replace('_', ' ').toUpperCase()}</h4>`;
        
        switch (appName) {
            case 'superconductivity':
                configHTML += `
                    <p>Rango de Temperatura: ${config.temperatureRange[0]} - ${config.temperatureRange[1]} K</p>
                    <p>Temperatura Crítica: ${config.criticalTemp} K</p>
                    <p>Material: ${config.materialType}</p>
                `;
                break;
            case 'gas_dynamics':
                configHTML += `
                    <p>Número de Reynolds: ${config.reynoldsNumber}</p>
                    <p>Número de Mach: ${config.machNumber}</p>
                    <p>Viscosidad: ${config.viscosity}</p>
                `;
                break;
            case 'cosmology':
                configHTML += `
                    <p>Constante de Hubble: ${config.hubbleConstant} km/s/Mpc</p>
                    <p>Ω<sub>m</sub>: ${config.omegaMatter}</p>
                    <p>Ω<sub>Λ</sub>: ${config.omegaLambda}</p>
                `;
                break;
        }
        
        configDiv.innerHTML = configHTML;
    }

    async startSimulation() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.updateButtonStates();
        
        try {
            // Enviar parámetros al backend
            const response = await fetch('/api/start_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.parameters)
            });
            
            if (!response.ok) {
                throw new Error(`Error en simulación: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                this.simulationData = result.data;
                this.currentStep = 0;
                this.animateSimulation();
            } else {
                throw new Error(result.error || 'Error desconocido en la simulación');
            }
            
        } catch (error) {
            console.error('Error iniciando simulación:', error);
            this.showNotification('Error iniciando simulación: ' + error.message, 'error');
            this.isRunning = false;
            this.updateButtonStates();
        }
    }

    pauseSimulation() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.updateButtonStates();
    }

    resetSimulation() {
        this.pauseSimulation();
        this.currentStep = 0;
        this.simulationData = null;
        this.clearPlots();
        this.updateProgressBar(0);
    }

    animateSimulation() {
        if (!this.isRunning || !this.simulationData) return;
        
        if (this.currentStep < this.simulationData.timeSteps.length) {
            this.updatePlots(this.currentStep);
            this.updateProgressBar((this.currentStep / this.simulationData.timeSteps.length) * 100);
            
            this.currentStep++;
            this.animationId = requestAnimationFrame(() => this.animateSimulation());
        } else {
            this.isRunning = false;
            this.updateButtonStates();
            this.showNotification('Simulación completada', 'success');
        }
    }

    updateSimulationParameters() {
        if (this.isRunning) {
            fetch('/api/update_parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.parameters)
            });
        }
    }

    initializePlots() {
        // Plot principal - Evolución temporal de ψ
        const mainPlot = document.getElementById('mainPlot');
        if (mainPlot) {
            Plotly.newPlot('mainPlot', [], {
                title: 'Evolución Temporal de ψ(x,t)',
                xaxis: { title: 'Posición (x)' },
                yaxis: { title: 'Amplitud |ψ|²' },
                showlegend: true
            });
        }
        
        // Plot de análisis espectral
        const spectrumPlot = document.getElementById('spectrumPlot');
        if (spectrumPlot) {
            Plotly.newPlot('spectrumPlot', [], {
                title: 'Análisis Espectral',
                xaxis: { title: 'Frecuencia', type: 'log' },
                yaxis: { title: 'Densidad Espectral', type: 'log' },
                showlegend: true
            });
        }
        
        // Plot de dimensión fractal
        const fractalPlot = document.getElementById('fractalPlot');
        if (fractalPlot) {
            Plotly.newPlot('fractalPlot', [], {
                title: 'Dimensión Fractal vs Tiempo',
                xaxis: { title: 'Tiempo' },
                yaxis: { title: 'Dimensión Fractal' },
                showlegend: false
            });
        }
    }

    updatePlots(timeStep) {
        if (!this.simulationData) return;
        
        const currentData = this.simulationData.timeSteps[timeStep];
        
        // Actualizar plot principal
        const mainTrace = {
            x: currentData.x,
            y: currentData.psi_squared,
            type: 'scatter',
            mode: 'lines',
            name: `t = ${currentData.time.toFixed(2)}`,
            line: { color: 'blue', width: 2 }
        };
        
        Plotly.animate('mainPlot', {
            data: [mainTrace]
        }, {
            transition: { duration: 50, easing: 'linear' },
            frame: { duration: 50, redraw: false }
        });
        
        // Actualizar análisis espectral
        if (currentData.spectrum) {
            const spectrumTrace = {
                x: currentData.spectrum.frequency,
                y: currentData.spectrum.power,
                type: 'scatter',
                mode: 'lines',
                name: 'Espectro de Potencia',
                line: { color: 'red', width: 2 }
            };
            
            Plotly.animate('spectrumPlot', {
                data: [spectrumTrace]
            });
        }
        
        // Actualizar dimensión fractal
        if (this.simulationData.fractalDimensions) {
            const fractalTrace = {
                x: this.simulationData.fractalDimensions.slice(0, timeStep + 1).map((_, i) => i * this.parameters.dt),
                y: this.simulationData.fractalDimensions.slice(0, timeStep + 1),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Dimensión Fractal',
                line: { color: 'green', width: 2 }
            };
            
            Plotly.animate('fractalPlot', {
                data: [fractalTrace]
            });
        }
    }

    clearPlots() {
        ['mainPlot', 'spectrumPlot', 'fractalPlot'].forEach(plotId => {
            const plot = document.getElementById(plotId);
            if (plot) {
                Plotly.purge(plotId);
                this.initializePlots();
            }
        });
    }

    updateButtonStates() {
        const startBtn = document.getElementById('startBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const resetBtn = document.getElementById('resetBtn');
        
        if (startBtn) startBtn.disabled = this.isRunning;
        if (pauseBtn) pauseBtn.disabled = !this.isRunning;
        if (resetBtn) resetBtn.disabled = this.isRunning;
    }

    updateProgressBar(percentage) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${percentage.toFixed(1)}%`;
        }
    }

    setInitialCondition(conditionType) {
        fetch('/api/set_initial_condition', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: conditionType,
                parameters: this.parameters
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.showNotification(`Condición inicial establecida: ${conditionType}`, 'success');
            }
        })
        .catch(error => {
            console.error('Error estableciendo condición inicial:', error);
            this.showNotification('Error estableciendo condición inicial', 'error');
        });
    }

    toggleRealTimeAnalysis(enabled) {
        this.realTimeAnalysis = enabled;
        const analysisPanel = document.getElementById('analysisPanel');
        if (analysisPanel) {
            analysisPanel.style.display = enabled ? 'block' : 'none';
        }
    }

    showAdvancedConfig() {
        const modal = document.getElementById('advancedConfigModal');
        if (modal) {
            modal.style.display = 'block';
        }
    }

    async exportData() {
        if (!this.simulationData) {
            this.showNotification('No hay datos para exportar', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/export_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: this.simulationData,
                    parameters: this.parameters,
                    format: 'json'
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `mfsu_simulation_${Date.now()}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showNotification('Datos exportados exitosamente', 'success');
            }
        } catch (error) {
            console.error('Error exportando datos:', error);
            this.showNotification('Error exportando datos', 'error');
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Mostrar notificación
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // Ocultar después de 3 segundos
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Métodos de análisis en tiempo real
    calculateFractalDimension(data) {
        // Implementación simplificada del método box-counting
        // En la implementación real esto se haría en el backend
        let dimension = 1.5; // Valor por defecto
        return dimension;
    }

    calculateSpectralDensity(data) {
        // Cálculo simplificado de la densidad espectral
        // En la implementación real esto se haría en el backend con FFT
        return {
            frequencies: Array.from({length: 50}, (_, i) => i * 0.1),
            power: Array.from({length: 50}, (_, i) => Math.exp(-i * 0.1))
        };
    }

    // Utilidades de validación
    validateParameters() {
        const errors = [];
        
        if (this.parameters.alpha <= 0 || this.parameters.alpha > 2) {
            errors.push('α debe estar entre 0 y 2');
        }
        
        if (this.parameters.beta < 0) {
            errors.push('β debe ser no negativo');
        }
        
        if (this.parameters.gamma < 0) {
            errors.push('γ debe ser no negativo');
        }
        
        if (this.parameters.hurst <= 0 || this.parameters.hurst >= 1) {
            errors.push('Exponente de Hurst debe estar entre 0 y 1');
        }
        
        return errors;
    }
}

// Inicializar el simulador cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    window.mfsuSimulator = new MFSUSimulator();
    
    // Configurar tooltips para parámetros
    const tooltips = {
        alpha: 'Controla la difusión fraccionaria del sistema',
        beta: 'Intensidad del ruido estocástico fractal',
        gamma: 'Fuerza de la no linealidad cúbica',
        hurst: 'Exponente de Hurst para el ruido fractal (0.5 = Browniano)'
    };
    
    Object.keys(tooltips).forEach(param => {
        const element = document.getElementById(`${param}Slider`);
        if (element) {
            element.title = tooltips[param];
        }
    });
    
    console.log('MFSU Simulator inicializado correctamente');
});

// Manejo de errores globales
window.addEventListener('error', function(event) {
    console.error('Error en MFSU Simulator:', event.error);
});

// API de utilidades para integración con otros scripts
window.MFSUUtils = {
    formatNumber: (num, decimals = 3) => num.toFixed(decimals),
    
    validateRange: (value, min, max) => Math.max(min, Math.min(max, value)),
    
    downloadJSON: (data, filename) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
};
