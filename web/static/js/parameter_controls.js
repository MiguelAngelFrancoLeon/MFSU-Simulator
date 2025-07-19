/**
 * MFSU Simulator - Parameter Controls
 * Manejo de controles de parámetros para la interfaz web
 * Basado en la ecuación: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξH(x,t)ψ - γψ³ + f(x,t)
 */

class ParameterControls {
    constructor() {
        this.parameters = {
            // Parámetros principales de la ecuación MFSU
            alpha: { value: 0.5, min: 0.1, max: 2.0, step: 0.01, description: 'Orden fraccionario del operador laplaciano' },
            beta: { value: 0.1, min: 0.0, max: 1.0, step: 0.01, description: 'Intensidad del ruido fractal' },
            gamma: { value: 0.01, min: 0.0, max: 0.1, step: 0.001, description: 'Coeficiente no lineal cúbico' },
            hurst: { value: 0.7, min: 0.1, max: 0.9, step: 0.01, description: 'Exponente de Hurst para ruido fractal' },
            
            // Parámetros numéricos
            dt: { value: 0.01, min: 0.001, max: 0.1, step: 0.001, description: 'Paso temporal' },
            dx: { value: 0.1, min: 0.01, max: 1.0, step: 0.01, description: 'Resolución espacial' },
            grid_size: { value: 100, min: 50, max: 500, step: 1, description: 'Tamaño de la grilla' },
            max_time: { value: 10.0, min: 1.0, max: 100.0, step: 0.5, description: 'Tiempo máximo de simulación' },
            
            // Parámetros específicos por aplicación
            temperature: { value: 77, min: 1, max: 300, step: 1, description: 'Temperatura (K) - Superconductividad' },
            reynolds: { value: 1000, min: 100, max: 10000, step: 100, description: 'Número de Reynolds - Gas Dynamics' },
            mach: { value: 0.3, min: 0.1, max: 2.0, step: 0.1, description: 'Número de Mach - Gas Dynamics' },
            hubble: { value: 70, min: 50, max: 100, step: 1, description: 'Constante de Hubble - Cosmología' },
            omega_matter: { value: 0.3, min: 0.1, max: 0.9, step: 0.01, description: 'Densidad de materia - Cosmología' }
        };
        
        this.currentApplication = 'general';
        this.callbacks = [];
        this.presets = this.initializePresets();
        this.validationRules = this.initializeValidationRules();
        
        this.init();
    }
    
    init() {
        this.createParameterPanels();
        this.attachEventListeners();
        this.loadFromLocalStorage();
    }
    
    initializePresets() {
        return {
            general: {
                name: 'General',
                params: { alpha: 0.5, beta: 0.1, gamma: 0.01, hurst: 0.7 }
            },
            superconductivity: {
                name: 'Superconductividad',
                params: { alpha: 1.5, beta: 0.05, gamma: 0.02, hurst: 0.8, temperature: 77 }
            },
            gas_dynamics: {
                name: 'Dinámica de Gases',
                params: { alpha: 0.8, beta: 0.15, gamma: 0.005, hurst: 0.6, reynolds: 1000, mach: 0.3 }
            },
            cosmology: {
                name: 'Cosmología',
                params: { alpha: 0.3, beta: 0.2, gamma: 0.001, hurst: 0.9, hubble: 70, omega_matter: 0.3 }
            },
            turbulence: {
                name: 'Turbulencia',
                params: { alpha: 1.2, beta: 0.3, gamma: 0.01, hurst: 0.5 }
            },
            soliton: {
                name: 'Solitón',
                params: { alpha: 2.0, beta: 0.0, gamma: 0.05, hurst: 0.7 }
            }
        };
    }
    
    initializeValidationRules() {
        return {
            stability: (params) => {
                // Criterio de estabilidad CFL
                const cfl = params.dt / (params.dx ** params.alpha);
                return cfl < 0.5 ? { valid: true } : { 
                    valid: false, 
                    message: `Condición CFL violada: dt/dx^α = ${cfl.toFixed(3)} > 0.5` 
                };
            },
            physical: (params) => {
                // Validaciones físicas básicas
                if (params.alpha <= 0 || params.alpha > 2) {
                    return { valid: false, message: 'α debe estar en (0, 2]' };
                }
                if (params.hurst <= 0 || params.hurst >= 1) {
                    return { valid: false, message: 'Exponente de Hurst debe estar en (0, 1)' };
                }
                return { valid: true };
            }
        };
    }
    
    createParameterPanels() {
        const container = document.getElementById('parameter-controls') || this.createContainer();
        
        // Panel principal
        const mainPanel = this.createPanel('Parámetros MFSU', [
            'alpha', 'beta', 'gamma', 'hurst'
        ]);
        
        // Panel numérico
        const numericalPanel = this.createPanel('Configuración Numérica', [
            'dt', 'dx', 'grid_size', 'max_time'
        ]);
        
        // Panel de aplicación
        const appPanel = this.createApplicationPanel();
        
        // Panel de presets
        const presetPanel = this.createPresetPanel();
        
        // Panel de validación
        const validationPanel = this.createValidationPanel();
        
        container.appendChild(mainPanel);
        container.appendChild(numericalPanel);
        container.appendChild(appPanel);
        container.appendChild(presetPanel);
        container.appendChild(validationPanel);
    }
    
    createContainer() {
        const container = document.createElement('div');
        container.id = 'parameter-controls';
        container.className = 'parameter-controls-container';
        document.body.appendChild(container);
        return container;
    }
    
    createPanel(title, paramNames) {
        const panel = document.createElement('div');
        panel.className = 'parameter-panel';
        
        const header = document.createElement('h3');
        header.textContent = title;
        header.className = 'panel-header';
        panel.appendChild(header);
        
        const content = document.createElement('div');
        content.className = 'panel-content';
        
        paramNames.forEach(name => {
            const control = this.createParameterControl(name);
            content.appendChild(control);
        });
        
        panel.appendChild(content);
        return panel;
    }
    
    createParameterControl(name) {
        const param = this.parameters[name];
        const container = document.createElement('div');
        container.className = 'parameter-control';
        container.dataset.param = name;
        
        // Label
        const label = document.createElement('label');
        label.textContent = this.getParameterLabel(name);
        label.className = 'parameter-label';
        label.title = param.description;
        
        // Input container
        const inputContainer = document.createElement('div');
        inputContainer.className = 'input-container';
        
        // Slider
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = param.min;
        slider.max = param.max;
        slider.step = param.step;
        slider.value = param.value;
        slider.className = 'parameter-slider';
        slider.id = `slider-${name}`;
        
        // Number input
        const numberInput = document.createElement('input');
        numberInput.type = 'number';
        numberInput.min = param.min;
        numberInput.max = param.max;
        numberInput.step = param.step;
        numberInput.value = param.value;
        numberInput.className = 'parameter-number';
        numberInput.id = `number-${name}`;
        
        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = '↺';
        resetBtn.className = 'reset-btn';
        resetBtn.title = 'Restablecer valor por defecto';
        resetBtn.onclick = () => this.resetParameter(name);
        
        inputContainer.appendChild(slider);
        inputContainer.appendChild(numberInput);
        inputContainer.appendChild(resetBtn);
        
        container.appendChild(label);
        container.appendChild(inputContainer);
        
        return container;
    }
    
    createApplicationPanel() {
        const panel = document.createElement('div');
        panel.className = 'parameter-panel application-panel';
        
        const header = document.createElement('h3');
        header.textContent = 'Aplicación';
        panel.appendChild(header);
        
        const select = document.createElement('select');
        select.id = 'application-select';
        select.className = 'application-select';
        
        const applications = [
            { value: 'general', text: 'General' },
            { value: 'superconductivity', text: 'Superconductividad' },
            { value: 'gas_dynamics', text: 'Dinámica de Gases' },
            { value: 'cosmology', text: 'Cosmología' }
        ];
        
        applications.forEach(app => {
            const option = document.createElement('option');
            option.value = app.value;
            option.textContent = app.text;
            select.appendChild(option);
        });
        
        panel.appendChild(select);
        
        // Container for application-specific parameters
        const appParams = document.createElement('div');
        appParams.id = 'app-specific-params';
        appParams.className = 'app-specific-params';
        panel.appendChild(appParams);
        
        return panel;
    }
    
    createPresetPanel() {
        const panel = document.createElement('div');
        panel.className = 'parameter-panel preset-panel';
        
        const header = document.createElement('h3');
        header.textContent = 'Presets';
        panel.appendChild(header);
        
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'preset-buttons';
        
        Object.keys(this.presets).forEach(key => {
            const btn = document.createElement('button');
            btn.textContent = this.presets[key].name;
            btn.className = 'preset-btn';
            btn.onclick = () => this.loadPreset(key);
            buttonContainer.appendChild(btn);
        });
        
        // Save current as preset
        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Guardar Actual';
        saveBtn.className = 'save-preset-btn';
        saveBtn.onclick = () => this.saveCurrentAsPreset();
        buttonContainer.appendChild(saveBtn);
        
        panel.appendChild(buttonContainer);
        return panel;
    }
    
    createValidationPanel() {
        const panel = document.createElement('div');
        panel.className = 'validation-panel';
        panel.id = 'validation-panel';
        
        const header = document.createElement('h4');
        header.textContent = 'Validación';
        panel.appendChild(header);
        
        const status = document.createElement('div');
        status.id = 'validation-status';
        status.className = 'validation-status';
        panel.appendChild(status);
        
        return panel;
    }
    
    attachEventListeners() {
        // Parameter control listeners
        Object.keys(this.parameters).forEach(name => {
            const slider = document.getElementById(`slider-${name}`);
            const numberInput = document.getElementById(`number-${name}`);
            
            if (slider) {
                slider.addEventListener('input', (e) => {
                    this.updateParameter(name, parseFloat(e.target.value));
                    if (numberInput) numberInput.value = e.target.value;
                });
            }
            
            if (numberInput) {
                numberInput.addEventListener('change', (e) => {
                    const value = this.validateParameterValue(name, parseFloat(e.target.value));
                    this.updateParameter(name, value);
                    if (slider) slider.value = value;
                    e.target.value = value;
                });
            }
        });
        
        // Application selector
        const appSelect = document.getElementById('application-select');
        if (appSelect) {
            appSelect.addEventListener('change', (e) => {
                this.switchApplication(e.target.value);
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 's':
                        e.preventDefault();
                        this.saveToLocalStorage();
                        this.showNotification('Parámetros guardados');
                        break;
                    case 'r':
                        e.preventDefault();
                        this.resetAllParameters();
                        break;
                }
            }
        });
    }
    
    updateParameter(name, value) {
        const param = this.parameters[name];
        if (!param) return;
        
        // Validate value
        value = this.validateParameterValue(name, value);
        param.value = value;
        
        // Update validation
        this.validateParameters();
        
        // Notify callbacks
        this.callbacks.forEach(callback => {
            callback({
                type: 'parameter_change',
                parameter: name,
                value: value,
                allParameters: this.getAllParameters()
            });
        });
        
        // Auto-save
        this.saveToLocalStorage();
    }
    
    validateParameterValue(name, value) {
        const param = this.parameters[name];
        return Math.max(param.min, Math.min(param.max, value));
    }
    
    validateParameters() {
        const params = this.getAllParameters();
        const statusDiv = document.getElementById('validation-status');
        if (!statusDiv) return;
        
        let isValid = true;
        let messages = [];
        
        // Run validation rules
        Object.values(this.validationRules).forEach(rule => {
            const result = rule(params);
            if (!result.valid) {
                isValid = false;
                messages.push(result.message);
            }
        });
        
        // Update UI
        statusDiv.className = `validation-status ${isValid ? 'valid' : 'invalid'}`;
        statusDiv.innerHTML = isValid 
            ? '<span class="valid-icon">✓</span> Configuración válida'
            : `<span class="invalid-icon">⚠</span> ${messages.join(', ')}`;
    }
    
    switchApplication(appType) {
        this.currentApplication = appType;
        this.updateApplicationSpecificUI();
        
        // Load application preset if available
        if (this.presets[appType]) {
            this.loadPreset(appType, false);
        }
    }
    
    updateApplicationSpecificUI() {
        const container = document.getElementById('app-specific-params');
        if (!container) return;
        
        container.innerHTML = '';
        
        const appParams = this.getApplicationSpecificParams(this.currentApplication);
        appParams.forEach(name => {
            const control = this.createParameterControl(name);
            container.appendChild(control);
        });
    }
    
    getApplicationSpecificParams(appType) {
        switch(appType) {
            case 'superconductivity':
                return ['temperature'];
            case 'gas_dynamics':
                return ['reynolds', 'mach'];
            case 'cosmology':
                return ['hubble', 'omega_matter'];
            default:
                return [];
        }
    }
    
    loadPreset(presetKey, notify = true) {
        const preset = this.presets[presetKey];
        if (!preset) return;
        
        Object.entries(preset.params).forEach(([name, value]) => {
            if (this.parameters[name]) {
                this.parameters[name].value = value;
                
                // Update UI elements
                const slider = document.getElementById(`slider-${name}`);
                const numberInput = document.getElementById(`number-${name}`);
                if (slider) slider.value = value;
                if (numberInput) numberInput.value = value;
            }
        });
        
        this.validateParameters();
        
        if (notify) {
            this.showNotification(`Preset "${preset.name}" cargado`);
            
            // Notify callbacks
            this.callbacks.forEach(callback => {
                callback({
                    type: 'preset_loaded',
                    preset: presetKey,
                    parameters: this.getAllParameters()
                });
            });
        }
    }
    
    resetParameter(name) {
        const param = this.parameters[name];
        if (!param || !param.default) return;
        
        this.updateParameter(name, param.default);
        
        // Update UI
        const slider = document.getElementById(`slider-${name}`);
        const numberInput = document.getElementById(`number-${name}`);
        if (slider) slider.value = param.default;
        if (numberInput) numberInput.value = param.default;
    }
    
    resetAllParameters() {
        Object.keys(this.parameters).forEach(name => {
            this.resetParameter(name);
        });
        
        this.showNotification('Todos los parámetros restablecidos');
    }
    
    saveCurrentAsPreset() {
        const name = prompt('Nombre del preset:');
        if (!name) return;
        
        const key = name.toLowerCase().replace(/\s+/g, '_');
        this.presets[key] = {
            name: name,
            params: { ...this.getAllParameters() }
        };
        
        // Update preset buttons
        this.updatePresetButtons();
        this.showNotification(`Preset "${name}" guardado`);
    }
    
    updatePresetButtons() {
        const container = document.querySelector('.preset-buttons');
        if (!container) return;
        
        // Clear existing preset buttons (keep save button)
        const saveBtn = container.querySelector('.save-preset-btn');
        container.innerHTML = '';
        
        // Recreate preset buttons
        Object.keys(this.presets).forEach(key => {
            const btn = document.createElement('button');
            btn.textContent = this.presets[key].name;
            btn.className = 'preset-btn';
            btn.onclick = () => this.loadPreset(key);
            container.appendChild(btn);
        });
        
        container.appendChild(saveBtn);
    }
    
    getAllParameters() {
        const result = {};
        Object.keys(this.parameters).forEach(key => {
            result[key] = this.parameters[key].value;
        });
        return result;
    }
    
    getParameterLabel(name) {
        const labels = {
            alpha: 'α (Orden fraccionario)',
            beta: 'β (Intensidad ruido)',
            gamma: 'γ (No linealidad)',
            hurst: 'H (Exponente Hurst)',
            dt: 'Δt (Paso temporal)',
            dx: 'Δx (Resolución espacial)',
            grid_size: 'Tamaño grilla',
            max_time: 'Tiempo máximo',
            temperature: 'Temperatura (K)',
            reynolds: 'Re (Reynolds)',
            mach: 'Ma (Mach)',
            hubble: 'H₀ (Hubble)',
            omega_matter: 'Ωₘ (Densidad materia)'
        };
        return labels[name] || name;
    }
    
    saveToLocalStorage() {
        const data = {
            parameters: this.getAllParameters(),
            application: this.currentApplication,
            presets: this.presets
        };
        localStorage.setItem('mfsu_parameters', JSON.stringify(data));
    }
    
    loadFromLocalStorage() {
        try {
            const data = JSON.parse(localStorage.getItem('mfsu_parameters'));
            if (data) {
                // Load parameters
                if (data.parameters) {
                    Object.entries(data.parameters).forEach(([name, value]) => {
                        if (this.parameters[name]) {
                            this.parameters[name].value = value;
                        }
                    });
                }
                
                // Load application
                if (data.application) {
                    this.currentApplication = data.application;
                    const appSelect = document.getElementById('application-select');
                    if (appSelect) appSelect.value = data.application;
                }
                
                // Load custom presets
                if (data.presets) {
                    this.presets = { ...this.presets, ...data.presets };
                }
                
                // Update UI
                this.updateParametersUI();
                this.updateApplicationSpecificUI();
            }
        } catch (e) {
            console.warn('Error loading parameters from localStorage:', e);
        }
    }
    
    updateParametersUI() {
        Object.keys(this.parameters).forEach(name => {
            const value = this.parameters[name].value;
            const slider = document.getElementById(`slider-${name}`);
            const numberInput = document.getElementById(`number-${name}`);
            
            if (slider) slider.value = value;
            if (numberInput) numberInput.value = value;
        });
        
        this.validateParameters();
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '10px 20px',
            backgroundColor: type === 'error' ? '#ff4444' : '#4CAF50',
            color: 'white',
            borderRadius: '4px',
            zIndex: '10000',
            opacity: '0',
            transition: 'opacity 0.3s'
        });
        
        document.body.appendChild(notification);
        
        // Animate in
        requestAnimationFrame(() => {
            notification.style.opacity = '1';
        });
        
        // Remove after delay
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    // Public API
    onParameterChange(callback) {
        this.callbacks.push(callback);
    }
    
    setParameter(name, value) {
        this.updateParameter(name, value);
        this.updateParametersUI();
    }
    
    getParameter(name) {
        return this.parameters[name]?.value;
    }
    
    exportParameters() {
        return {
            timestamp: new Date().toISOString(),
            application: this.currentApplication,
            parameters: this.getAllParameters(),
            equation: "∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξH(x,t)ψ - γψ³ + f(x,t)"
        };
    }
    
    importParameters(data) {
        if (data.parameters) {
            Object.entries(data.parameters).forEach(([name, value]) => {
                this.setParameter(name, value);
            });
        }
        
        if (data.application) {
            this.switchApplication(data.application);
        }
        
        this.showNotification('Parámetros importados correctamente');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.parameterControls = new ParameterControls();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ParameterControls;
}
