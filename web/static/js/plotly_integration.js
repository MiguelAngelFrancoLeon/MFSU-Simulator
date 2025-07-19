/**
 * MFSU Simulator - Plotly Integration
 * Integración avanzada de Plotly para visualización del Modelo Fractal Estocástico Unificado
 * 
 * Ecuación MFSU: ∂ψ/∂t = α(-Δ)^(α/2)ψ + β ξ_H(x,t)ψ - γψ³ + f(x,t)
 */

class MFSUPlotlyIntegration {
    constructor() {
        this.plots = new Map();
        this.activeSimulation = null;
        this.colorSchemes = {
            fractal: 'Viridis',
            stochastic: 'Plasma',
            phase: 'Rainbow',
            energy: 'Hot'
        };
        this.animationFrames = [];
        this.isAnimating = false;
        
        // Configuración por defecto
        this.defaultLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Arial, sans-serif', size: 12, color: '#333' },
            margin: { l: 60, r: 40, t: 60, b: 60 },
            showlegend: true,
            hovermode: 'closest'
        };
    }

    /**
     * Inicializa los contenedores de visualización
     */
    initializePlots() {
        // Plot principal - Evolución temporal de ψ
        this.createMainEvolutionPlot();
        
        // Plot de análisis espectral
        this.createSpectralAnalysisPlot();
        
        // Plot de análisis fractal
        this.createFractalAnalysisPlot();
        
        // Plot 3D de superficie
        this.create3DSurfacePlot();
        
        // Plot de fase portrait
        this.createPhasePortraitPlot();
        
        // Plot de energía temporal
        this.createEnergyPlot();
    }

    /**
     * Crear plot principal de evolución temporal
     */
    createMainEvolutionPlot() {
        const plotDiv = 'main-evolution-plot';
        
        const layout = {
            ...this.defaultLayout,
            title: {
                text: 'Evolución Temporal de ψ(x,t) - MFSU',
                font: { size: 16, color: '#2c3e50' }
            },
            xaxis: {
                title: 'Posición (x)',
                gridcolor: 'rgba(128,128,128,0.3)',
                zeroline: true
            },
            yaxis: {
                title: 'Amplitud ψ',
                gridcolor: 'rgba(128,128,128,0.3)',
                zeroline: true
            },
            annotations: [{
                text: 'α=' + (window.mfsuParams?.alpha || 0.5) + 
                      ', β=' + (window.mfsuParams?.beta || 0.1) + 
                      ', γ=' + (window.mfsuParams?.gamma || 0.01),
                xref: 'paper', yref: 'paper',
                x: 0.02, y: 0.98,
                showarrow: false,
                font: { size: 10, color: '#7f8c8d' }
            }]
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        };

        Plotly.newPlot(plotDiv, [], layout, config);
        this.plots.set('main', plotDiv);
    }

    /**
     * Crear plot de análisis espectral
     */
    createSpectralAnalysisPlot() {
        const plotDiv = 'spectral-analysis-plot';
        
        const layout = {
            ...this.defaultLayout,
            title: 'Análisis Espectral - Transformada de Fourier',
            xaxis: {
                title: 'Frecuencia (k)',
                type: 'log',
                gridcolor: 'rgba(128,128,128,0.3)'
            },
            yaxis: {
                title: 'Densidad Espectral |ψ̂(k)|²',
                type: 'log',
                gridcolor: 'rgba(128,128,128,0.3)'
            }
        };

        Plotly.newPlot(plotDiv, [], layout);
        this.plots.set('spectral', plotDiv);
    }

    /**
     * Crear plot de análisis fractal
     */
    createFractalAnalysisPlot() {
        const plotDiv = 'fractal-analysis-plot';
        
        const layout = {
            ...this.defaultLayout,
            title: 'Análisis de Dimensión Fractal',
            xaxis: {
                title: 'log(escala)',
                gridcolor: 'rgba(128,128,128,0.3)'
            },
            yaxis: {
                title: 'log(medida)',
                gridcolor: 'rgba(128,128,128,0.3)'
            },
            annotations: [{
                text: 'H = ' + (window.mfsuParams?.hurst || 0.7),
                xref: 'paper', yref: 'paper',
                x: 0.7, y: 0.9,
                showarrow: false,
                font: { size: 12, color: '#e74c3c' }
            }]
        };

        Plotly.newPlot(plotDiv, [], layout);
        this.plots.set('fractal', plotDiv);
    }

    /**
     * Crear plot 3D de superficie
     */
    create3DSurfacePlot() {
        const plotDiv = 'surface-3d-plot';
        
        const layout = {
            ...this.defaultLayout,
            title: 'Superficie 3D - ψ(x,y,t)',
            scene: {
                xaxis: { title: 'x' },
                yaxis: { title: 'y' },
                zaxis: { title: 'ψ' },
                camera: {
                    eye: { x: 1.2, y: 1.2, z: 0.6 }
                }
            }
        };

        Plotly.newPlot(plotDiv, [], layout);
        this.plots.set('surface3d', plotDiv);
    }

    /**
     * Crear plot de retrato de fase
     */
    createPhasePortraitPlot() {
        const plotDiv = 'phase-portrait-plot';
        
        const layout = {
            ...this.defaultLayout,
            title: 'Retrato de Fase - ψ vs ∂ψ/∂t',
            xaxis: {
                title: 'ψ',
                gridcolor: 'rgba(128,128,128,0.3)',
                zeroline: true
            },
            yaxis: {
                title: '∂ψ/∂t',
                gridcolor: 'rgba(128,128,128,0.3)',
                zeroline: true
            }
        };

        Plotly.newPlot(plotDiv, [], layout);
        this.plots.set('phase', plotDiv);
    }

    /**
     * Crear plot de energía
     */
    createEnergyPlot() {
        const plotDiv = 'energy-plot';
        
        const layout = {
            ...this.defaultLayout,
            title: 'Evolución de la Energía Total',
            xaxis: {
                title: 'Tiempo (t)',
                gridcolor: 'rgba(128,128,128,0.3)'
            },
            yaxis: {
                title: 'Energía E(t)',
                gridcolor: 'rgba(128,128,128,0.3)'
            }
        };

        Plotly.newPlot(plotDiv, [], layout);
        this.plots.set('energy', plotDiv);
    }

    /**
     * Actualizar plot principal con nuevos datos
     */
    updateMainPlot(x_data, psi_real, psi_imag, time) {
        const plotDiv = this.plots.get('main');
        
        const traces = [
            {
                x: x_data,
                y: psi_real,
                type: 'scatter',
                mode: 'lines',
                name: 'Re(ψ)',
                line: { color: '#3498db', width: 2 }
            },
            {
                x: x_data,
                y: psi_imag,
                type: 'scatter',
                mode: 'lines',
                name: 'Im(ψ)',
                line: { color: '#e74c3c', width: 2 }
            },
            {
                x: x_data,
                y: psi_real.map((re, i) => Math.sqrt(re*re + psi_imag[i]*psi_imag[i])),
                type: 'scatter',
                mode: 'lines',
                name: '|ψ|',
                line: { color: '#2ecc71', width: 2, dash: 'dash' }
            }
        ];

        // Actualizar anotación con tiempo actual
        const layout_update = {
            'annotations[1]': {
                text: `t = ${time.toFixed(3)}`,
                xref: 'paper', yref: 'paper',
                x: 0.98, y: 0.98,
                showarrow: false,
                font: { size: 14, color: '#2c3e50' }
            }
        };

        Plotly.react(plotDiv, traces, layout_update);
    }

    /**
     * Actualizar análisis espectral
     */
    updateSpectralPlot(frequencies, power_spectrum) {
        const plotDiv = this.plots.get('spectral');
        
        const trace = {
            x: frequencies,
            y: power_spectrum,
            type: 'scatter',
            mode: 'lines',
            name: 'Espectro de Potencia',
            line: { color: '#9b59b6', width: 2 }
        };

        Plotly.react(plotDiv, [trace]);
    }

    /**
     * Actualizar análisis fractal
     */
    updateFractalPlot(scales, measures, dimension) {
        const plotDiv = this.plots.get('fractal');
        
        const trace = {
            x: scales.map(s => Math.log10(s)),
            y: measures.map(m => Math.log10(m)),
            type: 'scatter',
            mode: 'markers+lines',
            name: `D = ${dimension.toFixed(3)}`,
            marker: { color: '#f39c12', size: 6 },
            line: { color: '#f39c12', width: 2 }
        };

        Plotly.react(plotDiv, [trace]);
    }

    /**
     * Actualizar superficie 3D
     */
    update3DSurface(x_grid, y_grid, z_data) {
        const plotDiv = this.plots.get('surface3d');
        
        const trace = {
            type: 'surface',
            x: x_grid,
            y: y_grid,
            z: z_data,
            colorscale: this.colorSchemes.fractal,
            showscale: true,
            colorbar: { title: '|ψ|²' }
        };

        Plotly.react(plotDiv, [trace]);
    }

    /**
     * Actualizar retrato de fase
     */
    updatePhasePortrait(psi_values, dpsi_dt_values) {
        const plotDiv = this.plots.get('phase');
        
        const trace = {
            x: psi_values,
            y: dpsi_dt_values,
            type: 'scatter',
            mode: 'markers',
            marker: {
                color: Array.from({length: psi_values.length}, (_, i) => i),
                colorscale: this.colorSchemes.phase,
                size: 4,
                opacity: 0.7
            },
            name: 'Trayectoria'
        };

        Plotly.react(plotDiv, [trace]);
    }

    /**
     * Actualizar plot de energía
     */
    updateEnergyPlot(time_array, energy_array) {
        const plotDiv = this.plots.get('energy');
        
        const trace = {
            x: time_array,
            y: energy_array,
            type: 'scatter',
            mode: 'lines',
            name: 'E(t)',
            line: { color: '#1abc9c', width: 2 }
        };

        Plotly.react(plotDiv, [trace]);
    }

    /**
     * Iniciar animación temporal
     */
    startAnimation(simulation_data) {
        if (this.isAnimating) return;
        
        this.isAnimating = true;
        this.animationFrames = simulation_data;
        let frameIndex = 0;

        const animate = () => {
            if (!this.isAnimating || frameIndex >= this.animationFrames.length) {
                this.isAnimating = false;
                return;
            }

            const frame = this.animationFrames[frameIndex];
            this.updateMainPlot(frame.x, frame.psi_real, frame.psi_imag, frame.time);
            
            frameIndex++;
            requestAnimationFrame(animate);
        };

        animate();
    }

    /**
     * Detener animación
     */
    stopAnimation() {
        this.isAnimating = false;
    }

    /**
     * Exportar plot como imagen
     */
    exportPlot(plotName, format = 'png', width = 800, height = 600) {
        const plotDiv = this.plots.get(plotName);
        if (!plotDiv) return;

        Plotly.downloadImage(plotDiv, {
            format: format,
            width: width,
            height: height,
            filename: `mfsu_${plotName}_${Date.now()}`
        });
    }

    /**
     * Configurar tema oscuro/claro
     */
    setTheme(isDark) {
        const bgColor = isDark ? '#2c3e50' : '#ffffff';
        const textColor = isDark ? '#ecf0f1' : '#2c3e50';
        const gridColor = isDark ? 'rgba(236,240,241,0.3)' : 'rgba(128,128,128,0.3)';

        const themeLayout = {
            paper_bgcolor: bgColor,
            plot_bgcolor: bgColor,
            font: { color: textColor },
            xaxis: { gridcolor: gridColor, zerolinecolor: gridColor },
            yaxis: { gridcolor: gridColor, zerolinecolor: gridColor }
        };

        // Aplicar tema a todos los plots
        this.plots.forEach((plotDiv) => {
            Plotly.relayout(plotDiv, themeLayout);
        });
    }

    /**
     * Limpiar todos los plots
     */
    clearAllPlots() {
        this.plots.forEach((plotDiv) => {
            Plotly.purge(plotDiv);
        });
        this.plots.clear();
    }

    /**
     * Redimensionar plots cuando cambia el tamaño de ventana
     */
    resizePlots() {
        this.plots.forEach((plotDiv) => {
            Plotly.Plots.resize(plotDiv);
        });
    }

    /**
     * Configurar callbacks para eventos de plotly
     */
    setupEventHandlers() {
        this.plots.forEach((plotDiv, plotName) => {
            // Evento de hover
            document.getElementById(plotDiv).on('plotly_hover', (data) => {
                this.onPlotHover(plotName, data);
            });

            // Evento de click
            document.getElementById(plotDiv).on('plotly_click', (data) => {
                this.onPlotClick(plotName, data);
            });

            // Evento de selección
            document.getElementById(plotDiv).on('plotly_selected', (data) => {
                this.onPlotSelected(plotName, data);
            });
        });
    }

    /**
     * Manejadores de eventos
     */
    onPlotHover(plotName, data) {
        // Implementar lógica de hover personalizada
        console.log(`Hover en ${plotName}:`, data);
    }

    onPlotClick(plotName, data) {
        // Implementar lógica de click personalizada
        console.log(`Click en ${plotName}:`, data);
    }

    onPlotSelected(plotName, data) {
        // Implementar lógica de selección personalizada
        console.log(`Selección en ${plotName}:`, data);
    }
}

// Instancia global
window.MFSUPlotly = new MFSUPlotlyIntegration();

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    window.MFSUPlotly.initializePlots();
    window.MFSUPlotly.setupEventHandlers();
    
    // Redimensionar plots cuando cambie el tamaño de ventana
    window.addEventListener('resize', () => {
        window.MFSUPlotly.resizePlots();
    });
});

// Funciones de utilidad para comunicación con el backend
window.updateMFSUVisualization = function(data) {
    if (data.main_plot) {
        window.MFSUPlotly.updateMainPlot(
            data.main_plot.x,
            data.main_plot.psi_real,
            data.main_plot.psi_imag,
            data.main_plot.time
        );
    }
    
    if (data.spectral_plot) {
        window.MFSUPlotly.updateSpectralPlot(
            data.spectral_plot.frequencies,
            data.spectral_plot.power_spectrum
        );
    }
    
    if (data.fractal_plot) {
        window.MFSUPlotly.updateFractalPlot(
            data.fractal_plot.scales,
            data.fractal_plot.measures,
            data.fractal_plot.dimension
        );
    }
    
    if (data.surface_3d) {
        window.MFSUPlotly.update3DSurface(
            data.surface_3d.x_grid,
            data.surface_3d.y_grid,
            data.surface_3d.z_data
        );
    }
    
    if (data.phase_portrait) {
        window.MFSUPlotly.updatePhasePortrait(
            data.phase_portrait.psi_values,
            data.phase_portrait.dpsi_dt_values
        );
    }
    
    if (data.energy_plot) {
        window.MFSUPlotly.updateEnergyPlot(
            data.energy_plot.time_array,
            data.energy_plot.energy_array
        );
    }
};
