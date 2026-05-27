# TP Final — Trend Following (Time Series Momentum)
## Ingeniería Financiera F414 — Universidad de San Andrés

**Grupo:** Agustín Millozi, Diego Lopez Fresco, Lucas Polakoff, Máximo Castro Darrigo  
**Materia:** Ingeniería Financiera F414 — tutoriales  
**Fecha:** Mayo 2026

---

## Descripción de la Estrategia

Implementación de una estrategia sistemática de **Time Series Momentum (TSMOM)** multi-activo, basada en cuatro papers académicos fundacionales. La estrategia opera en un universo de 18 ETFs de cuatro clases de activos (Equities, Fixed Income, Commodities, FX), con rebalanceo mensual, vol targeting en dos capas y un crash filter dinámico.

### Base Bibliográfica

| Paper | Contribución al sistema |
|-------|------------------------|
| **Moskowitz, Ooi & Pedersen (2012)**. *Time Series Momentum*. JFE. | Señal simple `sign(r_{t-12m})`, EWMA de volatilidad con COM=3, sizing `w_i = s_i × (40% / σ̂_i)` |
| **Baz, Granger, Harvey, Le Roux & Rattray (2015)**. *Dissecting Investment Strategies*. SSRN. | Señal CTA multi-horizonte (1/3/12m), función de respuesta `R(x) = x·exp(-x²/4)/0.89` |
| **Hurst, Ooi & Pedersen (2017)**. *A Century of Evidence on Trend-Following*. JPM. | Vol targeting a nivel portafolio `w_scaled = w × (10% / σ̂_port)` |
| **Daniel & Moskowitz (2016)**. *Momentum Crashes*. JFE. | Crash filter `w_final = w × min(1, 1.5/φ)` donde `φ = σ_21d / σ_252d` |

---

## Estructura del Repositorio

```
Estrategia Trend Following/
├── src/
│   └── estrategia.py          ← Sistema completo (señales, portafolio, backtest, broker, CLI)
├── analysis/
│   ├── strategy_documentation.ipynb   ← Documentación teórica con fórmulas y gráficos
│   ├── signal_comparison.png          ← Comparación señal simple vs CTA
│   └── risk_layers.png                ← Visualización de las capas de riesgo
├── requirements.txt
└── README.md
```

---

## Instalación

### Requisitos previos
- Python 3.9 o superior
- pip

### Pasos

```bash
# 1. Ir al directorio del proyecto
cd "Estrategia Trend Following"

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración de API Keys de Alpaca

Las credenciales **nunca** se hardcodean en el código. Se pasan como variables de entorno.

**1. Crear una cuenta en Alpaca:** https://app.alpaca.markets (usar Paper Trading)

**2. Obtener las keys:** Dashboard → API Keys → Generate New Key

**3. Configurar las variables de entorno:**

```powershell
# Windows PowerShell (sesión actual)
$env:ALPACA_API_KEY="PKXXXXXXXXXXXXXXXXXX"
$env:ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows — permanente (System Environment Variables)
[System.Environment]::SetEnvironmentVariable("ALPACA_API_KEY","tu_key","User")
[System.Environment]::SetEnvironmentVariable("ALPACA_API_SECRET","tu_secret","User")
```

```bash
# macOS / Linux (agregar al ~/.bashrc o ~/.zshrc para persistencia)
export ALPACA_API_KEY="PKXXXXXXXXXXXXXXXXXX"
export ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Comandos CLI

Todos los comandos se ejecutan desde el directorio raíz del proyecto:

```bash
# Backtest completo con señal CTA (Baz 2015) — recomendado
python src/estrategia.py --modo backtest --señal cta

# Backtest con señal simple (Moskowitz 2012)
python src/estrategia.py --modo backtest --señal simple

# Suite de robustez: subperíodos, stress testing, sensibilidad, out-of-sample
python src/estrategia.py --modo robustez

# Modo live: calcula pesos y ejecuta órdenes en Alpaca paper
python src/estrategia.py --modo live

# Monitor en tiempo real: posiciones y P&L (actualiza cada 30s)
python src/estrategia.py --modo monitor
```

### Opciones adicionales

```bash
# Personalizar período de backtest
python src/estrategia.py --modo backtest --señal cta --inicio 2015-01-01 --fin 2024-12-31

# Mostrar gráficos en pantalla (además de guardarlos)
python src/estrategia.py --modo backtest --señal cta --mostrar-graficos
```

Los gráficos se guardan automáticamente en `analysis/backtest_CTA.png` o `analysis/backtest_SIMPLE.png`.

---

## Arquitectura del Sistema

El archivo `src/estrategia.py` está organizado en 9 secciones independientes:

| Sección | Contenido |
|---------|-----------|
| **CONFIG** | Diccionario central con todos los parámetros (universo, fechas, vol targets, etc.) |
| **1. Datos** | `descargar_datos()`, `calcular_retornos()` — descarga vía yfinance |
| **2. Señales** | `señal_simple()` (Moskowitz), `señal_cta()` con `R(x)` (Baz) |
| **3. Portafolio** | `sizing_por_activo()`, `escalar_portafolio()` — dos capas de vol targeting |
| **4. Riesgo** | `crash_filter()` — ratio φ = σ_21d/σ_252d, ajuste continuo |
| **5. Backtesting** | `backtesting()`, `calcular_metricas()`, `tests_robustez()` |
| **6. Visualizaciones** | `plot_resultados()` — 4 paneles: equity curve, drawdown, rolling Sharpe, pesos |
| **7. Broker** | `conectar_alpaca()`, `calcular_ordenes()`, `ejecutar_ordenes()`, `monitorear_portfolio()` |
| **8-9. CLI + Main** | argparse con 4 modos; punto de entrada único |

---

## Nota de Transparencia sobre Uso de IA

Este trabajo utilizó herramientas de inteligencia artificial (Claude Code / Claude Sonnet) como asistente para la implementación del código en Python. Las decisiones de diseño, la base teórica, las fórmulas y la interpretación de los resultados son propias del grupo y están fundamentadas en los cuatro papers académicos citados, documentados en `analysis/strategy_documentation.ipynb`. El uso de IA se limitó a asistir en la traducción de fórmulas matemáticas a código y en la estructuración del sistema; el contenido académico y las elecciones metodológicas son responsabilidad de los autores.

---

## Papers de Referencia

1. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time series momentum*. Journal of Financial Economics, 104(2), 228–250. https://doi.org/10.1016/j.jfineco.2011.11.003

2. Baz, J., Granger, N., Harvey, C. R., Le Roux, N., & Rattray, S. (2015). *Dissecting investment strategies in the cross section and time series*. SSRN Working Paper. https://doi.org/10.2139/ssrn.2695101

3. Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). *A century of evidence on trend-following investing*. The Journal of Portfolio Management, 44(1), 15–29. https://doi.org/10.3905/jpm.2017.44.1.015

4. Daniel, K., & Moskowitz, T. J. (2016). *Momentum crashes*. Journal of Financial Economics, 122(2), 221–247. https://doi.org/10.1016/j.jfineco.2015.12.002
