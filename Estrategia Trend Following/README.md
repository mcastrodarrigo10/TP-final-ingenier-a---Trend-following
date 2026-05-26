# TP Final — Trend Following (Time Series Momentum)
## Ingeniería Financiera F414 — Universidad de San Andrés

**Autor:** Máximo Castro Darrigo  
**Materia:** Ingeniería Financiera F414  

---

## Estructura del Proyecto

```
TP final - Castro/
├── analysis/
│   └── strategy_documentation.ipynb   ← Documentación completa de la estrategia
├── requirements.txt                    ← Dependencias Python
└── README.md                           ← Este archivo
```

---

## Correr Localmente

### Requisitos previos
- Python 3.9 o superior
- pip

### Pasos

```bash
# 1. Clonar o descargar el proyecto
cd "TP final - Castro"

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar Jupyter
jupyter notebook analysis/strategy_documentation.ipynb
```

El notebook se abrirá automáticamente en el navegador. Ejecutar todas las celdas con **Kernel → Restart & Run All**.

---

## Correr en Google Colab

1. Ir a [colab.research.google.com](https://colab.research.google.com)
2. Seleccionar **File → Upload notebook** y subir `analysis/strategy_documentation.ipynb`
3. En la primera celda del notebook, agregar e instalar las dependencias:

```python
!pip install numpy pandas matplotlib seaborn yfinance scipy
```

4. Ejecutar todas las celdas con **Runtime → Run all**

### Alternativa: abrir directo desde GitHub

Si el repo está en GitHub, se puede abrir en Colab con el botón:

```
https://colab.research.google.com/github/<usuario>/<repo>/blob/main/analysis/strategy_documentation.ipynb
```

---

## Contenido del Notebook

| Sección | Contenido |
|---------|-----------|
| **1. Intuición Económica** | Underreaction, overreaction, fricción institucional. Tabla comparativa TSMOM vs XSMOM |
| **2. Base Bibliográfica** | Moskowitz (2012), Baz (2015), Hurst (2017), Daniel & Moskowitz (2016) — qué dice cada paper y qué tomamos |
| **3. Arquitectura del Sistema** | Diagrama del pipeline: signal → portfolio → risk → backtest → broker |
| **4. Diseño de la Señal** | Fórmulas LaTeX de señal simple y señal CTA multi-horizonte con visualización |
| **5. Portafolio y Riesgo** | Las 3 capas: vol targeting por activo, vol targeting portafolio, crash filter |
| **6. Referencias** | Bibliografía completa de los 4 papers + literatura adicional |

---

## Papers de Referencia

1. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time series momentum*. Journal of Financial Economics, 104(2), 228–250.
2. Baz, J., Granger, N., Harvey, C. R., Le Roux, N., & Rattray, S. (2015). *Dissecting investment strategies in the cross section and time series*. SSRN Working Paper.
3. Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). *A century of evidence on trend-following investing*. The Journal of Portfolio Management, 44(1), 15–29.
4. Daniel, K., & Moskowitz, T. J. (2016). *Momentum crashes*. Journal of Financial Economics, 122(2), 221–247.
