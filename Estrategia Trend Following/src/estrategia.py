"""
Estrategia Trend Following (Time Series Momentum) — TP Final F414
Universidad de San Andrés

Grupo: Agustín Millozi, Diego Lopez Fresco, Lucas Polakoff, Máximo Castro Darrigo

Base bibliográfica:
  [1] Moskowitz, Ooi & Pedersen (2012)         — señal simple + vol targeting por activo
  [2] Baz, Granger, Harvey, Le Roux & Rattray (2015) — señal CTA multi-horizonte
  [3] Hurst, Ooi & Pedersen (2017)              — vol targeting portafolio, evidencia centenaria
  [4] Daniel & Moskowitz (2016)                 — crash filter dinámico
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Compatibilidad de frecuencias entre versiones de pandas
# ─────────────────────────────────────────────────────────────────────────────
try:
    pd.date_range("2020-01-01", periods=2, freq="BME")
    _BIZ_MONTH_END = "BME"
    _MONTH_END = "ME"
except ValueError:
    _BIZ_MONTH_END = "BM"
    _MONTH_END = "M"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN CENTRAL — todos los parámetros del sistema aquí
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Universo de activos por clase de activo ────────────────────────────
    "universo": {
        "Equities":     ["SPY", "QQQ", "EFA", "EEM", "IWM"],
        "Fixed Income": ["TLT", "IEF", "LQD", "HYG"],
        "Commodities":  ["GLD", "SLV", "DBC", "USO"],
        "FX":           ["UUP", "FXE", "FXY", "FXB"],
    },
    # ── Período de análisis ───────────────────────────────────────────────
    "fecha_inicio":       "2010-01-01",
    "fecha_fin":           None,          # None = hoy
    # ── Vol targeting — Moskowitz (2012) y Hurst (2017) ──────────────────
    "vol_target_activo":   0.40,          # 40% por activo
    "vol_target_port":     0.10,          # 10% portafolio total
    # ── EWMA volatilidad — Moskowitz (2012) ──────────────────────────────
    "ewma_com":            3,             # COM=3 → lambda=0.75
    # ── Señal simple — Moskowitz (2012) ──────────────────────────────────
    "lookback_simple":     252,           # ~12 meses en días de trading
    # ── Señal CTA — Baz et al. (2015) ────────────────────────────────────
    "horizontes_cta":      [21, 63, 252], # 1, 3, 12 meses
    # ── Crash filter — Daniel & Moskowitz (2016) ─────────────────────────
    "vol_corta_dias":      21,
    "vol_larga_dias":      252,
    "crash_umbral":        1.5,           # umbral phi para reducir exposición
    # ── Límites de riesgo ────────────────────────────────────────────────
    "max_leverage":        3.0,
    "max_peso_activo":     0.30,
    # ── Covariance matrix (días EWMA) ────────────────────────────────────
    "cov_ventana":         60,
    # ── Backtest ─────────────────────────────────────────────────────────
    "costo_bps":           10,            # 10bps por unidad de turnover
    "benchmark_tickers":   ["SPY", "TLT"],
    "benchmark_pesos":     [0.60, 0.40],
    # ── Out-of-sample ────────────────────────────────────────────────────
    "oos_fecha_corte":     "2019-01-01",
    # ── Alpaca paper trading ──────────────────────────────────────────────
    "alpaca_paper":        True,
    "alpaca_base_url":     "https://paper-api.alpaca.markets",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def descargar_datos(universo=None, fecha_inicio=None, fecha_fin=None):
    """Descarga precios ajustados diarios vía yfinance para todo el universo."""
    universo = universo or CONFIG["universo"]
    fecha_inicio = fecha_inicio or CONFIG["fecha_inicio"]
    fecha_fin = fecha_fin or datetime.today().strftime("%Y-%m-%d")

    tickers = [t for clase in universo.values() for t in clase]
    print(f"[datos] Descargando {len(tickers)} tickers ({fecha_inicio} → {fecha_fin})...")

    raw = yf.download(
        tickers, start=fecha_inicio, end=fecha_fin,
        auto_adjust=True, progress=False
    )

    # yfinance puede devolver MultiIndex o Index simple según la versión
    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
        if isinstance(precios.columns, pd.MultiIndex):
            precios.columns = precios.columns.get_level_values(-1)
    else:
        precios = raw[["Close"]] if "Close" in raw.columns else raw

    if isinstance(precios, pd.Series):
        precios = precios.to_frame()

    precios = precios.sort_index().ffill().dropna(how="all")

    # Descartar activos con datos insuficientes para inicializar señales
    min_dias = max(CONFIG["lookback_simple"] + 50, 300)
    precios = precios.loc[:, precios.notna().sum() >= min_dias]

    print(f"[datos] {len(precios)} días | {precios.shape[1]} activos | "
          f"{precios.index[0].date()} → {precios.index[-1].date()}")
    return precios


def calcular_retornos(precios):
    """Retornos log-diarios: r_t = log(P_t / P_{t-1})."""
    return np.log(precios / precios.shift(1)).dropna(how="all")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SEÑALES
# ═══════════════════════════════════════════════════════════════════════════════

def _volatilidad_ewma(retornos, com=None):
    """
    Volatilidad anual EWMA usando r² (no centrado) — Moskowitz et al. (2012).
    lambda = COM/(COM+1) = 0.75 con COM=3.
    """
    com = com if com is not None else CONFIG["ewma_com"]
    return retornos.pow(2).ewm(com=com).mean().pow(0.5) * np.sqrt(252)


def señal_simple(retornos, lookback=None):
    """
    Señal binaria sign(r_{t-252:t}) — Moskowitz, Ooi & Pedersen (2012).
    Retorna DataFrame con valores en {-1, +1}.
    """
    lookback = lookback or CONFIG["lookback_simple"]
    ret_acum = retornos.rolling(lookback).sum()
    señal = np.sign(ret_acum)
    señal = señal.replace(0, 1)   # empate → long por convención
    return señal


def _funcion_respuesta(x):
    """R(x) = x · exp(-x²/4) / 0.89 — Baz et al. (2015)."""
    return x * np.exp(-x**2 / 4) / 0.89


def señal_cta(retornos, horizontes=None):
    """
    Señal CTA multi-horizonte — Baz, Granger, Harvey et al. (2015).
    Para cada horizonte h ∈ {21, 63, 252}: z = mean(r_h) / std(r_h), luego R(z).
    Señal final = promedio de R(z) en los tres horizontes, ∈ (-1, +1).
    """
    horizontes = horizontes or CONFIG["horizontes_cta"]
    señales_h = []
    for h in horizontes:
        mu_h = retornos.rolling(h).mean()
        sigma_h = retornos.rolling(h).std().replace(0, np.nan)
        z_h = mu_h / sigma_h
        señales_h.append(_funcion_respuesta(z_h))
    return sum(señales_h) / len(señales_h)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PORTAFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

def sizing_por_activo(señales, vol_activos, vol_target=None):
    """
    Capa 1 — Sizing inversamente proporcional a la volatilidad realizada.
    w_i = s_i × (σ* / σ̂_i) — Moskowitz et al. (2012).  σ* = 40%.
    Iguala la contribución de riesgo esperada de cada activo.
    """
    vol_target = vol_target if vol_target is not None else CONFIG["vol_target_activo"]
    señales_al, vol_al = señales.align(vol_activos, join="inner")
    return señales_al * (vol_target / vol_al.replace(0, np.nan))


def escalar_portafolio(pesos, retornos, vol_target_port=None, cov_ventana=None):
    """
    Capa 2 — Vol targeting a nivel portafolio.
    w_scaled = w × (σ_port / σ̂_port) — Hurst, Ooi & Pedersen (2017).
    σ̂_port estimada como vol EWMA del retorno del portafolio (proxy eficiente).
    """
    vol_target_port = vol_target_port if vol_target_port is not None else CONFIG["vol_target_port"]
    cov_ventana = cov_ventana or CONFIG["cov_ventana"]

    pesos_al, rets_al = pesos.align(retornos, join="inner")

    # Proxy de retorno del portafolio para estimar su volatilidad
    ret_port_proxy = (pesos_al * rets_al).sum(axis=1)
    vol_port = ret_port_proxy.ewm(span=cov_ventana).std() * np.sqrt(252)
    vol_port = vol_port.replace(0, np.nan)

    # Factor de escala: apunta a vol_target_port, capeado para evitar leverage excesivo
    factor = (vol_target_port / vol_port).clip(upper=5.0)
    return pesos_al.multiply(factor, axis=0)


def _aplicar_limites_riesgo(pesos):
    """Límites hard de concentración por activo y leverage bruto total."""
    max_peso = CONFIG["max_peso_activo"]
    max_lev  = CONFIG["max_leverage"]

    # Concentración por activo
    pesos = pesos.clip(lower=-max_peso, upper=max_peso)

    # Leverage bruto total
    gross = pesos.abs().sum(axis=1)
    scale = (max_lev / gross).clip(upper=1.0)
    return pesos.multiply(scale, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RIESGO
# ═══════════════════════════════════════════════════════════════════════════════

def crash_filter(pesos, retornos, umbral=None):
    """
    Capa 3 — Crash filter dinámico — Daniel & Moskowitz (2016).
    φ_i = σ_21d / σ_252d (ratio de vol reciente vs histórica).
    w_final = w × min(1, 1.5/φ) — ajuste continuo y proporcional.
    """
    umbral = umbral if umbral is not None else CONFIG["crash_umbral"]
    n_corta = CONFIG["vol_corta_dias"]
    n_larga = CONFIG["vol_larga_dias"]

    vol_corta = retornos.rolling(n_corta).std() * np.sqrt(252)
    vol_larga = retornos.rolling(n_larga).std() * np.sqrt(252)

    phi    = (vol_corta / vol_larga.replace(0, np.nan))
    factor = (umbral / phi).clip(upper=1.0)  # min(1, umbral/phi)

    pesos_al, factor_al = pesos.align(factor, join="inner")
    return pesos_al * factor_al


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

def _pipeline_pesos(retornos, tipo_señal="cta"):
    """Ejecuta el pipeline completo: señal → sizing → escala → crash filter → límites."""
    vol_activos = _volatilidad_ewma(retornos)

    if tipo_señal == "simple":
        señales = señal_simple(retornos)
    else:
        señales = señal_cta(retornos)

    pesos = sizing_por_activo(señales, vol_activos)
    pesos = escalar_portafolio(pesos, retornos)
    pesos = crash_filter(pesos, retornos)
    pesos = _aplicar_limites_riesgo(pesos)
    return pesos.dropna(how="all")


def _retorno_benchmark(precios):
    """Retorno diario del benchmark 60/40 (SPY + TLT)."""
    tickers_b = [t for t in CONFIG["benchmark_tickers"] if t in precios.columns]
    pesos_b   = CONFIG["benchmark_pesos"][:len(tickers_b)]
    rets_b    = calcular_retornos(precios[tickers_b])
    return (rets_b * pesos_b).sum(axis=1)


def backtesting(precios, tipo_señal="cta"):
    """
    Simula la estrategia con rebalanceo mensual y costos de transacción (10bps/turnover).

    Retorna dict con: retornos_port, retornos_bench, pesos_history,
    equity_curve, equity_bench, turnover_mensual, tipo_señal.
    """
    print(f"\n[backtest] Señal: {tipo_señal.upper()} | Rebalanceo mensual | "
          f"Costo: {CONFIG['costo_bps']}bps por turnover")

    retornos      = calcular_retornos(precios)
    pesos_diarios = _pipeline_pesos(retornos, tipo_señal)

    # Fechas de rebalanceo: último día hábil de cada mes con datos disponibles
    try:
        fechas_mes = retornos.resample(_BIZ_MONTH_END).last().index
    except Exception:
        fechas_mes = retornos.resample(_MONTH_END).last().index
    fechas_rebalanceo = fechas_mes[fechas_mes.isin(pesos_diarios.index)]

    # Pesos en fechas de rebalanceo, forward-filled a frecuencia diaria
    pesos_rebalan = pesos_diarios.loc[fechas_rebalanceo]
    pesos_ff      = pesos_rebalan.reindex(retornos.index, method="ffill")
    pesos_ff, retornos = pesos_ff.align(retornos, join="inner")

    # Sin look-ahead: posición del día t usa pesos calculados al cierre de t-1
    pesos_lag = pesos_ff.shift(1)

    # Retorno bruto diario
    ret_bruto = (pesos_lag * retornos).sum(axis=1)

    # Costos de transacción: solo en fechas de rebalanceo
    turnover_diario = pesos_ff.diff().abs().sum(axis=1)
    costo_diario    = pd.Series(0.0, index=retornos.index)
    mask_rebalan    = retornos.index.isin(fechas_rebalanceo)
    costo_diario[mask_rebalan] = (
        turnover_diario[mask_rebalan] * (CONFIG["costo_bps"] / 10_000)
    )

    ret_neto  = ret_bruto - costo_diario
    ret_bench = _retorno_benchmark(precios).reindex(ret_neto.index, fill_value=0.0)

    equity       = (1 + ret_neto).cumprod()
    equity_bench = (1 + ret_bench).cumprod()

    try:
        turnover_mens = costo_diario.resample(_MONTH_END).sum() / (CONFIG["costo_bps"] / 10_000)
    except Exception:
        turnover_mens = costo_diario.resample("ME").sum() / (CONFIG["costo_bps"] / 10_000)

    print(f"[backtest] Período: {equity.index[0].date()} → {equity.index[-1].date()} | "
          f"Rebalanceos: {len(fechas_rebalanceo)}")

    return {
        "retornos_port":    ret_neto,
        "retornos_bench":   ret_bench,
        "pesos_history":    pesos_ff,
        "equity_curve":     equity,
        "equity_bench":     equity_bench,
        "turnover_mensual": turnover_mens,
        "tipo_señal":       tipo_señal,
    }


def calcular_metricas(resultados, label="Estrategia"):
    """
    Calcula y muestra: Sharpe, Sortino, MaxDD, Calmar, turnover promedio, costo anual.
    Retorna dict con las métricas calculadas.
    """
    rets     = resultados["retornos_port"]
    equity   = resultados.get("equity_curve") or (1 + rets).cumprod()
    turnover = resultados["turnover_mensual"]

    mean_a  = rets.mean() * 252
    std_a   = rets.std() * np.sqrt(252)
    sharpe  = mean_a / std_a if std_a > 0 else np.nan

    down    = rets[rets < 0].std() * np.sqrt(252)
    sortino = mean_a / down if down > 0 else np.nan

    dd     = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = mean_a / abs(max_dd) if max_dd < 0 else np.nan

    turnover_prom = turnover.mean()
    costo_anual   = turnover_prom * 12 * (CONFIG["costo_bps"] / 10_000)

    try:
        rets_mens = rets.resample(_MONTH_END).sum()
    except Exception:
        rets_mens = rets.resample("ME").sum()
    hit_rate = (rets_mens > 0).mean()

    metricas = {
        "Retorno anualizado": f"{mean_a:.2%}",
        "Volatilidad anual":  f"{std_a:.2%}",
        "Sharpe ratio":       f"{sharpe:.3f}",
        "Sortino ratio":      f"{sortino:.3f}",
        "Max Drawdown":       f"{max_dd:.2%}",
        "Calmar ratio":       f"{calmar:.3f}",
        "Hit rate mensual":   f"{hit_rate:.2%}",
        "Turnover promedio":  f"{turnover_prom:.2%}",
        "Costo anual est.":   f"{costo_anual:.3%}",
    }

    print(f"\n{'─'*48}")
    print(f"  MÉTRICAS — {label}")
    print(f"{'─'*48}")
    for k, v in metricas.items():
        print(f"  {k:<25} {v:>10}")
    print(f"{'─'*48}\n")

    return metricas


def _metricas_subperiodo(rets, label):
    """Métricas resumidas para un subperíodo dado."""
    if len(rets) < 20:
        return None
    mean_a = rets.mean() * 252
    std_a  = rets.std() * np.sqrt(252)
    sharpe = mean_a / std_a if std_a > 0 else np.nan
    eq     = (1 + rets).cumprod()
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {
        "Período": label,
        "Retorno": f"{mean_a:.2%}",
        "Vol":     f"{std_a:.2%}",
        "Sharpe":  f"{sharpe:.3f}",
        "MaxDD":   f"{max_dd:.2%}",
    }


def tests_robustez(precios, tipo_señal="cta"):
    """
    Suite completa de tests de robustez:
      1. Análisis por subperíodos (pre/durante/post COVID, 2022, reciente)
      2. Stress testing en crisis (2020, 2022)
      3. Sensibilidad de parámetros (lookback, vol_target, umbral crash)
      4. Evaluación out-of-sample (train 2010-2018, test 2019-hoy)
    """
    print("\n" + "═" * 62)
    print("  TESTS DE ROBUSTEZ")
    print("═" * 62)

    resultados_base = backtesting(precios, tipo_señal)
    rets = resultados_base["retornos_port"]

    # ── 1. Subperíodos ──────────────────────────────────────────────────────
    print("\n── 1. ANÁLISIS POR SUBPERÍODOS ─────────────────────────────")
    subperiodos = {
        "Pre-COVID (2010-2019)":       ("2010-01-01", "2019-12-31"),
        "COVID crash (Q1-Q2 2020)":    ("2020-01-01", "2020-06-30"),
        "COVID recovery (2020-2021)":  ("2020-07-01", "2021-12-31"),
        "Bear market 2022":            ("2022-01-01", "2022-12-31"),
        "Reciente (2023-hoy)":         ("2023-01-01", "2099-01-01"),
    }
    for label, (ini, fin) in subperiodos.items():
        m = _metricas_subperiodo(rets.loc[ini:fin], label)
        if m:
            print(f"  {m['Período']:<35} Ret: {m['Retorno']:>7} | "
                  f"Sharpe: {m['Sharpe']:>7} | MaxDD: {m['MaxDD']:>7}")

    # ── 2. Stress testing ───────────────────────────────────────────────────
    print("\n── 2. STRESS TESTING EN CRISIS ────────────────────────────")
    crisis = {
        "COVID crash agudo (Feb-Mar 2020)":    ("2020-02-01", "2020-03-31"),
        "Selloff extendido (Feb-May 2020)":    ("2020-02-01", "2020-05-31"),
        "Bear 2022 completo":                   ("2022-01-01", "2022-10-31"),
        "Q1 2022 (suba de tasas)":             ("2022-01-01", "2022-03-31"),
    }
    for label, (ini, fin) in crisis.items():
        r = rets.loc[ini:fin]
        if len(r) > 5:
            ret_tot = (1 + r).prod() - 1
            vol_c   = r.std() * np.sqrt(252)
            print(f"  {label:<42} Ret total: {ret_tot:>7.2%} | Vol anual: {vol_c:.2%}")

    # ── 3. Sensibilidad de parámetros ───────────────────────────────────────
    print("\n── 3. SENSIBILIDAD DE PARÁMETROS ───────────────────────────")
    retornos  = calcular_retornos(precios)
    vol_base  = _volatilidad_ewma(retornos)

    def _sharpe_rapido(señales):
        p = sizing_por_activo(señales, vol_base)
        p = escalar_portafolio(p, retornos)
        p = crash_filter(p, retornos)
        p = _aplicar_limites_riesgo(p)
        r = (p.shift(1) * retornos).sum(axis=1).dropna()
        s = r.std() * np.sqrt(252)
        return (r.mean() * 252) / s if s > 0 else np.nan

    print("\n  a) Lookback señal simple (meses de trading):")
    for lb in [126, 189, 252, 315]:
        sh = _sharpe_rapido(señal_simple(retornos, lookback=lb))
        print(f"     Lookback {lb:>3}d (~{lb//21:>2}m): Sharpe = {sh:.3f}")

    orig_vt = CONFIG["vol_target_port"]
    print("\n  b) Vol target portafolio:")
    for vt in [0.05, 0.08, 0.10, 0.15, 0.20]:
        CONFIG["vol_target_port"] = vt
        señ = señal_cta(retornos) if tipo_señal == "cta" else señal_simple(retornos)
        sh  = _sharpe_rapido(señ)
        print(f"     Vol target {vt:.0%}: Sharpe = {sh:.3f}")
    CONFIG["vol_target_port"] = orig_vt

    orig_umbral = CONFIG["crash_umbral"]
    print("\n  c) Umbral crash filter (φ):")
    for u in [1.0, 1.25, 1.5, 2.0, 999]:
        CONFIG["crash_umbral"] = u
        señ = señal_cta(retornos) if tipo_señal == "cta" else señal_simple(retornos)
        sh  = _sharpe_rapido(señ)
        etq = f"{u:.2f}" if u < 100 else "sin filtro"
        print(f"     Umbral {etq:>9}: Sharpe = {sh:.3f}")
    CONFIG["crash_umbral"] = orig_umbral

    # ── 4. Out-of-sample ────────────────────────────────────────────────────
    print("\n── 4. OUT-OF-SAMPLE (train 2010-2018 / test 2019-hoy) ──────")
    fecha_corte = CONFIG["oos_fecha_corte"]

    precios_is = precios.loc[:fecha_corte]
    res_is     = backtesting(precios_is, tipo_señal)
    print(f"\n  In-sample (inicio → {fecha_corte[:4]}):")
    calcular_metricas(res_is, f"In-sample ({tipo_señal.upper()})")

    # Extender hacia atrás 300 días para warm-up de indicadores
    idx_corte   = precios.index.searchsorted(pd.Timestamp(fecha_corte))
    start_idx   = max(0, idx_corte - 300)
    precios_ext = precios.iloc[start_idx:]
    res_full    = backtesting(precios_ext, tipo_señal)
    rets_oos    = res_full["retornos_port"].loc[fecha_corte:]

    try:
        turn_oos = res_full["turnover_mensual"].loc[fecha_corte:]
    except Exception:
        turn_oos = res_full["turnover_mensual"]

    eq_oos    = (1 + rets_oos).cumprod()
    res_oos   = {"retornos_port": rets_oos, "equity_curve": eq_oos,
                 "turnover_mensual": turn_oos}
    print(f"\n  Out-of-sample (2019 → hoy):")
    calcular_metricas(res_oos, f"Out-of-sample ({tipo_señal.upper()})")

    print("═" * 62 + "\n")
    return resultados_base


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_resultados(resultados, precios, guardar=True, mostrar=False):
    """
    Panel de 4 gráficos:
      1. Equity curve (log) vs benchmark 60/40
      2. Drawdown
      3. Rolling Sharpe a 12 meses
      4. Pesos netos por clase de activo (barras mensuales apiladas)
    """
    eq       = resultados["equity_curve"]
    eq_bench = resultados["equity_bench"]
    rets     = resultados["retornos_port"]
    pesos_h  = resultados["pesos_history"]
    tipo     = resultados["tipo_señal"].upper()

    fig, axes = plt.subplots(4, 1, figsize=(14, 20), facecolor="#f8f9fa")
    fig.suptitle(
        f"Estrategia TSMOM — Señal {tipo}\n"
        f"Período: {eq.index[0].date()} → {eq.index[-1].date()}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    col = {
        "estrategia":  "#1f77b4",
        "benchmark":   "#ff7f0e",
        "drawdown":    "#d62728",
        "sharpe":      "#2ca02c",
        "Equities":    "#1f77b4",
        "Fixed Income":"#ff7f0e",
        "Commodities": "#2ca02c",
        "FX":          "#9467bd",
    }

    # ── 1. Equity curve ───────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eq.index,       eq,       color=col["estrategia"], lw=1.8, label=f"TSMOM {tipo}")
    ax.plot(eq_bench.index, eq_bench, color=col["benchmark"],  lw=1.4, ls="--", label="Benchmark 60/40")
    ax.set_title("Equity Curve (escala log, base = 1)", fontweight="bold")
    ax.set_ylabel("Valor del portafolio")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Drawdown ───────────────────────────────────────────────────────
    ax = axes[1]
    dd   = (eq       - eq.cummax())       / eq.cummax()
    dd_b = (eq_bench - eq_bench.cummax()) / eq_bench.cummax()
    ax.fill_between(dd.index,   dd,   0, color=col["drawdown"],   alpha=0.55, label="TSMOM")
    ax.fill_between(dd_b.index, dd_b, 0, color=col["benchmark"],  alpha=0.30, label="60/40")
    ax.set_title("Drawdown", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(); ax.grid(alpha=0.3)

    # ── 3. Rolling Sharpe 12m ─────────────────────────────────────────────
    ax = axes[2]
    v   = 252
    rs  = (rets.rolling(v).mean() * 252) / (rets.rolling(v).std() * np.sqrt(252))
    ax.plot(rs.index, rs, color=col["sharpe"], lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(1, color="grey",  lw=0.8, ls=":", alpha=0.7, label="Sharpe = 1")
    ax.fill_between(rs.index, rs, 0, where=(rs >= 0), alpha=0.25, color=col["sharpe"])
    ax.fill_between(rs.index, rs, 0, where=(rs <  0), alpha=0.25, color=col["drawdown"])
    ax.set_title("Rolling Sharpe ratio (12 meses)", fontweight="bold")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(); ax.grid(alpha=0.3)

    # ── 4. Pesos por clase de activo ──────────────────────────────────────
    ax = axes[3]
    universo = CONFIG["universo"]
    clases   = {}
    for clase, tickers in universo.items():
        cols = [t for t in tickers if t in pesos_h.columns]
        if cols:
            clases[clase] = pesos_h[cols].sum(axis=1)

    try:
        pesos_mens = pd.DataFrame(clases).resample(_MONTH_END).mean()
    except Exception:
        pesos_mens = pd.DataFrame(clases).resample("ME").mean()

    bottom_pos = np.zeros(len(pesos_mens))
    bottom_neg = np.zeros(len(pesos_mens))
    for clase in pesos_mens.columns:
        vals = pesos_mens[clase].fillna(0).values
        pos  = np.maximum(vals, 0)
        neg  = np.minimum(vals, 0)
        ax.bar(pesos_mens.index, pos, bottom=bottom_pos,
               color=col.get(clase, "grey"), label=clase, width=20, alpha=0.85)
        ax.bar(pesos_mens.index, neg, bottom=bottom_neg,
               color=col.get(clase, "grey"), width=20, alpha=0.85)
        bottom_pos = bottom_pos + pos
        bottom_neg = bottom_neg + neg

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Pesos netos por Clase de Activo (promedio mensual)", fontweight="bold")
    ax.set_ylabel("Peso del portafolio")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left"); ax.grid(alpha=0.3, axis="y")

    # Formato de fechas en todos los ejes
    for ax_ in axes:
        ax_.xaxis.set_major_locator(mdates.YearLocator())
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax_.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if guardar:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        fname = os.path.join(project_root, "analysis", f"backtest_{tipo.lower()}.png")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"[plot] Guardado: {fname}")
    if mostrar:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BROKER — Alpaca Paper Trading
# ═══════════════════════════════════════════════════════════════════════════════

def conectar_alpaca():
    """
    Conecta a Alpaca (paper por defecto) usando variables de entorno.
    Requiere: ALPACA_API_KEY y ALPACA_API_SECRET.
    Nunca hardcodear credenciales en el código.
    """
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        print("[broker] ERROR: pip install alpaca-trade-api")
        sys.exit(1)

    api_key    = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("[broker] ERROR: variables de entorno no configuradas.")
        print("  Windows PowerShell:")
        print("    $env:ALPACA_API_KEY='tu_key'")
        print("    $env:ALPACA_API_SECRET='tu_secret'")
        print("  Linux / macOS:")
        print("    export ALPACA_API_KEY='tu_key'")
        print("    export ALPACA_API_SECRET='tu_secret'")
        sys.exit(1)

    base_url = (CONFIG["alpaca_base_url"] if CONFIG["alpaca_paper"]
                else "https://api.alpaca.markets")
    api = tradeapi.REST(api_key, api_secret, base_url=base_url, api_version="v2")

    cuenta = api.get_account()
    modo   = "PAPER" if CONFIG["alpaca_paper"] else "LIVE"
    print(f"[broker] Conectado a Alpaca {modo}")
    print(f"[broker] Equity: ${float(cuenta.equity):,.2f} | "
          f"Cash: ${float(cuenta.cash):,.2f} | Status: {cuenta.status}")
    return api


def obtener_posiciones_actuales(api):
    """Retorna DataFrame con posiciones actuales (qty, valor, P&L)."""
    posiciones = api.list_positions()
    if not posiciones:
        print("[broker] Sin posiciones abiertas.")
        return pd.DataFrame()

    datos = [
        {
            "ticker":    p.symbol,
            "qty":       float(p.qty),
            "valor_mkt": float(p.market_value),
            "pnl":       float(p.unrealized_pl),
            "pnl_pct":   float(p.unrealized_plpc),
        }
        for p in posiciones
    ]
    df = pd.DataFrame(datos).set_index("ticker")
    print(f"[broker] {len(df)} posiciones abiertas:")
    print(df.to_string())
    return df


def calcular_ordenes(pesos_objetivo, capital_total, precios_actuales, posiciones_actuales):
    """
    Calcula el delta de shares entre la posición actual y la objetivo.

    pesos_objetivo:    Series (ticker → peso objetivo como fracción del capital)
    capital_total:     float (equity de la cuenta)
    precios_actuales:  dict (ticker → precio)
    posiciones_actuales: DataFrame indexado por ticker con columna 'qty'
    """
    ordenes = []
    for ticker, peso_obj in pesos_objetivo.items():
        precio = precios_actuales.get(ticker)
        if not precio or precio <= 0:
            continue

        qty_obj = int(peso_obj * capital_total / precio)
        qty_act = int(posiciones_actuales.loc[ticker, "qty"]) \
                  if ticker in posiciones_actuales.index else 0
        delta = qty_obj - qty_act

        if abs(delta) > 0:
            ordenes.append({
                "ticker":       ticker,
                "qty_actual":   qty_act,
                "qty_objetivo": qty_obj,
                "delta":        delta,
                "lado":         "buy" if delta > 0 else "sell",
            })

    df = pd.DataFrame(ordenes)
    if df.empty:
        print("[broker] Sin cambios en posiciones.")
    else:
        print(f"[broker] {len(df)} órdenes calculadas:")
        print(df.to_string(index=False))
    return df


def ejecutar_ordenes(api, df_ordenes):
    """Envía las órdenes a Alpaca como market orders y loguea cada una con timestamp."""
    if df_ordenes.empty:
        return pd.DataFrame()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = []
    for _, orden in df_ordenes.iterrows():
        try:
            resp = api.submit_order(
                symbol=orden["ticker"],
                qty=abs(int(orden["delta"])),
                side=orden["lado"],
                type="market",
                time_in_force="day",
            )
            print(f"[{ts}] {orden['lado'].upper():4} {abs(int(orden['delta'])):>6} "
                  f"{orden['ticker']:<6} → order_id={resp.id} status={resp.status}")
            log.append({"timestamp": ts, "ticker": orden["ticker"],
                        "lado": orden["lado"], "qty": abs(int(orden["delta"])),
                        "order_id": resp.id, "status": resp.status})
        except Exception as e:
            print(f"[{ts}] ERROR {orden['ticker']}: {e}")

    return pd.DataFrame(log)


def monitorear_portfolio(api):
    """Muestra posiciones y P&L en tiempo real, actualizando cada 30 segundos."""
    import time
    print("[monitor] Iniciando. Ctrl+C para detener.\n")
    try:
        while True:
            cuenta = api.get_account()
            ts     = datetime.now().strftime("%H:%M:%S")
            pnl_d  = float(cuenta.equity) - float(cuenta.last_equity)
            print(f"\n[{ts}] ── PORTFOLIO ────────────────────────────────────")
            print(f"  Equity:     ${float(cuenta.equity):>13,.2f}")
            print(f"  Cash:       ${float(cuenta.cash):>13,.2f}")
            print(f"  P&L diario: ${pnl_d:>+13,.2f}")
            posiciones = api.list_positions()
            if posiciones:
                print(f"\n  {'Ticker':<8} {'Qty':>8} {'Valor Mkt':>14} "
                      f"{'P&L $':>10} {'P&L%':>8}")
                print("  " + "─" * 52)
                for p in posiciones:
                    print(f"  {p.symbol:<8} {float(p.qty):>8.0f} "
                          f"${float(p.market_value):>12,.2f} "
                          f"${float(p.unrealized_pl):>9,.2f} "
                          f"{float(p.unrealized_plpc):>7.2%}")
            else:
                print("  Sin posiciones.")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[monitor] Detenido.")


def _precios_actuales_alpaca(api, tickers):
    """Obtiene el último precio de cierre para cada ticker vía Alpaca."""
    precios = {}
    for t in tickers:
        try:
            barra = api.get_latest_bar(t)
            precios[t] = barra.c
        except Exception:
            pass
    return precios


def ejecutar_live(precios_historicos, tipo_señal="cta"):
    """Modo live: calcula pesos objetivo y ejecuta las órdenes en Alpaca paper."""
    print("\n[live] ══ MODO LIVE ═══════════════════════════════════════")
    api     = conectar_alpaca()
    cuenta  = api.get_account()
    capital = float(cuenta.equity)

    retornos    = calcular_retornos(precios_historicos)
    vol_activos = _volatilidad_ewma(retornos)

    señales = señal_cta(retornos) if tipo_señal == "cta" else señal_simple(retornos)

    pesos = sizing_por_activo(señales, vol_activos)
    pesos = escalar_portafolio(pesos, retornos)
    pesos = crash_filter(pesos, retornos)
    pesos = _aplicar_limites_riesgo(pesos)

    # Pesos del último día disponible
    pesos_hoy = pesos.iloc[-1].dropna()
    pesos_hoy = pesos_hoy[pesos_hoy.abs() > 1e-3]

    print(f"\n[live] Pesos objetivo ({datetime.today().date()}):")
    for t, w in pesos_hoy.sort_values(key=abs, ascending=False).items():
        print(f"  {t:<6} {w:>+.4f}")

    posiciones     = obtener_posiciones_actuales(api)
    precios_ahora  = _precios_actuales_alpaca(api, list(pesos_hoy.index))
    df_ordenes     = calcular_ordenes(pesos_hoy, capital, precios_ahora, posiciones)
    ejecutar_ordenes(api, df_ordenes)
    print("[live] Ejecución completada.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _construir_parser():
    parser = argparse.ArgumentParser(
        prog="estrategia.py",
        description="TSMOM — Trend Following Strategy | TP Final F414 — UdeSA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python estrategia.py --modo backtest --señal cta
  python estrategia.py --modo backtest --señal simple
  python estrategia.py --modo robustez
  python estrategia.py --modo live
  python estrategia.py --modo monitor
        """,
    )
    parser.add_argument(
        "--modo",
        choices=["backtest", "robustez", "live", "monitor"],
        required=True,
        help="Modo de ejecución",
    )
    parser.add_argument(
        "--señal", dest="senal",
        choices=["cta", "simple"],
        default="cta",
        help="Tipo de señal: 'cta' (Baz 2015) o 'simple' (Moskowitz 2012). Default: cta",
    )
    parser.add_argument(
        "--inicio",
        default=CONFIG["fecha_inicio"],
        help=f"Fecha inicio del backtest (YYYY-MM-DD). Default: {CONFIG['fecha_inicio']}",
    )
    parser.add_argument(
        "--fin",
        default=None,
        help="Fecha fin (YYYY-MM-DD). Default: hoy",
    )
    parser.add_argument(
        "--guardar-graficos",
        action="store_true",
        default=True,
        help="Guardar gráficos en analysis/ (default: True)",
    )
    parser.add_argument(
        "--mostrar-graficos",
        action="store_true",
        default=False,
        help="Mostrar gráficos en pantalla (requiere display interactivo)",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = _construir_parser()
    args   = parser.parse_args()

    CONFIG["fecha_inicio"] = args.inicio
    if args.fin:
        CONFIG["fecha_fin"] = args.fin

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Time Series Momentum (TSMOM) — Trend Following Strategy    ║")
    print("║  TP Final F414 — Universidad de San Andrés — Mayo 2026      ║")
    print("║  Millozi · Lopez Fresco · Polakoff · Castro Darrigo         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    if args.modo == "monitor":
        api = conectar_alpaca()
        monitorear_portfolio(api)
        return

    precios = descargar_datos()

    if args.modo == "backtest":
        resultados = backtesting(precios, tipo_señal=args.senal)
        calcular_metricas(resultados, f"TSMOM {args.senal.upper()}")
        plot_resultados(resultados, precios,
                        guardar=args.guardar_graficos,
                        mostrar=args.mostrar_graficos)

    elif args.modo == "robustez":
        tests_robustez(precios, tipo_señal=args.senal)
        resultados = backtesting(precios, tipo_señal=args.senal)
        plot_resultados(resultados, precios,
                        guardar=args.guardar_graficos,
                        mostrar=args.mostrar_graficos)

    elif args.modo == "live":
        ejecutar_live(precios, tipo_señal=args.senal)


if __name__ == "__main__":
    main()
