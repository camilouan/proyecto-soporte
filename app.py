from __future__ import annotations

import io
import logging
import os
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, flash, redirect, render_template, request, send_file, url_for


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PLOT_DIR = STATIC_DIR / "plots"
TEMPLATE_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "support_tickets_bogota.csv"

DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
DAY_COLORS = {
    "Lunes": "#2F80ED",
    "Martes": "#56CCF2",
    "Miércoles": "#27AE60",
    "Jueves": "#F2994A",
    "Viernes": "#EB5757",
}
ZONES = [
    "Suba",
    "Engativá",
    "Chapinero",
    "Usaquén",
    "Kennedy",
]

WEEKDAY_BASE_LOAD = {
    "Lunes": 24,
    "Martes": 22,
    "Miércoles": 20,
    "Jueves": 21,
    "Viernes": 26,
}
DAY_DELAY = {
    "Lunes": 4,
    "Martes": 3,
    "Miércoles": 2,
    "Jueves": 3,
    "Viernes": 6,
}
ZONE_DELAY = {
    "Suba": 4,
    "Engativá": 3,
    "Chapinero": 2,
    "Usaquén": 3,
    "Kennedy": 5,
}
ZONE_WEIGHTS = [0.24, 0.2, 0.16, 0.17, 0.23]
PRIORITIES = ["Baja", "Media", "Alta", "Crítica"]
PRIORITY_WEIGHTS = [0.18, 0.46, 0.28, 0.08]
PRIORITY_EFFECT = {
    "Baja": 4,
    "Media": 2,
    "Alta": -1,
    "Crítica": -4,
}
CATEGORIES = ["Red", "Software", "Hardware", "Accesos", "Correo"]
CHANNELS = ["Portal", "Correo", "Teléfono", "Chat"]
MAP_MAX_POINTS = 220
MODEL_SAMPLE_X = 15

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

ZONE_NAME_MAP = {
    "Suba": "Suba",
    "Engativá": "Engativa",
    "Chapinero": "Chapinero",
    "Usaquén": "Usaquen",
    "Kennedy": "Kennedy",
}
BOGOTA_CENTROIDS = {
    "chapinero": (4.643856, -74.039600),
    "engativa": (4.708040, -74.124945),
    "kennedy": (4.627220, -74.158625),
    "suba": (4.795879, -74.089860),
    "usaquen": (4.741166, -74.026212),
}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "proyecto-electiva-secret")


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip().lower()


def weekday_name(input_date: date) -> str:
    names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    return names[input_date.weekday()]


def create_realistic_dataset(seed: int = 22, total_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = date.today() - timedelta(days=total_days)
    rows: list[dict[str, object]] = []
    ticket_seq = 10001

    for day_offset in range(total_days + 1):
        current_date = start_date + timedelta(days=day_offset)
        day_name = weekday_name(current_date)

        if day_name in {"Sábado", "Domingo"}:
            continue

        expected_load = WEEKDAY_BASE_LOAD[day_name]
        daily_records = int(rng.integers(expected_load - 5, expected_load + 6))

        for _ in range(daily_records):
            zone = str(rng.choice(ZONES, p=ZONE_WEIGHTS))
            priority = str(rng.choice(PRIORITIES, p=PRIORITY_WEIGHTS))
            category = str(rng.choice(CATEGORIES))
            channel = str(rng.choice(CHANNELS))

            tickets = int(rng.integers(1, 28))
            noise = int(rng.integers(-4, 5))
            base_time = 9 + (1.75 * tickets)
            time = int(
                round(
                    base_time
                    + DAY_DELAY[day_name]
                    + ZONE_DELAY[zone]
                    + PRIORITY_EFFECT[priority]
                    + noise
                )
            )
            time = max(8, min(time, 95))

            centroid = BOGOTA_CENTROIDS.get(_normalize_name(ZONE_NAME_MAP[zone]), (4.65, -74.10))
            latitude = centroid[0] + rng.uniform(-0.012, 0.012)
            longitude = centroid[1] + rng.uniform(-0.012, 0.012)

            rows.append(
                {
                    "ticket_id": f"TK-{ticket_seq}",
                    "fecha": current_date.isoformat(),
                    "dia": day_name,
                    "zona": zone,
                    "categoria": category,
                    "prioridad": priority,
                    "canal": channel,
                    "tickets": tickets,
                    "tiempo": time,
                    "lat": round(latitude, 6),
                    "lon": round(longitude, 6),
                }
            )
            ticket_seq += 1

    data = pd.DataFrame(rows)
    data["dia"] = pd.Categorical(data["dia"], categories=DAYS, ordered=True)
    return data.sort_values(["fecha", "dia", "tickets"]).reset_index(drop=True)


def load_or_create_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists():
        data = pd.read_csv(DATASET_PATH)
        logger.info("Dataset cargado desde %s con %s filas", DATASET_PATH, len(data))
    else:
        data = create_realistic_dataset()
        data.to_csv(DATASET_PATH, index=False)
        logger.info("Dataset creado en %s con %s filas", DATASET_PATH, len(data))

    data["dia"] = pd.Categorical(data["dia"], categories=DAYS, ordered=True)
    return data.sort_values(["dia", "tickets"]).reset_index(drop=True)


def build_dataset(seed: int = 7, records_per_day: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for day_index, day in enumerate(DAYS):
        day_offset = day_index * 3
        for item_index in range(records_per_day):
            tickets = int(rng.integers(2, 26))
            noise = int(rng.integers(-3, 4))
            time = max(5, 2 * tickets + 5 + day_offset + noise)
            zone = ZONES[(day_index + item_index) % len(ZONES)]
            centroid = BOGOTA_CENTROIDS.get(_normalize_name(ZONE_NAME_MAP[zone]), (4.65, -74.10))
            latitude = centroid[0] + rng.uniform(-0.012, 0.012)
            longitude = centroid[1] + rng.uniform(-0.012, 0.012)

            rows.append(
                {
                    "tickets": tickets,
                    "tiempo": time,
                    "dia": day,
                    "zona": zone,
                    "lat": round(latitude, 6),
                    "lon": round(longitude, 6),
                }
            )

    data = pd.DataFrame(rows)
    data["dia"] = pd.Categorical(data["dia"], categories=DAYS, ordered=True)
    return data.sort_values(["dia", "tickets"]).reset_index(drop=True)


def ensure_directories() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_chart(data: pd.DataFrame) -> str:
    averages = data.groupby("dia", observed=False)["tiempo"].mean().reindex(DAYS)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        averages.index,
        averages.values,
        color=[DAY_COLORS[day] for day in averages.index],
        edgecolor="#1f2937",
        linewidth=0.8,
    )

    ax.set_title("Tiempo promedio por día", fontsize=15, fontweight="bold")
    ax.set_xlabel("Día")
    ax.set_ylabel("Tiempo promedio (min)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, averages.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output = PLOT_DIR / "tiempo_por_dia.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(STATIC_DIR).as_posix())


def save_scatter_chart(data: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))

    for day in DAYS:
        subset = data[data["dia"] == day]
        ax.scatter(
            subset["tickets"],
            subset["tiempo"],
            s=55,
            alpha=0.85,
            label=day,
            color=DAY_COLORS[day],
            edgecolors="white",
            linewidths=0.5,
        )

    slope, intercept = np.polyfit(data["tickets"], data["tiempo"], 1)
    x_line = np.linspace(data["tickets"].min(), data["tickets"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=2.2, label="Tendencia")

    ax.set_title("Relación entre tickets y tiempo", fontsize=15, fontweight="bold")
    ax.set_xlabel("Tickets")
    ax.set_ylabel("Tiempo (min)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    output = PLOT_DIR / "tickets_vs_tiempo.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(STATIC_DIR).as_posix())


def save_category_chart(data: pd.DataFrame) -> str:
    category_avg = data.groupby("categoria", observed=False)["tiempo"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        category_avg.index,
        category_avg.values,
        color="#5DADE2",
        edgecolor="#1f2937",
        linewidth=0.8,
    )

    ax.set_title("Tiempo promedio por categoría", fontsize=15, fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Tiempo promedio (min)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, category_avg.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output = PLOT_DIR / "tiempo_por_categoria.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(STATIC_DIR).as_posix())


def save_trend_chart(data: pd.DataFrame) -> str:
    trend = data.groupby("fecha", observed=False)["tiempo"].mean().reset_index()
    trend["fecha"] = pd.to_datetime(trend["fecha"])
    trend["rolling"] = trend["tiempo"].rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(trend["fecha"], trend["tiempo"], color="#94a3b8", linewidth=1.6, alpha=0.8, label="Promedio diario")
    ax.plot(trend["fecha"], trend["rolling"], color="#10b981", linewidth=2.4, label="Tendencia 7 días")

    ax.set_title("Tendencia temporal del tiempo de atención", fontsize=15, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tiempo promedio (min)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    output = PLOT_DIR / "tendencia_tiempo.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(STATIC_DIR).as_posix())


def save_map_chart(data: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        data["lon"],
        data["lat"],
        c=data["tiempo"],
        cmap="viridis",
        s=80,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.6,
    )

    for zone in ZONES:
        sample = data[data["zona"] == zone].iloc[0]
        ax.text(sample["lon"] + 0.002, sample["lat"] + 0.002, zone, fontsize=9, weight="bold")

    ax.set_title("Mapa simulado de Bogotá por tiempo de atención", fontsize=15, fontweight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(alpha=0.15)
    fig.colorbar(scatter, ax=ax, label="Tiempo (min)")

    fig.tight_layout()
    output = PLOT_DIR / "mapa_bogota.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(STATIC_DIR).as_posix())


DATA = load_or_create_dataset()


def build_summary(data: pd.DataFrame) -> dict[str, object]:
    corr = float(data["tickets"].corr(data["tiempo"]))
    slope, intercept = np.polyfit(data["tickets"], data["tiempo"], 1)

    avg_by_day = (
        data.groupby("dia", observed=False)["tiempo"].mean().reindex(DAYS).round(1)
    )
    zone_summary = (
        data.groupby("zona", observed=False)["tiempo"].mean().sort_values(ascending=False).round(1)
    )

    busiest_day = avg_by_day.idxmax()
    busiest_zone = zone_summary.idxmax()
    peak_time = int(data["tiempo"].max())
    category_summary = (
        data.groupby("categoria", observed=False)["tiempo"].mean().sort_values(ascending=False).round(1)
    )
    priority_summary = (
        data.groupby("prioridad", observed=False)["tiempo"].mean().round(1).to_dict()
    )
    daily_trend = (
        data.groupby("fecha", observed=False)["tiempo"].mean().tail(7).round(1).to_list()
    )
    trend_delta = round(float(daily_trend[-1] - daily_trend[0]), 1) if len(daily_trend) >= 2 else 0.0

    return {
        "corr": round(corr, 2),
        "slope": round(float(slope), 2),
        "intercept": round(float(intercept), 2),
        "busiest_day": str(busiest_day),
        "busiest_zone": str(busiest_zone),
        "peak_time": peak_time,
        "avg_by_day": avg_by_day.to_dict(),
        "zone_summary": zone_summary.to_dict(),
        "category_summary": category_summary.to_dict(),
        "priority_summary": priority_summary,
        "trend_delta": trend_delta,
    }


def build_quality_report(data: pd.DataFrame) -> dict[str, object]:
    null_cells = int(data.isna().sum().sum())
    duplicate_ids = int(data.duplicated(subset=["ticket_id"]).sum()) if "ticket_id" in data.columns else 0
    invalid_ticket_rows = int((data["tickets"] <= 0).sum()) if "tickets" in data.columns else 0
    invalid_time_rows = int((data["tiempo"] <= 0).sum()) if "tiempo" in data.columns else 0

    q1 = float(data["tiempo"].quantile(0.25))
    q3 = float(data["tiempo"].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_rows = int(((data["tiempo"] < lower) | (data["tiempo"] > upper)).sum())

    total_rows = max(len(data), 1)
    quality_score = 100 - round(
        ((null_cells + duplicate_ids + invalid_ticket_rows + invalid_time_rows + outlier_rows) / total_rows) * 100,
        1,
    )

    return {
        "null_cells": null_cells,
        "duplicate_ids": duplicate_ids,
        "invalid_ticket_rows": invalid_ticket_rows,
        "invalid_time_rows": invalid_time_rows,
        "outlier_rows": outlier_rows,
        "quality_score": max(0.0, quality_score),
        "iqr_limits": {"lower": round(lower, 2), "upper": round(upper, 2)},
    }


def build_model_diagnostics(data: pd.DataFrame) -> dict[str, object]:
    x = data["tickets"].astype(float).to_numpy()
    y = data["tiempo"].astype(float).to_numpy()
    if len(x) < 3:
        return {
            "linear_rmse": 0.0,
            "poly2_rmse": 0.0,
            "recommended": "Lineal",
            "note": "Muestra insuficiente para comparar modelos.",
        }

    linear_coef = np.polyfit(x, y, 1)
    poly2_coef = np.polyfit(x, y, 2)
    pred_linear = np.polyval(linear_coef, x)
    pred_poly2 = np.polyval(poly2_coef, x)

    linear_rmse = float(np.sqrt(np.mean((y - pred_linear) ** 2)))
    poly2_rmse = float(np.sqrt(np.mean((y - pred_poly2) ** 2)))
    recommended = "Polinómico grado 2" if poly2_rmse + 0.5 < linear_rmse else "Lineal"
    note = (
        "El modelo lineal es suficiente y más interpretable para el objetivo del proyecto."
        if recommended == "Lineal"
        else "El modelo polinómico mejora el ajuste, pero reduce interpretabilidad operacional."
    )

    return {
        "linear_rmse": round(linear_rmse, 3),
        "poly2_rmse": round(poly2_rmse, 3),
        "recommended": recommended,
        "note": note,
    }


def apply_filters(data: pd.DataFrame, params: dict[str, str]) -> pd.DataFrame:
    filtered = data.copy()

    zone = params.get("zona", "")
    priority = params.get("prioridad", "")
    category = params.get("categoria", "")
    channel = params.get("canal", "")
    day = params.get("dia", "")
    start_date = params.get("fecha_inicio", "")
    end_date = params.get("fecha_fin", "")

    if zone:
        filtered = filtered[filtered["zona"] == zone]
    if priority:
        filtered = filtered[filtered["prioridad"] == priority]
    if category:
        filtered = filtered[filtered["categoria"] == category]
    if channel:
        filtered = filtered[filtered["canal"] == channel]
    if day:
        filtered = filtered[filtered["dia"] == day]
    if start_date:
        filtered = filtered[filtered["fecha"] >= start_date]
    if end_date:
        filtered = filtered[filtered["fecha"] <= end_date]

    return filtered.reset_index(drop=True)


def get_filter_options(data: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "zonas": sorted(data["zona"].dropna().unique().tolist()),
        "prioridades": sorted(data["prioridad"].dropna().unique().tolist()),
        "categorias": sorted(data["categoria"].dropna().unique().tolist()),
        "canales": sorted(data["canal"].dropna().unique().tolist()),
        "dias": DAYS,
    }


def _get_current_filters() -> dict[str, str]:
    return {
        "zona": request.args.get("zona", "").strip(),
        "prioridad": request.args.get("prioridad", "").strip(),
        "categoria": request.args.get("categoria", "").strip(),
        "canal": request.args.get("canal", "").strip(),
        "dia": request.args.get("dia", "").strip(),
        "fecha_inicio": request.args.get("fecha_inicio", "").strip(),
        "fecha_fin": request.args.get("fecha_fin", "").strip(),
    }


def _resolve_filtered_data() -> tuple[pd.DataFrame, dict[str, str], bool, str]:
    filters = _get_current_filters()
    if (
        filters.get("fecha_inicio")
        and filters.get("fecha_fin")
        and filters["fecha_inicio"] > filters["fecha_fin"]
    ):
        return DATA.copy(), filters, True, "Rango de fechas inválido: fecha inicio es mayor que fecha fin."

    filtered = apply_filters(DATA, filters)
    had_filters = any(bool(value) for value in filters.values())
    if filtered.empty:
        if had_filters:
            return DATA.copy(), filters, True, "No se encontraron registros con esos filtros. Se muestra el dataset completo."
        return DATA.copy(), filters, False, ""
    return filtered, filters, False, ""


def build_map_sample(data: pd.DataFrame, max_points: int = MAP_MAX_POINTS) -> pd.DataFrame:
    if len(data) <= max_points:
        return data.copy()

    per_zone = max(1, max_points // len(ZONES))
    sampled_parts: list[pd.DataFrame] = []
    selected_idx: set[int] = set()

    for zone in ZONES:
        zone_rows = data[data["zona"] == zone]
        take = min(len(zone_rows), per_zone)
        if take == 0:
            continue
        sample = zone_rows.sample(n=take, random_state=42)
        sampled_parts.append(sample)
        selected_idx.update(int(idx) for idx in sample.index)

    sampled = pd.concat(sampled_parts) if sampled_parts else pd.DataFrame(columns=data.columns)

    remaining = max_points - len(sampled)
    if remaining > 0:
        pool = data.loc[~data.index.isin(selected_idx)]
        if not pool.empty:
            extra = pool.sample(n=min(remaining, len(pool)), random_state=21)
            sampled = pd.concat([sampled, extra])

    return sampled.sort_values(["dia", "tickets"]).reset_index(drop=True)


@app.route("/")
def home():
    active_data, active_filters, fallback_used, fallback_message = _resolve_filtered_data()
    summary = build_summary(active_data)
    map_data = build_map_sample(active_data)
    quality_report = build_quality_report(active_data)
    model_diagnostics = build_model_diagnostics(active_data)
    charts = {
        "bar": save_bar_chart(active_data),
        "scatter": save_scatter_chart(active_data),
        "category": save_category_chart(active_data),
        "trend": save_trend_chart(active_data),
    }

    dataset_profile = {
        "rows": int(len(DATA)),
        "columns": list(DATA.columns),
        "date_min": str(DATA["fecha"].min()),
        "date_max": str(DATA["fecha"].max()),
        "source": str(DATASET_PATH.relative_to(BASE_DIR).as_posix()),
        "total_tickets": int(active_data["tickets"].sum()),
        "avg_tickets": round(float(active_data["tickets"].mean()), 2),
        "avg_time": round(float(active_data["tiempo"].mean()), 2),
        "min_time": int(active_data["tiempo"].min()),
        "max_time": int(active_data["tiempo"].max()),
        "corr": summary["corr"],
        "slope": summary["slope"],
        "intercept": summary["intercept"],
    }

    methods = [
        "Correlación: coeficiente de Pearson entre tickets y tiempo.",
        "Pendiente e intercepto: regresión lineal con mínimos cuadrados (numpy.polyfit).",
        "Promedios por grupo: media aritmética con groupby() por día, zona, categoría y prioridad.",
        "Rangos y totales: suma, mínimo y máximo directos sobre columnas numéricas.",
    ]

    stats = {
        "total_tickets": int(active_data["tickets"].sum()),
        "avg_tickets": round(float(active_data["tickets"].mean()), 1),
        "avg_time": round(float(active_data["tiempo"].mean()), 1),
        "max_time": int(active_data["tiempo"].max()),
        "min_time": int(active_data["tiempo"].min()),
    }

    model = {
        "function": f"T(x) = {summary['slope']}x + {summary['intercept']}",
        "derivative": f"T'(x) = {summary['slope']}",
        "integral": (
            f"\u222b({summary['slope']}x + {summary['intercept']})dx = "
            f"{round(summary['slope'] / 2, 2)}x^2 + {summary['intercept']}x + C"
        ),
        "interpretation": (
            f"El modelo ajustado al dataset indica que cada ticket adicional incrementa el tiempo en "
            f"aproximadamente {summary['slope']} minutos."
        ),
        "sample_x": MODEL_SAMPLE_X,
        "sample_t": round((summary["slope"] * MODEL_SAMPLE_X) + summary["intercept"], 1),
        "sample_increment": round(summary["slope"] * 5, 1),
    }

    insights = [
        {
            "label": "Correlación",
            "value": summary["corr"],
            "note": "Señala una relación positiva fuerte entre tickets y tiempo.",
        },
        {
            "label": "Pendiente estimada",
            "value": summary["slope"],
            "note": "Por cada ticket adicional, el tiempo crece aproximadamente así.",
        },
        {
            "label": "Día más cargado",
            "value": summary["busiest_day"],
            "note": "El día con mayor tiempo promedio de atención.",
        },
        {
            "label": "Zona crítica",
            "value": summary["busiest_zone"],
            "note": "La zona con mayor tiempo promedio observado.",
        },
        {
            "label": "Tendencia semanal",
            "value": f"{summary['trend_delta']} min",
            "note": "Cambio del promedio diario en los últimos 7 días del dataset.",
        },
    ]

    return render_template(
        "index.html",
        data=active_data.to_dict(orient="records"),
        map_data=map_data.to_dict(orient="records"),
        map_meta={
            "displayed_points": int(len(map_data)),
            "total_points": int(len(active_data)),
        },
        dataset_profile=dataset_profile,
        methods=methods,
        fallback_used=fallback_used,
        fallback_message=fallback_message,
        quality_report=quality_report,
        model_diagnostics=model_diagnostics,
        filter_options=get_filter_options(DATA),
        active_filters=active_filters,
        stats=stats,
        avg_by_day=summary["avg_by_day"],
        zone_summary=summary["zone_summary"],
        summary=summary,
        insights=insights,
        model=model,
        charts=charts,
    )


@app.route("/export/filtered.csv")
def export_filtered_csv():
    try:
        filtered, filters, _, _ = _resolve_filtered_data()
        output = io.StringIO()
        filtered.to_csv(output, index=False)
        output.seek(0)
        logger.info("Export CSV generado con %s filas", len(filtered))
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="tickets_filtrados.csv",
        )
    except Exception as exc:
        logger.exception("Error exportando CSV")
        flash(f"No se pudo exportar CSV: {exc}", "error")
        return redirect(url_for("home", **_get_current_filters()))


@app.route("/export/summary.xlsx")
def export_summary_excel():
    try:
        filtered, _, _, _ = _resolve_filtered_data()
        summary = build_summary(filtered)
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="data_filtrada")
            pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="resumen")

        buffer.seek(0)
        logger.info("Export Excel generado con %s filas", len(filtered))
        return send_file(
            buffer,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="resumen_tickets.xlsx",
        )
    except Exception as exc:
        logger.exception("Error exportando Excel")
        flash(f"No se pudo exportar Excel: {exc}", "error")
        return redirect(url_for("home", **_get_current_filters()))


@app.route("/export/report.pdf")
def export_report_pdf():
    try:
        filtered, _, _, _ = _resolve_filtered_data()
        summary = build_summary(filtered)

        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        y = height - 50
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(40, y, "Reporte de Soporte Técnico - Resumen Filtrado")
        y -= 30
        pdf.setFont("Helvetica", 11)
        lines = [
            f"Registros analizados: {len(filtered)}",
            f"Promedio de tiempo: {round(float(filtered['tiempo'].mean()), 2)} min",
            f"Correlación tickets-tiempo: {summary['corr']}",
            f"Modelo lineal: T(x) = {summary['slope']}x + {summary['intercept']}",
            f"Día más cargado: {summary['busiest_day']}",
            f"Zona crítica: {summary['busiest_zone']}",
        ]
        for line in lines:
            pdf.drawString(40, y, line)
            y -= 18

        pdf.save()
        buffer.seek(0)
        logger.info("Export PDF generado con %s filas", len(filtered))
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="reporte_soporte.pdf",
        )
    except Exception as exc:
        logger.exception("Error exportando PDF")
        flash(f"No se pudo exportar PDF: {exc}", "error")
        return redirect(url_for("home", **_get_current_filters()))


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
