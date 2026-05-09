from __future__ import annotations

import base64
import io
import logging
import os
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

# Rutas base del proyecto. Se calculan desde app.py para que funcionen igual
# en local y en Render sin depender de la carpeta desde donde se ejecute Python.
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PLOT_DIR = STATIC_DIR / "plots"
TEMPLATE_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "support_tickets_bogota.csv"

# Constantes del dominio: dias, zonas, prioridades y reglas usadas para
# construir un dataset realista cuando el CSV aun no existe.
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

# Configuracion de logs para ver en consola o Render que esta haciendo la app.
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Equivalencias y centroides locales de localidades. Se usan para ubicar puntos
# del mapa sin bloquear el arranque del servidor con una descarga externa.
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
    """Convierte nombres con tildes a una forma comparable y en minusculas."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip().lower()


def weekday_name(input_date: date) -> str:
    """Devuelve el nombre del dia de la semana en espanol para una fecha."""
    names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    return names[input_date.weekday()]


def create_realistic_dataset(seed: int = 22, total_days: int = 60) -> pd.DataFrame:
    """Crea datos sinteticos pero coherentes para soporte tecnico en Bogota.

    Cada fila representa un registro de atencion con fecha, localidad,
    categoria, prioridad, canal, numero de tickets y tiempo de respuesta.
    Las reglas de demora hacen que el tiempo suba cuando hay mas carga,
    ciertos dias o ciertas zonas.
    """
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
    """Carga el CSV principal o lo genera si el archivo no existe."""
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
    """Construye un dataset pequeno de respaldo para pruebas o demostraciones."""
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
    """Garantiza que exista la carpeta donde se guardan las graficas."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _svg_data_uri(svg: str) -> str:
    """Convierte un SVG en una data URI compacta para usarlo en <img>."""
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('utf-8')}"


def _build_svg_bar_chart(title: str, labels: list[str], values: list[float], colors: list[str], x_label: str, y_label: str) -> str:
    width = 900
    height = 460
    margin_left = 72
    margin_right = 24
    margin_top = 46
    margin_bottom = 82
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    bar_width = plot_width / max(len(values), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="28" font-size="22" font-weight="700" fill="#111827">{xml_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
        f'<text x="18" y="{margin_top + plot_height / 2}" font-size="14" fill="#374151" transform="rotate(-90 18,{margin_top + plot_height / 2})">{xml_escape(y_label)}</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" font-size="14" fill="#374151" text-anchor="middle">{xml_escape(x_label)}</text>',
    ]

    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        bar_height = (value / max_value) * (plot_height - 24)
        x = margin_left + idx * bar_width + 12
        y = margin_top + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 24, 18):.1f}" height="{bar_height:.1f}" rx="8" fill="{color}"/>')
        parts.append(f'<text x="{x + max(bar_width - 24, 18) / 2:.1f}" y="{y - 8:.1f}" font-size="12" text-anchor="middle" fill="#111827">{value:.1f}</text>')
        parts.append(f'<text x="{x + max(bar_width - 24, 18) / 2:.1f}" y="{height - 34}" font-size="12" text-anchor="middle" fill="#475569">{xml_escape(label)}</text>')

    parts.append('</svg>')
    return ''.join(parts)


def _build_svg_scatter_chart(title: str, x_values: list[float], y_values: list[float], x_label: str, y_label: str, points: list[dict[str, object]]) -> str:
    width = 900
    height = 460
    margin_left = 72
    margin_right = 24
    margin_top = 46
    margin_bottom = 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = min(x_values) if x_values else 0
    x_max = max(x_values) if x_values else 1
    y_min = min(y_values) if y_values else 0
    y_max = max(y_values) if y_values else 1
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1)

    def sx(x: float) -> float:
        return margin_left + ((x - x_min) / x_span) * plot_width

    def sy(y: float) -> float:
        return margin_top + plot_height - ((y - y_min) / y_span) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="28" font-size="22" font-weight="700" fill="#111827">{xml_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
        f'<text x="18" y="{margin_top + plot_height / 2}" font-size="14" fill="#374151" transform="rotate(-90 18,{margin_top + plot_height / 2})">{xml_escape(y_label)}</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" font-size="14" fill="#374151" text-anchor="middle">{xml_escape(x_label)}</text>',
    ]

    # Línea de tendencia simple
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x1 = x_min
    x2 = x_max
    parts.append(
        f'<line x1="{sx(x1):.1f}" y1="{sy(slope * x1 + intercept):.1f}" x2="{sx(x2):.1f}" y2="{sy(slope * x2 + intercept):.1f}" stroke="#111827" stroke-width="2.5" opacity="0.8"/>'
    )

    for point in points:
        parts.append(
            f'<circle cx="{sx(float(point["x"])): .1f}" cy="{sy(float(point["y"])): .1f}" r="4.2" fill="{point["color"]}" stroke="#ffffff" stroke-width="0.8" opacity="0.92"/>'
        )

    parts.append('</svg>')
    return ''.join(parts)


def _build_svg_line_chart(title: str, labels: list[str], values: list[float], line_color: str, x_label: str, y_label: str) -> str:
    width = 900
    height = 460
    margin_left = 72
    margin_right = 24
    margin_top = 46
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    x_step = plot_width / max(len(values) - 1, 1)

    points = []
    for idx, value in enumerate(values):
        x = margin_left + idx * x_step
        y = margin_top + plot_height - ((value / max_value) * (plot_height - 8))
        points.append(f'{x:.1f},{y:.1f}')

    polyline = ' '.join(points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="28" font-size="22" font-weight="700" fill="#111827">{xml_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
        f'<text x="18" y="{margin_top + plot_height / 2}" font-size="14" fill="#374151" transform="rotate(-90 18,{margin_top + plot_height / 2})">{xml_escape(y_label)}</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" font-size="14" fill="#374151" text-anchor="middle">{xml_escape(x_label)}</text>',
    ]

    parts.append(f'<polyline points="{polyline}" fill="none" stroke="{line_color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = margin_left + idx * x_step
        y = margin_top + plot_height - ((value / max_value) * (plot_height - 8))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{line_color}" stroke="#ffffff" stroke-width="0.8"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 36}" font-size="12" text-anchor="middle" fill="#475569">{xml_escape(label)}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" font-size="12" text-anchor="middle" fill="#111827">{value:.1f}</text>')

    parts.append('</svg>')
    return ''.join(parts)


def save_bar_chart(data: pd.DataFrame) -> str:
    """Genera la grafica de barras con tiempo promedio por dia y devuelve Base64 SVG."""
    averages = data.groupby("dia", observed=False)["tiempo"].mean().reindex(DAYS).round(1)
    svg = _build_svg_bar_chart(
        "Tiempo promedio por día",
        list(averages.index),
        [float(value) for value in averages.values],
        [DAY_COLORS[day] for day in averages.index],
        "Día",
        "Tiempo promedio (min)",
    )
    return _svg_data_uri(svg)


def save_scatter_chart(data: pd.DataFrame) -> str:
    """Genera la dispersion tickets-tiempo y dibuja la tendencia lineal como Base64 SVG."""
    sampled = data.sort_values(["tickets", "tiempo"]).groupby("dia", observed=False, group_keys=False).head(30)
    points = []
    for _, row in sampled.iterrows():
        points.append({"x": float(row["tickets"]), "y": float(row["tiempo"]), "color": DAY_COLORS.get(str(row["dia"]), "#64748b")})

    svg = _build_svg_scatter_chart(
        "Relación entre tickets y tiempo",
        [float(value) for value in data["tickets"].tolist()],
        [float(value) for value in data["tiempo"].tolist()],
        "Tickets",
        "Tiempo (min)",
        points,
    )
    return _svg_data_uri(svg)


def save_category_chart(data: pd.DataFrame) -> str:
    """Genera la grafica de tiempo promedio por categoria de incidencia como Base64 SVG."""
    category_avg = data.groupby("categoria", observed=False)["tiempo"].mean().sort_values(ascending=False).round(1)
    svg = _build_svg_bar_chart(
        "Tiempo promedio por categoría",
        list(category_avg.index),
        [float(value) for value in category_avg.values],
        ["#5DADE2"] * len(category_avg),
        "Categoría",
        "Tiempo promedio (min)",
    )
    return _svg_data_uri(svg)


def save_trend_chart(data: pd.DataFrame) -> str:
    """Genera una serie temporal con promedio diario y media movil de 7 dias como Base64 SVG."""
    trend = data.groupby("fecha", observed=False)["tiempo"].mean().reset_index()
    trend["fecha"] = pd.to_datetime(trend["fecha"])
    trend["rolling"] = trend["tiempo"].rolling(window=7, min_periods=1).mean()

    labels = [str(fecha.date()) for fecha in trend["fecha"].tolist()[-8:]] if len(trend) > 8 else [str(fecha.date()) for fecha in trend["fecha"].tolist()]
    values = trend["rolling"].tolist()[-8:] if len(trend) > 8 else trend["rolling"].tolist()
    svg = _build_svg_line_chart(
        "Tendencia temporal del tiempo de atención",
        labels,
        [float(value) for value in values],
        "#10b981",
        "Fecha",
        "Tiempo promedio (min)",
    )
    return _svg_data_uri(svg)


def save_map_chart(data: pd.DataFrame) -> str:
    """Genera un mapa estatico simple con latitud, longitud y tiempo como Base64 SVG."""
    if data.empty:
        return _svg_data_uri('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500"><rect width="100%" height="100%" fill="#fff"/><text x="24" y="40" font-size="22" fill="#111827">Sin datos</text></svg>')

    width = 900
    height = 520
    margin_left = 60
    margin_right = 28
    margin_top = 52
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    lon_min, lon_max = float(data["lon"].min()), float(data["lon"].max())
    lat_min, lat_max = float(data["lat"].min()), float(data["lat"].max())
    lon_span = max(lon_max - lon_min, 0.001)
    lat_span = max(lat_max - lat_min, 0.001)
    temp_min, temp_max = float(data["tiempo"].min()), float(data["tiempo"].max())
    temp_span = max(temp_max - temp_min, 1)

    def x_for(lon: float) -> float:
        return margin_left + ((lon - lon_min) / lon_span) * plot_width

    def y_for(lat: float) -> float:
        return margin_top + plot_height - ((lat - lat_min) / lat_span) * plot_height

    def color_for(temp: float) -> str:
        ratio = (temp - temp_min) / temp_span
        if ratio >= 0.8:
            return "#b4233f"
        if ratio >= 0.6:
            return "#e25555"
        if ratio >= 0.4:
            return "#f59e0b"
        if ratio >= 0.2:
            return "#84cc16"
        return "#22c55e"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-size="22" font-weight="700" fill="#111827">Mapa simulado de Bogotá por tiempo de atención</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" rx="12"/>',
    ]

    for zone in ZONES:
        subset = data[data["zona"] == zone]
        if subset.empty:
            continue
        sample = subset.iloc[0]
        parts.append(f'<text x="{x_for(float(sample["lon"])) + 8:.1f}" y="{y_for(float(sample["lat"])) - 8:.1f}" font-size="12" font-weight="700" fill="#0f172a">{xml_escape(zone)}</text>')

    for _, row in data.iterrows():
        x = x_for(float(row["lon"]))
        y = y_for(float(row["lat"]))
        radius = 3.5 + (float(row["tiempo"]) - temp_min) / temp_span * 4
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color_for(float(row["tiempo"]))}" opacity="0.88" stroke="#ffffff" stroke-width="0.6"/>')

    parts.append(f'<text x="{width / 2:.1f}" y="{height - 18}" font-size="14" text-anchor="middle" fill="#374151">Longitud</text>')
    parts.append(f'<text x="18" y="{margin_top + plot_height / 2:.1f}" font-size="14" fill="#374151" transform="rotate(-90 18,{margin_top + plot_height / 2:.1f})">Latitud</text>')
    parts.append('</svg>')
    return _svg_data_uri(''.join(parts))


# Dataset global: se carga una vez al iniciar la aplicacion y luego las rutas
# filtran sobre esta copia base.
DATA = load_or_create_dataset()


def build_summary(data: pd.DataFrame) -> dict[str, object]:
    """Calcula indicadores principales para tarjetas, conclusiones y modelo."""
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
    """Evalua calidad de datos: nulos, duplicados, invalidos y outliers."""
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
    """Compara modelo lineal contra polinomico grado 2 usando RMSE."""
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


# ============================================================================
# IA & MACHINE LEARNING - Predicción, Anomalías y Análisis Estadístico
# ============================================================================

def predict_response_time(data: pd.DataFrame, ticket_values: list[int] | None = None) -> dict[str, object]:
    """
    Predice tiempo de respuesta usando regresión lineal entrenada en el dataset.
    Retorna predicciones para diferentes volúmenes de tickets.
    """
    try:
        from sklearn.linear_model import LinearRegression

        x = data["tickets"].values.reshape(-1, 1)
        y = data["tiempo"].values

        model = LinearRegression()
        model.fit(x, y)

        if ticket_values is None:
            ticket_values = [5, 10, 15, 20, 25]

        predictions = []
        for tickets in ticket_values:
            pred_time = float(model.predict([[tickets]])[0])
            predictions.append({
                "tickets": tickets,
                "predicted_time": round(max(0, pred_time), 1),
                "confidence": "Alta" if abs(data["tickets"].mean() - tickets) < data["tickets"].std() * 2 else "Media"
            })

        return {
            "model_trained": True,
            "predictions": predictions,
            "r_squared": round(float(model.score(x, y)), 3),
            "coefficients": {
                "slope": round(float(model.coef_[0]), 3),
                "intercept": round(float(model.intercept_), 3)
            }
        }
    except Exception as e:
        logger.exception("Error en predicción de tiempos")
        return {"model_trained": False, "error": str(e)}


def detect_anomalies(data: pd.DataFrame, contamination: float = 0.05) -> dict[str, object]:
    """
    Detecta registros anómalos usando Isolation Forest (ML).
    Identifica tickets con tiempos inusualmente altos/bajos respecto a su volumen.
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        x = data[["tickets", "tiempo"]].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        anomaly_labels = iso_forest.fit_predict(x_scaled)

        anomalies = data[anomaly_labels == -1].copy()
        normal = data[anomaly_labels == 1].copy()

        anomaly_details = []
        for _, row in anomalies.iterrows():
            anomaly_details.append({
                "ticket_id": str(row.get("ticket_id", "N/A")),
                "tickets": int(row["tickets"]),
                "tiempo": int(row["tiempo"]),
                "zona": str(row.get("zona", "N/A")),
                "prioridad": str(row.get("prioridad", "N/A")),
                "reason": "Tiempo inusualmente alto" if row["tiempo"] > normal["tiempo"].mean() + normal["tiempo"].std() else "Tiempo inusualmente bajo"
            })

        return {
            "anomalies_detected": int(len(anomalies)),
            "total_records": int(len(data)),
            "anomaly_percentage": round((len(anomalies) / len(data)) * 100, 2),
            "anomaly_details": sorted(anomaly_details[:10], key=lambda x: x["tickets"], reverse=True),
            "statistics": {
                "mean_anomaly_time": round(float(anomalies["tiempo"].mean()), 1) if len(anomalies) > 0 else 0,
                "mean_normal_time": round(float(normal["tiempo"].mean()), 1),
                "deviation": round(float(anomalies["tiempo"].mean() - normal["tiempo"].mean()), 1) if len(anomalies) > 0 else 0
            }
        }
    except Exception as e:
        logger.exception("Error en detección de anomalías")
        return {"anomalies_detected": 0, "error": str(e)}


def advanced_statistical_analysis(data: pd.DataFrame) -> dict[str, object]:
    """
    Análisis estadístico avanzado: correlaciones, test ANOVA, distribuciones.
    """
    try:
        from scipy.stats import pearsonr, spearmanr, f_oneway, skew, kurtosis

        # Correlaciones
        pearson_corr, pearson_pval = pearsonr(data["tickets"], data["tiempo"])
        spearman_corr, spearman_pval = spearmanr(data["tickets"], data["tiempo"])

        # ANOVA: ¿Hay diferencias significativas entre días?
        groups_by_day = [group["tiempo"].values for name, group in data.groupby("dia", observed=False)]
        f_stat, anova_pval = f_oneway(*groups_by_day) if len(groups_by_day) > 1 else (0, 1)

        # Distribución por zonas
        zona_stats = data.groupby("zona", observed=False)["tiempo"].agg(["mean", "std", "min", "max", "count"])
        zona_dict = {str(idx): {
            "mean": round(float(row["mean"]), 1),
            "std": round(float(row["std"]), 1),
            "min": int(row["min"]),
            "max": int(row["max"]),
            "count": int(row["count"])
        } for idx, row in zona_stats.iterrows()}

        # Skewness y Kurtosis (forma de la distribución)
        skewness = float(skew(data["tiempo"]))
        kurt = float(kurtosis(data["tiempo"]))

        return {
            "correlations": {
                "pearson": {"coefficient": round(pearson_corr, 3), "p_value": round(pearson_pval, 5)},
                "spearman": {"coefficient": round(spearman_corr, 3), "p_value": round(spearman_pval, 5)}
            },
            "anova_test": {
                "f_statistic": round(f_stat, 3),
                "p_value": round(anova_pval, 5),
                "significant": anova_pval < 0.05,
                "interpretation": "Hay diferencias significativas en tiempo por día." if anova_pval < 0.05 else "No hay diferencias significativas por día."
            },
            "distribution_shape": {
                "skewness": round(skewness, 3),
                "kurtosis": round(kurt, 3),
                "shape": "Right-skewed" if skewness > 0.5 else ("Left-skewed" if skewness < -0.5 else "Symmetric")
            },
            "zone_statistics": zona_dict,
            "sample_size": int(len(data))
        }
    except Exception as e:
        logger.exception("Error en análisis estadístico")
        return {"error": str(e)}


def generate_recommendations(data: pd.DataFrame, summary: dict[str, object]) -> list[dict[str, str]]:
    """
    Genera recomendaciones automáticas basadas en patrones de IA detectados.
    """
    recommendations = []

    # Recomendación 1: Carga por zona
    zone_avg = data.groupby("zona", observed=False)["tiempo"].mean()
    slowest_zone = zone_avg.idxmax()
    slowest_time = round(float(zone_avg.max()), 1)
    recommendations.append({
        "title": "Zona crítica detectada",
        "description": f"La zona {slowest_zone} tiene el tiempo promedio más alto ({slowest_time} min). Considere recursos adicionales.",
        "priority": "Alta"
    })

    # Recomendación 2: Ticket para día específico
    day_avg = data.groupby("dia", observed=False)["tiempo"].mean()
    busiest_day = day_avg.idxmax()
    recommendations.append({
        "title": "Pico de demanda identificado",
        "description": f"El {busiest_day} es el día más cargado. Aumente personal este día.",
        "priority": "Media"
    })

    # Recomendación 3: Correlación fuerte
    if abs(float(summary.get("corr", 0))) > 0.7:
        recommendations.append({
            "title": "Correlación fuerte detectada",
            "description": "El volumen de tickets es el predictor principal del tiempo. Optimice velocidad de procesamiento.",
            "priority": "Alta"
        })

    # Recomendación 4: Prioridad vs Tiempo
    if "prioridad" in data.columns:
        priority_time = data.groupby("prioridad", observed=False)["tiempo"].mean()
        if len(priority_time) > 0:
            high_priority_time = priority_time.get("Alta", 0)
            if high_priority_time > 0:
                recommendations.append({
                    "title": "Tickets de alta prioridad",
                    "description": f"Los tickets 'Alta' demoran {high_priority_time:.1f} min en promedio. Revise SLA.",
                    "priority": "Media"
                })

    # Recomendación 5: Anomalías
    anomalies_report = detect_anomalies(data, contamination=0.05)
    if anomalies_report.get("anomalies_detected", 0) > 0:
        anomaly_pct = anomalies_report.get("anomaly_percentage", 0)
        recommendations.append({
            "title": "Registros anómalos detectados",
            "description": f"{anomaly_pct}% de registros tienen patrones inusionales. Investigue casos extremos.",
            "priority": "Baja"
        })

    return recommendations[:5]  # Máximo 5 recomendaciones



def apply_filters(data: pd.DataFrame, params: dict[str, str]) -> pd.DataFrame:
    """Aplica filtros enviados por query string sobre el DataFrame base.

    `params` es un diccionario con keys como `zona`, `prioridad`, `categoria`,
    `canal`, `dia`, `fecha_inicio`, `fecha_fin` (todas strings). Devuelve un
    DataFrame filtrado y reseteado en su índice.
    """
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
    """Obtiene las opciones disponibles para llenar los selects del formulario."""
    return {
        "zonas": sorted(data["zona"].dropna().unique().tolist()),
        "prioridades": sorted(data["prioridad"].dropna().unique().tolist()),
        "categorias": sorted(data["categoria"].dropna().unique().tolist()),
        "canales": sorted(data["canal"].dropna().unique().tolist()),
        "dias": DAYS,
    }


def _get_current_filters() -> dict[str, str]:
    """Lee los filtros actuales desde la URL y normaliza espacios."""
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
    """Devuelve datos filtrados y maneja rangos invalidos o filtros sin resultados."""
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
    """Reduce los puntos del mapa manteniendo representacion por zona."""
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
    """Renderiza el dashboard principal con filtros, metricas y graficas."""
    active_data, active_filters, fallback_used, fallback_message = _resolve_filtered_data()
    summary = build_summary(active_data)
    map_data = build_map_sample(active_data)
    quality_report = build_quality_report(active_data)
    model_diagnostics = build_model_diagnostics(active_data)
    
    # IA & Machine Learning (llamadas seguras: capturamos excepciones para evitar 500)
    prediction = {}
    anomalies = {"count": 0, "items": []}
    advanced_stats = {}
    recommendations = []
    try:
        prediction = predict_response_time(active_data)
    except Exception as e:
        logger.exception("Error durante la predicción IA")
        prediction = {"error": str(e)}

    try:
        anomalies = detect_anomalies(active_data, contamination=0.05)
    except Exception as e:
        logger.exception("Error durante la detección de anomalías")
        anomalies = {"error": str(e)}

    try:
        advanced_stats = advanced_statistical_analysis(active_data)
    except Exception as e:
        logger.exception("Error durante el análisis estadístico avanzado")
        advanced_stats = {"error": str(e)}

    try:
        recommendations = generate_recommendations(active_data, summary)
    except Exception as e:
        logger.exception("Error generando recomendaciones de IA")
        recommendations = []
    
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
        # IA & ML
        prediction=prediction,
        anomalies=anomalies,
        advanced_stats=advanced_stats,
        recommendations=recommendations,
    )


@app.route("/export/filtered.csv")
def export_filtered_csv():
    """Exporta el subconjunto filtrado como CSV descargable."""
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
    """Exporta el subconjunto filtrado y su resumen en un archivo Excel."""
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
    """Genera un PDF ejecutivo con los principales resultados filtrados."""
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
    """Endpoint liviano para comprobar que el servidor esta vivo."""
    return {"status": "ok"}


# ============================================================================
# API ENDPOINTS - IA & Machine Learning (JSON)
# ============================================================================

@app.route("/api/predictions")
def api_predictions():
    """API: Predicciones de tiempo basadas en volumen de tickets (ML)."""
    filtered, _, _, _ = _resolve_filtered_data()
    
    # Parámetro opcional: ticket_count
    try:
        custom_tickets = request.args.get("tickets")
        ticket_values = None
        if custom_tickets:
            ticket_values = [int(t) for t in custom_tickets.split(",")]
    except (ValueError, TypeError):
        ticket_values = None
    
    prediction = predict_response_time(filtered, ticket_values)
    return prediction


@app.route("/api/anomalies")
def api_anomalies():
    """API: Detección de anomalías en registros de tickets (Isolation Forest)."""
    filtered, _, _, _ = _resolve_filtered_data()
    anomalies = detect_anomalies(filtered, contamination=0.05)
    return anomalies


@app.route("/api/analytics")
def api_analytics():
    """API: Análisis estadístico avanzado con test ANOVA y correlaciones."""
    filtered, _, _, _ = _resolve_filtered_data()
    stats = advanced_statistical_analysis(filtered)
    return stats


@app.route("/api/recommendations")
def api_recommendations():
    """API: Recomendaciones automáticas basadas en IA."""
    filtered, _, _, _ = _resolve_filtered_data()
    summary = build_summary(filtered)
    recommendations = generate_recommendations(filtered, summary)
    return {"recommendations": recommendations, "count": len(recommendations)}


@app.route("/api/full-ai-report")
def api_full_ai_report():
    """API: Reporte completo de IA (predicciones + anomalías + estadísticas + recomendaciones)."""
    filtered, _, _, _ = _resolve_filtered_data()
    summary = build_summary(filtered)
    
    report = {
        "timestamp": date.today().isoformat(),
        "dataset_size": len(filtered),
        "predictions": predict_response_time(filtered),
        "anomalies": detect_anomalies(filtered, contamination=0.05),
        "analytics": advanced_statistical_analysis(filtered),
        "recommendations": generate_recommendations(filtered, summary),
    }
    return report



    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
