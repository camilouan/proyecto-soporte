# Proyecto integrador: soporte técnico

Aplicación Flask lista para presentar y desplegar en Render. Analiza cómo la cantidad de tickets impacta el tiempo de respuesta en soporte técnico usando un dataset CSV realista, visualizaciones y modelo matemático ajustado.

## Qué incluye
- Dataset CSV realista en `data/support_tickets_bogota.csv`.
- Filtros interactivos por zona, prioridad, categoría, canal, día y rango de fechas.
- Gráficas: tiempo por día, dispersión tickets-tiempo, tiempo por categoría, tendencia temporal y mapa real de Bogotá.
- Modelo matemático ajustado al dataset (`T(x)=ax+b`) con derivada e integral explicadas.
- Calidad de datos: score, nulos, duplicados y outliers (IQR).
- Exportables: CSV filtrado, Excel resumen y PDF ejecutivo.
- Pruebas automatizadas básicas con `unittest`.
- Archivos listos para Render.

## Ejecutar en local
```bash
pip install -r requirements.txt
python app.py
```

Luego abre `http://127.0.0.1:5000`.

## Ejecutar pruebas
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Exportaciones
- `/export/filtered.csv`: dataset filtrado.
- `/export/summary.xlsx`: resumen y datos filtrados en Excel.
- `/export/report.pdf`: reporte ejecutivo en PDF.

## Despliegue en Render
1. Sube el proyecto a GitHub.
2. Crea un Web Service en Render.
3. Conecta el repositorio.
4. Usa el comando de inicio:
```bash
gunicorn app:app
```

## Estructura
- `app.py`: backend Flask y generación de gráficas.
- `templates/index.html`: presentación visual del proyecto.
- `data/support_tickets_bogota.csv`: dataset analizado.
- `tests/test_app.py`: pruebas básicas.
- `static/plots/`: imágenes generadas automáticamente.
- `requirements.txt`: dependencias.
- `Procfile` y `render.yaml`: despliegue.
