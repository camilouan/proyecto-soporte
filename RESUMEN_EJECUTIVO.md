# Resumen Ejecutivo - Proyecto Soporte Técnico
**Para presentación oral - 25 de abril**

---

## 📊 Estado Actual
**✅ COMPLETADO Y DEPLOYABLE EN RENDER**

- Backend Flask funcional con dataset de 500+ tickets de Bogotá
- 5 gráficas interactivas + mapa geográfico Leaflet
- Sistema de filtros por zona, prioridad, categoría, canal, día y fecha
- Modelo matemático T(x) = ax + b con análisis de derivada e integral
- Exportaciones: CSV, Excel, PDF
- Tests automatizados pasando
- Despliegue en Render configurado

---

## 🛠 Decisiones Técnicas Clave

| Decisión | Razón |
|----------|-------|
| **Flask monolítico** | Simplifica deployment en Render sin dependencias externas |
| **Gráficas generadas al arrancar** | Evita latencia en solicitudes posteriores |
| **Leaflet.js + GeoJSON desde GitHub** | Mapa ligero (~100KB) sin servidor externo |
| **Rutas absolutas desde app.py** | Funciona idéntico en local y en producción |
| **Matplotlib backend "Agg"** | Genera PNGs sin servidor X11 |
| **Dataset sintético en memoria** | App funciona incluso sin CSV pre-existente |

---

## ⚠️ Dificultades Resueltas

### 1. **Rutas de archivos inconsistentes** → Cálculo dinámico desde `app.py` ✅
### 2. **Encoding UTF-8 en GeoJSON** → Normalización Unicode + diccionario de equivalencias ✅
### 3. **Matplotlib bloqueaba servidor** → Backend "Agg" + pre-generación ✅
### 4. **CSV obligatorio no existía** → Generador sintético realista ✅
### 5. **Mapa pesado (5MB Folium)** → Cambio a Leaflet.js cliente (~100KB) ✅
### 6. **Filtros inconsistentes** → Función centralizada + tests unitarios ✅

---

## 📈 Métricas

- **Tiempo arranque:** ~2-3 segundos
- **Tamaño assets:** ~100KB (HTML + JS + CSS + GeoJSON)
- **Cobertura de tests:** Validaciones básicas de dataset, filtros y rutas
- **Soporta:** Windows local + Render (Linux)
- **Python:** 3.10+ requerido

---

## 🎯 Funcionalidades Adicionales (Beyond Scope)

✨ Mapa interactivo con localizaciones reales  
✨ Exportaciones PDF ejecutivas  
✨ Análisis de calidad de datos (outliers, nulos, duplicados)  
✨ Modelo matemático con derivada e integral  
✨ Tests automatizados  

---

## ✅ Checklist Entrega

- [x] Código versionado en GitHub/Git
- [x] README.md completo con instrucciones
- [x] requirements.txt con dependencias
- [x] Tests en tests/test_app.py
- [x] Ejecutable en local (`python app.py`)
- [x] Archivos Render listos (Procfile, render.yaml)
- [x] 5+ visualizaciones
- [x] Filtros funcionales
- [x] Modelo matemático implementado
- [x] Documentación de decisiones técnicas
- [x] Balance de avance con dificultades documentadas
