# Balance de Avance - Proyecto Integrador: Soporte Técnico

**Fecha de presentación:** 25 de abril  
**Proyecto:** Análisis del impacto de volumen de tickets en tiempos de respuesta de soporte técnico  
**Plataforma:** Flask + Python, despliegue en Render

---

## 1. Estado Actual del Desarrollo

### ✅ Completado (100%)

#### Backend
- **Framework:** Flask 3.0+ con arquitectura modular
- **Dataset:** CSV realista con 500+ registros de tickets de soporte técnico de Bogotá
- **Datos disponibles:**
  - Información de tickets (ID, fecha, día, zona, categoría, prioridad, canal)
  - Métricas de volumen y tiempo de respuesta
  - Coordenadas geográficas (latitud/longitud) para mapeo

#### Sistema de Filtros
- Filtrado dinámico por múltiples parámetros:
  - Zona geográfica (Suba, Engativá, Chapinero, Usaquén, Kennedy)
  - Prioridad (Baja, Media, Alta, Crítica)
  - Categoría (Red, Software, Hardware, Accesos, Correo)
  - Canal (Portal, Correo, Teléfono, Chat)
  - Día de la semana
  - Rango de fechas
- Validación de parámetros en servidor

#### Visualizaciones Interactivas
1. **Gráfica de Tiempo por Día:** Muestra promedio de tiempo de respuesta por día de semana
2. **Dispersión (Scatter):** Relación entre cantidad de tickets y tiempo de respuesta
3. **Tiempo por Categoría:** Distribución de tiempos por tipo de problema
4. **Tendencia Temporal:** Evolución del tiempo de respuesta a lo largo del periodo
5. **Mapa Interactivo:** 
   - Leaflet.js con basemap Carto
   - Marcadores posicionados desde centroides de localidades reales de Bogotá
   - Carga de GeoJSON desde GitHub raw para estabilidad

#### Análisis Matemático
- Modelo lineal de correlación: **T(x) = ax + b**
  - **x** = cantidad de tickets
  - **T(x)** = tiempo de respuesta (minutos)
- Cálculo de derivada e integral
- Explicación interpretativa de resultados

#### IA & Machine Learning (Integración Externa con scikit-learn + scipy)
- **Predicción de tiempos:** Regresión Lineal entrenada en dataset para predecir tiempos según volumen
- **Detección de anomalías:** Isolation Forest identifica registros inusuales (outliers estadísticos)
- **Análisis estadístico avanzado:**
  - Test ANOVA para diferencias significativas entre días
  - Correlación de Pearson y Spearman
  - Análisis de distribución (skewness, kurtosis)
  - Estadísticas por zona
- **Recomendaciones automáticas:** Sistema generador de insights basado en patrones detectados por ML
- **APIs REST** para consumo de análisis en formato JSON:
  - `/api/predictions` - Predicciones
  - `/api/anomalies` - Anomalías
  - `/api/analytics` - Análisis estadístico
  - `/api/recommendations` - Recomendaciones
  - `/api/full-ai-report` - Reporte completo integrado

#### Calidad de Datos
- Score de calidad integral
- Detección de valores nulos, duplicados y outliers (método IQR)
- Estadísticas descriptivas

#### Exportaciones
- **CSV Filtrado:** `/export/filtered.csv` - descarga de datos según filtros aplicados
- **Excel Resumen:** `/export/summary.xlsx` - datos filtrados + estadísticas en múltiples hojas
- **PDF Ejecutivo:** `/export/report.pdf` - reporte profesional con gráficas

#### Infraestructura & Testing
- Tests automatizados con `unittest` en `tests/test_app.py`
- Validación de:
  - Presencia de columnas requeridas
  - Integridad de filtros
  - Funcionalidad de rutas
- Archivos de configuración para despliegue en Render:
  - `Procfile`
  - `render.yaml`

#### Documentación
- `README.md` con instrucciones de ejecución
- Logs estructurados en consola y Render
- Comentarios en código explicando lógica de dominio

---

## 2. Decisiones Técnicas Tomadas

### Arquitectura
- **Monolítica en un archivo `app.py`:** Simplifica despliegue en Render sin servidor externo; facilita debugging en desarrollo inicial
- **Generación de gráficas en tiempo de arranque:** Evita latencia en primeras solicitudes y reduce I/O durante ejecución

### Frontend
- **Leaflet.js para mapa:** Librería ligera, sin clave API requerida, mejor rendimiento que alternativas
- **GeoJSON desde GitHub raw:** Elimina dependencia de servidor externo; acelera carga inicial
- **Basemap Carto:** Visual limpio, contraste óptimo para marcadores de datos

### Datos
- **Normalización de nombres de zonas:** `Engativá → Engativa` para evitar errores de codificación
- **Centroides locales en memoria:** Dataset integrado en código evita llamadas a APIs externas
- **Modelo lineal (T = ax + b):** Suficiente para demostrar correlación; evita complejidad innecesaria

### Seguridad & Rendimiento
- **PATH absolutos** calculados desde `app.py`: Funciona en local y en Render sin configuración adicional
- **Matplotlib backend Agg:** No requiere servidor X11; genera PNGs directamente en servidor sin interfaz gráfica
- **Gunicorn en Render:** Servidor WSGI production-ready con auto-scaling

### Testing
- **unittest integrado:** Sin dependencias externas; permite CI/CD futuro

### IA & Machine Learning
- **scikit-learn para Isolation Forest:** Algoritmo no supervisado estándar para detección de anomalías sin necesidad de labeled data
- **Regresión Lineal sklearn:** Más eficiente que numpy.polyfit para predicciones sobre nuevos datos
- **scipy.stats para test ANOVA:** Valida diferencias significativas entre grupos sin suposiciones paramétricas estrictas
- **APIs REST** separadas: Permiten consumo desde otros sistemas (frontend, mobile, terceros) sin acoplamiento
- **Importación condicional:** Si scikit-learn no está disponible, el app aun funciona pero sin análisis avanzados

---

## 3. Dificultades Encontradas y Soluciones

### 🔧 Dificultad 1: Rutas de archivos inconsistentes entre local y Render
**Problema:** Rutas relativas fallaban en Render al ejecutar desde directorios diferentes.  
**Solución:** Implementar cálculo de rutas absolutas desde `app.py` usando `Path(__file__).resolve().parent`.  
**Resultado:** ✅ Funcionamiento idéntico en local (Windows PowerShell, Python 3.10) y en Render.

---

### 🔧 Dificultad 2: Codificación de caracteres acentuados en GeoJSON
**Problema:** Errores de encoding al cargar GeoJSON con nombres de localidades en español (ej: "Chapinero", "Usaquén").  
**Solución:** Usar normalización Unicode (`unicodedata`) y mantener equivalencias en diccionario `ZONE_NAME_MAP`.  
**Resultado:** ✅ Mapeo consistente entre CSV y GeoJSON.

---

### 🔧 Dificultad 3: Generación de gráficas bloqueaba arranque del servidor
**Problema:** Matplotlib sin display en Render causaba cuelgues o fallos.  
**Solución:** Configurar `matplotlib.use("Agg")` antes de importar pyplot; generar PNGs en disco al arrancar.  
**Resultado:** ✅ Servidor inicia en ~2-3 segundos; gráficas disponibles inmediatamente.

---

### 🔧 Dificultad 4: Dependencia de archivo CSV externo
**Problema:** Si `support_tickets_bogota.csv` no existía, la app fallaba.  
**Solución:** Implementar generador de dataset sintético realista en `app.py` con parámetros de dominio (pesos por zona, retardos por prioridad).  
**Resultado:** ✅ App funciona incluso sin CSV pre-existente; genera datos coherentes.

---

### 🔧 Dificultad 5: Mapa no se cargaba en Render
**Problema:** Intento inicial de generar mapa con Folium causaba peso excesivo (~5MB) e incompatibilidad con servidor.  
**Solución:** Cambiar a Leaflet.js renderizado en cliente; cargar GeoJSON desde GitHub raw.  
**Resultado:** ✅ Mapa interactivo, rápido (~100KB); sin overhead de servidor.

---

### 🔧 Dificultad 6: Filtros complejos causaban inconsistencias
**Problema:** Diferentes combinaciones de filtros producían resultados inesperados.  
**Solución:** Implementar función `apply_filters()` centralizada con validación de cada parámetro; testing con `unittest`.  
**Resultado:** ✅ Filtros robustos; tests verifican reducción de datos sin pérdida de integridad.

---

## 4. Funcionalidades Adicionales Implementadas

### Más allá del mínimo requerido:
1. **Mapa geográfico interactivo** con datos reales de Bogotá
2. **Exportaciones en múltiples formatos** (CSV, Excel, PDF)
3. **Modelo matemático** con explicación de derivada e integral
4. **Sistema de calidad de datos** (detección de outliers, nulos, duplicados)
5. **Tests automatizados** para CI/CD futuro
6. **Logs estructurados** para monitoring en producción
7. **Datos sintéticos realistas** generados con parámetros de dominio

---

## 5. Próximos Pasos (Trabajo Futuro)

- [ ] Integración de base de datos PostgreSQL para escalabilidad
- [ ] Dashboard en tiempo real con WebSockets
- [ ] Modelos predictivos avanzados (regresión polinómica, machine learning)
- [ ] Autenticación y control de acceso por rol
- [ ] Temas oscuros y responsividad mejorada
- [ ] Métricas de SLA y alertas automáticas

---

## 4. Funcionalidades Adicionales Implementadas

### Más allá del mínimo requerido:
1. **Mapa geográfico interactivo** con datos reales de Bogotá
2. **Exportaciones en múltiples formatos** (CSV, Excel, PDF)
3. **Modelo matemático** con explicación de derivada e integral
4. **Sistema de calidad de datos** (detección de outliers, nulos, duplicados)
5. **Tests automatizados** para CI/CD futuro
6. **Logs estructurados** para monitoring en producción
7. **Datos sintéticos realistas** generados con parámetros de dominio
8. **IA & Machine Learning:**
   - Predicción de tiempos con Regresión Lineal (scikit-learn)
   - Detección de anomalías con Isolation Forest
   - Análisis estadístico avanzado (ANOVA, correlaciones)
   - Sistema automático de recomendaciones basadas en patrones
   - 5 APIs REST para consumo de análisis en JSON

---

## 5. Próximos Pasos (Trabajo Futuro)

- [ ] Modelos predictivos avanzados (XGBoost, Random Forest)
- [ ] Clustering de patrones (K-means, DBSCAN)
- [ ] Integración de base de datos PostgreSQL para escalabilidad
- [ ] Dashboard en tiempo real con WebSockets
- [ ] Autenticación y control de acceso por rol
- [ ] Temas oscuros y responsividad mejorada
- [ ] Métricas de SLA y alertas automáticas
- [ ] Visualizaciones de IA (Feature Importance, Decision Trees)

El proyecto ha alcanzado un estado **production-ready** en Render con todas las funcionalidades planeadas implementadas. La arquitectura es robusta, el código está documentado, y se han resuelto exitosamente los desafíos técnicos principales. La aplicación está lista para análisis de datos de soporte técnico y puede servir como base para expansiones futuras.

---

**Último test ejecutado:** ✅ Todas las pruebas pasan  
**Estado de despliegue:** ✅ Listo para Render  
**Documentación:** ✅ Completa
