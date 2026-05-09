# Documentación: IA & Machine Learning en el Proyecto

**Última actualización:** 9 de mayo de 2026  
**Estado:** Integrado completamente en producción

---

## Resumen Ejecutivo

El proyecto integra **IA y Machine Learning** utilizando bibliotecas estándar de Python (**scikit-learn** y **scipy**) para:
- Predecir tiempos de respuesta
- Detectar anomalías automáticamente
- Realizar análisis estadístico avanzado
- Generar recomendaciones basadas en patrones

Todas estas capacidades están disponibles tanto en la interfaz web como a través de **APIs REST**.

---

## Funcionalidades de IA Implementadas

### 1. Predicción de Tiempos (Regresión Lineal)

**¿Qué hace?**  
Entrena un modelo de Regresión Lineal en los datos históricos y predice el tiempo de respuesta para cualquier volumen de tickets.

**Modelo:** `T(tiempo) = slope × tickets + intercept`

**Resultado:**
```json
{
  "model_trained": true,
  "r_squared": 0.926,
  "coefficients": {
    "slope": 1.75,
    "intercept": 9.24
  },
  "predictions": [
    {"tickets": 5, "predicted_time": 17.1, "confidence": "Alta"},
    {"tickets": 10, "predicted_time": 26.8, "confidence": "Alta"},
    {"tickets": 15, "predicted_time": 36.5, "confidence": "Alta"}
  ]
}
```

**Uso en API:**
```bash
GET /api/predictions?tickets=5,10,15,20
```

**Implementación:**
- Librería: `scikit-learn` (LinearRegression)
- Características: Coeficientes, R² score, predicciones
- Entrada: DataFrame con columnas 'tickets' y 'tiempo'

---

### 2. Detección de Anomalías (Isolation Forest)

**¿Qué hace?**  
Identifica automáticamente registros inusuales que se desvían del patrón normal de datos usando un algoritmo no supervisado.

**Algoritmo:** Isolation Forest (scikit-learn)  
**Contamination default:** 5% (identifica ~5% como anomalías)

**Resultado:**
```json
{
  "anomalies_detected": 50,
  "total_records": 1010,
  "anomaly_percentage": 4.95,
  "statistics": {
    "mean_anomaly_time": 24.3,
    "mean_normal_time": 42.1,
    "deviation": -17.8
  },
  "anomaly_details": [
    {
      "ticket_id": "TK-10245",
      "tickets": 22,
      "tiempo": 8,
      "zona": "Suba",
      "prioridad": "Baja",
      "reason": "Tiempo inusualmente bajo"
    }
  ]
}
```

**Uso en API:**
```bash
GET /api/anomalies
```

**Ventajas:**
- No requiere labeled data (sin supervisión)
- Detecta outliers multidimensionales
- Maneja características escaladas automáticamente

---

### 3. Análisis Estadístico Avanzado

**¿Qué incluye?**

#### 3.1 Correlaciones
- **Pearson:** Correlación lineal paramétrica
- **Spearman:** Correlación ordinal no paramétrica
- Incluye p-values para significancia estadística

#### 3.2 Test ANOVA
- Valida si hay diferencias significativas entre grupos (ej: días de la semana)
- Retorna F-statistic y p-value
- Null hypothesis: "No hay diferencias entre grupos"

#### 3.3 Análisis de Distribución
- **Skewness:** Sesgo de la distribución (-0.5 a 0.5 = simétrica)
- **Kurtosis:** Altura de la distribución (> 3 = colas pesadas)
- Interpretación automática de forma

#### 3.4 Estadísticas por Zona
- Media, desviación estándar, min, max, count
- Agrupación por localidad

**Resultado:**
```json
{
  "correlations": {
    "pearson": {
      "coefficient": 0.962,
      "p_value": 2.8e-100
    },
    "spearman": {
      "coefficient": 0.949,
      "p_value": 1.2e-95
    }
  },
  "anova_test": {
    "f_statistic": 2.487,
    "p_value": 0.02456,
    "significant": true,
    "interpretation": "Hay diferencias significativas en tiempo por día."
  },
  "distribution_shape": {
    "skewness": -0.072,
    "kurtosis": 0.351,
    "shape": "Symmetric"
  },
  "zone_statistics": {
    "Kennedy": {
      "mean": 44.2,
      "std": 12.5,
      "min": 10,
      "max": 72,
      "count": 242
    }
  }
}
```

**Uso en API:**
```bash
GET /api/analytics
```

---

### 4. Recomendaciones Automáticas

**¿Qué hace?**  
Genera 5 recomendaciones accionables analizando patrones detectados por IA.

**Tipos de recomendaciones:**
1. **Zona crítica:** Identifica la zona con mayor tiempo promedio
2. **Pico de demanda:** Detecta el día más cargado
3. **Correlación fuerte:** Sugiere optimizar si la correlación > 0.7
4. **Prioridad:** Valida SLA de tickets de alta prioridad
5. **Anomalías:** Advierte sobre registros inusuales

**Resultado:**
```json
{
  "recommendations": [
    {
      "title": "Zona crítica detectada",
      "description": "La zona Kennedy tiene el tiempo promedio más alto (44.2 min). Considere recursos adicionales.",
      "priority": "Alta"
    },
    {
      "title": "Pico de demanda identificado",
      "description": "El Viernes es el día más cargado. Aumente personal este día.",
      "priority": "Media"
    },
    {
      "title": "Correlación fuerte detectada",
      "description": "El volumen de tickets es el predictor principal del tiempo. Optimice velocidad de procesamiento.",
      "priority": "Alta"
    }
  ],
  "count": 3
}
```

**Uso en API:**
```bash
GET /api/recommendations
```

---

## APIs REST Disponibles

### 1. `/api/predictions`
**Método:** GET  
**Parámetros:** `tickets` (opcional, CSV: "5,10,15")  
**Retorna:** Predicciones de tiempo para volúmenes especificados

```bash
curl "http://localhost:5000/api/predictions?tickets=5,10,15,20"
```

### 2. `/api/anomalies`
**Método:** GET  
**Retorna:** Anomalías detectadas en el dataset actual (con filtros aplicados)

```bash
curl "http://localhost:5000/api/anomalies"
```

### 3. `/api/analytics`
**Método:** GET  
**Retorna:** Análisis estadístico completo (correlaciones, ANOVA, distribuciones)

```bash
curl "http://localhost:5000/api/analytics"
```

### 4. `/api/recommendations`
**Método:** GET  
**Retorna:** Recomendaciones automáticas basadas en datos

```bash
curl "http://localhost:5000/api/recommendations"
```

### 5. `/api/full-ai-report`
**Método:** GET  
**Retorna:** Reporte completo (todas las análisis + predicciones + anomalías + recomendaciones)

```bash
curl "http://localhost:5000/api/full-ai-report"
```

---

## Librerías Externas Utilizadas

| Librería | Versión | Función |
|----------|---------|---------|
| **scikit-learn** | >=1.3 | Regresión, Isolation Forest |
| **scipy** | >=1.11 | Test ANOVA, cálculos estadísticos |

---

## Ejemplos de Uso

### Python
```python
from app import predict_response_time, detect_anomalies, advanced_statistical_analysis, DATA

# Predicción
pred = predict_response_time(DATA, [5, 10, 15])
print(f"R²: {pred['r_squared']}")

# Anomalías
anomalies = detect_anomalies(DATA, contamination=0.05)
print(f"Detectadas: {anomalies['anomalies_detected']}")

# Análisis
stats = advanced_statistical_analysis(DATA)
print(f"Correlación Pearson: {stats['correlations']['pearson']['coefficient']}")
```

### JavaScript (desde frontend)
```javascript
// Fetch predicciones
fetch('/api/predictions?tickets=10,15,20')
  .then(r => r.json())
  .then(data => console.log(data.predictions));

// Fetch recomendaciones
fetch('/api/recommendations')
  .then(r => r.json())
  .then(data => data.recommendations.forEach(rec => {
    console.log(`[${rec.priority}] ${rec.title}`);
  }));
```

### cURL
```bash
# Reporte completo
curl -X GET "http://proyecto-soporte.onrender.com/api/full-ai-report"

# Con filtros aplicados (ej: solo Kennedy)
curl -X GET "http://proyecto-soporte.onrender.com/api/analytics?zona=Kennedy"
```

---

## Consideraciones de Rendimiento

| Operación | Tiempo Aprox | Dataset |
|-----------|--------------|---------|
| Predicción | < 10ms | Cualquier tamaño |
| Anomalías (IF) | ~100ms | 1000 registros |
| ANOVA | ~50ms | 1000 registros |
| Reporte completo | ~200ms | 1000 registros |

**Nota:** Isolation Forest es O(n log n), escalable hasta millones de registros.

---

## Limitaciones Actuales

1. **Predicción:** Solo soporta regresión lineal (y(x) = ax + b)
   - Mejora futura: Polinomial, XGBoost, Neural Networks

2. **Anomalías:** Usa contamination fijo (5%)
   - Mejora futura: Parámetro dinámico, múltiples algoritmos

3. **ANOVA:** Solo entre días; no considera interacciones
   - Mejora futura: Factorial ANOVA, ANCOVA

4. **Recomendaciones:** Basadas en reglas; no usa NLP
   - Mejora futura: LLM integration para recomendaciones textuales

---

## Próximos Pasos

- [ ] Modelos predictivos avanzados (XGBoost, Random Forest)
- [ ] Clustering de patrones (K-means, DBSCAN, Hierarchical)
- [ ] Análisis de series temporales (ARIMA, Prophet)
- [ ] Explicabilidad de modelos (SHAP, Feature Importance)
- [ ] Reentrenamiento automático de modelos
- [ ] Dashboard de monitore de modelos (Model Monitoring)

---

## Conclusión

La integración de IA y ML proporciona **capacidades analíticas avanzadas** sin aumentar significativamente la complejidad del código. Los modelos son interpretables, eficientes y listos para producción.

