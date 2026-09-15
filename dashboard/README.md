# Anomaly Detection Dashboard

Dashboard interactivo desarrollado con Streamlit para explorar e interpretar anomalías detectadas en alojamientos turísticos de España obtenidos de Booking.com.

La aplicación constituye la capa de visualización e interpretación del pipeline de datos y machine learning. Permite analizar las predicciones de Isolation Forest y Local Outlier Factor (LOF), compararlas con reglas de negocio y estudiar la evolución temporal de los precios y las anomalías.

## Objetivo

El dashboard está diseñado para responder a las siguientes preguntas:

- ¿Cuántos alojamientos presentan comportamientos atípicos?
- ¿Qué anomalías detecta cada modelo?
- ¿En qué regiones y tipos de alojamiento se concentran?
- ¿Las anomalías están relacionadas con el precio, el tamaño o el contexto geográfico?
- ¿Coinciden los modelos con reglas de negocio interpretables?
- ¿Cómo evolucionan los precios y las anomalías entre diferentes extracciones?
- ¿Qué alojamientos concretos requieren una inspección más detallada?

La aplicación no realiza scraping ni entrena modelos. Utiliza los resultados generados previamente por el pipeline.

## Ejecución

Desde la raíz del proyecto:

```powershell
cd dashboard
..\venv\Scripts\python.exe -m streamlit run app.py
```

También puede ejecutarse si el entorno virtual está activado:

```powershell
cd dashboard
streamlit run app.py
```

Por defecto, Streamlit inicia la aplicación en:

```text
http://localhost:8501
```

## Fuentes de datos

El dashboard utiliza dos tipos de archivos.

### Último resultado procesado

```text
data/results/model_results_full.csv
```

Este archivo contiene el último snapshot procesado y se utiliza en los módulos generales de exploración.

### Resultados históricos

```text
data/results/history/model_results_full_<fecha>_<periodo>.csv
```

Estos archivos se utilizan en el módulo de evolución temporal. Cada archivo representa una extracción independiente y conserva los siguientes metadatos:

- `extraction_date`: fecha en la que se realizó la extracción.
- `lookahead_period`: horizonte de reserva, normalmente `1m` o `3m`.
- `checkin_date`: fecha prevista de entrada.
- `checkout_date`: fecha prevista de salida.

También puede existir el resumen agregado:

```text
data/results/temporal_evolution.csv
```

Este archivo contiene métricas agregadas por fecha de extracción, horizonte y fechas de estancia.

El histórico se carga seleccionando como máximo un archivo por semana ISO y
horizonte. Si hay varias extracciones en la misma semana para `1m` o `3m`, se
utiliza la más reciente. Los archivos antiguos no se borran y pueden conservarse
para auditoría, pero no se muestran en las vistas temporales.

## Preparación de los datos

Durante la carga, la aplicación realiza varias transformaciones:

1. Lee los resultados separados por `;`.
2. Convierte las predicciones de los modelos a indicadores binarios.
3. En los archivos de los modelos, `-1` representa una anomalía y `1` un registro normal.
4. Calcula el estado conjunto de cada alojamiento.
5. Calcula los percentiles de las puntuaciones de IF y LOF.
6. Genera un ranking combinado mediante la media de ambos percentiles.
7. Reconstruye variables categóricas codificadas mediante one-hot cuando es necesario.
8. Genera señales agregadas relacionadas con anomalías de precio.

La carga de datos utiliza la caché de Streamlit para evitar repetir operaciones costosas durante la navegación.

El botón `Refresh Data Pipeline` limpia la caché de datos y vuelve a ejecutar la
aplicación. No ejecuta el scraper ni modifica los CSV.

## Estados de detección

Cada alojamiento recibe uno de los siguientes estados:

| Estado | Interpretación |
| --- | --- |
| `Normal` | Ninguno de los modelos lo identifica como anomalía. |
| `IF` | Detectado únicamente por Isolation Forest. |
| `LOF` | Detectado únicamente por LOF. |
| `Both` | Detectado simultáneamente por ambos modelos. |

Isolation Forest representa una perspectiva global y busca observaciones aisladas en el espacio de características. LOF representa una perspectiva local y analiza desviaciones respecto a los vecinos más cercanos.

## Filtros globales

La barra lateral ofrece filtros para los módulos generales:

- Región.
- Tipo de alojamiento.
- Estado de detección.
- Rango de precio.
- Rango de superficie.
- Anomalía contextual relacionada con el precio.

El filtro de anomalía contextual combina reglas de precio y desviaciones extremas respecto al tipo de alojamiento o la provincia.

El botón `Refresh Data Pipeline` limpia la caché y vuelve a cargar los resultados almacenados. No ejecuta el scraper ni vuelve a entrenar los modelos.

## Módulos

### 1. Pipeline Overview

Ofrece una visión general del resultado de la detección.

Incluye:

- Total de alojamientos analizados.
- Número de anomalías detectadas por Isolation Forest.
- Número de anomalías detectadas por LOF.
- Número de anomalías de consenso.
- Índice de similitud de Jaccard.
- Correlación de rangos de Spearman entre las puntuaciones de ambos modelos.
- Gráfico de distribución de estados.
- Representación de la intersección entre IF y LOF.

El índice de Jaccard mide la coincidencia entre los conjuntos de anomalías de ambos modelos:

```text
Jaccard = intersección de anomalías / unión de anomalías
```

La correlación de Spearman mide la relación monótona entre las puntuaciones generadas por ambos algoritmos.

### 2. Contextual Price Anomalies

Analiza la relación entre las anomalías y las variables de precio.

Incluye diagramas de caja y violín para comparar por estado:

- Precio absoluto.
- Precio por metro cuadrado.
- Precio relativo al tipo de alojamiento.
- Precio relativo a la provincia.

El objetivo es evitar interpretar automáticamente un precio alto o bajo como anomalía. El precio debe analizarse en relación con el tipo de propiedad, el tamaño y el contexto geográfico.

### 3. Market Anomaly Explorer

Muestra los alojamientos anómalos con mayor desviación de precio.

Se divide en dos grupos:

- `Low-price anomaly candidates`: alojamientos anómalos con precios especialmente bajos.
- `Premium opportunities`: alojamientos anómalos con precios especialmente altos.

Cada candidato muestra:

- Nombre.
- Tipo de alojamiento.
- Región.
- Precio.
- Superficie.
- Habitaciones.
- Precio por metro cuadrado.
- Diferencia respecto a la mediana.

La aplicación calcula una puntuación auxiliar para ordenar los candidatos combinando precio, superficie, habitaciones y precio por metro cuadrado. Esta puntuación es una ayuda visual y no sustituye a las puntuaciones de Isolation Forest o LOF.

### 4. Listing Lookup

Permite buscar un alojamiento por nombre, utilizando coincidencias parciales.

Para el alojamiento seleccionado muestra:

- Información básica de la propiedad.
- Tipo, habitaciones y baños.
- Región y provincia.
- Superficie.
- Precio de la estancia.
- Precio por metro cuadrado.
- Diferencia respecto al precio habitual del tipo.
- Diferencia respecto al precio habitual de la región.
- Estado de detección.
- Puntuaciones de IF y LOF.
- Ranking combinado.
- Reglas de negocio activadas.

También proporciona un veredicto interpretativo sobre si el alojamiento es normal, una anomalía global, una anomalía local o un caso de consenso.

### 5. Market Segment Analysis

Permite analizar las anomalías por diferentes segmentos:

- Tipo de alojamiento.
- Región.
- Provincia.

Para cada segmento calcula:

- Número total de registros.
- Número de anomalías de IF.
- Número de anomalías de LOF.
- Número de anomalías de consenso.
- Tasa de anomalías de cada modelo.

Incluye un filtro de observaciones mínimas para evitar interpretar segmentos con muestras demasiado pequeñas.

### 6. Empirical Heuristic Validation

Compara las predicciones de los modelos con reglas de negocio interpretables.

Las reglas disponibles incluyen:

- Precio extremadamente alto.
- Precio extremadamente bajo.
- Precio por metro cuadrado extremo.
- Número excesivo de baños.
- Villa con superficie anormalmente pequeña.
- Apartamento con superficie anormalmente grande.

Para cada regla se muestra:

- Número de alojamientos afectados.
- Porcentaje sobre los datos filtrados.
- Tasa de detección de IF.
- Tasa de detección de LOF.
- Tasa de consenso.

Las reglas no se consideran etiquetas reales. Funcionan como referencias de validación en un problema no supervisado sin ground truth.

### 7. Final Consensus Ranking

Genera un ranking de los alojamientos más anómalos.

Permite seleccionar tres estrategias:

- `Strict Consensus`: solo casos detectados por ambos modelos.
- `IF-centric`: casos detectados únicamente por Isolation Forest.
- `LOF-centric`: casos detectados únicamente por LOF.

El usuario puede seleccionar entre 10 y 100 alojamientos.

El ranking combinado se calcula con percentiles de las puntuaciones de IF y LOF. Esto permite combinar modelos con escalas diferentes sin comparar directamente sus valores originales.

Los tres primeros casos se pueden desplegar para consultar:

- Tipo y provincia.
- Precio.
- Puntuaciones de IF y LOF.
- Estado de detección.
- Explicación del criterio utilizado.

### 8. Market Timeline

Analiza la evolución del mercado y la persistencia de anomalías con tres vistas
intencionadamente reducidas.

La vista `Weekly market` utiliza el resumen agregado y es la opción recomendada
para una lectura rápida. Las vistas de anomalías persistentes, estacionalidad y
comparación detallada cargan históricos a nivel de registro y pueden tardar más.

#### Weekly market

Utiliza `extraction_date` como eje temporal y muestra:

- Mediana semanal del precio.
- Banda intercuartílica entre los percentiles 25 y 75.
- Heatmap de la tasa de anomalías de consenso por semana y horizonte.
- Indicadores del último snapshot y comparación con el anterior.

La banda intercuartílica muestra si el mercado se concentra o se dispersa,
mientras que el heatmap permite localizar rápidamente semanas y horizontes con
mayor intensidad de anomalías.

#### Persistent anomalies

Agrupa por alojamiento y muestra un gráfico de dispersión que relaciona:

- Número de snapshots en los que aparece.
- Número de detecciones de consenso.
- Precio mediano.

La tabla permite localizar propiedades que aparecen repetidamente como atípicas.
Esta persistencia puede señalar un comportamiento estructural, un problema de
calidad de datos o una característica estable del mercado.

#### Seasonal market

Es la tercera vista y analiza la estacionalidad del mercado junto con la
concentración de anomalías.

Incluye:

- Precio mediano por mes previsto de entrada.
- Comparación entre horizontes `1m` y `3m`.
- Comparación del nivel de precios por mes previsto de entrada.
- Heatmap de la tasa de consenso por región y mes.
- Comparación del snapshot más reciente con la línea base histórica.
- Filtro por horizonte y tipo de alojamiento.

Esta vista permite responder:

- ¿Qué meses presentan los precios más altos?
- ¿La tasa de anomalías aumenta en temporada alta?
- ¿Qué regiones son más sensibles a la estacionalidad?
- ¿El snapshot actual se comporta como el histórico?

Un precio elevado en julio o agosto no se considera automáticamente una anomalía:
los modelos también tienen en cuenta el tipo de alojamiento, la geografía y el
resto de variables contextuales.

La vista `1m vs 3m` ofrece dos comparaciones: una comparación descriptiva entre
horizontes observados en la misma extracción y una comparación de reserva
anticipada cuando la misma estancia aparece en semanas diferentes. La segunda
es observacional y no debe interpretarse como un experimento causal.

La fecha de extracción y la fecha de entrada no representan lo mismo:

- `extraction_date`: cuándo se observó el precio.
- `checkin_date`: para qué estancia era válido el precio.
- `lookahead_period`: con qué antelación se realizó la consulta.

El módulo también muestra indicadores asociados a la última fecha disponible:

- Número de alojamientos.
- Precio mediano.
- Porcentaje de anomalías de consenso.
- Porcentaje de señales de reglas de negocio.

## Flujo de datos

```text
Booking.com
    ↓
Scraper
    ↓
Limpieza y enriquecimiento geográfico
    ↓
ETL y feature engineering
    ↓
Isolation Forest + LOF
    ↓
Evaluación y consenso
    ↓
CSV de resultados
    ↓
Dashboard Streamlit
```

El dashboard es una capa de exploración. Las predicciones deben generarse previamente mediante el pipeline.

## Interpretación de resultados

### Isolation Forest

Detecta observaciones globalmente aisladas en el espacio de características. Es útil para identificar alojamientos que se separan notablemente del comportamiento general del mercado.

Puede detectar, por ejemplo:

- Precios extremos.
- Configuraciones estructurales poco frecuentes.
- Combinaciones inusuales de precio, superficie y características.

### LOF

Detecta observaciones cuya densidad local es diferente a la de sus vecinos. Es útil para encontrar anomalías dentro de mercados concretos, regiones o tipos de alojamiento.

### Consenso

Cuando ambos modelos coinciden, la señal suele ser más robusta desde el punto de vista interpretativo. Sin embargo, el consenso no constituye una etiqueta verdadera ni garantiza que el registro sea un error.

## Limitaciones

- El problema es no supervisado y no dispone necesariamente de etiquetas reales.
- Una anomalía representa un comportamiento atípico, no un error confirmado.
- Las tasas individuales de anomalías dependen de la contaminación configurada en los modelos.
- Las comparaciones temporales dependen de que la cobertura de alojamientos sea comparable entre semanas.
- Un cambio en la composición regional o en los tipos de alojamiento puede afectar a los precios agregados.
- `1m` y `3m` son horizontes de reserva diferentes, no simplemente dos fechas históricas.
- La comparación por fecha de entrada solo representa una evolución del mismo periodo si las fechas de entrada se repiten en diferentes extracciones.
- El dashboard no realiza scraping, no ejecuta el ETL y no reentrena modelos.

## Actualización de resultados

Para generar o actualizar los resultados históricos, deben procesarse los snapshots mediante el pipeline:

```powershell
..\venv\Scripts\python.exe ..\pipeline\run_pipeline.py
```

Los resultados históricos se almacenan en:

```text
data/results/history/
```

Y el resumen temporal se genera en:

```text
data/results/temporal_evolution.csv
```

Después de actualizar los datos, utiliza el botón de refresco de la aplicación o reinicia Streamlit para limpiar la caché.

Para ejecutar una actualización completa desde la raíz del proyecto en Windows:

```powershell
.\venv\Scripts\python.exe main.py
```

Para regenerar únicamente el resumen temporal después de revisar el histórico:

```powershell
.\venv\Scripts\python.exe pipeline\src\temporal_summary.py
```

No es necesario borrar manualmente los duplicados semanales: el pipeline y el
dashboard seleccionan automáticamente la extracción más reciente de cada semana
ISO y horizonte.
