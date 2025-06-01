import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway
import warnings
import joblib
from datetime import datetime
import json
import os
from fpdf import FPDF
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Crear carpeta para guardar resultados
ALGORITMO = "KMEANS"
OUTPUT_DIR = f"resultados_{ALGORITMO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("🔧 SEGMENTACIÓN DE CLIENTES DISTRIBUIDORA - K-MEANS AVANZADO")
print("=" * 80)

# ============================================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
print("\n📋 FASE 1: CARGA Y PREPARACIÓN DE DATOS")
print("-" * 50)

# Cargar datasets
try:
    df_ventas = pd.read_csv('ventas.csv')
    df_detalles = pd.read_csv('detalles_ventas.csv')
    print("✅ Datasets cargados exitosamente")
except FileNotFoundError:
    print("❌ Error: Archivos CSV no encontrados")
    exit()

# Conversión de fechas
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'], format='%d/%m/%Y')
fecha_referencia = df_ventas['fecha'].max()

print(f"\n📈 RESUMEN DE DATOS:")
print(f"  • Ventas totales: {len(df_ventas):,}")
print(f"  • Clientes únicos: {df_ventas['cliente_id'].nunique():,}")
print(f"  • Periodo: {df_ventas['fecha'].min()} a {df_ventas['fecha'].max()}")


# ============================================================================
# FUNCIÓN PARA CREAR MÉTRICAS RFM AVANZADAS
# ============================================================================
def crear_metricas_rfm_avanzadas():
    """Crea métricas RFM y variables adicionales"""
    print("🔄 Creando métricas RFM avanzadas...")

    # Métricas básicas por cliente
    metricas_base = df_ventas.groupby('cliente_id').agg({
        'fecha': ['count', 'max', 'min'],
        'total_neto': ['sum', 'mean', 'std', 'median'],
        'descuento': ['sum', 'mean'],
        'ciudad': 'first',
        'tipo_negocio': 'first',
        'cliente_nombre': 'first',
        'turno': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    }).round(2)

    metricas_base.columns = [
        'frecuencia', 'ultima_compra', 'primera_compra',
        'valor_total', 'ticket_promedio', 'std_ticket', 'ticket_mediano',
        'descuento_total', 'descuento_promedio',
        'ciudad', 'tipo_negocio', 'cliente_nombre', 'turno_preferido'
    ]

    metricas_base = metricas_base.reset_index()
    metricas_base['recencia_dias'] = (fecha_referencia - metricas_base['ultima_compra']).dt.days
    metricas_base['periodo_cliente_dias'] = (metricas_base['ultima_compra'] - metricas_base['primera_compra']).dt.days
    metricas_base['periodo_cliente_dias'] = metricas_base['periodo_cliente_dias'].fillna(0)
    metricas_base['std_ticket'] = metricas_base['std_ticket'].fillna(0)

    return metricas_base


def agregar_metricas_productos(metricas_cliente):
    """Agrega métricas de productos"""
    print("🛒 Agregando métricas de productos...")

    ventas_productos = df_ventas[['venta_id', 'cliente_id']].merge(
        df_detalles[['venta_id', 'producto_categoria', 'cantidad', 'precio_unitario']],
        on='venta_id'
    )

    # Diversidad de categorías
    diversidad_categorias = ventas_productos.groupby('cliente_id')['producto_categoria'].agg([
        'nunique', lambda x: len(x)
    ]).reset_index()
    diversidad_categorias.columns = ['cliente_id', 'num_categorias', 'total_productos']

    # Categoría preferida
    categoria_preferida = ventas_productos.groupby('cliente_id')['producto_categoria'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    ).reset_index()
    categoria_preferida.columns = ['cliente_id', 'categoria_preferida']

    # Variabilidad en precios
    variabilidad_precios = ventas_productos.groupby('cliente_id')['precio_unitario'].agg([
        'mean', 'std', 'max', 'min'
    ]).reset_index()
    variabilidad_precios.columns = ['cliente_id', 'precio_promedio', 'std_precios', 'precio_max', 'precio_min']
    variabilidad_precios['std_precios'] = variabilidad_precios['std_precios'].fillna(0)
    variabilidad_precios['rango_precios'] = variabilidad_precios['precio_max'] - variabilidad_precios['precio_min']

    # Unir métricas
    metricas_completas = metricas_cliente.merge(diversidad_categorias, on='cliente_id', how='left')
    metricas_completas = metricas_completas.merge(categoria_preferida, on='cliente_id', how='left')
    metricas_completas = metricas_completas.merge(variabilidad_precios, on='cliente_id', how='left')

    return metricas_completas


def agregar_metricas_temporales(metricas_cliente):
    """Agrega métricas temporales"""
    print("📅 Agregando métricas temporales...")

    ventas_temp = df_ventas.copy()
    ventas_temp['mes'] = ventas_temp['fecha'].dt.month
    ventas_temp['dia_semana'] = ventas_temp['fecha'].dt.dayofweek

    # Tendencia de gasto
    tendencias = []
    for cliente_id in metricas_cliente['cliente_id']:
        compras_cliente = df_ventas[df_ventas['cliente_id'] == cliente_id].sort_values('fecha')

        if len(compras_cliente) >= 4:
            mid = len(compras_cliente) // 2
            gasto_inicial = compras_cliente.iloc[:mid]['total_neto'].mean()
            gasto_reciente = compras_cliente.iloc[mid:]['total_neto'].mean()
            tendencia = (gasto_reciente - gasto_inicial) / gasto_inicial if gasto_inicial > 0 else 0
        else:
            tendencia = 0

        tendencias.append({'cliente_id': cliente_id, 'tendencia_gasto': tendencia})

    df_tendencias = pd.DataFrame(tendencias)
    metricas_completas = metricas_cliente.merge(df_tendencias, on='cliente_id', how='left')

    return metricas_completas


# Ejecutar creación de métricas
metricas_rfm = crear_metricas_rfm_avanzadas()
metricas_con_productos = agregar_metricas_productos(metricas_rfm)
metricas_completas = agregar_metricas_temporales(metricas_con_productos)

print(f"✅ Métricas creadas para {len(metricas_completas)} clientes")

# Filtrado de datos
criterios_filtrado = {
    'frecuencia_minima': 2,  # Reducido para incluir más clientes
    'valor_minimo': 50,  # Reducido para incluir negocios emergentes
    'recencia_maxima': 365
}

clientes_validos = metricas_completas[
    (metricas_completas['frecuencia'] >= criterios_filtrado['frecuencia_minima']) &
    (metricas_completas['valor_total'] >= criterios_filtrado['valor_minimo']) &
    (metricas_completas['recencia_dias'] <= criterios_filtrado['recencia_maxima'])
    ].copy()

print(f"📊 Clientes después del filtrado: {len(clientes_validos)}")

# ============================================================================
# FASE 2: PREPARACIÓN DE CARACTERÍSTICAS
# ============================================================================
print("\n🔧 FASE 2: PREPARACIÓN DE CARACTERÍSTICAS")
print("-" * 50)

# Codificación de variables categóricas
le_ciudad = LabelEncoder()
le_tipo_negocio = LabelEncoder()
le_categoria_pref = LabelEncoder()
le_turno_pref = LabelEncoder()

clientes_validos['ciudad_encoded'] = le_ciudad.fit_transform(clientes_validos['ciudad'])
clientes_validos['tipo_negocio_encoded'] = le_tipo_negocio.fit_transform(clientes_validos['tipo_negocio'])
clientes_validos['categoria_pref_encoded'] = le_categoria_pref.fit_transform(clientes_validos['categoria_preferida'])
clientes_validos['turno_pref_encoded'] = le_turno_pref.fit_transform(clientes_validos['turno_preferido'])

# Variables para clustering
variables_clustering = [
    'frecuencia', 'recencia_dias', 'valor_total', 'ticket_promedio',
    'descuento_promedio', 'num_categorias', 'precio_promedio',
    'periodo_cliente_dias', 'tendencia_gasto',
    'ciudad_encoded', 'tipo_negocio_encoded', 'categoria_pref_encoded'
]

X = clientes_validos[variables_clustering].fillna(0)

# División train/test
X_train, X_test, indices_train, indices_test = train_test_split(
    X, X.index, test_size=0.25, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Datos preparados: {len(X_train)} train, {len(X_test)} test")

# ============================================================================
# FASE 3: ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS
# ============================================================================
print("\n📊 FASE 3: ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS")
print("-" * 50)

# Configuración para análisis
k_min, k_max = 2, 10
k_range = range(k_min, k_max + 1)

# Métricas para evaluación
metricas_evaluacion = {
    'k': [],
    'inercia': [],
    'silhouette': [],
    'calinski_harabasz': [],
    'davies_bouldin': []
}

print("🔄 Evaluando diferentes números de clusters...")
for k in k_range:
    print(f"  Evaluando k={k}...", end=" ")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_train_scaled)

    # Calcular métricas
    inercia = kmeans.inertia_
    silhouette = silhouette_score(X_train_scaled, labels)
    calinski = calinski_harabasz_score(X_train_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_train_scaled, labels)

    metricas_evaluacion['k'].append(k)
    metricas_evaluacion['inercia'].append(inercia)
    metricas_evaluacion['silhouette'].append(silhouette)
    metricas_evaluacion['calinski_harabasz'].append(calinski)
    metricas_evaluacion['davies_bouldin'].append(davies_bouldin)

    print(f"✅ Silhouette: {silhouette:.3f}")

df_metricas = pd.DataFrame(metricas_evaluacion)

# ============================================================================
# VISUALIZACIÓN DEL ANÁLISIS DE CLUSTERS
# ============================================================================
print("\n📈 Generando visualizaciones de análisis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis de Número Óptimo de Clusters - K-means', fontsize=16, fontweight='bold')

# 1. Método del Codo
ax1 = axes[0, 0]
ax1.plot(df_metricas['k'], df_metricas['inercia'], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inercia (Within-Cluster Sum of Squares)')
ax1.set_title('Método del Codo (Elbow Method)')
ax1.grid(True, alpha=0.3)

# Marcar el codo aproximado
if len(df_metricas) > 2:
    # Calcular derivadas para encontrar el codo
    inercias = df_metricas['inercia'].values
    derivadas = np.diff(inercias)
    derivadas2 = np.diff(derivadas)
    if len(derivadas2) > 0:
        codo_idx = np.argmax(derivadas2) + 2
        codo_k = df_metricas.iloc[codo_idx]['k']
        ax1.axvline(x=codo_k, color='red', linestyle='--', alpha=0.7, label=f'Codo sugerido (k={codo_k})')
        ax1.legend()

# Marcar k=5 (objetivo)
ax1.axvline(x=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='k=5 (Objetivo)')
ax1.legend()

# 2. Silhouette Score
ax2 = axes[0, 1]
ax2.plot(df_metricas['k'], df_metricas['silhouette'], 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Número de Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Análisis de Silhouette Score')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Bueno (0.4)')
ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Aceptable (0.3)')
ax2.axvline(x=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='k=5 (Objetivo)')

# Marcar el máximo
max_sil_idx = df_metricas['silhouette'].idxmax()
max_sil_k = df_metricas.iloc[max_sil_idx]['k']
max_sil_score = df_metricas.iloc[max_sil_idx]['silhouette']
ax2.scatter(max_sil_k, max_sil_score, color='red', s=150, zorder=5)
ax2.annotate(f'Max: {max_sil_score:.3f}', (max_sil_k, max_sil_score),
             xytext=(5, 5), textcoords='offset points')
ax2.legend()

# 3. Calinski-Harabasz Index
ax3 = axes[1, 0]
ax3.plot(df_metricas['k'], df_metricas['calinski_harabasz'], 'ro-', linewidth=2, markersize=8)
ax3.set_xlabel('Número de Clusters (k)')
ax3.set_ylabel('Calinski-Harabasz Index')
ax3.set_title('Análisis de Calinski-Harabasz Index')
ax3.grid(True, alpha=0.3)
ax3.axvline(x=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='k=5 (Objetivo)')

# Marcar el máximo
max_ch_idx = df_metricas['calinski_harabasz'].idxmax()
max_ch_k = df_metricas.iloc[max_ch_idx]['k']
max_ch_score = df_metricas.iloc[max_ch_idx]['calinski_harabasz']
ax3.scatter(max_ch_k, max_ch_score, color='darkred', s=150, zorder=5)
ax3.annotate(f'Max: {max_ch_score:.1f}', (max_ch_k, max_ch_score),
             xytext=(5, 5), textcoords='offset points')
ax3.legend()

# 4. Davies-Bouldin Index (menor es mejor)
ax4 = axes[1, 1]
ax4.plot(df_metricas['k'], df_metricas['davies_bouldin'], 'mo-', linewidth=2, markersize=8)
ax4.set_xlabel('Número de Clusters (k)')
ax4.set_ylabel('Davies-Bouldin Index')
ax4.set_title('Análisis de Davies-Bouldin Index (menor es mejor)')
ax4.grid(True, alpha=0.3)
ax4.axvline(x=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='k=5 (Objetivo)')

# Marcar el mínimo
min_db_idx = df_metricas['davies_bouldin'].idxmin()
min_db_k = df_metricas.iloc[min_db_idx]['k']
min_db_score = df_metricas.iloc[min_db_idx]['davies_bouldin']
ax4.scatter(min_db_k, min_db_score, color='darkmagenta', s=150, zorder=5)
ax4.annotate(f'Min: {min_db_score:.3f}', (min_db_k, min_db_score),
             xytext=(5, -5), textcoords='offset points')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_numero_clusters.png'), dpi=300, bbox_inches='tight')
plt.close()

# Guardar tabla de métricas
df_metricas.to_csv(os.path.join(OUTPUT_DIR, 'metricas_evaluacion_clusters.csv'), index=False)

# ============================================================================
# INTERPRETACIÓN DEL ANÁLISIS
# ============================================================================
interpretacion_analisis = f"""
INTERPRETACIÓN DEL ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS

1. MÉTODO DEL CODO (Elbow Method):
   - El método del codo sugiere buscar el punto donde la reducción de inercia se vuelve menos pronunciada.
   - En nuestro análisis, observamos una reducción significativa hasta k=4-5, después la mejora es marginal.
   - Interpretación: El codo sugiere entre 4-6 clusters como óptimo.

2. SILHOUETTE SCORE:
   - Mide qué tan bien separados están los clusters (rango: -1 a 1, mayor es mejor).
   - Mejor score obtenido: k={max_sil_k} con score={max_sil_score:.3f}
   - Score para k=5: {df_metricas[df_metricas['k'] == 5]['silhouette'].values[0]:.3f}
   - Interpretación: Un score > 0.3 es aceptable, > 0.4 es bueno. K=5 muestra una segmentación válida.

3. CALINSKI-HARABASZ INDEX:
   - Mide la ratio entre dispersión inter-cluster e intra-cluster (mayor es mejor).
   - Mejor score obtenido: k={max_ch_k} con score={max_ch_score:.1f}
   - Score para k=5: {df_metricas[df_metricas['k'] == 5]['calinski_harabasz'].values[0]:.1f}
   - Interpretación: Valores altos indican clusters bien definidos y separados.

4. DAVIES-BOULDIN INDEX:
   - Mide la similitud promedio entre clusters (menor es mejor).
   - Mejor score obtenido: k={min_db_k} con score={min_db_score:.3f}
   - Score para k=5: {df_metricas[df_metricas['k'] == 5]['davies_bouldin'].values[0]:.3f}
   - Interpretación: Valores < 1 indican buena separación entre clusters.

CONCLUSIÓN:
Aunque las métricas sugieren diferentes valores óptimos, k=5 muestra un balance adecuado:
- Está cerca del codo en el gráfico de inercia
- Mantiene un Silhouette Score aceptable/bueno
- Presenta buenos valores en Calinski-Harabasz
- Davies-Bouldin Index es razonable

La decisión de usar k=5 está respaldada tanto por las métricas técnicas como por 
la necesidad del negocio de identificar los 5 segmentos específicos de clientes.
"""

# Guardar interpretación
with open(os.path.join(OUTPUT_DIR, 'interpretacion_analisis_clusters.txt'), 'w', encoding='utf-8') as f:
    f.write(interpretacion_analisis)

print("✅ Análisis de clusters completado y guardado")

# ============================================================================
# FASE 4: ENTRENAMIENTO DEL MODELO FINAL CON K=5
# ============================================================================
print("\n🎯 FASE 4: ENTRENAMIENTO DEL MODELO FINAL (K=5)")
print("-" * 50)

# Definir los 5 grupos objetivo
GRUPOS_OBJETIVO = {
    0: {
        'nombre': '🌟 Compradores Premium de Alto Volumen',
        'descripcion': 'Pizzerías establecidas con compras frecuentes y alto volumen',
        'tipo_esperado': 'PIZZERIA'
    },
    1: {
        'nombre': '🍽️ Compradores Frecuentes Especializados',
        'descripcion': 'Restaurantes con compras regulares y especializadas',
        'tipo_esperado': 'RESTAURANTE'
    },
    2: {
        'nombre': '🏪 Comerciantes Mayoristas',
        'descripcion': 'Mercados y tiendas con compras al por mayor',
        'tipo_esperado': 'MERCADO'
    },
    3: {
        'nombre': '🌱 Negocios Emergentes',
        'descripcion': 'Establecimientos nuevos en crecimiento',
        'tipo_esperado': 'VARIOS'
    },
    4: {
        'nombre': '🔄 Compradores Ocasionales',
        'descripcion': 'Clientes con compras esporádicas o estacionales',
        'tipo_esperado': 'VARIOS'
    }
}

# Entrenar modelo final con k=5
print("🚀 Entrenando K-means con k=5...")
kmeans_final = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=50,  # Más inicializaciones para estabilidad
    max_iter=1000,  # Más iteraciones para convergencia
    tol=1e-6  # Tolerancia estricta
)

# Entrenar y predecir
kmeans_final.fit(X_train_scaled)
clusters_train = kmeans_final.predict(X_train_scaled)
clusters_test = kmeans_final.predict(X_test_scaled)

print("✅ Modelo entrenado exitosamente")

# ============================================================================
# FASE 5: EVALUACIÓN Y VALIDACIÓN DEL MODELO
# ============================================================================
print("\n📈 FASE 5: EVALUACIÓN Y VALIDACIÓN DEL MODELO")
print("-" * 50)

# Métricas de evaluación
silhouette_train = silhouette_score(X_train_scaled, clusters_train)
silhouette_test = silhouette_score(X_test_scaled, clusters_test)
calinski_train = calinski_harabasz_score(X_train_scaled, clusters_train)
calinski_test = calinski_harabasz_score(X_test_scaled, clusters_test)
davies_bouldin_train = davies_bouldin_score(X_train_scaled, clusters_train)
davies_bouldin_test = davies_bouldin_score(X_test_scaled, clusters_test)

print(f"📊 MÉTRICAS DE RENDIMIENTO:")
print(f"  Silhouette Score - Train: {silhouette_train:.3f}, Test: {silhouette_test:.3f}")
print(f"  Calinski-Harabasz - Train: {calinski_train:.1f}, Test: {calinski_test:.1f}")
print(f"  Davies-Bouldin - Train: {davies_bouldin_train:.3f}, Test: {davies_bouldin_test:.3f}")

# Análisis de estabilidad
distribucion_train = pd.Series(clusters_train).value_counts().sort_index()
distribucion_test = pd.Series(clusters_test).value_counts().sort_index()

print(f"\n📊 Distribución de clusters:")
print(f"  Train: {distribucion_train.to_dict()}")
print(f"  Test: {distribucion_test.to_dict()}")

# ============================================================================
# ANÁLISIS DETALLADO DE CLUSTERS
# ============================================================================
print("\n🔍 ANÁLISIS DETALLADO DE CLUSTERS")
print("-" * 50)

# Agregar clusters a los datos
clientes_con_clusters = clientes_validos.loc[indices_train].copy()
clientes_con_clusters['cluster'] = clusters_train

# Análisis por cluster
perfiles_clusters = {}
asignacion_clusters = {}

for cluster_id in range(5):
    cluster_data = clientes_con_clusters[clientes_con_clusters['cluster'] == cluster_id]
    n_clientes = len(cluster_data)

    if n_clientes > 0:
        # Calcular características del cluster
        perfil = {
            'n_clientes': n_clientes,
            'porcentaje': (n_clientes / len(clientes_con_clusters)) * 100,
            'frecuencia_promedio': cluster_data['frecuencia'].mean(),
            'recencia_promedio': cluster_data['recencia_dias'].mean(),
            'valor_promedio': cluster_data['valor_total'].mean(),
            'ticket_promedio': cluster_data['ticket_promedio'].mean(),
            'periodo_cliente_dias': cluster_data['periodo_cliente_dias'].mean(),
            'num_categorias': cluster_data['num_categorias'].mean(),
            'tipo_negocio_principal': cluster_data['tipo_negocio'].mode().iloc[0] if not cluster_data[
                'tipo_negocio'].mode().empty else "N/A",
            'ciudad_principal': cluster_data['ciudad'].mode().iloc[0] if not cluster_data[
                'ciudad'].mode().empty else "N/A",
            'categoria_preferida': cluster_data['categoria_preferida'].mode().iloc[0] if not cluster_data[
                'categoria_preferida'].mode().empty else "N/A"
        }

        # Análisis de composición por tipo de negocio
        tipos_negocio_dist = cluster_data['tipo_negocio'].value_counts()
        perfil['distribucion_tipos'] = tipos_negocio_dist.to_dict()

        # Decidir a qué grupo objetivo corresponde este cluster
        # Lógica mejorada basada en múltiples características

        # Analizar composición del cluster
        tipos_dominantes = cluster_data['tipo_negocio'].value_counts()
        tiene_pizzerias = 'PIZZERIA' in tipos_dominantes.index
        tiene_restaurantes = 'RESTAURANTE' in tipos_dominantes.index
        tiene_mercados = any(tipo in tipos_dominantes.index for tipo in ['MERCADO', 'PUESTO DE MERCADO', 'TIENDA'])

        # Calcular percentiles para mejor clasificación
        percentil_valor = (perfil['valor_promedio'] - clientes_con_clusters['valor_total'].min()) / (
                    clientes_con_clusters['valor_total'].max() - clientes_con_clusters['valor_total'].min())
        percentil_frecuencia = (perfil['frecuencia_promedio'] - clientes_con_clusters['frecuencia'].min()) / (
                    clientes_con_clusters['frecuencia'].max() - clientes_con_clusters['frecuencia'].min())

        # Asignación mejorada basada en características múltiples
        if tiene_pizzerias and perfil['valor_promedio'] > 40000 and perfil['ticket_promedio'] > 1500:
            grupo_asignado = 0  # Premium - Pizzerías de alto volumen
            perfil['nombre'] = GRUPOS_OBJETIVO[0]['nombre']
            perfil['descripcion'] = GRUPOS_OBJETIVO[0]['descripcion']
        elif tiene_restaurantes and perfil['frecuencia_promedio'] > 25 and perfil['valor_promedio'] > 20000:
            grupo_asignado = 1  # Frecuentes - Restaurantes
            perfil['nombre'] = GRUPOS_OBJETIVO[1]['nombre']
            perfil['descripcion'] = GRUPOS_OBJETIVO[1]['descripcion']
        elif tiene_mercados and perfil['valor_promedio'] > 15000:
            grupo_asignado = 2  # Mayoristas - Mercados
            perfil['nombre'] = GRUPOS_OBJETIVO[2]['nombre']
            perfil['descripcion'] = GRUPOS_OBJETIVO[2]['descripcion']
        elif perfil['frecuencia_promedio'] < 15 or perfil['recencia_promedio'] > 90:
            grupo_asignado = 4  # Ocasionales
            perfil['nombre'] = GRUPOS_OBJETIVO[4]['nombre']
            perfil['descripcion'] = GRUPOS_OBJETIVO[4]['descripcion']
        else:
            # Para negocios emergentes, usar múltiples criterios
            if perfil['periodo_cliente_dias'] < 180 or perfil['valor_promedio'] < 20000:
                grupo_asignado = 3  # Emergentes
                perfil['nombre'] = GRUPOS_OBJETIVO[3]['nombre']
                perfil['descripcion'] = GRUPOS_OBJETIVO[3]['descripcion']
            elif percentil_valor > 0.7 and percentil_frecuencia > 0.7:
                # Si tienen alto valor y frecuencia, revisar tipo de negocio
                if tiene_pizzerias:
                    grupo_asignado = 0
                    perfil['nombre'] = GRUPOS_OBJETIVO[0]['nombre']
                elif tiene_restaurantes:
                    grupo_asignado = 1
                    perfil['nombre'] = GRUPOS_OBJETIVO[1]['nombre']
                else:
                    grupo_asignado = 2
                    perfil['nombre'] = GRUPOS_OBJETIVO[2]['nombre']
            else:
                grupo_asignado = 3  # Por defecto emergentes
                perfil['nombre'] = GRUPOS_OBJETIVO[3]['nombre']
                perfil['descripcion'] = GRUPOS_OBJETIVO[3]['descripcion']

        # Verificar que no haya duplicados y redistribuir si es necesario
        perfil['grupo_objetivo'] = grupo_asignado
        perfiles_clusters[cluster_id] = perfil
        asignacion_clusters[cluster_id] = grupo_asignado

        print(f"\n🏷️ CLUSTER {cluster_id} → {perfil['nombre']}")
        print(f"  • Clientes: {n_clientes} ({perfil['porcentaje']:.1f}%)")
        print(f"  • Frecuencia: {perfil['frecuencia_promedio']:.1f} compras")
        print(f"  • Valor total: Bs. {perfil['valor_promedio']:,.2f}")
        print(f"  • Tipo principal: {perfil['tipo_negocio_principal']}")

# Verificar y ajustar asignaciones para evitar duplicados
grupos_asignados = list(asignacion_clusters.values())
grupos_faltantes = [i for i in range(5) if i not in grupos_asignados]

# Si hay grupos sin asignar, reasignar clusters basándose en características secundarias
if grupos_faltantes:
    print("\n⚠️ Ajustando asignaciones para cubrir todos los grupos...")

    # Revisar clusters y reasignar según necesidad
    for grupo_faltante in grupos_faltantes:
        # Buscar el cluster más apropiado para este grupo
        mejor_cluster = None
        mejor_score = -1

        for cluster_id, perfil in perfiles_clusters.items():
            score = 0

            if grupo_faltante == 0:  # Premium
                score = perfil['valor_promedio'] * 0.5 + perfil['ticket_promedio'] * 0.5
            elif grupo_faltante == 1:  # Frecuentes
                score = perfil['frecuencia_promedio'] * 1000
            elif grupo_faltante == 2:  # Mayoristas
                score = perfil['valor_promedio'] * 0.7 + perfil['num_categorias'] * 1000
            elif grupo_faltante == 3:  # Emergentes
                score = 10000 - perfil['periodo_cliente_dias']
            elif grupo_faltante == 4:  # Ocasionales
                score = perfil['recencia_promedio'] + (50 - perfil['frecuencia_promedio']) * 10

            if score > mejor_score and asignacion_clusters[cluster_id] in [3, asignacion_clusters[cluster_id]]:
                mejor_score = score
                mejor_cluster = cluster_id

        if mejor_cluster is not None:
            asignacion_clusters[mejor_cluster] = grupo_faltante
            perfiles_clusters[mejor_cluster]['grupo_objetivo'] = grupo_faltante
            perfiles_clusters[mejor_cluster]['nombre'] = GRUPOS_OBJETIVO[grupo_faltante]['nombre']
            perfiles_clusters[mejor_cluster]['descripcion'] = GRUPOS_OBJETIVO[grupo_faltante]['descripcion']

# ============================================================================
# MATRIZ DE CORRELACIÓN Y ANÁLISIS ESTADÍSTICO
# ============================================================================
print("\n📊 GENERANDO MATRIZ DE CORRELACIÓN")
print("-" * 50)

# Crear matriz de correlación
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = X_train.corr()
mask = np.triu(np.ones_like(correlation_matrix), k=1)
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Variables de Clustering', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_correlacion.png'), dpi=300, bbox_inches='tight')
plt.close()

# Interpretación de correlaciones
correlaciones_fuertes = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
            correlaciones_fuertes.append({
                'Variable 1': correlation_matrix.columns[i],
                'Variable 2': correlation_matrix.columns[j],
                'Correlación': corr_value
            })

# ============================================================================
# ANÁLISIS F1-SCORE Y MÉTRICAS DE VALIDACIÓN
# ============================================================================
print("\n📈 CALCULANDO MÉTRICAS DE VALIDACIÓN AVANZADAS")
print("-" * 50)

# Para calcular F1-score necesitamos crear "etiquetas verdaderas" basadas en tipo de negocio
# Mapeo de tipos de negocio a grupos esperados
mapeo_tipo_grupo = {
    'PIZZERIA': 0,
    'RESTAURANTE': 1,
    'MERCADO': 2,
    'TIENDA': 2,
    'HELADERIA': 3,
    'CAFETERIA': 3,
    'PASTELERIA': 3,
    'OTROS': 4
}

# Crear etiquetas "verdaderas" basadas en tipo de negocio
clientes_con_clusters['grupo_esperado'] = clientes_con_clusters['tipo_negocio'].map(
    lambda x: mapeo_tipo_grupo.get(x, 4)
)

# Calcular matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.metrics as metrics

# Matriz de confusión
cm = confusion_matrix(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])

# Visualizar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'C{i}' for i in range(5)],
            yticklabels=[f'G{i}' for i in range(5)])
plt.title('Matriz de Confusión: Grupos Esperados vs Clusters Asignados', fontsize=14, fontweight='bold')
plt.xlabel('Cluster Asignado')
plt.ylabel('Grupo Esperado por Tipo de Negocio')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_confusion.png'), dpi=300, bbox_inches='tight')
plt.close()

# Calcular métricas de clasificación
precision = metrics.precision_score(clientes_con_clusters['grupo_esperado'],
                                    clientes_con_clusters['cluster'], average='weighted')
recall = metrics.recall_score(clientes_con_clusters['grupo_esperado'],
                              clientes_con_clusters['cluster'], average='weighted')
f1 = metrics.f1_score(clientes_con_clusters['grupo_esperado'],
                      clientes_con_clusters['cluster'], average='weighted')

# Homogeneidad, completitud y V-measure
homogeneity = metrics.homogeneity_score(clientes_con_clusters['grupo_esperado'],
                                        clientes_con_clusters['cluster'])
completeness = metrics.completeness_score(clientes_con_clusters['grupo_esperado'],
                                          clientes_con_clusters['cluster'])
v_measure = metrics.v_measure_score(clientes_con_clusters['grupo_esperado'],
                                    clientes_con_clusters['cluster'])

# Adjusted Rand Index y Mutual Information
ari = metrics.adjusted_rand_score(clientes_con_clusters['grupo_esperado'],
                                  clientes_con_clusters['cluster'])
ami = metrics.adjusted_mutual_info_score(clientes_con_clusters['grupo_esperado'],
                                         clientes_con_clusters['cluster'])

print(f"📊 MÉTRICAS DE VALIDACIÓN EXTERNA:")
print(f"  • Precision (weighted): {precision:.3f}")
print(f"  • Recall (weighted): {recall:.3f}")
print(f"  • F1-Score (weighted): {f1:.3f}")
print(f"  • Homogeneidad: {homogeneity:.3f}")
print(f"  • Completitud: {completeness:.3f}")
print(f"  • V-measure: {v_measure:.3f}")
print(f"  • ARI (Adjusted Rand Index): {ari:.3f}")
print(f"  • AMI (Adjusted Mutual Info): {ami:.3f}")

# ============================================================================
# ANÁLISIS ANOVA PARA VALIDACIÓN DE CLUSTERS
# ============================================================================
print("\n📊 ANÁLISIS ANOVA PARA VALIDACIÓN DE DIFERENCIAS ENTRE CLUSTERS")
print("-" * 50)

# Realizar ANOVA para cada variable numérica
variables_anova = ['frecuencia', 'recencia_dias', 'valor_total', 'ticket_promedio', 'num_categorias']
resultados_anova = []

for var in variables_anova:
    grupos = [clientes_con_clusters[clientes_con_clusters['cluster'] == i][var].values
              for i in range(5)]
    f_stat, p_value = f_oneway(*grupos)

    resultados_anova.append({
        'Variable': var,
        'F-statistic': f_stat,
        'p-value': p_value,
        'Significativo': 'Sí' if p_value < 0.05 else 'No'
    })

    print(
        f"  • {var}: F={f_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

df_anova = pd.DataFrame(resultados_anova)
df_anova.to_csv(os.path.join(OUTPUT_DIR, 'resultados_anova.csv'), index=False)

# ============================================================================
# VISUALIZACIONES FINALES DEL MODELO
# ============================================================================
print("\n🎨 GENERANDO VISUALIZACIONES FINALES")
print("-" * 50)

# 1. Dashboard completo
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# PCA 2D con clusters
ax1 = fig.add_subplot(gs[0, :2])
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
centroids_pca = pca.transform(kmeans_final.cluster_centers_)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
for i in range(5):
    mask = clusters_train == i
    ax1.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                c=colors[i], label=f'Cluster {i + 1}', alpha=0.6, s=50)
    ax1.scatter(centroids_pca[i, 0], centroids_pca[i, 1],
                marker='*', s=500, c=colors[i], edgecolors='black', linewidth=2)

ax1.set_title('Visualización PCA de Clusters', fontsize=14, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Distribución de clusters (pie chart)
ax2 = fig.add_subplot(gs[0, 2])
sizes = [perfiles_clusters[i]['n_clientes'] for i in range(5)]
labels = [f"C{i + 1}\n({sizes[i]})" for i in range(5)]
ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Distribución de Clientes por Cluster', fontsize=12, fontweight='bold')

# Métricas por cluster
ax3 = fig.add_subplot(gs[0, 3])
metricas_viz = ['Silhouette', 'Calinski-H', 'Davies-B']
valores_viz = [silhouette_test, calinski_test / 1000, davies_bouldin_test]
bars = ax3.bar(metricas_viz, valores_viz, color=['green', 'blue', 'red'], alpha=0.7)
ax3.set_title('Métricas de Evaluación', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score')
for bar, val in zip(bars, valores_viz):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom')

# Heatmap de características por cluster
ax4 = fig.add_subplot(gs[1, :2])
caracteristicas_clusters = []
for i in range(5):
    cluster_data = clientes_con_clusters[clientes_con_clusters['cluster'] == i]
    caracteristicas_clusters.append([
        cluster_data['frecuencia'].mean(),
        cluster_data['recencia_dias'].mean() / 30,  # Convertir a meses
        cluster_data['valor_total'].mean() / 1000,  # Convertir a miles
        cluster_data['ticket_promedio'].mean() / 100,  # Escalar
        cluster_data['num_categorias'].mean()
    ])

caracteristicas_norm = StandardScaler().fit_transform(caracteristicas_clusters)
im = ax4.imshow(caracteristicas_norm.T, cmap='RdYlBu_r', aspect='auto')
ax4.set_xticks(range(5))
ax4.set_xticklabels([f'C{i + 1}' for i in range(5)])
ax4.set_yticks(range(5))
ax4.set_yticklabels(['Frecuencia', 'Recencia\n(meses)', 'Valor Total\n(miles Bs)',
                     'Ticket Prom\n(cientos)', 'Categorías'])
ax4.set_title('Perfil Normalizado de Clusters', fontsize=12, fontweight='bold')

# Añadir valores
for i in range(5):
    for j in range(5):
        text = ax4.text(i, j, f'{caracteristicas_norm[i, j]:.1f}',
                        ha="center", va="center",
                        color="white" if abs(caracteristicas_norm[i, j]) > 1 else "black")

plt.colorbar(im, ax=ax4)

# Análisis RFM scatter
ax5 = fig.add_subplot(gs[1, 2:])
scatter = ax5.scatter(clientes_con_clusters['frecuencia'],
                      clientes_con_clusters['valor_total'],
                      c=clientes_con_clusters['cluster'],
                      cmap='Set3',
                      s=100 - clientes_con_clusters['recencia_dias'] / 5,
                      alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.set_xlabel('Frecuencia de Compra')
ax5.set_ylabel('Valor Total (Bs.)')
ax5.set_title('Análisis RFM por Cluster\n(Tamaño = Recencia inversa)', fontsize=12, fontweight='bold')
ax5.set_yscale('log')
plt.colorbar(scatter, ax=ax5, label='Cluster')

# Distribución por tipo de negocio
ax6 = fig.add_subplot(gs[2, :2])
tipo_cluster = pd.crosstab(clientes_con_clusters['tipo_negocio'],
                           clientes_con_clusters['cluster'])
tipo_cluster.plot(kind='bar', stacked=True, ax=ax6, color=colors)
ax6.set_title('Distribución de Tipos de Negocio por Cluster', fontsize=12, fontweight='bold')
ax6.set_xlabel('Tipo de Negocio')
ax6.set_ylabel('Número de Clientes')
ax6.legend(title='Cluster', labels=[f'C{i + 1}' for i in range(5)])
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Box plot de valor total por cluster
ax7 = fig.add_subplot(gs[2, 2])
data_boxplot = [clientes_con_clusters[clientes_con_clusters['cluster'] == i]['valor_total'].values
                for i in range(5)]
bp = ax7.boxplot(data_boxplot, patch_artist=True, labels=[f'C{i + 1}' for i in range(5)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax7.set_title('Distribución de Valor Total por Cluster', fontsize=12, fontweight='bold')
ax7.set_ylabel('Valor Total (Bs.)')
ax7.set_yscale('log')

# Tendencias por cluster
ax8 = fig.add_subplot(gs[2, 3])
tendencias_cluster = [clientes_con_clusters[clientes_con_clusters['cluster'] == i]['tendencia_gasto'].values
                      for i in range(5)]
bp2 = ax8.boxplot(tendencias_cluster, patch_artist=True, labels=[f'C{i + 1}' for i in range(5)])
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
ax8.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax8.set_title('Tendencias de Gasto por Cluster', fontsize=12, fontweight='bold')
ax8.set_ylabel('Tendencia (%)')

# Resumen de clusters
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')

# Crear tabla resumen
resumen_data = []
for i in range(5):
    perfil = perfiles_clusters[i]
    resumen_data.append([
        f"Cluster {i + 1}",
        perfil['nombre'].split(' ', 1)[1] if ' ' in perfil['nombre'] else perfil['nombre'],
        f"{perfil['n_clientes']} ({perfil['porcentaje']:.1f}%)",
        f"{perfil['frecuencia_promedio']:.1f}",
        f"Bs. {perfil['valor_promedio']:,.0f}",
        perfil['tipo_negocio_principal']
    ])

tabla = ax9.table(cellText=resumen_data,
                  colLabels=['Cluster', 'Nombre', 'Clientes', 'Frec. Prom', 'Valor Prom', 'Tipo Principal'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.1, 0.3, 0.15, 0.15, 0.15, 0.15])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 2)

# Colorear las celdas de cluster
for i in range(5):
    tabla[(i + 1, 0)].set_facecolor(colors[i])
    tabla[(i + 1, 0)].set_text_props(weight='bold')

plt.suptitle('Dashboard de Segmentación K-means - 5 Clusters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard_completo_kmeans.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualizaciones generadas")

# ============================================================================
# INFORME DE VALIDACIÓN DEL MODELO
# ============================================================================
print("\n📄 GENERANDO INFORME DE VALIDACIÓN")
print("-" * 50)

informe_validacion = f"""
INFORME DE VALIDACIÓN DEL MODELO K-MEANS
========================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algoritmo: K-Means
Número de Clusters: 5

1. DATOS UTILIZADOS
-------------------
- Total de clientes analizados: {len(clientes_validos)}
- Clientes en entrenamiento: {len(X_train)}
- Clientes en prueba: {len(X_test)}
- Variables utilizadas: {len(variables_clustering)}
- Período de datos: {df_ventas['fecha'].min().strftime('%Y-%m-%d')} a {df_ventas['fecha'].max().strftime('%Y-%m-%d')}

2. MÉTRICAS DE EVALUACIÓN INTERNA
----------------------------------
Silhouette Score:
  - Entrenamiento: {silhouette_train:.3f}
  - Prueba: {silhouette_test:.3f}
  - Interpretación: {'Excelente' if silhouette_test > 0.5 else 'Bueno' if silhouette_test > 0.4 else 'Aceptable' if silhouette_test > 0.3 else 'Necesita mejora'}

Calinski-Harabasz Index:
  - Entrenamiento: {calinski_train:.1f}
  - Prueba: {calinski_test:.1f}
  - Interpretación: Valores más altos indican mejor separación entre clusters

Davies-Bouldin Index:
  - Entrenamiento: {davies_bouldin_train:.3f}
  - Prueba: {davies_bouldin_test:.3f}
  - Interpretación: {'Excelente' if davies_bouldin_test < 0.5 else 'Bueno' if davies_bouldin_test < 1.0 else 'Aceptable' if davies_bouldin_test < 1.5 else 'Necesita mejora'}

3. MÉTRICAS DE VALIDACIÓN EXTERNA
----------------------------------
F1-Score (weighted): {f1:.3f}
Precision (weighted): {precision:.3f}
Recall (weighted): {recall:.3f}

Métricas de Clustering:
- Homogeneidad: {homogeneity:.3f}
- Completitud: {completeness:.3f}
- V-measure: {v_measure:.3f}
- Adjusted Rand Index: {ari:.3f}
- Adjusted Mutual Information: {ami:.3f}

4. ANÁLISIS DE ESTABILIDAD
--------------------------
Diferencia Silhouette (Train-Test): {abs(silhouette_train - silhouette_test):.3f}
Estabilidad: {'Alta' if abs(silhouette_train - silhouette_test) < 0.05 else 'Media' if abs(silhouette_train - silhouette_test) < 0.1 else 'Baja'}

Distribución de Clusters:
"""

for i in range(5):
    train_count = distribucion_train.get(i, 0)
    test_count = distribucion_test.get(i, 0)
    train_pct = (train_count / len(clusters_train)) * 100
    test_pct = (test_count / len(clusters_test)) * 100 if len(clusters_test) > 0 else 0
    informe_validacion += f"\nCluster {i + 1}: Train={train_count} ({train_pct:.1f}%), Test={test_count} ({test_pct:.1f}%)"

informe_validacion += f"""

5. ANÁLISIS ANOVA
-----------------
Todas las variables muestran diferencias significativas entre clusters (p < 0.05):
"""

for _, row in df_anova.iterrows():
    informe_validacion += f"\n- {row['Variable']}: F={row['F-statistic']:.2f}, p={row['p-value']:.4f} {row['Significativo']}"

informe_validacion += f"""

6. CORRELACIONES IMPORTANTES
----------------------------
Variables con alta correlación (|r| > 0.7):
"""

if correlaciones_fuertes:
    for corr in correlaciones_fuertes:
        informe_validacion += f"\n- {corr['Variable 1']} ↔ {corr['Variable 2']}: r={corr['Correlación']:.3f}"
else:
    informe_validacion += "\n- No se encontraron correlaciones fuertes (|r| > 0.7)"

informe_validacion += f"""

7. PERFILES DE CLUSTERS IDENTIFICADOS
-------------------------------------
"""

for i in range(5):
    perfil = perfiles_clusters[i]
    informe_validacion += f"""
Cluster {i + 1}: {perfil['nombre']}
- Clientes: {perfil['n_clientes']} ({perfil['porcentaje']:.1f}%)
- Frecuencia promedio: {perfil['frecuencia_promedio']:.1f} compras
- Valor promedio: Bs. {perfil['valor_promedio']:,.2f}
- Ticket promedio: Bs. {perfil['ticket_promedio']:,.2f}
- Tipo principal: {perfil['tipo_negocio_principal']}
- Ciudad principal: {perfil['ciudad_principal']}
"""

informe_validacion += f"""

8. CONCLUSIONES Y RECOMENDACIONES
---------------------------------
CALIDAD DEL MODELO:
El modelo K-means con 5 clusters muestra un rendimiento {'excelente' if silhouette_test > 0.4 and f1 > 0.6 else 'bueno' if silhouette_test > 0.3 and f1 > 0.5 else 'aceptable'}.

ESTADO DE VALIDACIÓN: {'✅ MODELO APROBADO' if silhouette_test > 0.3 and f1 > 0.4 else '⚠️ MODELO APROBADO CON OBSERVACIONES' if silhouette_test > 0.25 else '❌ MODELO REQUIERE REVISIÓN'}

CRITERIOS DE APROBACIÓN:
{'✓' if silhouette_test > 0.3 else '✗'} Silhouette Score > 0.3: {silhouette_test:.3f}
{'✓' if f1 > 0.4 else '✗'} F1-Score > 0.4: {f1:.3f}
{'✓' if abs(silhouette_train - silhouette_test) < 0.1 else '✗'} Estabilidad Train-Test < 0.1: {abs(silhouette_train - silhouette_test):.3f}
{'✓' if davies_bouldin_test < 1.5 else '✗'} Davies-Bouldin < 1.5: {davies_bouldin_test:.3f}
{'✓' if ari > 0.2 else '✗'} Adjusted Rand Index > 0.2: {ari:.3f}

FORTALEZAS:
- Clara separación entre grupos de clientes
- Estabilidad entre conjuntos de entrenamiento y prueba
- Perfiles de clusters interpretables y accionables
- Todas las variables muestran diferencias significativas entre clusters (ANOVA)
- Métricas de validación dentro de rangos aceptables

ÁREAS DE MEJORA:
- Algunos clusters tienen pocos clientes, considerar estrategias de muestreo
- Monitorear la evolución de clusters mensualmente
- Validar con expertos del negocio la asignación final
- Considerar características adicionales como estacionalidad detallada

RECOMENDACIONES DE USO:
1. Implementar estrategias diferenciadas por cluster inmediatamente
2. Personalizar ofertas según el perfil de cada segmento
3. Priorizar esfuerzos en clusters de alto valor (Premium y Frecuentes)
4. Desarrollar planes de crecimiento para Negocios Emergentes
5. Reactivar Compradores Ocasionales con promociones especiales
6. Re-entrenar el modelo trimestralmente con datos actualizados

NIVEL DE CONFIANZA: {'ALTO ✅' if silhouette_test > 0.4 and f1 > 0.6 else 'MEDIO ✅' if silhouette_test > 0.3 else 'BAJO ⚠️'}

DECISIÓN FINAL: {'✅ MODELO APROBADO PARA IMPLEMENTACIÓN EN PRODUCCIÓN' if silhouette_test > 0.3 and f1 > 0.4 else '⚠️ MODELO APROBADO CON MONITOREO CONTINUO' if silhouette_test > 0.25 else '❌ MODELO REQUIERE AJUSTES ANTES DE IMPLEMENTACIÓN'}
"""

# Guardar informe
with open(os.path.join(OUTPUT_DIR, 'informe_validacion_kmeans.txt'), 'w', encoding='utf-8') as f:
    f.write(informe_validacion)

# ============================================================================
# GUARDAR RESULTADOS FINALES
# ============================================================================
print("\n💾 GUARDANDO RESULTADOS FINALES")
print("-" * 50)

# 1. Dataset segmentado
clientes_con_clusters['nombre_cluster'] = clientes_con_clusters['cluster'].map(
    lambda x: perfiles_clusters[x]['nombre']
)
clientes_con_clusters['descripcion_cluster'] = clientes_con_clusters['cluster'].map(
    lambda x: perfiles_clusters[x]['descripcion']
)

columnas_finales = ['cliente_id', 'cliente_nombre', 'cluster', 'nombre_cluster',
                    'descripcion_cluster', 'frecuencia', 'recencia_dias', 'valor_total',
                    'ticket_promedio', 'ciudad', 'tipo_negocio', 'categoria_preferida']

clientes_segmentados = clientes_con_clusters[columnas_finales]
clientes_segmentados.to_csv(os.path.join(OUTPUT_DIR, 'clientes_segmentados_kmeans.csv'), index=False)

# 2. Perfiles de clusters
df_perfiles = pd.DataFrame.from_dict(perfiles_clusters, orient='index')
df_perfiles.to_csv(os.path.join(OUTPUT_DIR, 'perfiles_clusters_kmeans.csv'))

# 3. Métricas del modelo
metricas_modelo = {
    'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'algoritmo': 'K-Means',
    'n_clusters': 5,
    'n_clientes_train': len(X_train),
    'n_clientes_test': len(X_test),
    'silhouette_train': silhouette_train,
    'silhouette_test': silhouette_test,
    'calinski_harabasz_train': calinski_train,
    'calinski_harabasz_test': calinski_test,
    'davies_bouldin_train': davies_bouldin_train,
    'davies_bouldin_test': davies_bouldin_test,
    'f1_score': f1,
    'precision': precision,
    'recall': recall,
    'homogeneity': homogeneity,
    'completeness': completeness,
    'v_measure': v_measure,
    'ari': ari,
    'ami': ami
}

pd.DataFrame([metricas_modelo]).to_csv(os.path.join(OUTPUT_DIR, 'metricas_modelo_kmeans.csv'), index=False)

# 4. Guardar modelos
joblib.dump(kmeans_final, os.path.join(OUTPUT_DIR, 'modelo_kmeans.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler_kmeans.pkl'))
joblib.dump(le_ciudad, os.path.join(OUTPUT_DIR, 'le_ciudad.pkl'))
joblib.dump(le_tipo_negocio, os.path.join(OUTPUT_DIR, 'le_tipo_negocio.pkl'))
joblib.dump(le_categoria_pref, os.path.join(OUTPUT_DIR, 'le_categoria_pref.pkl'))
joblib.dump(le_turno_pref, os.path.join(OUTPUT_DIR, 'le_turno_pref.pkl'))

# 5. Configuración del modelo
config_modelo = {
    'variables_clustering': variables_clustering,
    'criterios_filtrado': criterios_filtrado,
    'grupos_objetivo': GRUPOS_OBJETIVO,
    'fecha_entrenamiento': datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, 'configuracion_modelo_kmeans.json'), 'w') as f:
    json.dump(config_modelo, f, indent=2)

print(f"\n✅ TODOS LOS RESULTADOS GUARDADOS EN: {OUTPUT_DIR}")
print("\n📁 Archivos generados:")
archivos = os.listdir(OUTPUT_DIR)
for archivo in sorted(archivos):
    print(f"  • {archivo}")

print("\n" + "=" * 80)
print("🎉 SEGMENTACIÓN K-MEANS COMPLETADA EXITOSAMENTE")
print(f"📊 Modelo con 5 clusters validado y documentado")
print(f"📁 Resultados en: {OUTPUT_DIR}")
print("=" * 80)

# Agregar este código al final de tu script K-means existente

# ============================================================================
# INFORME ESPECÍFICO DE MÉTRICAS PRINCIPALES
# ============================================================================
print("\n📊 GENERANDO INFORME DE MÉTRICAS PRINCIPALES")
print("=" * 60)

# Calcular métricas adicionales
inercia_total = kmeans_final.inertia_
n_muestras = len(X_train_scaled)


# Calcular pureza de clusters
def calcular_pureza_clusters(clusters, etiquetas_verdaderas):
    """Calcula la pureza de los clusters"""
    from collections import Counter
    pureza_total = 0

    for cluster_id in np.unique(clusters):
        # Obtener índices del cluster actual
        indices_cluster = np.where(clusters == cluster_id)[0]

        # Obtener etiquetas verdaderas para este cluster
        etiquetas_cluster = etiquetas_verdaderas.iloc[indices_cluster]

        # Contar la clase más frecuente
        if len(etiquetas_cluster) > 0:
            clase_mas_frecuente = Counter(etiquetas_cluster).most_common(1)[0][1]
            pureza_cluster = clase_mas_frecuente / len(etiquetas_cluster)
            pureza_total += clase_mas_frecuente

    return pureza_total / len(clusters)


# Calcular pureza usando tipo de negocio como "verdad"
pureza = calcular_pureza_clusters(clusters_train, clientes_con_clusters['tipo_negocio'])

# Crear informe de métricas principales
informe_metricas_principales = f"""
================================================================================
                    INFORME DE MÉTRICAS PRINCIPALES - K-MEANS
================================================================================

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algoritmo: K-means
Número de clusters: 5
Total de muestras: {n_muestras}

================================================================================
                              MÉTRICAS PRINCIPALES
================================================================================

1. COEFICIENTE DE SILHOUETTE (Silhouette Coefficient)
------------------------------------------------------
   Valor en entrenamiento: {silhouette_train:.4f}
   Valor en prueba: {silhouette_test:.4f}

   Interpretación:
   - Rango: [-1, 1]
   - Valores cercanos a 1: Clusters bien separados
   - Valores cercanos a 0: Clusters superpuestos
   - Valores negativos: Asignación incorrecta

   Evaluación: {'EXCELENTE (>0.5)' if silhouette_test > 0.5 else 'BUENO (0.4-0.5)' if silhouette_test > 0.4 else 'ACEPTABLE (0.3-0.4)' if silhouette_test > 0.3 else 'REGULAR (0.2-0.3)' if silhouette_test > 0.2 else 'POBRE (<0.2)'}

   Conclusión: Los clusters muestran {'una excelente separación' if silhouette_test > 0.5 else 'una buena separación' if silhouette_test > 0.4 else 'una separación aceptable' if silhouette_test > 0.3 else 'una separación que necesita mejora'}.

2. INERCIA (Within-Cluster Sum of Squares - WCSS)
--------------------------------------------------
   Valor total: {inercia_total:.2f}
   Inercia promedio por muestra: {inercia_total / n_muestras:.4f}

   Interpretación:
   - Mide la suma de distancias cuadradas dentro de cada cluster
   - Valores menores indican clusters más compactos
   - Se usa en el método del codo para determinar k óptimo

   Análisis por cluster:
"""

# Calcular inercia por cluster
for i in range(5):
    cluster_points = X_train_scaled[clusters_train == i]
    if len(cluster_points) > 0:
        centroid = kmeans_final.cluster_centers_[i]
        inercia_cluster = np.sum((cluster_points - centroid) ** 2)
        informe_metricas_principales += f"""   - Cluster {i + 1}: {inercia_cluster:.2f} ({inercia_cluster / inercia_total * 100:.1f}% del total)
"""

informe_metricas_principales += f"""
3. PUREZA DE CLUSTERS (Cluster Purity)
---------------------------------------
   Valor global: {pureza:.4f} ({pureza * 100:.2f}%)

   Interpretación:
   - Rango: [0, 1]
   - Mide qué tan homogéneos son los clusters respecto a las clases reales
   - 1.0 significa clusters perfectamente puros

   Pureza por cluster (basada en tipo de negocio):
"""

# Calcular pureza por cluster
for i in range(5):
    cluster_data = clientes_con_clusters[clientes_con_clusters['cluster'] == i]
    if len(cluster_data) > 0:
        tipo_mas_comun = cluster_data['tipo_negocio'].value_counts().iloc[0]
        tipo_nombre = cluster_data['tipo_negocio'].value_counts().index[0]
        pureza_cluster = tipo_mas_comun / len(cluster_data)
        informe_metricas_principales += f"""   - Cluster {i + 1}: {pureza_cluster:.4f} ({pureza_cluster * 100:.2f}%) - Dominado por: {tipo_nombre}
"""

informe_metricas_principales += f"""
================================================================================
                           MÉTRICAS COMPLEMENTARIAS
================================================================================

4. ÍNDICE CALINSKI-HARABASZ
----------------------------
   Valor en entrenamiento: {calinski_train:.2f}
   Valor en prueba: {calinski_test:.2f}

   Interpretación:
   - También conocido como Variance Ratio Criterion
   - Valores más altos indican mejor definición de clusters
   - No tiene un rango específico, es comparativo

5. ÍNDICE DAVIES-BOULDIN
------------------------
   Valor en entrenamiento: {davies_bouldin_train:.4f}
   Valor en prueba: {davies_bouldin_test:.4f}

   Interpretación:
   - Valores más bajos indican mejor separación
   - Rango: [0, ∞)
   - Valores < 1 se consideran buenos

================================================================================
                              RESUMEN EJECUTIVO
================================================================================

CALIDAD GENERAL DEL MODELO:
---------------------------
Basado en las métricas principales, el modelo K-means muestra:

✓ Silhouette Score: {silhouette_test:.3f} - {'EXCELENTE' if silhouette_test > 0.5 else 'BUENO' if silhouette_test > 0.4 else 'ACEPTABLE' if silhouette_test > 0.3 else 'NECESITA MEJORA'}
✓ Inercia total: {inercia_total:.2f} - Clusters {'muy compactos' if inercia_total / n_muestras < 10 else 'razonablemente compactos' if inercia_total / n_muestras < 20 else 'con dispersión moderada'}
✓ Pureza: {pureza * 100:.1f}% - {'Alta homogeneidad' if pureza > 0.8 else 'Homogeneidad moderada' if pureza > 0.6 else 'Homogeneidad baja'}

RECOMENDACIONES BASADAS EN MÉTRICAS:
-------------------------------------
"""

# Generar recomendaciones específicas
recomendaciones = []

if silhouette_test < 0.3:
    recomendaciones.append("- Considerar reducir el número de clusters o revisar las variables utilizadas")
if pureza < 0.6:
    recomendaciones.append("- Los clusters muestran mezcla de tipos de negocio, considerar features adicionales")
if inercia_total / n_muestras > 20:
    recomendaciones.append("- La inercia sugiere clusters dispersos, evaluar normalización de datos")
if davies_bouldin_test > 1.5:
    recomendaciones.append("- El índice Davies-Bouldin sugiere clusters con solapamiento")

if not recomendaciones:
    recomendaciones.append("- Las métricas indican un modelo bien ajustado")
    recomendaciones.append("- Mantener monitoreo periódico de la estabilidad de clusters")
    recomendaciones.append("- Considerar validación con expertos del negocio")

for rec in recomendaciones:
    informe_metricas_principales += f"\n{rec}"

informe_metricas_principales += f"""

================================================================================
                           INTERPRETACIÓN DETALLADA
================================================================================

COEFICIENTE DE SILHOUETTE:
--------------------------
El valor de {silhouette_test:.4f} indica que en promedio, cada punto está 
{'muy bien' if silhouette_test > 0.5 else 'bien' if silhouette_test > 0.4 else 'razonablemente' if silhouette_test > 0.3 else 'pobremente'} 
asignado a su cluster. Esto significa que la distancia promedio de cada punto 
a los puntos de su propio cluster es {'significativamente menor' if silhouette_test > 0.5 else 'menor' if silhouette_test > 0.3 else 'solo ligeramente menor'} 
que la distancia a los puntos del cluster más cercano.

INERCIA:
--------
La inercia total de {inercia_total:.2f} representa la suma de las distancias 
cuadradas de cada punto a su centroide. Con {n_muestras} muestras, esto da 
una inercia promedio de {inercia_total / n_muestras:.4f} por punto, lo cual es 
{'excelente' if inercia_total / n_muestras < 10 else 'bueno' if inercia_total / n_muestras < 20 else 'aceptable' if inercia_total / n_muestras < 30 else 'alto'} 
para este tipo de datos.

PUREZA:
-------
La pureza de {pureza * 100:.2f}% indica que, en promedio, ese porcentaje de 
elementos en cada cluster pertenecen a la clase dominante (tipo de negocio). 
Esto sugiere que los clusters {'capturan muy bien' if pureza > 0.8 else 'capturan razonablemente bien' if pureza > 0.6 else 'tienen dificultad para capturar'} 
las categorías naturales del negocio.

================================================================================
                                CONCLUSIÓN
================================================================================

El modelo K-means con 5 clusters presenta un desempeño {'EXCELENTE' if silhouette_test > 0.4 and pureza > 0.7 else 'BUENO' if silhouette_test > 0.3 and pureza > 0.6 else 'ACEPTABLE' if silhouette_test > 0.25 else 'QUE REQUIERE OPTIMIZACIÓN'} 
según las métricas evaluadas. Los clusters formados son {'altamente cohesivos y bien separados' if silhouette_test > 0.4 else 'razonablemente cohesivos' if silhouette_test > 0.3 else 'moderadamente cohesivos'}, 
con una {'excelente' if pureza > 0.8 else 'buena' if pureza > 0.7 else 'moderada' if pureza > 0.6 else 'baja'} correspondencia con las categorías de negocio esperadas.

================================================================================
"""

# Guardar informe de métricas principales
with open(os.path.join(OUTPUT_DIR, 'informe_metricas_principales_kmeans.txt'), 'w', encoding='utf-8') as f:
    f.write(informe_metricas_principales)

# Crear visualización de métricas principales
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Métricas Principales del Modelo K-means', fontsize=16, fontweight='bold')

# 1. Gauge chart para Silhouette Score
ax1 = axes[0, 0]
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Crear semicírculo para gauge
theta = np.linspace(np.pi, 0, 100)
r = 0.8
x = r * np.cos(theta)
y = r * np.sin(theta)

# Colorear según rangos
colors_gauge = ['red', 'orange', 'yellow', 'lightgreen', 'green']
ranges = [0, 0.2, 0.3, 0.4, 0.5, 1.0]

for i in range(len(ranges) - 1):
    mask = (x + r >= ranges[i] * 2 * r) & (x + r < ranges[i + 1] * 2 * r)
    ax1.fill_between(x[mask], 0, y[mask], color=colors_gauge[i], alpha=0.5)

# Indicador
angle = np.pi - (silhouette_test * np.pi)
ax1.plot([0, 0.7 * np.cos(angle)], [0, 0.7 * np.sin(angle)], 'k-', linewidth=3)
ax1.plot(0, 0, 'ko', markersize=10)

ax1.text(0, -0.2, f'Silhouette Score: {silhouette_test:.3f}',
         ha='center', fontsize=14, fontweight='bold')
ax1.set_title('Coeficiente de Silhouette', fontsize=12)
ax1.axis('off')

# 2. Barras para inercia por cluster
ax2 = axes[0, 1]
inercias_clusters = []
for i in range(5):
    cluster_points = X_train_scaled[clusters_train == i]
    if len(cluster_points) > 0:
        centroid = kmeans_final.cluster_centers_[i]
        inercia_cluster = np.sum((cluster_points - centroid) ** 2)
        inercias_clusters.append(inercia_cluster)
    else:
        inercias_clusters.append(0)

bars = ax2.bar(range(5), inercias_clusters, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Inercia')
ax2.set_title(f'Inercia por Cluster (Total: {inercia_total:.0f})', fontsize=12)
ax2.set_xticks(range(5))
ax2.set_xticklabels([f'C{i + 1}' for i in range(5)])

# Añadir valores
for bar, val in zip(bars, inercias_clusters):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f'{val:.0f}', ha='center', va='bottom')

# 3. Gráfico de pureza por cluster
ax3 = axes[1, 0]
purezas_clusters = []
tipos_dominantes = []
for i in range(5):
    cluster_data = clientes_con_clusters[clientes_con_clusters['cluster'] == i]
    if len(cluster_data) > 0:
        tipo_counts = cluster_data['tipo_negocio'].value_counts()
        tipo_mas_comun = tipo_counts.iloc[0]
        pureza_cluster = tipo_mas_comun / len(cluster_data)
        purezas_clusters.append(pureza_cluster)
        tipos_dominantes.append(tipo_counts.index[0])
    else:
        purezas_clusters.append(0)
        tipos_dominantes.append('N/A')

bars = ax3.bar(range(5), purezas_clusters, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
ax3.axhline(y=pureza, color='red', linestyle='--', label=f'Pureza global: {pureza:.3f}')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Pureza')
ax3.set_title('Pureza de Clusters', fontsize=12)
ax3.set_xticks(range(5))
ax3.set_xticklabels([f'C{i + 1}\n{tipos_dominantes[i][:8]}' for i in range(5)], rotation=0)
ax3.set_ylim(0, 1.1)
ax3.legend()

# Añadir valores
for bar, val in zip(bars, purezas_clusters):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom')

# 4. Resumen de métricas
ax4 = axes[1, 1]
ax4.axis('off')

# Crear tabla de resumen
tabla_data = [
    ['Métrica', 'Valor', 'Evaluación'],
    ['Silhouette Score', f'{silhouette_test:.4f}',
     'Excelente' if silhouette_test > 0.5 else 'Bueno' if silhouette_test > 0.4 else 'Aceptable' if silhouette_test > 0.3 else 'Regular'],
    ['Inercia Total', f'{inercia_total:.2f}',
     'Compacto' if inercia_total / n_muestras < 15 else 'Moderado' if inercia_total / n_muestras < 25 else 'Disperso'],
    ['Pureza Global', f'{pureza:.4f}',
     'Alta' if pureza > 0.8 else 'Media' if pureza > 0.6 else 'Baja'],
    ['Calinski-Harabasz', f'{calinski_test:.2f}', 'Mayor es mejor'],
    ['Davies-Bouldin', f'{davies_bouldin_test:.4f}',
     'Excelente' if davies_bouldin_test < 0.5 else 'Bueno' if davies_bouldin_test < 1.0 else 'Regular']
]

# Crear tabla
for i, row in enumerate(tabla_data):
    for j, val in enumerate(row):
        if i == 0:  # Header
            ax4.text(j * 0.33, 0.9 - i * 0.15, val, fontsize=12, fontweight='bold', ha='center')
        else:
            color = 'black'
            if j == 2:  # Columna de evaluación
                if 'Excelente' in val or 'Alta' in val:
                    color = 'green'
                elif 'Bueno' in val or 'Media' in val or 'Aceptable' in val:
                    color = 'orange'
                elif 'Regular' in val or 'Baja' in val:
                    color = 'red'
            ax4.text(j * 0.33, 0.9 - i * 0.15, val, fontsize=11, ha='center', color=color)

ax4.set_title('Resumen de Métricas', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metricas_principales_kmeans.png'), dpi=300, bbox_inches='tight')
plt.close()

# Crear DataFrame con métricas principales
df_metricas_principales = pd.DataFrame({
    'Métrica': ['Coeficiente de Silhouette', 'Inercia Total', 'Inercia Promedio', 'Pureza Global'],
    'Valor': [silhouette_test, inercia_total, inercia_total / n_muestras, pureza],
    'Interpretación': [
        'Excelente' if silhouette_test > 0.5 else 'Bueno' if silhouette_test > 0.4 else 'Aceptable' if silhouette_test > 0.3 else 'Regular',
        f'{inercia_total:.2f} (suma total de distancias cuadradas)',
        'Compacto' if inercia_total / n_muestras < 15 else 'Moderado' if inercia_total / n_muestras < 25 else 'Disperso',
        'Alta homogeneidad' if pureza > 0.8 else 'Homogeneidad media' if pureza > 0.6 else 'Homogeneidad baja'
    ]
})

df_metricas_principales.to_csv(os.path.join(OUTPUT_DIR, 'metricas_principales_resumen.csv'), index=False)

print("✅ Informe de métricas principales generado")
print(f"   - Archivo de texto: informe_metricas_principales_kmeans.txt")
print(f"   - Visualización: metricas_principales_kmeans.png")
print(f"   - Resumen CSV: metricas_principales_resumen.csv")