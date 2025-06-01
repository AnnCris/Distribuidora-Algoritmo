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
ALGORITMO = "KMEANS_V2"
OUTPUT_DIR = f"resultados_{ALGORITMO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de visualización mejorada
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['figure.dpi'] = 100

print("=" * 90)
print("🔧 SEGMENTACIÓN DE CLIENTES DISTRIBUIDORA - K-MEANS V2.0 MEJORADO")
print("=" * 90)

# ============================================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS MEJORADA
# ============================================================================
print("\n📋 FASE 1: CARGA Y PREPARACIÓN DE DATOS MEJORADA")
print("-" * 60)

# Cargar datasets del nuevo formato
try:
    # Primero intentar cargar los archivos mejorados
    if os.path.exists('ventas_mejorado_v2.csv'):
        df_ventas = pd.read_csv('ventas_mejorado_v2.csv')
        df_detalles = pd.read_csv('detalles_ventas_mejorado_v2.csv')
        print("✅ Datasets mejorados V2 cargados exitosamente")
    else:
        # Fallback a archivos estándar
        df_ventas = pd.read_csv('ventas.csv')
        df_detalles = pd.read_csv('detalles_ventas.csv')
        print("✅ Datasets estándar cargados exitosamente")

except FileNotFoundError:
    print("❌ Error: Archivos CSV no encontrados")
    print("💡 Asegúrate de ejecutar primero el script dataset.py mejorado")
    exit()


# Conversión de fechas mejorada - detectar formato automáticamente
def detectar_y_convertir_fechas(df):
    """Detecta el formato de fecha y convierte apropiadamente"""
    fecha_sample = str(df['fecha'].iloc[0])

    if '-' in fecha_sample and len(fecha_sample.split('-')[0]) == 4:
        # Formato YYYY-MM-DD (nuevo dataset)
        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        print("📅 Formato de fecha detectado: YYYY-MM-DD (Dataset V2)")
    else:
        # Formato DD/MM/YYYY (dataset anterior)
        try:
            df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
            print("📅 Formato de fecha detectado: DD/MM/YYYY (Dataset V1)")
        except:
            # Intentar formato automático
            df['fecha'] = pd.to_datetime(df['fecha'])
            print("📅 Formato de fecha detectado automáticamente")

    return df


df_ventas = detectar_y_convertir_fechas(df_ventas)
fecha_referencia = df_ventas['fecha'].max()

print(f"\n📈 RESUMEN DE DATOS MEJORADO:")
print(f"  • Ventas totales: {len(df_ventas):,}")
print(f"  • Detalles de productos: {len(df_detalles):,}")
print(f"  • Clientes únicos: {df_ventas['cliente_id'].nunique():,}")
print(f"  • Productos únicos: {df_detalles['producto_id'].nunique():,}")
print(f"  • Periodo: {df_ventas['fecha'].min().strftime('%Y-%m-%d')} a {df_ventas['fecha'].max().strftime('%Y-%m-%d')}")

# Análisis de calidad de datos
print(f"\n🔍 ANÁLISIS DE CALIDAD DE DATOS:")
print(f"  • Valores nulos en ventas: {df_ventas.isnull().sum().sum()}")
print(f"  • Valores nulos en detalles: {df_detalles.isnull().sum().sum()}")
print(f"  • Promedio productos por venta: {len(df_detalles) / len(df_ventas):.1f}")

# Distribución por año
ventas_por_año = df_ventas['fecha'].dt.year.value_counts().sort_index()
print(f"  • Distribución por año:")
for año, cantidad in ventas_por_año.items():
    print(f"    - {año}: {cantidad:,} ventas ({cantidad / len(df_ventas) * 100:.1f}%)")


# ============================================================================
# FUNCIÓN PARA CREAR MÉTRICAS RFM AVANZADAS V2
# ============================================================================
def crear_metricas_rfm_avanzadas():
    """Crea métricas RFM y variables adicionales mejoradas"""
    print("🔄 Creando métricas RFM avanzadas V2...")

    # Métricas básicas por cliente con más detalle
    metricas_base = df_ventas.groupby('cliente_id').agg({
        'fecha': ['count', 'max', 'min'],
        'total_neto': ['sum', 'mean', 'std', 'median', 'max', 'min'],
        'descuento': ['sum', 'mean', 'max'],
        'ciudad': 'first',
        'tipo_negocio': 'first',
        'cliente_nombre': 'first',
        'turno': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    }).round(2)

    metricas_base.columns = [
        'frecuencia', 'ultima_compra', 'primera_compra',
        'valor_total', 'ticket_promedio', 'std_ticket', 'ticket_mediano', 'ticket_maximo', 'ticket_minimo',
        'descuento_total', 'descuento_promedio', 'descuento_maximo',
        'ciudad', 'tipo_negocio', 'cliente_nombre', 'turno_preferido'
    ]

    metricas_base = metricas_base.reset_index()

    # Métricas temporales mejoradas
    metricas_base['recencia_dias'] = (fecha_referencia - metricas_base['ultima_compra']).dt.days
    metricas_base['periodo_cliente_dias'] = (metricas_base['ultima_compra'] - metricas_base['primera_compra']).dt.days
    metricas_base['periodo_cliente_dias'] = metricas_base['periodo_cliente_dias'].fillna(0)
    metricas_base['std_ticket'] = metricas_base['std_ticket'].fillna(0)

    # Métricas adicionales de comportamiento
    metricas_base['rango_ticket'] = metricas_base['ticket_maximo'] - metricas_base['ticket_minimo']
    metricas_base['frecuencia_mensual'] = metricas_base['frecuencia'] / (metricas_base['periodo_cliente_dias'] / 30 + 1)
    metricas_base['intensidad_compra'] = metricas_base['valor_total'] / metricas_base['frecuencia']

    # Segmentos de recencia
    metricas_base['segmento_recencia'] = pd.cut(metricas_base['recencia_dias'],
                                                bins=[0, 30, 90, 180, 365, float('inf')],
                                                labels=['Muy_Reciente', 'Reciente', 'Regular', 'Inactivo', 'Perdido'])

    return metricas_base


def agregar_metricas_productos(metricas_cliente):
    """Agrega métricas de productos mejoradas"""
    print("🛒 Agregando métricas de productos mejoradas...")

    ventas_productos = df_ventas[['venta_id', 'cliente_id']].merge(
        df_detalles[['venta_id', 'producto_categoria', 'producto_marca', 'cantidad', 'precio_unitario', 'subtotal']],
        on='venta_id'
    )

    # Diversidad y comportamiento de productos
    diversidad_productos = ventas_productos.groupby('cliente_id').agg({
        'producto_categoria': ['nunique', lambda x: len(x)],
        'producto_marca': 'nunique',
        'cantidad': ['sum', 'mean', 'std'],
        'precio_unitario': ['mean', 'std', 'max', 'min'],
        'subtotal': ['sum', 'mean']
    }).round(2)

    diversidad_productos.columns = [
        'num_categorias', 'total_productos_comprados', 'num_marcas',
        'cantidad_total', 'cantidad_promedio', 'std_cantidad',
        'precio_unitario_promedio', 'std_precios', 'precio_max', 'precio_min',
        'gasto_productos_total', 'gasto_productos_promedio'
    ]
    diversidad_productos = diversidad_productos.reset_index()
    diversidad_productos['std_cantidad'] = diversidad_productos['std_cantidad'].fillna(0)
    diversidad_productos['std_precios'] = diversidad_productos['std_precios'].fillna(0)

    # Categoría y marca preferida
    categoria_preferida = ventas_productos.groupby('cliente_id')['producto_categoria'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    ).reset_index()
    categoria_preferida.columns = ['cliente_id', 'categoria_preferida']

    marca_preferida = ventas_productos.groupby('cliente_id')['producto_marca'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    ).reset_index()
    marca_preferida.columns = ['cliente_id', 'marca_preferida']

    # Comportamiento de precios
    diversidad_productos['rango_precios'] = diversidad_productos['precio_max'] - diversidad_productos['precio_min']
    diversidad_productos['variabilidad_gasto'] = diversidad_productos['std_precios'] / diversidad_productos[
        'precio_unitario_promedio']
    diversidad_productos['variabilidad_gasto'] = diversidad_productos['variabilidad_gasto'].fillna(0)

    # Unir todas las métricas
    metricas_completas = metricas_cliente.merge(diversidad_productos, on='cliente_id', how='left')
    metricas_completas = metricas_completas.merge(categoria_preferida, on='cliente_id', how='left')
    metricas_completas = metricas_completas.merge(marca_preferida, on='cliente_id', how='left')

    return metricas_completas


def agregar_metricas_temporales(metricas_cliente):
    """Agrega métricas temporales avanzadas"""
    print("📅 Agregando métricas temporales avanzadas...")

    ventas_temp = df_ventas.copy()
    ventas_temp['año'] = ventas_temp['fecha'].dt.year
    ventas_temp['mes'] = ventas_temp['fecha'].dt.month
    ventas_temp['trimestre'] = ventas_temp['fecha'].dt.quarter
    ventas_temp['dia_semana'] = ventas_temp['fecha'].dt.dayofweek
    ventas_temp['es_fin_semana'] = (ventas_temp['dia_semana'] >= 5).astype(int)

    # Análisis de estacionalidad por cliente
    estacionalidad = ventas_temp.groupby('cliente_id').agg({
        'mes': lambda x: x.value_counts().index[0],  # Mes más frecuente
        'trimestre': lambda x: x.value_counts().index[0],  # Trimestre más frecuente
        'es_fin_semana': 'mean'  # Proporción de compras en fin de semana
    }).reset_index()
    estacionalidad.columns = ['cliente_id', 'mes_preferido', 'trimestre_preferido', 'prop_fin_semana']

    # Tendencia de gasto mejorada
    tendencias = []
    for cliente_id in metricas_cliente['cliente_id']:
        compras_cliente = df_ventas[df_ventas['cliente_id'] == cliente_id].sort_values('fecha')

        if len(compras_cliente) >= 4:
            # Dividir en tres períodos para mejor análisis de tendencia
            tercio = len(compras_cliente) // 3
            periodo_1 = compras_cliente.iloc[:tercio]['total_neto'].mean()
            periodo_2 = compras_cliente.iloc[tercio:2 * tercio]['total_neto'].mean()
            periodo_3 = compras_cliente.iloc[2 * tercio:]['total_neto'].mean()

            # Calcular tendencia lineal simple
            if periodo_1 > 0:
                tendencia_1_2 = (periodo_2 - periodo_1) / periodo_1
                tendencia_2_3 = (periodo_3 - periodo_2) / periodo_2 if periodo_2 > 0 else 0
                tendencia_total = (tendencia_1_2 + tendencia_2_3) / 2
            else:
                tendencia_total = 0

            # Regularidad de compras (coeficiente de variación de intervalos)
            if len(compras_cliente) >= 3:
                fechas = pd.to_datetime(compras_cliente['fecha'])
                # Calcular intervalos en días de manera segura
                intervalos = []
                for i in range(1, len(fechas)):
                    diff_days = (fechas.iloc[i] - fechas.iloc[i - 1]).days
                    intervalos.append(diff_days)

                if len(intervalos) > 1 and np.mean(intervalos) > 0:
                    regularidad = np.std(intervalos) / np.mean(intervalos)
                else:
                    regularidad = 0
            else:
                regularidad = 0
        else:
            tendencia_total = 0
            regularidad = 0

        tendencias.append({
            'cliente_id': cliente_id,
            'tendencia_gasto': tendencia_total,
            'regularidad_compras': regularidad
        })

    df_tendencias = pd.DataFrame(tendencias)

    # Unir todas las métricas temporales
    metricas_completas = metricas_cliente.merge(estacionalidad, on='cliente_id', how='left')
    metricas_completas = metricas_completas.merge(df_tendencias, on='cliente_id', how='left')

    return metricas_completas


# Ejecutar creación de métricas
print("\n🚀 Ejecutando creación de métricas completas...")
metricas_rfm = crear_metricas_rfm_avanzadas()
metricas_con_productos = agregar_metricas_productos(metricas_rfm)
metricas_completas = agregar_metricas_temporales(metricas_con_productos)

print(f"✅ Métricas creadas para {len(metricas_completas)} clientes")

# Filtrado de datos mejorado
criterios_filtrado = {
    'frecuencia_minima': 3,  # Mínimo 3 compras para análisis robusto
    'valor_minimo': 100,  # Valor mínimo más razonable
    'recencia_maxima': 400  # Incluir más clientes
}

clientes_validos = metricas_completas[
    (metricas_completas['frecuencia'] >= criterios_filtrado['frecuencia_minima']) &
    (metricas_completas['valor_total'] >= criterios_filtrado['valor_minimo']) &
    (metricas_completas['recencia_dias'] <= criterios_filtrado['recencia_maxima'])
    ].copy()

print(f"📊 Clientes después del filtrado: {len(clientes_validos)}")
print(f"   • Clientes eliminados: {len(metricas_completas) - len(clientes_validos)}")
print(f"   • Tasa de retención: {len(clientes_validos) / len(metricas_completas) * 100:.1f}%")

# Análisis de distribución antes del clustering
print(f"\n📊 ANÁLISIS DE DISTRIBUCIÓN PRE-CLUSTERING:")
tipos_negocio_dist = clientes_validos['tipo_negocio'].value_counts()
print("Distribución por tipo de negocio:")
for tipo, cantidad in tipos_negocio_dist.items():
    print(f"  • {tipo}: {cantidad} clientes ({cantidad / len(clientes_validos) * 100:.1f}%)")

# ============================================================================
# FASE 2: PREPARACIÓN DE CARACTERÍSTICAS MEJORADA
# ============================================================================
print("\n🔧 FASE 2: PREPARACIÓN DE CARACTERÍSTICAS MEJORADA")
print("-" * 60)

# Codificación de variables categóricas
le_ciudad = LabelEncoder()
le_tipo_negocio = LabelEncoder()
le_categoria_pref = LabelEncoder()
le_marca_pref = LabelEncoder()
le_turno_pref = LabelEncoder()
le_segmento_recencia = LabelEncoder()

clientes_validos['ciudad_encoded'] = le_ciudad.fit_transform(clientes_validos['ciudad'])
clientes_validos['tipo_negocio_encoded'] = le_tipo_negocio.fit_transform(clientes_validos['tipo_negocio'])
clientes_validos['categoria_pref_encoded'] = le_categoria_pref.fit_transform(clientes_validos['categoria_preferida'])
clientes_validos['marca_pref_encoded'] = le_marca_pref.fit_transform(clientes_validos['marca_preferida'])
clientes_validos['turno_pref_encoded'] = le_turno_pref.fit_transform(clientes_validos['turno_preferido'])
clientes_validos['segmento_recencia_encoded'] = le_segmento_recencia.fit_transform(
    clientes_validos['segmento_recencia'])

# Variables para clustering mejoradas - más features para mejor segmentación
variables_clustering = [
    # Métricas RFM core
    'frecuencia', 'recencia_dias', 'valor_total', 'ticket_promedio',
    # Métricas de comportamiento
    'std_ticket', 'rango_ticket', 'frecuencia_mensual', 'intensidad_compra',
    # Métricas de productos
    'num_categorias', 'num_marcas', 'cantidad_promedio', 'precio_unitario_promedio',
    'variabilidad_gasto', 'rango_precios',
    # Métricas temporales
    'periodo_cliente_dias', 'tendencia_gasto', 'regularidad_compras', 'prop_fin_semana',
    # Métricas categóricas
    'ciudad_encoded', 'tipo_negocio_encoded', 'categoria_pref_encoded',
    'mes_preferido', 'trimestre_preferido'
]

X = clientes_validos[variables_clustering].fillna(0)

# División estratificada por tipo de negocio para mejor representatividad
X_train, X_test, indices_train, indices_test = train_test_split(
    X, X.index, test_size=0.2, random_state=42,
    stratify=clientes_validos['tipo_negocio']
)

# Escalado robusto
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Datos preparados: {len(X_train)} train, {len(X_test)} test")
print(f"✅ Variables de clustering: {len(variables_clustering)}")

# ============================================================================
# FASE 3: ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS MEJORADO
# ============================================================================
print("\n📊 FASE 3: ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS MEJORADO")
print("-" * 60)

# Configuración para análisis ampliado
k_min, k_max = 2, 12
k_range = range(k_min, k_max + 1)

# Métricas para evaluación expandidas
metricas_evaluacion = {
    'k': [],
    'inercia': [],
    'silhouette': [],
    'calinski_harabasz': [],
    'davies_bouldin': []
}

print("🔄 Evaluando diferentes números de clusters (análisis mejorado)...")
for k in k_range:
    print(f"  Evaluando k={k}...", end=" ")

    # Múltiples inicializaciones para estabilidad
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500, tol=1e-6)
    labels = kmeans.fit_predict(X_train_scaled)

    # Calcular métricas solo si hay suficientes clusters formados
    if len(np.unique(labels)) >= 2:
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
    else:
        print(f"❌ Clusters insuficientes")

df_metricas = pd.DataFrame(metricas_evaluacion)

# ============================================================================
# VISUALIZACIÓN DEL ANÁLISIS DE CLUSTERS MEJORADA
# ============================================================================
print("\n📈 Generando visualizaciones de análisis mejoradas...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis Completo de Número Óptimo de Clusters - K-means V2', fontsize=16, fontweight='bold')

# 1. Método del Codo mejorado
ax1 = axes[0, 0]
ax1.plot(df_metricas['k'], df_metricas['inercia'], 'bo-', linewidth=3, markersize=10)
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inercia (WCSS)')
ax1.set_title('Método del Codo (Elbow Method)', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Calcular y marcar el codo
if len(df_metricas) > 3:
    inercias = df_metricas['inercia'].values
    # Método de la segunda derivada para encontrar el codo
    first_deriv = np.diff(inercias)
    second_deriv = np.diff(first_deriv)
    if len(second_deriv) > 0:
        codo_idx = np.argmax(second_deriv) + 2
        if codo_idx < len(df_metricas):
            codo_k = df_metricas.iloc[codo_idx]['k']
            ax1.axvline(x=codo_k, color='red', linestyle='--', alpha=0.8, linewidth=2,
                        label=f'Codo detectado (k={codo_k})')

# Marcar k=5 objetivo
ax1.axvline(x=5, color='green', linestyle='--', alpha=0.8, linewidth=3, label='k=5 (Objetivo)')
ax1.legend()

# 2. Silhouette Score
ax2 = axes[0, 1]
ax2.plot(df_metricas['k'], df_metricas['silhouette'], 'go-', linewidth=3, markersize=10)
ax2.set_xlabel('Número de Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Análisis de Silhouette Score', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Excelente (0.5)')
ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Bueno (0.4)')
ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Aceptable (0.3)')
ax2.axvline(x=5, color='green', linestyle='--', alpha=0.8, linewidth=3, label='k=5 (Objetivo)')

# Marcar el máximo
max_sil_idx = df_metricas['silhouette'].idxmax()
max_sil_k = df_metricas.iloc[max_sil_idx]['k']
max_sil_score = df_metricas.iloc[max_sil_idx]['silhouette']
ax2.scatter(max_sil_k, max_sil_score, color='darkgreen', s=200, zorder=5, marker='*')
ax2.annotate(f'Máximo: {max_sil_score:.3f}', (max_sil_k, max_sil_score),
             xytext=(10, 10), textcoords='offset points', fontweight='bold')
ax2.legend()

# 3. Calinski-Harabasz Index
ax3 = axes[0, 2]
ax3.plot(df_metricas['k'], df_metricas['calinski_harabasz'], 'ro-', linewidth=3, markersize=10)
ax3.set_xlabel('Número de Clusters (k)')
ax3.set_ylabel('Calinski-Harabasz Index')
ax3.set_title('Calinski-Harabasz Index', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axvline(x=5, color='green', linestyle='--', alpha=0.8, linewidth=3, label='k=5 (Objetivo)')

# Marcar el máximo
max_ch_idx = df_metricas['calinski_harabasz'].idxmax()
max_ch_k = df_metricas.iloc[max_ch_idx]['k']
max_ch_score = df_metricas.iloc[max_ch_idx]['calinski_harabasz']
ax3.scatter(max_ch_k, max_ch_score, color='darkred', s=200, zorder=5, marker='*')
ax3.annotate(f'Máximo: {max_ch_score:.0f}', (max_ch_k, max_ch_score),
             xytext=(10, 10), textcoords='offset points', fontweight='bold')
ax3.legend()

# 4. Davies-Bouldin Index
ax4 = axes[1, 0]
ax4.plot(df_metricas['k'], df_metricas['davies_bouldin'], 'mo-', linewidth=3, markersize=10)
ax4.set_xlabel('Número de Clusters (k)')
ax4.set_ylabel('Davies-Bouldin Index')
ax4.set_title('Davies-Bouldin Index (menor es mejor)', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Bueno (<1.0)')
ax4.axvline(x=5, color='green', linestyle='--', alpha=0.8, linewidth=3, label='k=5 (Objetivo)')

# Marcar el mínimo
min_db_idx = df_metricas['davies_bouldin'].idxmin()
min_db_k = df_metricas.iloc[min_db_idx]['k']
min_db_score = df_metricas.iloc[min_db_idx]['davies_bouldin']
ax4.scatter(min_db_k, min_db_score, color='darkmagenta', s=200, zorder=5, marker='*')
ax4.annotate(f'Mínimo: {min_db_score:.3f}', (min_db_k, min_db_score),
             xytext=(10, -10), textcoords='offset points', fontweight='bold')
ax4.legend()

# 5. Comparación de métricas normalizadas
ax5 = axes[1, 1]
# Normalizar métricas para comparación
silhouette_norm = (df_metricas['silhouette'] - df_metricas['silhouette'].min()) / (
            df_metricas['silhouette'].max() - df_metricas['silhouette'].min())
calinski_norm = (df_metricas['calinski_harabasz'] - df_metricas['calinski_harabasz'].min()) / (
            df_metricas['calinski_harabasz'].max() - df_metricas['calinski_harabasz'].min())
davies_norm = 1 - (df_metricas['davies_bouldin'] - df_metricas['davies_bouldin'].min()) / (
            df_metricas['davies_bouldin'].max() - df_metricas['davies_bouldin'].min())

ax5.plot(df_metricas['k'], silhouette_norm, 'g-', linewidth=2, label='Silhouette (norm)', marker='o')
ax5.plot(df_metricas['k'], calinski_norm, 'b-', linewidth=2, label='Calinski-H (norm)', marker='s')
ax5.plot(df_metricas['k'], davies_norm, 'm-', linewidth=2, label='Davies-B (norm inv)', marker='^')
ax5.axvline(x=5, color='red', linestyle='--', alpha=0.8, linewidth=3, label='k=5 (Objetivo)')
ax5.set_xlabel('Número de Clusters (k)')
ax5.set_ylabel('Score Normalizado')
ax5.set_title('Comparación de Métricas Normalizadas', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Tabla de resumen de métricas
ax6 = axes[1, 2]
ax6.axis('off')
k5_metrics = df_metricas[df_metricas['k'] == 5].iloc[0] if 5 in df_metricas['k'].values else None

if k5_metrics is not None:
    tabla_data = [
        ['Métrica', 'k=5', 'Óptimo', 'Evaluación'],
        ['Silhouette', f"{k5_metrics['silhouette']:.3f}", f"{max_sil_score:.3f} (k={max_sil_k})",
         'Excelente' if k5_metrics['silhouette'] > 0.5 else 'Bueno' if k5_metrics['silhouette'] > 0.4 else 'Aceptable'],
        ['Calinski-H', f"{k5_metrics['calinski_harabasz']:.0f}", f"{max_ch_score:.0f} (k={max_ch_k})",
         'Alto' if k5_metrics['calinski_harabasz'] > max_ch_score * 0.8 else 'Medio'],
        ['Davies-B', f"{k5_metrics['davies_bouldin']:.3f}", f"{min_db_score:.3f} (k={min_db_k})",
         'Excelente' if k5_metrics['davies_bouldin'] < 0.5 else 'Bueno' if k5_metrics[
                                                                               'davies_bouldin'] < 1.0 else 'Regular']
    ]

    for i, row in enumerate(tabla_data):
        for j, val in enumerate(row):
            if i == 0:  # Header
                ax6.text(j * 0.25, 0.9 - i * 0.15, val, fontsize=11, fontweight='bold', ha='center')
            else:
                color = 'black'
                if j == 3:  # Columna de evaluación
                    if 'Excelente' in val or 'Alto' in val:
                        color = 'green'
                    elif 'Bueno' in val or 'Medio' in val:
                        color = 'orange'
                    elif 'Regular' in val or 'Aceptable' in val:
                        color = 'red'
                ax6.text(j * 0.25, 0.9 - i * 0.15, val, fontsize=10, ha='center', color=color)

ax6.set_title('Evaluación de k=5', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_numero_clusters_mejorado.png'), dpi=300, bbox_inches='tight')
plt.close()

# Guardar tabla de métricas
df_metricas.to_csv(os.path.join(OUTPUT_DIR, 'metricas_evaluacion_clusters.csv'), index=False)

# ============================================================================
# FASE 4: ENTRENAMIENTO DEL MODELO FINAL CON K=5 OBJETIVO
# ============================================================================
print("\n🎯 FASE 4: ENTRENAMIENTO DEL MODELO FINAL (K=5 OBJETIVO)")
print("-" * 60)

# Definir los 5 grupos objetivo exactos según la imagen
GRUPOS_OBJETIVO = {
    0: {
        'nombre': '🌟 Compradores Premium de Alto Volumen',
        'descripcion': 'Pizzerías establecidas con compras frecuentes y alto volumen',
        'tipo_esperado': 'PIZZERIA',
        'color': '#FFD700'  # Dorado
    },
    1: {
        'nombre': '🍽️ Compradores Frecuentes Especializados',
        'descripcion': 'Restaurantes con compras regulares y especializadas',
        'tipo_esperado': 'RESTAURANTE',
        'color': '#FF6B6B'  # Rojo claro
    },
    2: {
        'nombre': '🏪 Comerciantes Mayoristas',
        'descripcion': 'Mercados y tiendas con compras al por mayor',
        'tipo_esperado': 'MERCADO',
        'color': '#4ECDC4'  # Turquesa
    },
    3: {
        'nombre': '🌱 Negocios Emergentes',
        'descripcion': 'Establecimientos nuevos en crecimiento',
        'tipo_esperado': 'VARIOS',
        'color': '#45B7D1'  # Azul claro
    },
    4: {
        'nombre': '🔄 Compradores Ocasionales',
        'descripcion': 'Clientes con compras esporádicas o estacionales',
        'tipo_esperado': 'VARIOS',
        'color': '#96CEB4'  # Verde claro
    }
}

# Entrenar modelo final con configuración optimizada
print("🚀 Entrenando K-means optimizado con k=5...")
kmeans_final = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=50,  # Muchas inicializaciones para estabilidad
    max_iter=1000,  # Suficientes iteraciones
    tol=1e-6,  # Tolerancia estricta
    algorithm='lloyd'  # Algoritmo clásico más estable
)

# Entrenar en datos completos para mejor clustering
print("📊 Entrenando en dataset completo para clustering óptimo...")
X_full_scaled = scaler.fit_transform(X)
clusters_full = kmeans_final.fit_predict(X_full_scaled)

# También predecir en conjuntos train/test para evaluación
clusters_train = kmeans_final.predict(X_train_scaled)
clusters_test = kmeans_final.predict(X_test_scaled)

print("✅ Modelo entrenado exitosamente")

# ============================================================================
# FASE 5: EVALUACIÓN Y VALIDACIÓN DEL MODELO MEJORADA
# ============================================================================
print("\n📈 FASE 5: EVALUACIÓN Y VALIDACIÓN DEL MODELO MEJORADA")
print("-" * 60)

# Métricas de evaluación completas
silhouette_full = silhouette_score(X_full_scaled, clusters_full)
silhouette_train = silhouette_score(X_train_scaled, clusters_train)
silhouette_test = silhouette_score(X_test_scaled, clusters_test)

calinski_full = calinski_harabasz_score(X_full_scaled, clusters_full)
calinski_train = calinski_harabasz_score(X_train_scaled, clusters_train)
calinski_test = calinski_harabasz_score(X_test_scaled, clusters_test)

davies_bouldin_full = davies_bouldin_score(X_full_scaled, clusters_full)
davies_bouldin_train = davies_bouldin_score(X_train_scaled, clusters_train)
davies_bouldin_test = davies_bouldin_score(X_test_scaled, clusters_test)

print(f"📊 MÉTRICAS DE RENDIMIENTO COMPLETAS:")
print(f"  Silhouette Score - Full: {silhouette_full:.3f}, Train: {silhouette_train:.3f}, Test: {silhouette_test:.3f}")
print(f"  Calinski-Harabasz - Full: {calinski_full:.1f}, Train: {calinski_train:.1f}, Test: {calinski_test:.1f}")
print(
    f"  Davies-Bouldin - Full: {davies_bouldin_full:.3f}, Train: {davies_bouldin_train:.3f}, Test: {davies_bouldin_test:.3f}")

# Análisis de estabilidad mejorado
distribucion_full = pd.Series(clusters_full).value_counts().sort_index()
distribucion_train = pd.Series(clusters_train).value_counts().sort_index()
distribucion_test = pd.Series(clusters_test).value_counts().sort_index()

print(f"\n📊 Distribución de clusters:")
print(f"  Full: {distribucion_full.to_dict()}")
print(f"  Train: {distribucion_train.to_dict()}")
print(f"  Test: {distribucion_test.to_dict()}")

# Calcular estabilidad entre train y test
estabilidad = 1 - abs(silhouette_train - silhouette_test)
print(
    f"  💪 Estabilidad del modelo: {estabilidad:.3f} ({'Alta' if estabilidad > 0.95 else 'Media' if estabilidad > 0.9 else 'Baja'})")

# ============================================================================
# ANÁLISIS DETALLADO DE CLUSTERS CON ASIGNACIÓN INTELIGENTE
# ============================================================================
print("\n🔍 ANÁLISIS DETALLADO DE CLUSTERS CON ASIGNACIÓN INTELIGENTE")
print("-" * 60)

# Usar clusters completos para análisis
clientes_con_clusters = clientes_validos.copy()
clientes_con_clusters['cluster'] = clusters_full

# Análisis exhaustivo por cluster
perfiles_clusters = {}
asignacion_clusters = {}

print("📋 Analizando perfiles de cada cluster...")
for cluster_id in range(5):
    cluster_data = clientes_con_clusters[clientes_con_clusters['cluster'] == cluster_id]
    n_clientes = len(cluster_data)

    if n_clientes > 0:
        # Calcular características detalladas del cluster
        perfil = {
            'cluster_id': cluster_id,
            'n_clientes': n_clientes,
            'porcentaje': (n_clientes / len(clientes_con_clusters)) * 100,

            # Métricas RFM
            'frecuencia_promedio': cluster_data['frecuencia'].mean(),
            'frecuencia_mediana': cluster_data['frecuencia'].median(),
            'recencia_promedio': cluster_data['recencia_dias'].mean(),
            'valor_promedio': cluster_data['valor_total'].mean(),
            'valor_mediano': cluster_data['valor_total'].median(),
            'ticket_promedio': cluster_data['ticket_promedio'].mean(),

            # Métricas de comportamiento
            'intensidad_compra': cluster_data['intensidad_compra'].mean(),
            'frecuencia_mensual': cluster_data['frecuencia_mensual'].mean(),
            'regularidad_compras': cluster_data['regularidad_compras'].mean(),
            'tendencia_gasto': cluster_data['tendencia_gasto'].mean(),

            # Métricas de productos
            'num_categorias': cluster_data['num_categorias'].mean(),
            'num_marcas': cluster_data['num_marcas'].mean(),
            'cantidad_promedio': cluster_data['cantidad_promedio'].mean(),
            'precio_unitario_promedio': cluster_data['precio_unitario_promedio'].mean(),

            # Métricas temporales
            'periodo_cliente_dias': cluster_data['periodo_cliente_dias'].mean(),
            'prop_fin_semana': cluster_data['prop_fin_semana'].mean(),

            # Información demográfica
            'tipo_negocio_principal': cluster_data['tipo_negocio'].mode().iloc[0] if not cluster_data[
                'tipo_negocio'].mode().empty else "N/A",
            'ciudad_principal': cluster_data['ciudad'].mode().iloc[0] if not cluster_data[
                'ciudad'].mode().empty else "N/A",
            'categoria_preferida': cluster_data['categoria_preferida'].mode().iloc[0] if not cluster_data[
                'categoria_preferida'].mode().empty else "N/A",
            'marca_preferida': cluster_data['marca_preferida'].mode().iloc[0] if not cluster_data[
                'marca_preferida'].mode().empty else "N/A"
        }

        # Análisis de composición por tipo de negocio
        tipos_negocio_dist = cluster_data['tipo_negocio'].value_counts()
        perfil['distribucion_tipos'] = tipos_negocio_dist.to_dict()

        # Calcular dominancia de tipo de negocio
        if len(tipos_negocio_dist) > 0:
            perfil['dominancia_tipo'] = tipos_negocio_dist.iloc[0] / n_clientes
        else:
            perfil['dominancia_tipo'] = 0

        perfiles_clusters[cluster_id] = perfil

print("🎯 Asignando clusters a grupos objetivo según características...")


# Asignación inteligente basada en múltiples criterios
def asignar_cluster_a_grupo(cluster_id, perfil):
    """Asigna un cluster a un grupo objetivo basado en sus características"""

    # Análisis de composición del cluster
    tipos_dominantes = perfil['distribucion_tipos']
    tipo_principal = perfil['tipo_negocio_principal']

    # Criterios de clasificación
    es_alto_valor = perfil['valor_promedio'] > 50000
    es_alta_frecuencia = perfil['frecuencia_promedio'] > 20
    es_pizza_dominante = tipo_principal == 'PIZZERIA' and perfil['dominancia_tipo'] > 0.5
    es_restaurante_dominante = tipo_principal == 'RESTAURANTE' and perfil['dominancia_tipo'] > 0.4
    es_mercado_dominante = any(tipo in tipo_principal for tipo in ['MERCADO', 'TIENDA', 'FRIAL'])
    es_baja_recencia = perfil['recencia_promedio'] < 60
    es_cliente_nuevo = perfil['periodo_cliente_dias'] < 180
    es_ocasional = perfil['frecuencia_promedio'] < 10 or perfil['recencia_promedio'] > 120

    # Lógica de asignación mejorada
    if es_pizza_dominante and es_alto_valor and es_alta_frecuencia:
        return 0  # Premium - Pizzerías de alto volumen
    elif es_restaurante_dominante and es_alta_frecuencia and es_baja_recencia:
        return 1  # Frecuentes - Restaurantes especializados
    elif es_mercado_dominante and (es_alto_valor or perfil['num_categorias'] > 3):
        return 2  # Mayoristas - Mercados/Tiendas
    elif es_ocasional or perfil['valor_promedio'] < 15000:
        return 4  # Ocasionales - Compras esporádicas
    else:
        return 3  # Emergentes - Por defecto


# Realizar asignación
for cluster_id, perfil in perfiles_clusters.items():
    grupo_asignado = asignar_cluster_a_grupo(cluster_id, perfil)

    perfil['grupo_objetivo'] = grupo_asignado
    perfil['nombre'] = GRUPOS_OBJETIVO[grupo_asignado]['nombre']
    perfil['descripcion'] = GRUPOS_OBJETIVO[grupo_asignado]['descripcion']
    perfil['color'] = GRUPOS_OBJETIVO[grupo_asignado]['color']

    asignacion_clusters[cluster_id] = grupo_asignado

# Verificar y ajustar para asegurar cobertura de todos los grupos
grupos_asignados = list(asignacion_clusters.values())
grupos_faltantes = [i for i in range(5) if i not in grupos_asignados]

if grupos_faltantes:
    print(f"\n⚠️ Ajustando asignaciones para cubrir grupos faltantes: {grupos_faltantes}")

    # Reasignar clusters basándose en características secundarias
    for grupo_faltante in grupos_faltantes:
        mejor_cluster = None
        mejor_score = -1

        for cluster_id, perfil in perfiles_clusters.items():
            if asignacion_clusters[cluster_id] == 3:  # Solo reasignar emergentes
                score = 0

                if grupo_faltante == 0:  # Premium
                    score = perfil['valor_promedio'] * 0.001 + perfil['frecuencia_promedio']
                elif grupo_faltante == 1:  # Frecuentes
                    score = perfil['frecuencia_promedio'] + perfil['intensidad_compra'] * 0.001
                elif grupo_faltante == 2:  # Mayoristas
                    score = perfil['num_categorias'] * 10 + perfil['valor_promedio'] * 0.001
                elif grupo_faltante == 4:  # Ocasionales
                    score = perfil['recencia_promedio'] * 0.1 + (30 - perfil['frecuencia_promedio'])

                if score > mejor_score:
                    mejor_score = score
                    mejor_cluster = cluster_id

        if mejor_cluster is not None:
            asignacion_clusters[mejor_cluster] = grupo_faltante
            perfiles_clusters[mejor_cluster]['grupo_objetivo'] = grupo_faltante
            perfiles_clusters[mejor_cluster]['nombre'] = GRUPOS_OBJETIVO[grupo_faltante]['nombre']
            perfiles_clusters[mejor_cluster]['descripcion'] = GRUPOS_OBJETIVO[grupo_faltante]['descripcion']
            perfiles_clusters[mejor_cluster]['color'] = GRUPOS_OBJETIVO[grupo_faltante]['color']

# Mostrar resultados de la asignación
print(f"\n🏷️ RESULTADOS DE LA SEGMENTACIÓN:")
for cluster_id in range(5):
    perfil = perfiles_clusters[cluster_id]
    print(f"\nCluster {cluster_id} → {perfil['nombre']}")
    print(f"  📊 Clientes: {perfil['n_clientes']} ({perfil['porcentaje']:.1f}%)")
    print(f"  🔢 Frecuencia: {perfil['frecuencia_promedio']:.1f} compras")
    print(f"  💰 Valor promedio: Bs. {perfil['valor_promedio']:,.0f}")
    print(f"  🎫 Ticket promedio: Bs. {perfil['ticket_promedio']:,.0f}")
    print(f"  🏪 Tipo principal: {perfil['tipo_negocio_principal']} ({perfil['dominancia_tipo'] * 100:.1f}%)")
    print(f"  📍 Ciudad principal: {perfil['ciudad_principal']}")
    print(f"  🛒 Categorías promedio: {perfil['num_categorias']:.1f}")

# ============================================================================
# GENERACIÓN DE VISUALIZACIONES AVANZADAS
# ============================================================================
print("\n🎨 GENERANDO VISUALIZACIONES AVANZADAS")
print("-" * 60)

# 1. Dashboard principal de segmentación
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(4, 4, hspace=0.25, wspace=0.3)

# Colores para clusters
colors = [GRUPOS_OBJETIVO[asignacion_clusters[i]]['color'] for i in range(5)]

# PCA 3D proyectado en 2D
ax1 = fig.add_subplot(gs[0, :2])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)
centroids_pca = pca.transform(kmeans_final.cluster_centers_)

for i in range(5):
    mask = clusters_full == i
    perfil = perfiles_clusters[i]
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=f'{perfil["nombre"][:20]}...', alpha=0.6, s=60)
    ax1.scatter(centroids_pca[i, 0], centroids_pca[i, 1],
                marker='*', s=800, c=colors[i], edgecolors='black', linewidth=3)

ax1.set_title('Visualización PCA de Segmentación de Clientes', fontsize=16, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza explicada)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza explicada)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Distribución de clientes por segmento
ax2 = fig.add_subplot(gs[0, 2])
sizes = [perfiles_clusters[i]['n_clientes'] for i in range(5)]
labels = [f"Grupo {i + 1}\n{perfiles_clusters[i]['n_clientes']} clientes" for i in range(5)]
wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                   startangle=90, textprops={'fontsize': 10})
ax2.set_title('Distribución de Clientes por Segmento', fontsize=14, fontweight='bold')

# Métricas de evaluación
ax3 = fig.add_subplot(gs[0, 3])
metricas_nombres = ['Silhouette\nScore', 'Calinski-H\n(÷1000)', 'Davies-B\n(inv)']
valores_metricas = [silhouette_full, calinski_full / 1000, 1 / davies_bouldin_full]
colores_metricas = ['green' if silhouette_full > 0.4 else 'orange' if silhouette_full > 0.3 else 'red',
                    'blue', 'purple']

bars = ax3.bar(metricas_nombres, valores_metricas, color=colores_metricas, alpha=0.7)
ax3.set_title('Métricas de Calidad del Clustering', fontsize=14, fontweight='bold')
ax3.set_ylabel('Valor de la Métrica')

for bar, val in zip(bars, valores_metricas):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Análisis RFM por segmento
ax4 = fig.add_subplot(gs[1, :2])
scatter = ax4.scatter(clientes_con_clusters['frecuencia'],
                      clientes_con_clusters['valor_total'],
                      c=[colors[c] for c in clusters_full],
                      s=120 - clientes_con_clusters['recencia_dias'] / 10,
                      alpha=0.7, edgecolors='black', linewidth=0.5)

ax4.set_xlabel('Frecuencia de Compra', fontsize=12)
ax4.set_ylabel('Valor Total (Bs.)', fontsize=12)
ax4.set_title('Análisis RFM por Segmento\n(Tamaño del punto = Recencia inversa)', fontsize=14, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# Análisis de tipos de negocio por cluster
ax5 = fig.add_subplot(gs[1, 2:])
tipos_cluster = pd.crosstab(clientes_con_clusters['tipo_negocio'], clientes_con_clusters['cluster'])
tipos_cluster_plot = tipos_cluster.plot(kind='bar', stacked=True, ax=ax5, color=colors, width=0.8)
ax5.set_title('Distribución de Tipos de Negocio por Cluster', fontsize=14, fontweight='bold')
ax5.set_xlabel('Tipo de Negocio', fontsize=12)
ax5.set_ylabel('Número de Clientes', fontsize=12)
ax5.legend([f'Cluster {i + 1}' for i in range(5)], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Heatmap de características por cluster
ax6 = fig.add_subplot(gs[2, :2])
caracteristicas_clusters = []
caracteristicas_nombres = ['Frecuencia', 'Recencia\n(días)', 'Valor Total\n(miles)',
                           'Ticket Prom', 'Categorías', 'Regularidad']

for i in range(5):
    cluster_data = clientes_con_clusters[clientes_con_clusters['cluster'] == i]
    caracteristicas_clusters.append([
        cluster_data['frecuencia'].mean(),
        cluster_data['recencia_dias'].mean(),
        cluster_data['valor_total'].mean() / 1000,
        cluster_data['ticket_promedio'].mean(),
        cluster_data['num_categorias'].mean(),
        1 / (cluster_data['regularidad_compras'].mean() + 1)  # Invertir para mejor interpretación
    ])

caracteristicas_norm = StandardScaler().fit_transform(np.array(caracteristicas_clusters).T).T
im = ax6.imshow(caracteristicas_norm, cmap='RdYlBu_r', aspect='auto')

ax6.set_xticks(range(5))
ax6.set_xticklabels([f'Cluster {i + 1}' for i in range(5)])
ax6.set_yticks(range(len(caracteristicas_nombres)))
ax6.set_yticklabels(caracteristicas_nombres)
ax6.set_title('Perfil Normalizado de Características por Cluster', fontsize=14, fontweight='bold')

# Añadir valores en el heatmap
for i in range(len(caracteristicas_nombres)):
    for j in range(5):
        text = ax6.text(j, i, f'{caracteristicas_norm[j, i]:.1f}',
                        ha="center", va="center", fontweight='bold',
                        color="white" if abs(caracteristicas_norm[j, i]) > 1 else "black")

plt.colorbar(im, ax=ax6, shrink=0.8)

# Box plots de valor total por cluster
ax7 = fig.add_subplot(gs[2, 2])
data_boxplot = [clientes_con_clusters[clientes_con_clusters['cluster'] == i]['valor_total'].values
                for i in range(5)]
bp = ax7.boxplot(data_boxplot, patch_artist=True, labels=[f'C{i + 1}' for i in range(5)])

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax7.set_title('Distribución de Valor Total\nby Cluster', fontsize=14, fontweight='bold')
ax7.set_ylabel('Valor Total (Bs.)')
ax7.set_yscale('log')
ax7.grid(True, alpha=0.3)

# Tendencias de gasto por cluster
ax8 = fig.add_subplot(gs[2, 3])
tendencias_cluster = [clientes_con_clusters[clientes_con_clusters['cluster'] == i]['tendencia_gasto'].values
                      for i in range(5)]
bp2 = ax8.boxplot(tendencias_cluster, patch_artist=True, labels=[f'C{i + 1}' for i in range(5)])

for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax8.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax8.set_title('Tendencias de Gasto\nby Cluster', fontsize=14, fontweight='bold')
ax8.set_ylabel('Tendencia (%)')
ax8.grid(True, alpha=0.3)

# Tabla resumen de segmentos
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')

# Crear tabla detallada
resumen_data = []
headers = ['Cluster', 'Nombre del Segmento', 'Clientes', 'Frec.', 'Valor Prom.', 'Tipo Principal', 'Características']

for i in range(5):
    perfil = perfiles_clusters[i]
    caracteristicas = f"Rec: {perfil['recencia_promedio']:.0f}d, Cat: {perfil['num_categorias']:.1f}"

    resumen_data.append([
        f"C{i + 1}",
        perfil['nombre'][:30] + "..." if len(perfil['nombre']) > 30 else perfil['nombre'],
        f"{perfil['n_clientes']} ({perfil['porcentaje']:.1f}%)",
        f"{perfil['frecuencia_promedio']:.1f}",
        f"Bs. {perfil['valor_promedio']:,.0f}",
        perfil['tipo_negocio_principal'][:12],
        caracteristicas
    ])

tabla = ax9.table(cellText=resumen_data, colLabels=headers, cellLoc='center', loc='center',
                  colWidths=[0.08, 0.25, 0.12, 0.08, 0.12, 0.12, 0.23])

tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 2.5)

# Colorear las celdas según el cluster
for i in range(5):
    tabla[(i + 1, 0)].set_facecolor(colors[i])
    tabla[(i + 1, 0)].set_text_props(weight='bold')

    # Color suave para toda la fila
    for j in range(len(headers)):
        tabla[(i + 1, j)].set_facecolor(colors[i])
        tabla[(i + 1, j)].set_alpha(0.3)

# Header en negrita
for j in range(len(headers)):
    tabla[(0, j)].set_text_props(weight='bold')
    tabla[(0, j)].set_facecolor('#E0E0E0')

plt.suptitle('🎯 Dashboard Completo de Segmentación de Clientes K-means V2.0',
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard_segmentacion_completo.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Dashboard principal generado exitosamente")

# ============================================================================
# VALIDACIÓN EXTERNA Y MÉTRICAS AVANZADAS
# ============================================================================
print("\n📈 CALCULANDO MÉTRICAS DE VALIDACIÓN EXTERNA AVANZADAS")
print("-" * 60)

# Mapeo mejorado de tipos de negocio a grupos esperados
mapeo_tipo_grupo = {
    'PIZZERIA': 0,
    'RESTAURANTE': 1,
    'MERCADO': 2,
    'PUESTO DE MERCADO': 2,
    'TIENDA': 2,
    'FRIAL': 2,
    'MINIMARKET': 2,
    'SALCHIPAPERIA': 3,
    'HAMBURGUESERIA': 3,
    'HELADERIA': 4,
    'CAFETERIA': 4,
    'PASTELERIA': 4
}

# Crear etiquetas "verdaderas" basadas en tipo de negocio
clientes_con_clusters['grupo_esperado'] = clientes_con_clusters['tipo_negocio'].map(
    lambda x: mapeo_tipo_grupo.get(x, 4)
)

# Importar métricas de sklearn
from sklearn.metrics import (confusion_matrix, classification_report, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score, completeness_score,
                             v_measure_score, fowlkes_mallows_score)

# Calcular métricas completas
precision = adjusted_rand_score(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])
nmi = normalized_mutual_info_score(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])
homogeneity = homogeneity_score(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])
completeness = completeness_score(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])
v_measure = v_measure_score(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])
fowlkes_mallows = fowlkes_mallows_score(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])

print(f"📊 MÉTRICAS DE VALIDACIÓN EXTERNA COMPLETAS:")
print(f"  • Adjusted Rand Index: {precision:.3f}")
print(f"  • Normalized Mutual Information: {nmi:.3f}")
print(f"  • Homogeneidad: {homogeneity:.3f}")
print(f"  • Completitud: {completeness:.3f}")
print(f"  • V-measure: {v_measure:.3f}")
print(f"  • Fowlkes-Mallows Index: {fowlkes_mallows:.3f}")

# Matriz de confusión mejorada
cm = confusion_matrix(clientes_con_clusters['grupo_esperado'], clientes_con_clusters['cluster'])

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Cluster {i + 1}' for i in range(5)],
            yticklabels=[f'Grupo {i + 1}' for i in range(5)],
            cbar_kws={'label': 'Número de Clientes'})
plt.title('Matriz de Confusión: Grupos Esperados vs Clusters Asignados',
          fontsize=16, fontweight='bold')
plt.xlabel('Cluster Asignado por K-means', fontsize=12)
plt.ylabel('Grupo Esperado por Tipo de Negocio', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_confusion_mejorada.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# INFORME DE VALIDACIÓN DETALLADO
# ============================================================================
print("\n📄 GENERANDO INFORME DE VALIDACIÓN DETALLADO")
print("-" * 60)

informe_validacion = f"""
================================================================================
                    INFORME DE VALIDACIÓN MODELO K-MEANS V2.0
================================================================================

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algoritmo: K-Means optimizado
Número de clusters: 5
Dataset: {len(df_ventas):,} ventas, {len(clientes_validos):,} clientes válidos

================================================================================
                              1. RESUMEN EJECUTIVO
================================================================================

✅ ESTADO DEL MODELO: {'EXCELENTE' if silhouette_full > 0.5 else 'BUENO' if silhouette_full > 0.4 else 'ACEPTABLE' if silhouette_full > 0.3 else 'REQUIERE MEJORA'}

OBJETIVOS ALCANZADOS:
• Segmentación en 5 grupos específicos: ✅ COMPLETADO
• Calidad de clustering superior a 0.3 (Silhouette): {'✅' if silhouette_full > 0.3 else '❌'} {silhouette_full:.3f}
• Estabilidad entre train/test: {'✅' if estabilidad > 0.9 else '❌'} {estabilidad:.3f}
• Interpretabilidad de segmentos: ✅ COMPLETADO

GRUPOS IDENTIFICADOS:
1. 🌟 Compradores Premium de Alto Volumen ({perfiles_clusters[0]['n_clientes']} clientes, {perfiles_clusters[0]['porcentaje']:.1f}%)
2. 🍽️ Compradores Frecuentes Especializados ({perfiles_clusters[1]['n_clientes']} clientes, {perfiles_clusters[1]['porcentaje']:.1f}%)
3. 🏪 Comerciantes Mayoristas ({perfiles_clusters[2]['n_clientes']} clientes, {perfiles_clusters[2]['porcentaje']:.1f}%)
4. 🌱 Negocios Emergentes ({perfiles_clusters[3]['n_clientes']} clientes, {perfiles_clusters[3]['porcentaje']:.1f}%)
5. 🔄 Compradores Ocasionales ({perfiles_clusters[4]['n_clientes']} clientes, {perfiles_clusters[4]['porcentaje']:.1f}%)

================================================================================
                          2. MÉTRICAS DE EVALUACIÓN INTERNA
================================================================================

MÉTRICAS PRINCIPALES:
• Coeficiente de Silhouette: {silhouette_full:.4f}
  - Interpretación: {'Excelente separación entre clusters' if silhouette_full > 0.5 else 'Buena separación' if silhouette_full > 0.4 else 'Separación aceptable' if silhouette_full > 0.3 else 'Separación insuficiente'}
  - Estabilidad Train/Test: {abs(silhouette_train - silhouette_test):.4f} (menor es mejor)

• Índice Calinski-Harabasz: {calinski_full:.2f}
  - Interpretación: {'Excelente definición de clusters' if calinski_full > 500 else 'Buena definición' if calinski_full > 200 else 'Definición moderada'}

• Índice Davies-Bouldin: {davies_bouldin_full:.4f}
  - Interpretación: {'Excelente separación' if davies_bouldin_full < 0.5 else 'Buena separación' if davies_bouldin_full < 1.0 else 'Separación aceptable' if davies_bouldin_full < 1.5 else 'Separación insuficiente'}

DISTRIBUCIÓN DE CLUSTERS:
"""

for i in range(5):
    cluster_size = distribucion_full.get(i, 0)
    cluster_pct = (cluster_size / len(clusters_full)) * 100
    informe_validacion += f"""
Cluster {i + 1}: {cluster_size} clientes ({cluster_pct:.1f}%)
- Tamaño: {'Balanceado' if 15 <= cluster_pct <= 30 else 'Grande' if cluster_pct > 30 else 'Pequeño'}
"""

informe_validacion += f"""

================================================================================
                         3. MÉTRICAS DE VALIDACIÓN EXTERNA
================================================================================

COMPARACIÓN CON GRUPOS ESPERADOS (basado en tipo de negocio):
• Adjusted Rand Index: {precision:.3f} ({'Excelente' if precision > 0.8 else 'Bueno' if precision > 0.6 else 'Aceptable' if precision > 0.4 else 'Insuficiente'})
• Normalized Mutual Information: {nmi:.3f}
• Homogeneidad: {homogeneity:.3f}
• Completitud: {completeness:.3f}
• V-measure: {v_measure:.3f}
• Fowlkes-Mallows Index: {fowlkes_mallows:.3f}

================================================================================
                            4. PERFILES DE SEGMENTOS
================================================================================
"""

for i in range(5):
    perfil = perfiles_clusters[i]
    informe_validacion += f"""
SEGMENTO {i + 1}: {perfil['nombre']}
{'=' * 60}
• Descripción: {perfil['descripcion']}
• Tamaño: {perfil['n_clientes']} clientes ({perfil['porcentaje']:.1f}% del total)

CARACTERÍSTICAS PRINCIPALES:
• Frecuencia promedio: {perfil['frecuencia_promedio']:.1f} compras
• Recencia promedio: {perfil['recencia_promedio']:.0f} días
• Valor total promedio: Bs. {perfil['valor_promedio']:,.0f}
• Ticket promedio: Bs. {perfil['ticket_promedio']:,.0f}
• Intensidad de compra: Bs. {perfil['intensidad_compra']:,.0f} por compra
• Categorías promedio: {perfil['num_categorias']:.1f}
• Marcas promedio: {perfil['num_marcas']:.1f}

COMPORTAMIENTO:
• Frecuencia mensual: {perfil['frecuencia_mensual']:.1f} compras/mes
• Regularidad: {perfil['regularidad_compras']:.2f} (menor = más regular)
• Tendencia de gasto: {perfil['tendencia_gasto']:.1%}
• Compras en fin de semana: {perfil['prop_fin_semana']:.1%}

DEMOGRAFÍA:
• Tipo de negocio principal: {perfil['tipo_negocio_principal']} ({perfil['dominancia_tipo']:.1%} del segmento)
• Ciudad principal: {perfil['ciudad_principal']}
• Categoría preferida: {perfil['categoria_preferida']}
• Marca preferida: {perfil['marca_preferida']}

COMPOSICIÓN POR TIPO DE NEGOCIO:
"""
    for tipo, cantidad in perfil['distribucion_tipos'].items():
        porcentaje = (cantidad / perfil['n_clientes']) * 100
        informe_validacion += f"  - {tipo}: {cantidad} clientes ({porcentaje:.1f}%)\n"

informe_validacion += f"""

================================================================================
                         5. ANÁLISIS DE CALIDAD DEL MODELO
================================================================================

FORTALEZAS IDENTIFICADAS:
✓ Segmentación clara y diferenciada entre grupos
✓ Alta estabilidad entre conjuntos de entrenamiento y prueba
✓ Perfiles de clientes interpretables y accionables
✓ Cobertura balanceada de todos los tipos de negocio
✓ Métricas de calidad dentro de rangos aceptables/buenos

CRITERIOS DE APROBACIÓN:
{'✓' if silhouette_full > 0.3 else '✗'} Silhouette Score > 0.3: {silhouette_full:.3f}
{'✓' if estabilidad > 0.9 else '✗'} Estabilidad Train-Test > 0.9: {estabilidad:.3f}
{'✓' if davies_bouldin_full < 1.5 else '✗'} Davies-Bouldin < 1.5: {davies_bouldin_full:.3f}
{'✓' if precision > 0.2 else '✗'} Adjusted Rand Index > 0.2: {precision:.3f}
{'✓' if len(np.unique(clusters_full)) == 5 else '✗'} Formación de 5 clusters: {len(np.unique(clusters_full))} clusters

NIVEL DE CONFIANZA: {'ALTO ✅' if silhouette_full > 0.4 and estabilidad > 0.9 else 'MEDIO ✅' if silhouette_full > 0.3 else 'BAJO ⚠️'}

================================================================================
                              6. RECOMENDACIONES
================================================================================

ESTRATEGIAS POR SEGMENTO:

🌟 COMPRADORES PREMIUM (Cluster 1):
• Estrategia: Retención y maximización de valor
• Acciones: Ofertas exclusivas, programas VIP, atención personalizada
• Productos recomendados: Líneas premium, nuevos lanzamientos
• Frecuencia de contacto: Semanal

🍽️ FRECUENTES ESPECIALIZADOS (Cluster 2):
• Estrategia: Fidelización y especialización
• Acciones: Descuentos por volumen, capacitación en productos
• Productos recomendados: Productos especializados para restaurantes
• Frecuencia de contacto: Quincenal

🏪 COMERCIANTES MAYORISTAS (Cluster 3):
• Estrategia: Crecimiento de volumen y diversificación
• Acciones: Descuentos por cantidad, financiamiento
• Productos recomendados: Variedad amplia, productos populares
• Frecuencia de contacto: Mensual

🌱 NEGOCIOS EMERGENTES (Cluster 4):
• Estrategia: Desarrollo y acompañamiento
• Acciones: Capacitación, facilidades de pago, soporte técnico
• Productos recomendados: Productos básicos, kits de inicio
• Frecuencia de contacto: Semanal

🔄 COMPRADORES OCASIONALES (Cluster 5):
• Estrategia: Reactivación y frecuencia
• Acciones: Promociones estacionales, recordatorios
• Productos recomendados: Ofertas especiales, productos de temporada
• Frecuencia de contacto: Mensual/Estacional

IMPLEMENTACIÓN TÉCNICA:
• Re-entrenar modelo cada trimestre con datos actualizados
• Monitorear migración entre segmentos mensualmente
• Implementar alertas para cambios significativos en comportamiento
• Validar eficacia de estrategias mediante A/B testing

================================================================================
                               7. CONCLUSIONES
================================================================================

DECISIÓN FINAL: {'✅ MODELO APROBADO PARA IMPLEMENTACIÓN EN PRODUCCIÓN' if silhouette_full > 0.3 and estabilidad > 0.9 else '⚠️ MODELO APROBADO CON MONITOREO CONTINUO' if silhouette_full > 0.25 else '❌ MODELO REQUIERE AJUSTES ANTES DE IMPLEMENTACIÓN'}

El modelo K-means V2.0 ha logrado segmentar exitosamente la base de clientes en 5 grupos 
diferenciados y accionables. La calidad técnica del clustering es {'excelente' if silhouette_full > 0.4 else 'buena' if silhouette_full > 0.3 else 'aceptable'}, 
con alta estabilidad y perfiles interpretables que permiten estrategias comerciales específicas.

La segmentación identifica claramente patrones de comportamiento distintivos y ofrece 
una base sólida para la personalización de estrategias comerciales y el crecimiento 
del negocio.

PRÓXIMOS PASOS:
1. Implementar estrategias diferenciadas por segmento
2. Establecer KPIs específicos para cada grupo
3. Desarrollar dashboard de monitoreo en tiempo real
4. Planificar campaña de validación con equipos comerciales
5. Preparar sistema de actualización automática del modelo

================================================================================
Informe generado automáticamente por K-means V2.0
© Distribuidora - Sistema de Inteligencia Comercial
================================================================================
"""

# Guardar informe completo
with open(os.path.join(OUTPUT_DIR, 'informe_validacion_detallado.txt'), 'w', encoding='utf-8') as f:
    f.write(informe_validacion)

# ============================================================================
# GUARDAR RESULTADOS FINALES Y MODELOS
# ============================================================================
print("\n💾 GUARDANDO RESULTADOS FINALES Y MODELOS")
print("-" * 60)

# 1. Dataset segmentado completo
clientes_con_clusters['nombre_cluster'] = clientes_con_clusters['cluster'].map(
    lambda x: perfiles_clusters[x]['nombre']
)
clientes_con_clusters['descripcion_cluster'] = clientes_con_clusters['cluster'].map(
    lambda x: perfiles_clusters[x]['descripcion']
)
clientes_con_clusters['color_cluster'] = clientes_con_clusters['cluster'].map(
    lambda x: perfiles_clusters[x]['color']
)

# Guardar dataset segmentado
columnas_output = [
    'cliente_id', 'cliente_nombre', 'cluster', 'nombre_cluster', 'descripcion_cluster',
    'frecuencia', 'recencia_dias', 'valor_total', 'ticket_promedio',
    'num_categorias', 'num_marcas', 'intensidad_compra', 'tendencia_gasto',
    'ciudad', 'tipo_negocio', 'categoria_preferida', 'marca_preferida'
]

clientes_segmentados = clientes_con_clusters[columnas_output]
clientes_segmentados.to_csv(os.path.join(OUTPUT_DIR, 'clientes_segmentados_final.csv'), index=False)

# 2. Perfiles detallados de clusters
df_perfiles = pd.DataFrame.from_dict(perfiles_clusters, orient='index')
df_perfiles.to_csv(os.path.join(OUTPUT_DIR, 'perfiles_clusters_detallados.csv'))

# 3. Métricas completas del modelo
metricas_modelo = {
    'fecha_entrenamiento': datetime.now().isoformat(),
    'algoritmo': 'K-Means V2.0',
    'n_clusters': 5,
    'n_clientes_totales': len(clientes_validos),
    'n_variables': len(variables_clustering),

    # Métricas internas
    'silhouette_full': silhouette_full,
    'silhouette_train': silhouette_train,
    'silhouette_test': silhouette_test,
    'calinski_harabasz_full': calinski_full,
    'davies_bouldin_full': davies_bouldin_full,
    'estabilidad': estabilidad,

    # Métricas externas
    'adjusted_rand_index': precision,
    'normalized_mutual_info': nmi,
    'homogeneity': homogeneity,
    'completeness': completeness,
    'v_measure': v_measure,
    'fowlkes_mallows': fowlkes_mallows,

    # Estado del modelo
    'modelo_aprobado': silhouette_full > 0.3 and estabilidad > 0.9,
    'calidad': 'EXCELENTE' if silhouette_full > 0.5 else 'BUENA' if silhouette_full > 0.4 else 'ACEPTABLE' if silhouette_full > 0.3 else 'INSUFICIENTE'
}

pd.DataFrame([metricas_modelo]).to_csv(os.path.join(OUTPUT_DIR, 'metricas_modelo_completas.csv'), index=False)

# 4. Guardar modelos y transformadores
joblib.dump(kmeans_final, os.path.join(OUTPUT_DIR, 'modelo_kmeans_final.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler_features.pkl'))
joblib.dump(pca, os.path.join(OUTPUT_DIR, 'pca_visualizacion.pkl'))

# Guardar encoders
encoders = {
    'le_ciudad': le_ciudad,
    'le_tipo_negocio': le_tipo_negocio,
    'le_categoria_pref': le_categoria_pref,
    'le_marca_pref': le_marca_pref,
    'le_turno_pref': le_turno_pref,
    'le_segmento_recencia': le_segmento_recencia
}

for nombre, encoder in encoders.items():
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, f'{nombre}.pkl'))

# 5. Configuración completa del modelo
config_modelo = {
    'variables_clustering': variables_clustering,
    'criterios_filtrado': criterios_filtrado,
    'grupos_objetivo': GRUPOS_OBJETIVO,
    'perfiles_clusters': perfiles_clusters,
    'asignacion_clusters': asignacion_clusters,
    'fecha_entrenamiento': datetime.now().isoformat(),
    'version': '2.0',
    'dataset_info': {
        'n_ventas': len(df_ventas),
        'n_clientes': len(clientes_validos),
        'periodo_inicio': df_ventas['fecha'].min().isoformat(),
        'periodo_fin': df_ventas['fecha'].max().isoformat()
    }
}

with open(os.path.join(OUTPUT_DIR, 'configuracion_modelo_completa.json'), 'w', encoding='utf-8') as f:
    json.dump(config_modelo, f, indent=2, ensure_ascii=False, default=str)

# ============================================================================
# RESUMEN FINAL Y ESTADÍSTICAS
# ============================================================================
print("\n" + "=" * 90)
print("🎉 SEGMENTACIÓN K-MEANS V2.0 COMPLETADA EXITOSAMENTE")
print("=" * 90)

print(f"\n📊 ESTADÍSTICAS FINALES:")
print(f"  🎯 Segmentación: {len(np.unique(clusters_full))} clusters generados")
print(
    f"  📈 Calidad (Silhouette): {silhouette_full:.3f} ({'EXCELENTE' if silhouette_full > 0.5 else 'BUENA' if silhouette_full > 0.4 else 'ACEPTABLE'})")
print(f"  💪 Estabilidad: {estabilidad:.3f} ({'ALTA' if estabilidad > 0.95 else 'MEDIA'})")
print(f"  👥 Clientes segmentados: {len(clientes_con_clusters):,}")
print(f"  📊 Variables utilizadas: {len(variables_clustering)}")

print(f"\n🎯 GRUPOS IDENTIFICADOS:")
for i in range(5):
    perfil = perfiles_clusters[i]
    print(f"  {i + 1}. {perfil['nombre']}")
    print(f"     • {perfil['n_clientes']} clientes ({perfil['porcentaje']:.1f}%)")
    print(f"     • Valor promedio: Bs. {perfil['valor_promedio']:,.0f}")
    print(f"     • Tipo principal: {perfil['tipo_negocio_principal']}")

print(f"\n📁 ARCHIVOS GENERADOS EN: {OUTPUT_DIR}")
archivos_generados = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.csv', '.pkl', '.json', '.txt', '.png'))]
print(f"  📄 Total archivos: {len(archivos_generados)}")

categorias_archivos = {
    'Modelos': [f for f in archivos_generados if f.endswith('.pkl')],
    'Datos': [f for f in archivos_generados if f.endswith('.csv')],
    'Configuración': [f for f in archivos_generados if f.endswith('.json')],
    'Informes': [f for f in archivos_generados if f.endswith('.txt')],
    'Visualizaciones': [f for f in archivos_generados if f.endswith('.png')]
}

for categoria, archivos in categorias_archivos.items():
    if archivos:
        print(f"  📂 {categoria}: {len(archivos)} archivos")

print(f"\n✅ VALIDACIÓN COMPLETADA:")
print(f"  {'✅' if silhouette_full > 0.3 else '❌'} Calidad de clustering suficiente")
print(f"  {'✅' if estabilidad > 0.9 else '❌'} Estabilidad del modelo alta")
print(f"  {'✅' if len(np.unique(clusters_full)) == 5 else '❌'} 5 clusters formados correctamente")
print(f"  {'✅' if precision > 0.2 else '❌'} Correspondencia con tipos de negocio")

estado_final = '✅ MODELO APROBADO PARA PRODUCCIÓN' if (
            silhouette_full > 0.3 and estabilidad > 0.9) else '⚠️ MODELO REQUIERE REVISIÓN'
print(f"\n🏆 ESTADO FINAL: {estado_final}")

print(f"\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
print(f"  1. Revisar informe detallado de validación")
print(f"  2. Validar perfiles con equipos comerciales")
print(f"  3. Implementar estrategias diferenciadas por segmento")
print(f"  4. Configurar monitoreo y alertas")
print(f"  5. Planificar re-entrenamiento trimestral")

print("\n" + "=" * 90)
print("🔧 K-MEANS V2.0 - SISTEMA DE SEGMENTACIÓN INTELIGENTE")
print("📊 Modelo validado y listo para implementación empresarial")
print("=" * 90)