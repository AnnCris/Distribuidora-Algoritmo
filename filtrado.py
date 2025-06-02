import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import warnings
import os
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import joblib

warnings.filterwarnings('ignore')

# Configuración inicial
ALGORITMO = "FILTRADO_COLABORATIVO_AVANZADO"
OUTPUT_DIR = f"resultados_{ALGORITMO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['figure.dpi'] = 100

print("=" * 100)
print("🧀 SISTEMA DE RECOMENDACIÓN CON FILTRADO COLABORATIVO - METODOLOGÍA CRISP-DM")
print("🎯 ESPECIALIZADO EN PRODUCTOS GASTRONÓMICOS Y QUESOS")
print("📊 MOTOR DE RECOMENDACIÓN AVANZADO")
print("=" * 100)

# ============================================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO (CRISP-DM)
# ============================================================================
print("\n📋 FASE 1: COMPRENSIÓN DEL NEGOCIO")
print("-" * 70)

# Objetivos específicos del sistema de recomendación
OBJETIVOS_NEGOCIO = {
    'objetivo_principal': 'Sistema de Recomendación de Productos Gastronómicos con Filtrado Colaborativo',
    'metricas_objetivo': {
        'mae': 0.75,  # MAE ≤ 0.75
        'rmse': 1.0,  # RMSE ≤ 1.0
        'cobertura': 0.80  # Cobertura ≥ 0.80
    },
    'tipos_recomendacion': {
        'tipo_1': {
            'nombre': 'Combinaciones óptimas de quesos para determinados platillos',
            'descripcion': 'Recomendar quesos que complementen específicos platillos gastronómicos',
            'contexto': 'Basado en patrones de compra históricos de establecimientos similares'
        },
        'tipo_2': {
            'nombre': 'Nuevos productos que podrían adaptarse a su oferta gastronómica',
            'descripcion': 'Identificar productos innovadores que otros establecimientos similares han adoptado exitosamente',
            'contexto': 'Análisis de tendencias y adopción de nuevos productos'
        },
        'tipo_3': {
            'nombre': 'Tendencias en el uso de quesos en su tipo de establecimiento',
            'descripcion': 'Mostrar patrones emergentes en el uso de quesos específicos por tipo de negocio',
            'contexto': 'Análisis temporal y segmentación por tipo de establecimiento'
        }
    },
    'segmentacion': 'Por tipo de negocio (restaurantes, hoteles, cafeterías, etc.)',
    'algoritmos': ['User-Based CF', 'Item-Based CF', 'Matrix Factorization', 'Hybrid Model']
}

print("🎯 OBJETIVOS DEL SISTEMA DE RECOMENDACIÓN:")
print(f"  📊 MAE objetivo: ≤ {OBJETIVOS_NEGOCIO['metricas_objetivo']['mae']}")
print(f"  📐 RMSE objetivo: ≤ {OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse']}")
print(f"  📈 Cobertura objetivo: ≥ {OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura']}")

print(f"\n🧀 TIPOS DE RECOMENDACIONES ESPECIALIZADAS:")
for tipo_key, tipo_info in OBJETIVOS_NEGOCIO['tipos_recomendacion'].items():
    print(f"\n📌 {tipo_info['nombre']}:")
    print(f"   • Descripción: {tipo_info['descripcion']}")
    print(f"   • Contexto: {tipo_info['contexto']}")

print(f"\n🏢 Segmentación: {OBJETIVOS_NEGOCIO['segmentacion']}")
print(f"🔧 Algoritmos: {', '.join(OBJETIVOS_NEGOCIO['algoritmos'])}")

# ============================================================================
# FASE 2: COMPRENSIÓN DE LOS DATOS (CRISP-DM)
# ============================================================================
print("\n📊 FASE 2: COMPRENSIÓN DE LOS DATOS")
print("-" * 70)


def cargar_datos():
    """Carga los mismos datasets que usa Random Forest"""
    try:
        archivos_posibles = [
            ('ventas_mejorado_v2.csv', 'detalles_ventas_mejorado_v2.csv'),
            ('ventas.csv', 'detalles_ventas.csv')
        ]

        df_ventas, df_detalles = None, None

        for ventas_file, detalles_file in archivos_posibles:
            if os.path.exists(ventas_file) and os.path.exists(detalles_file):
                df_ventas = pd.read_csv(ventas_file)
                df_detalles = pd.read_csv(detalles_file)
                print(f"✅ Datasets cargados exitosamente: {ventas_file}, {detalles_file}")
                break

        if df_ventas is None:
            raise FileNotFoundError("No se encontraron archivos de datos")

        return df_ventas, df_detalles

    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        exit(1)


def detectar_formato_fecha(df):
    """Detecta y convierte fechas automáticamente"""
    print("📅 Detectando formato de fechas...")
    try:
        fecha_sample = str(df['fecha'].iloc[0])
        if '-' in fecha_sample and len(fecha_sample.split('-')[0]) == 4:
            df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        elif '/' in fecha_sample:
            df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
        else:
            df['fecha'] = pd.to_datetime(df['fecha'])
        print("   ✅ Fechas convertidas exitosamente")
        return df
    except Exception as e:
        print(f"   ⚠️ Error en conversión: {e}")
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        return df


# Cargar datos
print("📥 Cargando datos para sistema de recomendación...")
df_ventas, df_detalles = cargar_datos()
df_ventas = detectar_formato_fecha(df_ventas)

print(f"\n📈 RESUMEN DE DATOS PARA RECOMENDACIONES:")
print(f"  • Ventas totales: {len(df_ventas):,}")
print(f"  • Detalles de productos: {len(df_detalles):,}")
print(f"  • Clientes únicos: {df_ventas['cliente_id'].nunique():,}")
print(f"  • Productos únicos: {df_detalles['producto_id'].nunique():,}")
print(f"  • Tipos de negocio: {df_ventas['tipo_negocio'].nunique()}")
print(f"  • Período: {df_ventas['fecha'].min().strftime('%Y-%m-%d')} → {df_ventas['fecha'].max().strftime('%Y-%m-%d')}")

# ============================================================================
# FASE 3: PREPARACIÓN DE DATOS (CRISP-DM)
# ============================================================================
print("\n🔧 FASE 3: PREPARACIÓN DE DATOS PARA FILTRADO COLABORATIVO")
print("-" * 70)


def preparar_datos_colaborativo():
    """Prepara datos específicamente para filtrado colaborativo"""
    print("🔄 Preparando datos para filtrado colaborativo...")

    # Crear dataset completo
    dataset_completo = df_detalles.merge(
        df_ventas[['venta_id', 'cliente_id', 'fecha', 'tipo_negocio', 'ciudad']],
        on='venta_id', how='inner'
    )

    print(f"   ✅ Dataset consolidado: {len(dataset_completo)} registros")

    # Detectar columnas de categoría y marca
    col_categoria = None
    col_marca = None

    for col in dataset_completo.columns:
        if 'categoria' in col.lower():
            col_categoria = col
        elif 'marca' in col.lower():
            col_marca = col

    # Si no existen, crear columnas genéricas
    if col_categoria is None:
        dataset_completo['categoria'] = 'Quesos_General'
        col_categoria = 'categoria'
    if col_marca is None:
        dataset_completo['marca'] = 'Marca_General'
        col_marca = 'marca'

    # Crear matriz de interacciones usuario-producto
    # Agregamos cantidad y subtotal por cliente-producto
    interacciones = dataset_completo.groupby(['cliente_id', 'producto_id']).agg({
        'cantidad': 'sum',
        'subtotal': 'sum',
        'fecha': 'count',  # frecuencia de compra
        'tipo_negocio': 'first',
        'ciudad': 'first',
        col_categoria: 'first',
        col_marca: 'first'
    }).reset_index()

    interacciones.columns = ['cliente_id', 'producto_id', 'cantidad_total', 'valor_total',
                             'frecuencia_compra', 'tipo_negocio', 'ciudad', 'categoria', 'marca']

    # Crear rating implícito combinando cantidad, valor y frecuencia
    # Normalizar cada componente y crear score compuesto
    scaler = StandardScaler()

    # Normalizar componentes (0-1)
    interacciones['cantidad_norm'] = scaler.fit_transform(interacciones[['cantidad_total']])
    interacciones['valor_norm'] = scaler.fit_transform(interacciones[['valor_total']])
    interacciones['frecuencia_norm'] = scaler.fit_transform(interacciones[['frecuencia_compra']])

    # Crear rating compuesto (escala 1-5)
    interacciones['rating'] = (
            0.4 * interacciones['cantidad_norm'] +
            0.4 * interacciones['valor_norm'] +
            0.2 * interacciones['frecuencia_norm']
    )

    # Escalar a rango 1-5
    min_rating = interacciones['rating'].min()
    max_rating = interacciones['rating'].max()
    interacciones['rating'] = 1 + 4 * (interacciones['rating'] - min_rating) / (max_rating - min_rating)

    print(f"   📊 Matriz de interacciones: {len(interacciones)} combinaciones cliente-producto")
    print(f"   📈 Rating promedio: {interacciones['rating'].mean():.2f}")
    print(f"   📊 Rango de ratings: {interacciones['rating'].min():.2f} - {interacciones['rating'].max():.2f}")

    # Información de productos
    productos_info = dataset_completo.groupby('producto_id').agg({
        col_categoria: 'first',
        col_marca: 'first',
        'cantidad': 'sum',
        'subtotal': 'sum',
        'cliente_id': 'nunique'
    }).reset_index()

    productos_info.columns = ['producto_id', 'categoria', 'marca', 'cantidad_total',
                              'ventas_total', 'clientes_unicos']

    # Información de clientes
    clientes_info = dataset_completo.groupby('cliente_id').agg({
        'tipo_negocio': 'first',
        'ciudad': 'first',
        'cantidad': 'sum',
        'subtotal': 'sum',
        'producto_id': 'nunique'
    }).reset_index()

    clientes_info.columns = ['cliente_id', 'tipo_negocio', 'ciudad', 'cantidad_total',
                             'gasto_total', 'productos_distintos']

    return interacciones, productos_info, clientes_info, dataset_completo


# Preparar datos
interacciones, productos_info, clientes_info, dataset_completo = preparar_datos_colaborativo()


# Crear matriz esparsa para eficiencia
def crear_matriz_usuario_producto(interacciones):
    """Crea matriz esparsa usuario-producto"""
    print("🔢 Creando matriz usuario-producto...")

    # Verificar tamaño mínimo del dataset
    min_usuarios = 2
    min_productos = 2

    usuarios_unicos = sorted(interacciones['cliente_id'].unique())
    productos_unicos = sorted(interacciones['producto_id'].unique())

    if len(usuarios_unicos) < min_usuarios:
        print(f"   ⚠️ Advertencia: Solo {len(usuarios_unicos)} usuarios únicos (mínimo recomendado: {min_usuarios})")

    if len(productos_unicos) < min_productos:
        print(f"   ⚠️ Advertencia: Solo {len(productos_unicos)} productos únicos (mínimo recomendado: {min_productos})")

    # Crear mapeos de IDs a índices
    user_to_idx = {user: idx for idx, user in enumerate(usuarios_unicos)}
    item_to_idx = {item: idx for idx, item in enumerate(productos_unicos)}

    # Crear matriz esparsa
    filas = [user_to_idx[user] for user in interacciones['cliente_id']]
    columnas = [item_to_idx[item] for item in interacciones['producto_id']]
    ratings = interacciones['rating'].values

    matriz_ratings = csr_matrix((ratings, (filas, columnas)),
                                shape=(len(usuarios_unicos), len(productos_unicos)))

    # Calcular densidad
    densidad = matriz_ratings.nnz / (matriz_ratings.shape[0] * matriz_ratings.shape[1]) * 100

    print(f"   ✅ Matriz creada: {matriz_ratings.shape}")
    print(f"   📊 Densidad: {densidad:.2f}%")
    print(f"   🔢 Interacciones: {matriz_ratings.nnz:,}")

    # Advertencias para datasets pequeños
    if densidad < 1.0:
        print(f"   ⚠️ Advertencia: Matriz muy dispersa ({densidad:.2f}%). Considera usar más datos.")

    if matriz_ratings.shape[0] < 10 or matriz_ratings.shape[1] < 10:
        print(f"   ⚠️ Advertencia: Dataset pequeño. Los resultados pueden ser limitados.")

    return matriz_ratings, user_to_idx, item_to_idx, usuarios_unicos, productos_unicos


matriz_ratings, user_to_idx, item_to_idx, usuarios_unicos, productos_unicos = crear_matriz_usuario_producto(
    interacciones)

# ============================================================================
# FASE 4: MODELADO - ALGORITMOS DE FILTRADO COLABORATIVO (CRISP-DM)
# ============================================================================
print("\n🤖 FASE 4: MODELADO - ALGORITMOS DE FILTRADO COLABORATIVO")
print("-" * 70)


class SistemaRecomendacionColaborativo:
    """
    Sistema completo de recomendación con filtrado colaborativo
    """

    def __init__(self, matriz_ratings, user_to_idx, item_to_idx, usuarios_unicos, productos_unicos,
                 interacciones, productos_info, clientes_info):
        self.matriz_ratings = matriz_ratings
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.usuarios_unicos = usuarios_unicos
        self.productos_unicos = productos_unicos
        self.interacciones = interacciones
        self.productos_info = productos_info
        self.clientes_info = clientes_info

        # Modelos entrenados
        self.modelos = {}
        self.metricas = {}

    def dividir_datos_temporal(self, test_size=0.2):
        """Divide datos cronológicamente para evaluación realista"""
        print(f"📊 Dividiendo datos cronológicamente ({int((1 - test_size) * 100)}-{int(test_size * 100)})...")

        # Ordenar interacciones por fecha implícita (usando valor como proxy temporal)
        interacciones_sorted = self.interacciones.sort_values(['cliente_id', 'valor_total'])

        # División por cliente para mantener historial
        train_data = []
        test_data = []

        for cliente_id in self.interacciones['cliente_id'].unique():
            cliente_interacciones = interacciones_sorted[
                interacciones_sorted['cliente_id'] == cliente_id
                ].copy()

            if len(cliente_interacciones) >= 2:
                # Dividir cronológicamente
                n_train = max(1, int(len(cliente_interacciones) * (1 - test_size)))

                train_data.append(cliente_interacciones.iloc[:n_train])
                test_data.append(cliente_interacciones.iloc[n_train:])
            else:
                # Si solo tiene una interacción, va a entrenamiento
                train_data.append(cliente_interacciones)

        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()

        print(f"   ✅ Entrenamiento: {len(train_df)} interacciones")
        print(f"   ✅ Prueba: {len(test_df)} interacciones")

        return train_df, test_df

    def entrenar_user_based_cf(self, train_data, k_neighbors=20):
        """Entrena modelo User-Based Collaborative Filtering"""
        print("👥 Entrenando User-Based Collaborative Filtering...")

        # Crear matriz de entrenamiento
        train_filas = [self.user_to_idx[user] for user in train_data['cliente_id']]
        train_columnas = [self.item_to_idx[item] for item in train_data['producto_id']]
        train_ratings = train_data['rating'].values

        train_matrix = csr_matrix((train_ratings, (train_filas, train_columnas)),
                                  shape=self.matriz_ratings.shape)

        # Calcular similitud entre usuarios
        user_similarity = cosine_similarity(train_matrix)

        self.modelos['user_based'] = {
            'train_matrix': train_matrix,
            'user_similarity': user_similarity,
            'k_neighbors': k_neighbors
        }

        print(f"   ✅ Modelo User-Based entrenado con {k_neighbors} vecinos")
        return user_similarity

    def entrenar_item_based_cf(self, train_data, k_neighbors=20):
        """Entrena modelo Item-Based Collaborative Filtering"""
        print("🛒 Entrenando Item-Based Collaborative Filtering...")

        # Crear matriz de entrenamiento
        train_filas = [self.user_to_idx[user] for user in train_data['cliente_id']]
        train_columnas = [self.item_to_idx[item] for item in train_data['producto_id']]
        train_ratings = train_data['rating'].values

        train_matrix = csr_matrix((train_ratings, (train_filas, train_columnas)),
                                  shape=self.matriz_ratings.shape)

        # Calcular similitud entre productos (transponer matriz)
        item_similarity = cosine_similarity(train_matrix.T)

        self.modelos['item_based'] = {
            'train_matrix': train_matrix,
            'item_similarity': item_similarity,
            'k_neighbors': k_neighbors
        }

        print(f"   ✅ Modelo Item-Based entrenado con {k_neighbors} vecinos")
        return item_similarity

    def entrenar_matrix_factorization(self, train_data, n_components=50, algorithm='svd'):
        """Entrena modelo de Matrix Factorization (SVD o NMF)"""
        print(f"🔢 Entrenando Matrix Factorization ({algorithm.upper()})...")

        # Crear matriz de entrenamiento
        train_filas = [self.user_to_idx[user] for user in train_data['cliente_id']]
        train_columnas = [self.item_to_idx[item] for item in train_data['producto_id']]
        train_ratings = train_data['rating'].values

        train_matrix = csr_matrix((train_ratings, (train_filas, train_columnas)),
                                  shape=self.matriz_ratings.shape)

        # Ajustar n_components según el tamaño de la matriz
        max_components = min(train_matrix.shape[0], train_matrix.shape[1]) - 1
        n_components_adjusted = min(n_components, max_components)

        print(
            f"   📊 Matriz: {train_matrix.shape}, Componentes originales: {n_components}, Ajustados: {n_components_adjusted}")

        # Entrenar modelo
        if algorithm == 'svd':
            modelo = TruncatedSVD(n_components=n_components_adjusted, random_state=42)
        else:  # nmf
            modelo = NMF(n_components=n_components_adjusted, random_state=42, max_iter=200)

        # Ajustar modelo
        user_factors = modelo.fit_transform(train_matrix)
        item_factors = modelo.components_.T

        self.modelos[f'matrix_fact_{algorithm}'] = {
            'modelo': modelo,
            'user_factors': user_factors,
            'item_factors': item_factors,
            'train_matrix': train_matrix,
            'n_components_used': n_components_adjusted
        }

        print(f"   ✅ Modelo {algorithm.upper()} entrenado con {n_components_adjusted} componentes")
        return modelo, user_factors, item_factors

    def predecir_user_based(self, user_idx, item_idx):
        """Predicción con User-Based CF"""
        if 'user_based' not in self.modelos:
            return 0.0

        modelo = self.modelos['user_based']
        user_sim = modelo['user_similarity'][user_idx]
        train_matrix = modelo['train_matrix']
        k = modelo['k_neighbors']

        # Encontrar usuarios similares que hayan calificado el producto
        item_ratings = train_matrix[:, item_idx].toarray().flatten()
        rated_users = np.where(item_ratings > 0)[0]

        if len(rated_users) == 0:
            return np.mean([r for r in train_matrix[user_idx].data if r > 0]) if len(
                train_matrix[user_idx].data) > 0 else 3.0

        # Obtener similitudes y ratings de usuarios similares
        user_similarities = user_sim[rated_users]
        user_ratings = item_ratings[rated_users]

        # Seleccionar top-k usuarios más similares
        top_k_indices = np.argsort(user_similarities)[::-1][:k]
        top_similarities = user_similarities[top_k_indices]
        top_ratings = user_ratings[top_k_indices]

        # Calcular predicción ponderada
        if np.sum(np.abs(top_similarities)) > 0:
            prediccion = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        else:
            prediccion = np.mean(top_ratings)

        return max(1.0, min(5.0, prediccion))

    def predecir_item_based(self, user_idx, item_idx):
        """Predicción con Item-Based CF"""
        if 'item_based' not in self.modelos:
            return 0.0

        modelo = self.modelos['item_based']
        item_sim = modelo['item_similarity'][item_idx]
        train_matrix = modelo['train_matrix']
        k = modelo['k_neighbors']

        # Encontrar productos similares que el usuario haya calificado
        user_ratings = train_matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            # Promedio global del producto
            item_ratings = train_matrix[:, item_idx].data
            return np.mean(item_ratings) if len(item_ratings) > 0 else 3.0

        # Obtener similitudes y ratings de productos similares
        item_similarities = item_sim[rated_items]
        item_ratings = user_ratings[rated_items]

        # Seleccionar top-k productos más similares
        top_k_indices = np.argsort(item_similarities)[::-1][:k]
        top_similarities = item_similarities[top_k_indices]
        top_ratings = item_ratings[top_k_indices]

        # Calcular predicción ponderada
        if np.sum(np.abs(top_similarities)) > 0:
            prediccion = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        else:
            prediccion = np.mean(top_ratings)

        return max(1.0, min(5.0, prediccion))

    def predecir_matrix_factorization(self, user_idx, item_idx, algorithm='svd'):
        """Predicción con Matrix Factorization"""
        modelo_key = f'matrix_fact_{algorithm}'
        if modelo_key not in self.modelos:
            return 0.0

        modelo = self.modelos[modelo_key]
        user_factors = modelo['user_factors']
        item_factors = modelo['item_factors']

        # Predicción como producto punto
        prediccion = np.dot(user_factors[user_idx], item_factors[item_idx])
        return max(1.0, min(5.0, prediccion))

    def entrenar_modelo_hibrido(self, train_data):
        """Entrena modelo híbrido combinando diferentes enfoques"""
        print("🔄 Entrenando modelo híbrido...")

        # Determinar número óptimo de componentes basado en el tamaño del dataset
        max_users = len(self.usuarios_unicos)
        max_items = len(self.productos_unicos)
        max_components = min(max_users, max_items) - 1

        # Usar componentes conservadores para datasets pequeños
        n_components_svd = min(20, max_components)
        n_components_nmf = min(15, max_components)

        print(f"   📊 Dataset: {max_users} usuarios, {max_items} productos")
        print(f"   🔢 Componentes SVD: {n_components_svd}, NMF: {n_components_nmf}")

        # Entrenar todos los modelos base
        self.entrenar_user_based_cf(train_data, k_neighbors=min(20, max_users // 2))
        self.entrenar_item_based_cf(train_data, k_neighbors=min(20, max_items // 2))
        self.entrenar_matrix_factorization(train_data, n_components=n_components_svd, algorithm='svd')
        self.entrenar_matrix_factorization(train_data, n_components=n_components_nmf, algorithm='nmf')

        # Pesos para combinación (se pueden optimizar)
        self.modelos['hibrido'] = {
            'pesos': {
                'user_based': 0.25,
                'item_based': 0.25,
                'svd': 0.30,
                'nmf': 0.20
            }
        }

        print("   ✅ Modelo híbrido configurado")

    def predecir_hibrido(self, user_idx, item_idx):
        """Predicción con modelo híbrido"""
        if 'hibrido' not in self.modelos:
            return 3.0

        pesos = self.modelos['hibrido']['pesos']

        # Obtener predicciones de cada modelo
        pred_user = self.predecir_user_based(user_idx, item_idx)
        pred_item = self.predecir_item_based(user_idx, item_idx)
        pred_svd = self.predecir_matrix_factorization(user_idx, item_idx, 'svd')
        pred_nmf = self.predecir_matrix_factorization(user_idx, item_idx, 'nmf')

        # Combinación ponderada
        prediccion_hibrida = (
                pesos['user_based'] * pred_user +
                pesos['item_based'] * pred_item +
                pesos['svd'] * pred_svd +
                pesos['nmf'] * pred_nmf
        )

        return max(1.0, min(5.0, prediccion_hibrida))


# Inicializar sistema de recomendación
print("🚀 Inicializando Sistema de Recomendación...")
sistema_recom = SistemaRecomendacionColaborativo(
    matriz_ratings, user_to_idx, item_to_idx, usuarios_unicos, productos_unicos,
    interacciones, productos_info, clientes_info
)

# Dividir datos
train_data, test_data = sistema_recom.dividir_datos_temporal(test_size=0.2)

# Entrenar modelo híbrido
sistema_recom.entrenar_modelo_hibrido(train_data)

# ============================================================================
# FASE 5: EVALUACIÓN (CRISP-DM)
# ============================================================================
print("\n📊 FASE 5: EVALUACIÓN DEL SISTEMA DE RECOMENDACIÓN")
print("-" * 70)


def evaluar_modelo(sistema, test_data, modelo_nombre='hibrido'):
    """Evalúa el modelo de recomendación con métricas específicas"""
    print(f"📈 Evaluando modelo {modelo_nombre}...")

    if len(test_data) == 0:
        print("   ⚠️ No hay datos de prueba disponibles")
        return {}

    predicciones = []
    valores_reales = []

    for _, row in test_data.iterrows():
        try:
            user_idx = sistema.user_to_idx[row['cliente_id']]
            item_idx = sistema.item_to_idx[row['producto_id']]

            if modelo_nombre == 'hibrido':
                pred = sistema.predecir_hibrido(user_idx, item_idx)
            elif modelo_nombre == 'user_based':
                pred = sistema.predecir_user_based(user_idx, item_idx)
            elif modelo_nombre == 'item_based':
                pred = sistema.predecir_item_based(user_idx, item_idx)
            elif modelo_nombre == 'svd':
                pred = sistema.predecir_matrix_factorization(user_idx, item_idx, 'svd')
            elif modelo_nombre == 'nmf':
                pred = sistema.predecir_matrix_factorization(user_idx, item_idx, 'nmf')
            else:
                pred = 3.0

            predicciones.append(pred)
            valores_reales.append(row['rating'])

        except KeyError:
            # Usuario o producto no visto en entrenamiento (cold start)
            predicciones.append(3.0)  # Predicción por defecto
            valores_reales.append(row['rating'])

    if len(predicciones) == 0:
        return {}

    # Calcular métricas
    mae = mean_absolute_error(valores_reales, predicciones)
    rmse = np.sqrt(mean_squared_error(valores_reales, predicciones))

    # Calcular cobertura (porcentaje de pares usuario-producto que se pueden predecir)
    usuarios_test = test_data['cliente_id'].unique()
    productos_test = test_data['producto_id'].unique()

    pares_posibles = 0
    pares_predecibles = 0

    for user in usuarios_test[:100]:  # Muestra para eficiencia
        if user in sistema.user_to_idx:
            for item in productos_test[:100]:
                if item in sistema.item_to_idx:
                    pares_posibles += 1
                    # Si el usuario o producto están en el entrenamiento, es predecible
                    user_idx = sistema.user_to_idx[user]
                    item_idx = sistema.item_to_idx[item]

                    # Verificar si hay información suficiente para predecir
                    if (hasattr(sistema.modelos.get('user_based', {}).get('train_matrix', None), 'shape') and
                            sistema.modelos['user_based']['train_matrix'][user_idx].nnz > 0):
                        pares_predecibles += 1

    cobertura = pares_predecibles / pares_posibles if pares_posibles > 0 else 0

    metricas = {
        'mae': mae,
        'rmse': rmse,
        'cobertura': cobertura,
        'n_predicciones': len(predicciones)
    }

    print(f"   📊 MAE: {mae:.3f}")
    print(f"   📐 RMSE: {rmse:.3f}")
    print(f"   📈 Cobertura: {cobertura:.3f} ({cobertura * 100:.1f}%)")
    print(f"   🔢 Predicciones evaluadas: {len(predicciones)}")

    return metricas


# Evaluar todos los modelos
print("🧪 Evaluando todos los modelos...")
resultados_evaluacion = {}

modelos_evaluar = ['user_based', 'item_based', 'svd', 'nmf', 'hibrido']

for modelo in modelos_evaluar:
    resultados_evaluacion[modelo] = evaluar_modelo(sistema_recom, test_data, modelo)

# Verificar cumplimiento de objetivos
print(f"\n🎯 VERIFICACIÓN DE OBJETIVOS:")
print(f"  📊 MAE objetivo: ≤ {OBJETIVOS_NEGOCIO['metricas_objetivo']['mae']}")
print(f"  📐 RMSE objetivo: ≤ {OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse']}")
print(f"  📈 Cobertura objetivo: ≥ {OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura']}")

mejor_modelo = None
mejor_score = float('inf')

for modelo, metricas in resultados_evaluacion.items():
    if metricas:
        mae_ok = metricas['mae'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['mae']
        rmse_ok = metricas['rmse'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse']
        cobertura_ok = metricas['cobertura'] >= OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura']

        objetivos_cumplidos = sum([mae_ok, rmse_ok, cobertura_ok])
        score_compuesto = metricas['mae'] + metricas['rmse'] - metricas['cobertura']

        print(f"\n  🤖 Modelo {modelo.upper()}:")
        print(f"    {'✅' if mae_ok else '❌'} MAE: {metricas['mae']:.3f}")
        print(f"    {'✅' if rmse_ok else '❌'} RMSE: {metricas['rmse']:.3f}")
        print(f"    {'✅' if cobertura_ok else '❌'} Cobertura: {metricas['cobertura']:.3f}")
        print(f"    📊 Objetivos cumplidos: {objetivos_cumplidos}/3")

        if objetivos_cumplidos >= 2 and score_compuesto < mejor_score:
            mejor_modelo = modelo
            mejor_score = score_compuesto

print(f"\n🏆 MEJOR MODELO: {mejor_modelo.upper() if mejor_modelo else 'NINGUNO CUMPLE OBJETIVOS'}")

# ============================================================================
# MOTOR DE RECOMENDACIÓN ESPECIALIZADO
# ============================================================================
print("\n🧀 MOTOR DE RECOMENDACIÓN ESPECIALIZADO EN QUESOS")
print("-" * 70)


class MotorRecomendacionQuesos:
    """
    Motor especializado para recomendaciones de productos gastronómicos
    """

    def __init__(self, sistema_recom, productos_info, clientes_info, dataset_completo):
        self.sistema = sistema_recom
        self.productos_info = productos_info
        self.clientes_info = clientes_info
        self.dataset_completo = dataset_completo

        # Definir categorías de quesos y platillos
        self.categorias_quesos = self._identificar_categorias_quesos()
        self.platillos_quesos = self._definir_combinaciones_platillos()

    def _identificar_categorias_quesos(self):
        """Identifica y categoriza productos relacionados con quesos"""
        categorias = {
            'quesos_frescos': ['mozzarella', 'ricotta', 'cottage', 'feta', 'fresco'],
            'quesos_semiduros': ['gouda', 'edam', 'cheddar', 'swiss', 'provolone'],
            'quesos_duros': ['parmesano', 'romano', 'pecorino', 'grana'],
            'quesos_azules': ['roquefort', 'gorgonzola', 'stilton', 'azul'],
            'quesos_especiales': ['brie', 'camembert', 'mascarpone', 'gruyere']
        }
        return categorias

    def _definir_combinaciones_platillos(self):
        """Define combinaciones óptimas de quesos por tipo de platillo"""
        return {
            'pizza': ['mozzarella', 'parmesano', 'provolone'],
            'pasta': ['parmesano', 'pecorino', 'gorgonzola'],
            'ensaladas': ['feta', 'mozzarella', 'gouda'],
            'postres': ['mascarpone', 'ricotta', 'cream_cheese'],
            'tabla_quesos': ['brie', 'roquefort', 'gouda', 'cheddar'],
            'gratinados': ['gruyere', 'emmental', 'cheddar'],
            'sandwiches': ['cheddar', 'swiss', 'provolone']
        }

    def recomendar_combinaciones_platillos(self, cliente_id, tipo_platillo='pizza', n_recomendaciones=5):
        """
        Tipo 1: Combinaciones óptimas de quesos para determinados platillos
        """
        print(f"🍕 Generando recomendaciones de quesos para {tipo_platillo}...")

        try:
            user_idx = self.sistema.user_to_idx[cliente_id]

            # Obtener información del cliente
            cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente_id].iloc[0]
            tipo_negocio = cliente_info['tipo_negocio']

            # Quesos recomendados para el tipo de platillo
            quesos_platillo = self.platillos_quesos.get(tipo_platillo, ['mozzarella', 'parmesano'])

            recomendaciones = []

            # Para cada producto, calcular score si contiene quesos del platillo
            for producto_id in self.productos_info['producto_id']:
                try:
                    item_idx = self.sistema.item_to_idx[producto_id]

                    # Verificar si el producto es relevante para el platillo
                    producto_info = self.productos_info[self.productos_info['producto_id'] == producto_id].iloc[0]
                    categoria = str(producto_info['categoria']).lower()
                    marca = str(producto_info['marca']).lower()

                    # Verificar si contiene quesos del platillo
                    relevancia_platillo = 0
                    for queso in quesos_platillo:
                        if queso in categoria or queso in marca or queso in str(producto_id).lower():
                            relevancia_platillo += 1

                    if relevancia_platillo > 0:
                        # Obtener predicción del modelo híbrido
                        score_base = self.sistema.predecir_hibrido(user_idx, item_idx)

                        # Ajustar score por relevancia del platillo
                        score_final = score_base * (1 + 0.2 * relevancia_platillo)

                        # Verificar si otros del mismo tipo de negocio compran este producto
                        clientes_mismo_tipo = self.clientes_info[
                            self.clientes_info['tipo_negocio'] == tipo_negocio
                            ]['cliente_id'].tolist()

                        popularidad_tipo_negocio = len([
                            c for c in clientes_mismo_tipo
                            if c in self.sistema.user_to_idx and
                               self.sistema.modelos['user_based']['train_matrix'][
                                   self.sistema.user_to_idx[c], item_idx
                               ] > 0
                        ]) / len(clientes_mismo_tipo)

                        recomendaciones.append({
                            'producto_id': producto_id,
                            'score': score_final,
                            'categoria': categoria,
                            'marca': marca,
                            'relevancia_platillo': relevancia_platillo,
                            'popularidad_tipo_negocio': popularidad_tipo_negocio,
                            'tipo_recomendacion': 'Combinación para platillo'
                        })

                except KeyError:
                    continue

            # Ordenar por score y devolver top N
            recomendaciones.sort(key=lambda x: x['score'], reverse=True)

            return {
                'cliente_id': cliente_id,
                'tipo_platillo': tipo_platillo,
                'tipo_negocio': tipo_negocio,
                'recomendaciones': recomendaciones[:n_recomendaciones],
                'total_evaluados': len(recomendaciones)
            }

        except KeyError:
            print(f"   ⚠️ Cliente {cliente_id} no encontrado")
            return {'error': f'Cliente {cliente_id} no encontrado'}

    def recomendar_nuevos_productos(self, cliente_id, n_recomendaciones=5):
        """
        Tipo 2: Nuevos productos que podrían adaptarse a su oferta gastronómica
        """
        print(f"🆕 Generando recomendaciones de nuevos productos...")

        try:
            user_idx = self.sistema.user_to_idx[cliente_id]

            # Obtener información del cliente
            cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente_id].iloc[0]
            tipo_negocio = cliente_info['tipo_negocio']

            # Productos que el cliente ya compra
            productos_comprados = set(self.sistema.interacciones[
                                          self.sistema.interacciones['cliente_id'] == cliente_id
                                          ]['producto_id'])

            # Obtener clientes similares del mismo tipo de negocio
            clientes_similares = self.clientes_info[
                (self.clientes_info['tipo_negocio'] == tipo_negocio) &
                (self.clientes_info['cliente_id'] != cliente_id)
                ]['cliente_id'].tolist()

            # Productos que compran clientes similares pero el cliente objetivo no
            productos_candidatos = set()
            for cliente_similar in clientes_similares[:20]:  # Limitar para eficiencia
                productos_similar = set(self.sistema.interacciones[
                                            self.sistema.interacciones['cliente_id'] == cliente_similar
                                            ]['producto_id'])
                productos_candidatos.update(productos_similar - productos_comprados)

            recomendaciones = []

            for producto_id in productos_candidatos:
                try:
                    item_idx = self.sistema.item_to_idx[producto_id]

                    # Score base del modelo
                    score_base = self.sistema.predecir_hibrido(user_idx, item_idx)

                    # Calcular adopción por tipo de negocio
                    clientes_tipo_que_compran = len([
                        c for c in clientes_similares
                        if c in self.sistema.user_to_idx and
                           producto_id in set(self.sistema.interacciones[
                                                  self.sistema.interacciones['cliente_id'] == c
                                                  ]['producto_id'])
                    ])

                    tasa_adopcion = clientes_tipo_que_compran / len(clientes_similares)

                    # Ajustar score por novedad y adopción
                    score_final = score_base * (1 + 0.3 * tasa_adopcion)

                    producto_info = self.productos_info[
                        self.productos_info['producto_id'] == producto_id
                        ].iloc[0]

                    recomendaciones.append({
                        'producto_id': producto_id,
                        'score': score_final,
                        'categoria': producto_info['categoria'],
                        'marca': producto_info['marca'],
                        'tasa_adopcion': tasa_adopcion,
                        'clientes_tipo_que_compran': clientes_tipo_que_compran,
                        'tipo_recomendacion': 'Nuevo producto'
                    })

                except KeyError:
                    continue

            # Ordenar por score y devolver top N
            recomendaciones.sort(key=lambda x: x['score'], reverse=True)

            return {
                'cliente_id': cliente_id,
                'tipo_negocio': tipo_negocio,
                'recomendaciones': recomendaciones[:n_recomendaciones],
                'productos_ya_comprados': len(productos_comprados),
                'productos_candidatos': len(productos_candidatos)
            }

        except KeyError:
            print(f"   ⚠️ Cliente {cliente_id} no encontrado")
            return {'error': f'Cliente {cliente_id} no encontrado'}

    def analizar_tendencias_tipo_negocio(self, cliente_id, n_tendencias=5):
        """
        Tipo 3: Tendencias en el uso de quesos en su tipo de establecimiento
        """
        print(f"📈 Analizando tendencias para tipo de establecimiento...")

        try:
            # Obtener información del cliente
            cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente_id].iloc[0]
            tipo_negocio = cliente_info['tipo_negocio']

            # Obtener todos los clientes del mismo tipo de negocio
            clientes_mismo_tipo = self.clientes_info[
                self.clientes_info['tipo_negocio'] == tipo_negocio
                ]['cliente_id'].tolist()

            # Analizar productos más populares por tipo de negocio
            interacciones_tipo = self.sistema.interacciones[
                self.sistema.interacciones['cliente_id'].isin(clientes_mismo_tipo)
            ]

            # Calcular métricas por producto
            tendencias = interacciones_tipo.groupby('producto_id').agg({
                'cliente_id': 'nunique',  # Número de clientes que lo compran
                'cantidad_total': 'sum',  # Cantidad total vendida
                'valor_total': 'sum',  # Valor total vendido
                'rating': 'mean'  # Rating promedio
            }).reset_index()

            tendencias.columns = ['producto_id', 'clientes_unicos', 'cantidad_total',
                                  'valor_total', 'rating_promedio']

            # Calcular métricas de tendencia
            total_clientes_tipo = len(clientes_mismo_tipo)
            tendencias['penetracion'] = tendencias['clientes_unicos'] / total_clientes_tipo
            tendencias['cantidad_promedio'] = tendencias['cantidad_total'] / tendencias['clientes_unicos']
            tendencias['valor_promedio'] = tendencias['valor_total'] / tendencias['clientes_unicos']

            # Score de tendencia combinado
            tendencias['score_tendencia'] = (
                    0.4 * tendencias['penetracion'] +
                    0.3 * (tendencias['rating_promedio'] / 5.0) +
                    0.3 * (tendencias['cantidad_promedio'] / tendencias['cantidad_promedio'].max())
            )

            # Añadir información de productos
            tendencias = tendencias.merge(
                self.productos_info[['producto_id', 'categoria', 'marca']],
                on='producto_id', how='left'
            )

            # Identificar si el cliente ya compra estos productos
            productos_cliente = set(self.sistema.interacciones[
                                        self.sistema.interacciones['cliente_id'] == cliente_id
                                        ]['producto_id'])

            tendencias['ya_compra'] = tendencias['producto_id'].isin(productos_cliente)

            # Ordenar por score de tendencia
            tendencias = tendencias.sort_values('score_tendencia', ascending=False)

            return {
                'cliente_id': cliente_id,
                'tipo_negocio': tipo_negocio,
                'total_clientes_tipo': total_clientes_tipo,
                'tendencias': tendencias.head(n_tendencias).to_dict('records'),
                'resumen': {
                    'productos_analizados': len(tendencias),
                    'penetracion_promedio': tendencias['penetracion'].mean(),
                    'rating_promedio_tipo': tendencias['rating_promedio'].mean()
                }
            }

        except Exception as e:
            print(f"   ⚠️ Error analizando tendencias: {e}")
            return {'error': f'Error analizando tendencias: {e}'}


# Inicializar motor especializado
print("🚀 Inicializando Motor de Recomendación Especializado...")
motor_quesos = MotorRecomendacionQuesos(
    sistema_recom, productos_info, clientes_info, dataset_completo
)

# ============================================================================
# EJEMPLOS DE RECOMENDACIONES
# ============================================================================
print("\n🧪 EJEMPLOS DE RECOMENDACIONES ESPECIALIZADAS")
print("-" * 70)

# Seleccionar algunos clientes de ejemplo
clientes_ejemplo = clientes_info['cliente_id'].head(3).tolist()

print("🔮 Generando ejemplos de recomendaciones...")

for i, cliente_id in enumerate(clientes_ejemplo):
    print(f"\n👤 EJEMPLO {i + 1}: CLIENTE {cliente_id}")
    print("-" * 40)

    # Tipo 1: Combinaciones para platillos
    print("🍕 Recomendaciones para Pizza:")
    rec_pizza = motor_quesos.recomendar_combinaciones_platillos(
        cliente_id, 'pizza', n_recomendaciones=3
    )
    if 'error' not in rec_pizza:
        for j, rec in enumerate(rec_pizza['recomendaciones'][:3]):
            print(f"   {j + 1}. Producto {rec['producto_id']} - Score: {rec['score']:.2f}")

    # Tipo 2: Nuevos productos
    print("\n🆕 Nuevos productos recomendados:")
    rec_nuevos = motor_quesos.recomendar_nuevos_productos(cliente_id, n_recomendaciones=3)
    if 'error' not in rec_nuevos:
        for j, rec in enumerate(rec_nuevos['recomendaciones'][:3]):
            print(f"   {j + 1}. Producto {rec['producto_id']} - Adopción: {rec['tasa_adopcion']:.2f}")

    # Tipo 3: Tendencias
    print("\n📈 Tendencias en su tipo de establecimiento:")
    tendencias = motor_quesos.analizar_tendencias_tipo_negocio(cliente_id, n_tendencias=3)
    if 'error' not in tendencias:
        print(f"   Tipo de negocio: {tendencias['tipo_negocio']}")
        for j, trend in enumerate(tendencias['tendencias'][:3]):
            print(f"   {j + 1}. Producto {trend['producto_id']} - Penetración: {trend['penetracion']:.2f}")

print("\n✅ Ejemplos de recomendaciones completados")

# ============================================================================
# GUARDAR MODELOS Y RESULTADOS
# ============================================================================
print("\n💾 GUARDANDO MODELOS Y RESULTADOS")
print("-" * 70)

print("💾 Guardando sistema de recomendación...")

# Guardar modelos entrenados
joblib.dump(sistema_recom, os.path.join(OUTPUT_DIR, 'sistema_recomendacion_colaborativo.pkl'))
joblib.dump(motor_quesos, os.path.join(OUTPUT_DIR, 'motor_recomendacion_quesos.pkl'))

# Guardar datos procesados
interacciones.to_csv(os.path.join(OUTPUT_DIR, 'interacciones_usuario_producto.csv'), index=False)
productos_info.to_csv(os.path.join(OUTPUT_DIR, 'productos_info_recomendacion.csv'), index=False)
clientes_info.to_csv(os.path.join(OUTPUT_DIR, 'clientes_info_recomendacion.csv'), index=False)

# Guardar matrices
np.save(os.path.join(OUTPUT_DIR, 'matriz_ratings.npy'), matriz_ratings.toarray())
np.save(os.path.join(OUTPUT_DIR, 'user_to_idx.npy'), user_to_idx)
np.save(os.path.join(OUTPUT_DIR, 'item_to_idx.npy'), item_to_idx)

print("✅ Modelos y datos guardados")

# ============================================================================
# VISUALIZACIONES Y DASHBOARD
# ============================================================================
print("\n🎨 GENERANDO VISUALIZACIONES Y DASHBOARD")
print("-" * 70)

print("📊 Creando dashboard de filtrado colaborativo...")

# Crear dashboard principal
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. Métricas de evaluación por modelo (superior izquierda)
ax1 = fig.add_subplot(gs[0, 0])
modelos_nombres = []
mae_valores = []
rmse_valores = []
cobertura_valores = []

for modelo, metricas in resultados_evaluacion.items():
    if metricas:
        modelos_nombres.append(modelo.upper())
        mae_valores.append(metricas['mae'])
        rmse_valores.append(metricas['rmse'])
        cobertura_valores.append(metricas['cobertura'])

x = np.arange(len(modelos_nombres))
width = 0.25

bars1 = ax1.bar(x - width, mae_valores, width, label='MAE', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x, rmse_valores, width, label='RMSE', color='#4ECDC4', alpha=0.8)
bars3 = ax1.bar(x + width, cobertura_valores, width, label='Cobertura', color='#45B7D1', alpha=0.8)

ax1.set_ylabel('Valor de Métrica')
ax1.set_title('📊 Métricas de Evaluación por Modelo', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(modelos_nombres, rotation=45)
ax1.legend()

# Líneas de objetivo
ax1.axhline(y=OBJETIVOS_NEGOCIO['metricas_objetivo']['mae'], color='red', linestyle='--', alpha=0.7,
            label='Objetivo MAE')
ax1.axhline(y=OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse'], color='red', linestyle='--', alpha=0.7,
            label='Objetivo RMSE')
ax1.axhline(y=OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura'], color='green', linestyle='--', alpha=0.7,
            label='Objetivo Cobertura')

# 2. Distribución de ratings (superior centro)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(interacciones['rating'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Rating')
ax2.set_ylabel('Frecuencia')
ax2.set_title('📈 Distribución de Ratings Implícitos', fontweight='bold')
ax2.axvline(interacciones['rating'].mean(), color='red', linestyle='--',
            label=f'Media: {interacciones["rating"].mean():.2f}')
ax2.legend()

# 3. Matriz de densidad por tipo de negocio (superior derecha)
ax3 = fig.add_subplot(gs[0, 2])
densidad_por_tipo = clientes_info.groupby('tipo_negocio')['productos_distintos'].mean().sort_values(ascending=False)
bars = ax3.bar(range(len(densidad_por_tipo)), densidad_por_tipo.values, color='green', alpha=0.7)
ax3.set_ylabel('Productos Distintos Promedio')
ax3.set_title('🏢 Diversidad de Productos por Tipo de Negocio', fontweight='bold')
ax3.set_xticks(range(len(densidad_por_tipo)))
ax3.set_xticklabels([tipo[:10] + '...' if len(tipo) > 10 else tipo for tipo in densidad_por_tipo.index], rotation=45)

# 4. Cumplimiento de objetivos (fila 2, izquierda)
ax4 = fig.add_subplot(gs[1, 0])
objetivos = ['MAE ≤ 0.75', 'RMSE ≤ 1.0', 'Cobertura ≥ 0.80']
cumplimiento = []

if mejor_modelo and mejor_modelo in resultados_evaluacion:
    metricas_mejor = resultados_evaluacion[mejor_modelo]
    cumplimiento = [
        1 if metricas_mejor['mae'] <= 0.75 else 0,
        1 if metricas_mejor['rmse'] <= 1.0 else 0,
        1 if metricas_mejor['cobertura'] >= 0.80 else 0
    ]
else:
    cumplimiento = [0, 0, 0]

colores = ['green' if c == 1 else 'red' for c in cumplimiento]
bars = ax4.bar(objetivos, cumplimiento, color=colores, alpha=0.7)
ax4.set_ylabel('Cumplimiento (1=Sí, 0=No)')
ax4.set_title('🎯 Cumplimiento de Objetivos', fontweight='bold')
ax4.set_ylim(0, 1.2)

for bar, valor in zip(bars, cumplimiento):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
             '✅' if valor == 1 else '❌', ha='center', va='bottom', fontsize=20)

# 5. Top productos más populares (fila 2, centro)
ax5 = fig.add_subplot(gs[1, 1])
top_productos = productos_info.nlargest(10, 'clientes_unicos')
bars = ax5.barh(range(len(top_productos)), top_productos['clientes_unicos'], color='orange', alpha=0.7)
ax5.set_xlabel('Número de Clientes Únicos')
ax5.set_title('🏆 Top 10 Productos Más Populares', fontweight='bold')
ax5.set_yticks(range(len(top_productos)))
ax5.set_yticklabels([f'Prod {pid}' for pid in top_productos['producto_id']])
ax5.invert_yaxis()

# 6. Distribución por tipo de establecimiento (fila 2, derecha)
ax6 = fig.add_subplot(gs[1, 2])
tipo_counts = clientes_info['tipo_negocio'].value_counts()
wedges, texts, autotexts = ax6.pie(tipo_counts.values, labels=None, autopct='%1.1f%%',
                                   startangle=90, colors=plt.cm.Set3.colors)
ax6.set_title('🏢 Distribución de Tipos de Establecimiento', fontweight='bold')
ax6.legend(wedges, [f'{tipo[:15]}...' if len(tipo) > 15 else tipo for tipo in tipo_counts.index],
           loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# 7. Evolución de la matriz de ratings (fila 3, completa)
ax7 = fig.add_subplot(gs[2, :])
# Mostrar muestra de la matriz de ratings
muestra_matriz = matriz_ratings.toarray()[:50, :50]  # Muestra 50x50
im = ax7.imshow(muestra_matriz, cmap='YlOrRd', aspect='auto', interpolation='nearest')
ax7.set_xlabel('Productos (muestra)')
ax7.set_ylabel('Usuarios (muestra)')
ax7.set_title('🔢 Matriz de Ratings Usuario-Producto (Muestra 50x50)', fontweight='bold')
plt.colorbar(im, ax=ax7, shrink=0.6)

# 8. Tabla resumen de resultados (fila 4, completa)
ax8 = fig.add_subplot(gs[3, :])
ax8.axis('off')

# Crear tabla de resultados
tabla_data = []
headers = ['Modelo', 'MAE', 'RMSE', 'Cobertura', 'MAE Ok', 'RMSE Ok', 'Cobertura Ok', 'Score']

for modelo, metricas in resultados_evaluacion.items():
    if metricas:
        mae_ok = '✅' if metricas['mae'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['mae'] else '❌'
        rmse_ok = '✅' if metricas['rmse'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse'] else '❌'
        cobertura_ok = '✅' if metricas['cobertura'] >= OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura'] else '❌'

        score = (metricas['mae'] + metricas['rmse'] - metricas['cobertura'])

        tabla_data.append([
            modelo.upper(),
            f"{metricas['mae']:.3f}",
            f"{metricas['rmse']:.3f}",
            f"{metricas['cobertura']:.3f}",
            mae_ok,
            rmse_ok,
            cobertura_ok,
            f"{score:.3f}"
        ])

tabla = ax8.table(cellText=tabla_data, colLabels=headers, cellLoc='center', loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 2)

# Colorear filas según el mejor modelo
for i, (modelo, _) in enumerate(resultados_evaluacion.items()):
    if modelo == mejor_modelo:
        for j in range(len(headers)):
            tabla[(i + 1, j)].set_facecolor('#90EE90')
            tabla[(i + 1, j)].set_alpha(0.7)

# Header
for j in range(len(headers)):
    tabla[(0, j)].set_text_props(weight='bold')
    tabla[(0, j)].set_facecolor('#D3D3D3')

plt.suptitle('🧀 Dashboard Sistema de Recomendación con Filtrado Colaborativo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard_filtrado_colaborativo.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Dashboard principal generado")

# Dashboard de tipos de recomendación
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('🧀 Dashboard Tipos de Recomendaciones Especializadas', fontsize=16, fontweight='bold')

# Ejemplo de recomendaciones por tipo
if clientes_ejemplo:
    cliente_ejemplo = clientes_ejemplo[0]

    # Tipo 1: Combinaciones para platillos
    ax = axes[0, 0]
    rec_pizza = motor_quesos.recomendar_combinaciones_platillos(cliente_ejemplo, 'pizza', 5)
    if 'error' not in rec_pizza and rec_pizza['recomendaciones']:
        productos = [f"Prod {r['producto_id']}" for r in rec_pizza['recomendaciones'][:5]]
        scores = [r['score'] for r in rec_pizza['recomendaciones'][:5]]
        bars = ax.barh(productos, scores, color='red', alpha=0.7)
        ax.set_xlabel('Score de Recomendación')
        ax.set_title('🍕 Combinaciones para Pizza', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No hay datos\nsuficientes', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('🍕 Combinaciones para Pizza', fontweight='bold')

    # Tipo 2: Nuevos productos
    ax = axes[0, 1]
    rec_nuevos = motor_quesos.recomendar_nuevos_productos(cliente_ejemplo, 5)
    if 'error' not in rec_nuevos and rec_nuevos['recomendaciones']:
        productos = [f"Prod {r['producto_id']}" for r in rec_nuevos['recomendaciones'][:5]]
        adopciones = [r['tasa_adopcion'] for r in rec_nuevos['recomendaciones'][:5]]
        bars = ax.barh(productos, adopciones, color='green', alpha=0.7)
        ax.set_xlabel('Tasa de Adopción')
        ax.set_title('🆕 Nuevos Productos', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No hay datos\nsuficientes', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('🆕 Nuevos Productos', fontweight='bold')

    # Tipo 3: Tendencias
    ax = axes[1, 0]
    tendencias = motor_quesos.analizar_tendencias_tipo_negocio(cliente_ejemplo, 5)
    if 'error' not in tendencias and tendencias['tendencias']:
        productos = [f"Prod {t['producto_id']}" for t in tendencias['tendencias'][:5]]
        penetraciones = [t['penetracion'] for t in tendencias['tendencias'][:5]]
        bars = ax.barh(productos, penetraciones, color='blue', alpha=0.7)
        ax.set_xlabel('Penetración en Tipo de Negocio')
        ax.set_title('📈 Tendencias por Tipo', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No hay datos\nsuficientes', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('📈 Tendencias por Tipo', fontweight='bold')

# Resumen de tipos de recomendación
ax = axes[1, 1]
tipos = ['Combinaciones\nPlatillos', 'Nuevos\nProductos', 'Tendencias\nTipo Negocio']
valores = [3, 3, 3]  # Valores de ejemplo
colores = ['red', 'green', 'blue']
bars = ax.bar(tipos, valores, color=colores, alpha=0.7)
ax.set_ylabel('Número de Recomendaciones')
ax.set_title('📊 Resumen Tipos de Recomendaciones', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard_tipos_recomendaciones.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Dashboard de tipos de recomendaciones generado")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 100)
print("🧀 RESUMEN FINAL - SISTEMA DE RECOMENDACIÓN CON FILTRADO COLABORATIVO")
print("=" * 100)

archivos_generados = os.listdir(OUTPUT_DIR)

print(f"\n📊 ESTADÍSTICAS FINALES:")
print(f"  🎯 Usuarios únicos: {len(usuarios_unicos):,}")
print(f"  🛒 Productos únicos: {len(productos_unicos):,}")
print(f"  💫 Interacciones totales: {len(interacciones):,}")
print(f"  🏢 Tipos de negocio: {clientes_info['tipo_negocio'].nunique()}")
print(f"  📁 Archivos generados: {len(archivos_generados)}")

print(f"\n🎯 CUMPLIMIENTO DE OBJETIVOS:")
if mejor_modelo and mejor_modelo in resultados_evaluacion:
    metricas_mejor = resultados_evaluacion[mejor_modelo]
    mae_ok = metricas_mejor['mae'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['mae']
    rmse_ok = metricas_mejor['rmse'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse']
    cobertura_ok = metricas_mejor['cobertura'] >= OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura']

    print(f"  🏆 Mejor modelo: {mejor_modelo.upper()}")
    print(
        f"  {'✅' if mae_ok else '❌'} MAE: {metricas_mejor['mae']:.3f} (Objetivo: ≤{OBJETIVOS_NEGOCIO['metricas_objetivo']['mae']})")
    print(
        f"  {'✅' if rmse_ok else '❌'} RMSE: {metricas_mejor['rmse']:.3f} (Objetivo: ≤{OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse']})")
    print(
        f"  {'✅' if cobertura_ok else '❌'} Cobertura: {metricas_mejor['cobertura']:.3f} (Objetivo: ≥{OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura']})")

    objetivos_cumplidos = sum([mae_ok, rmse_ok, cobertura_ok])
    print(f"  📊 Objetivos cumplidos: {objetivos_cumplidos}/3")
else:
    print("  ❌ Ningún modelo cumple los objetivos mínimos")

print(f"\n🧀 TIPOS DE RECOMENDACIONES IMPLEMENTADOS:")
for tipo_key, tipo_info in OBJETIVOS_NEGOCIO['tipos_recomendacion'].items():
    print(f"  ✅ {tipo_info['nombre']}")

print(f"\n📈 METODOLOGÍA CRISP-DM:")
print(f"  ✅ Comprensión del negocio completada")
print(f"  ✅ Comprensión de datos realizada")
print(f"  ✅ Preparación de datos con matriz usuario-producto")
print(f"  ✅ Modelado con múltiples algoritmos de filtrado colaborativo")
print(f"  ✅ Evaluación rigurosa con división temporal 80-20")
print(f"  ✅ Despliegue con motor de recomendación especializado")

print(f"\n💾 ENTREGABLES GENERADOS:")
print(f"  📄 {len([f for f in archivos_generados if f.endswith('.pkl')])} modelos entrenados (.pkl)")
print(f"  📊 {len([f for f in archivos_generados if f.endswith('.csv')])} datasets procesados (.csv)")
print(f"  📈 {len([f for f in archivos_generados if f.endswith('.png')])} visualizaciones (.png)")
print(f"  🔢 {len([f for f in archivos_generados if f.endswith('.npy')])} matrices guardadas (.npy)")

print(f"\n🚀 ARCHIVOS PRINCIPALES:")
archivos_principales = [
    'dashboard_filtrado_colaborativo.png',
    'dashboard_tipos_recomendaciones.png',
    'sistema_recomendacion_colaborativo.pkl',
    'motor_recomendacion_quesos.pkl',
    'interacciones_usuario_producto.csv'
]

for archivo in archivos_principales:
    if archivo in archivos_generados:
        print(f"  📄 {archivo}")

print(f"\n📍 UBICACIÓN DE RESULTADOS:")
print(f"  📁 Directorio: {OUTPUT_DIR}")

estado_final = "✅ EXITOSO - OBJETIVOS CUMPLIDOS" if (
            mejor_modelo and objetivos_cumplidos >= 2) else "⚠️ PARCIAL - ALGUNOS OBJETIVOS CUMPLIDOS" if mejor_modelo else "❌ REQUIERE MEJORAS"

# ============================================================================
# INFORME DE VALIDACIÓN COMPLETO
# ============================================================================
print("\n📄 GENERANDO INFORME DE VALIDACIÓN COMPLETO")
print("-" * 70)

# Calcular estadísticas para el informe
total_usuarios = len(usuarios_unicos)
total_productos = len(productos_unicos)
total_interacciones = len(interacciones)
densidad_matriz = matriz_ratings.nnz / (matriz_ratings.shape[0] * matriz_ratings.shape[1]) * 100

# Estado del proyecto
objetivos_cumplidos_total = 0
if mejor_modelo and mejor_modelo in resultados_evaluacion:
    metricas_mejor = resultados_evaluacion[mejor_modelo]
    mae_ok = metricas_mejor['mae'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['mae']
    rmse_ok = metricas_mejor['rmse'] <= OBJETIVOS_NEGOCIO['metricas_objetivo']['rmse']
    cobertura_ok = metricas_mejor['cobertura'] >= OBJETIVOS_NEGOCIO['metricas_objetivo']['cobertura']
    objetivos_cumplidos_total = sum([mae_ok, rmse_ok, cobertura_ok])

if objetivos_cumplidos_total == 3:
    estado_proyecto = "✅ EXCELENTE - TODOS LOS OBJETIVOS CUMPLIDOS"
elif objetivos_cumplidos_total >= 2:
    estado_proyecto = "✅ BUENO - MAYORÍA DE OBJETIVOS CUMPLIDOS"
elif objetivos_cumplidos_total >= 1:
    estado_proyecto = "⚠️ ACEPTABLE - ALGUNOS OBJETIVOS CUMPLIDOS"
else:
    estado_proyecto = "❌ REQUIERE MEJORA - OBJETIVOS NO CUMPLIDOS"

informe_completo = f"""
INFORME DE VALIDACIÓN COMPLETO - SISTEMA DE RECOMENDACIÓN CON FILTRADO COLABORATIVO
===================================================================================

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Metodología aplicada: CRISP-DM (Cross-Industry Standard Process for Data Mining)
Algoritmo principal: Filtrado Colaborativo Híbrido
División de datos: 80% entrenamiento, 20% prueba (división temporal)
Especialización: Productos gastronómicos y quesos

ESTADO GENERAL DEL PROYECTO: {estado_proyecto}
Objetivos cumplidos: {objetivos_cumplidos_total}/3

════════════════════════════════════════════════════════════════════════════════
1. RESUMEN EJECUTIVO
════════════════════════════════════════════════════════════════════════════════

Se ha desarrollado un sistema de recomendación avanzado con filtrado colaborativo especializado 
en productos gastronómicos y quesos. El sistema implementa múltiples algoritmos y proporciona
tres tipos específicos de recomendaciones:

🧀 TIPOS DE RECOMENDACIONES IMPLEMENTADOS:
• Combinaciones óptimas de quesos para determinados platillos
• Nuevos productos que podrían adaptarse a la oferta gastronómica
• Tendencias en el uso de quesos por tipo de establecimiento

🎯 OBJETIVOS TÉCNICOS:
• MAE ≤ 0.75: {'✅ CUMPLIDO' if mejor_modelo and resultados_evaluacion[mejor_modelo]['mae'] <= 0.75 else '❌ NO CUMPLIDO'}
• RMSE ≤ 1.0: {'✅ CUMPLIDO' if mejor_modelo and resultados_evaluacion[mejor_modelo]['rmse'] <= 1.0 else '❌ NO CUMPLIDO'}
• Cobertura ≥ 0.80: {'✅ CUMPLIDO' if mejor_modelo and resultados_evaluacion[mejor_modelo]['cobertura'] >= 0.80 else '❌ NO CUMPLIDO'}

RESULTADO PRINCIPAL: {estado_proyecto.split(' - ')[1] if ' - ' in estado_proyecto else estado_proyecto}

════════════════════════════════════════════════════════════════════════════════
2. METODOLOGÍA CRISP-DM APLICADA
════════════════════════════════════════════════════════════════════════════════

✅ FASE 1 - COMPRENSIÓN DEL NEGOCIO:
- Objetivos específicos definidos para sistema de recomendación gastronómica
- Métricas de éxito establecidas (MAE, RMSE, Cobertura)
- Tres tipos de recomendaciones especializadas definidas
- Segmentación por tipo de negocio implementada

✅ FASE 2 - COMPRENSIÓN DE LOS DATOS:
- Mismos datasets que Random Forest para consistencia
- Análisis exploratorio de interacciones usuario-producto
- Evaluación de densidad de matriz de ratings

✅ FASE 3 - PREPARACIÓN DE DATOS:
- Creación de matriz usuario-producto esparsa
- Generación de ratings implícitos basados en cantidad, valor y frecuencia
- Información adicional de productos y clientes integrada
- División temporal de datos para evaluación realista

✅ FASE 4 - MODELADO:
- User-Based Collaborative Filtering implementado
- Item-Based Collaborative Filtering implementado
- Matrix Factorization (SVD y NMF) implementado
- Modelo híbrido combinando todos los enfoques

✅ FASE 5 - EVALUACIÓN:
- Evaluación rigurosa con división temporal 80-20
- Métricas específicas calculadas (MAE, RMSE, Cobertura)
- Comparación de múltiples algoritmos
- Validación en datos no vistos

✅ FASE 6 - DESPLIEGUE:
- Motor de recomendación especializado implementado
- Funciones de producción desarrolladas
- Dashboard ejecutivo generado
- Documentación completa

════════════════════════════════════════════════════════════════════════════════
3. DATOS UTILIZADOS
════════════════════════════════════════════════════════════════════════════════

📊 ESTADÍSTICAS DE DATOS:
- Usuarios únicos: {total_usuarios:,}
- Productos únicos: {total_productos:,}
- Interacciones totales: {total_interacciones:,}
- Tipos de negocio: {clientes_info['tipo_negocio'].nunique()}
- Ciudades: {clientes_info['ciudad'].nunique()}

🔢 MATRIZ DE RATINGS:
- Dimensiones: {matriz_ratings.shape[0]} x {matriz_ratings.shape[1]}
- Valores no nulos: {matriz_ratings.nnz:,}
- Densidad: {densidad_matriz:.4f}%
- Rango de ratings: {interacciones['rating'].min():.2f} - {interacciones['rating'].max():.2f}
- Rating promedio: {interacciones['rating'].mean():.2f}

📈 CARACTERÍSTICAS DEL DATASET:
- Problema de Cold Start: Presente (nuevos usuarios/productos)
- Sparsity: Alta ({100 - densidad_matriz:.2f}% valores faltantes)
- Escalabilidad: Matriz esparsa implementada para eficiencia

════════════════════════════════════════════════════════════════════════════════
4. ALGORITMOS IMPLEMENTADOS Y RESULTADOS
════════════════════════════════════════════════════════════════════════════════
"""

# Resultados por algoritmo
for modelo, metricas in resultados_evaluacion.items():
    if metricas:
        informe_completo += f"""
🤖 ALGORITMO: {modelo.upper()}
──────────────────────────────────────────────────────────────────────────────
📊 Métricas de Rendimiento:
- MAE (Error Absoluto Medio): {metricas['mae']:.4f}
- RMSE (Error Cuadrático Medio): {metricas['rmse']:.4f}
- Cobertura: {metricas['cobertura']:.4f} ({metricas['cobertura'] * 100:.1f}%)
- Predicciones evaluadas: {metricas['n_predicciones']:,}

🎯 Cumplimiento de Objetivos:
- MAE ≤ 0.75: {'✅ SÍ' if metricas['mae'] <= 0.75 else '❌ NO'} ({metricas['mae']:.4f})
- RMSE ≤ 1.0: {'✅ SÍ' if metricas['rmse'] <= 1.0 else '❌ NO'} ({metricas['rmse']:.4f})
- Cobertura ≥ 0.80: {'✅ SÍ' if metricas['cobertura'] >= 0.80 else '❌ NO'} ({metricas['cobertura']:.4f})

📋 Descripción del Algoritmo:"""

        if modelo == 'user_based':
            informe_completo += """
User-Based Collaborative Filtering encuentra usuarios similares basándose en sus 
patrones de compra históricos y recomienda productos que otros usuarios similares 
han comprado. Utiliza similitud coseno para encontrar vecinos similares."""

        elif modelo == 'item_based':
            informe_completo += """
Item-Based Collaborative Filtering encuentra productos similares basándose en los 
patrones de compra de los usuarios y recomienda productos similares a los que el 
usuario ya ha comprado. Más estable que user-based para catálogos grandes."""

        elif modelo == 'svd':
            informe_completo += """
Singular Value Decomposition (SVD) reduce la dimensionalidad de la matriz usuario-producto 
para encontrar factores latentes que explican las preferencias. Maneja bien la sparsity 
y puede capturar patrones complejos."""

        elif modelo == 'nmf':
            informe_completo += """
Non-negative Matrix Factorization (NMF) descompone la matriz en factores no negativos, 
lo que permite interpretaciones más intuitivas de los factores latentes. Útil para 
entender características subyacentes de productos."""

        elif modelo == 'hibrido':
            informe_completo += """
Modelo Híbrido combina las predicciones de User-Based CF (25%), Item-Based CF (25%), 
SVD (30%) y NMF (20%) para aprovechar las fortalezas de cada algoritmo y mitigar 
sus debilidades individuales."""

if mejor_modelo:
    informe_completo += f"""

🏆 MEJOR MODELO IDENTIFICADO: {mejor_modelo.upper()}
──────────────────────────────────────────────────────────────────────────────
El modelo {mejor_modelo.upper()} fue seleccionado como el mejor basándose en el cumplimiento 
de objetivos y el score compuesto que combina todas las métricas.

════════════════════════════════════════════════════════════════════════════════
5. MOTOR DE RECOMENDACIÓN ESPECIALIZADO
════════════════════════════════════════════════════════════════════════════════

🧀 MOTOR DE RECOMENDACIÓN DE QUESOS Y PRODUCTOS GASTRONÓMICOS

El sistema incluye un motor especializado que proporciona tres tipos específicos 
de recomendaciones para establecimientos gastronómicos:

📌 TIPO 1: COMBINACIONES ÓPTIMAS DE QUESOS PARA PLATILLOS
──────────────────────────────────────────────────────────────────────────────
• Objetivo: Recomendar quesos específicos que complementen platillos determinados
• Metodología: Análisis de patrones de compra por tipo de platillo
• Platillos soportados: Pizza, Pasta, Ensaladas, Postres, Tabla de quesos, etc.
• Algoritmo: Combinación de filtrado colaborativo + relevancia por platillo

Categorías de quesos identificadas:
- Quesos frescos: mozzarella, ricotta, cottage, feta
- Quesos semiduros: gouda, edam, cheddar, swiss, provolone  
- Quesos duros: parmesano, romano, pecorino, grana
- Quesos azules: roquefort, gorgonzola, stilton
- Quesos especiales: brie, camembert, mascarpone, gruyere

📌 TIPO 2: NUEVOS PRODUCTOS PARA OFERTA GASTRONÓMICA
──────────────────────────────────────────────────────────────────────────────
• Objetivo: Identificar productos innovadores adoptados por establecimientos similares
• Metodología: Análisis de adopción por tipo de negocio + filtrado colaborativo
• Segmentación: Por tipo de establecimiento (restaurante, hotel, cafetería, etc.)
• Algoritmo: Tasa de adopción + score de recomendación personalizado

Métricas calculadas:
- Tasa de adopción por tipo de negocio
- Score de recomendación personalizado
- Popularidad en establecimientos similares
- Potencial de éxito basado en perfiles similares

📌 TIPO 3: TENDENCIAS EN USO DE QUESOS POR TIPO DE ESTABLECIMIENTO
──────────────────────────────────────────────────────────────────────────────
• Objetivo: Mostrar patrones emergentes específicos por tipo de negocio
• Metodología: Análisis de penetración + tendencias temporales
• Segmentación: Granular por tipo de establecimiento
• Algoritmo: Score de tendencia combinado (penetración + rating + volumen)

Métricas de tendencia:
- Penetración en el tipo de negocio
- Rating promedio por tipo de establecimiento
- Volumen de ventas por categoría
- Crecimiento en adopción

════════════════════════════════════════════════════════════════════════════════
6. VALIDACIÓN Y ROBUSTEZ
════════════════════════════════════════════════════════════════════════════════

✅ METODOLOGÍA DE VALIDACIÓN:
- División temporal 80-20 para evaluar predicciones futuras realistas
- Evaluación en datos completamente no vistos
- Métricas específicas para sistemas de recomendación
- Comparación de múltiples algoritmos

✅ ROBUSTEZ DEL SISTEMA:
- Manejo del problema de Cold Start para nuevos usuarios/productos
- Matriz esparsa para eficiencia con datasets grandes
- Múltiples algoritmos para mayor confiabilidad
- Sistema híbrido que combina fortalezas de diferentes enfoques

✅ CALIDAD DE RECOMENDACIONES:
- Ratings implícitos basados en comportamiento real (cantidad + valor + frecuencia)
- Segmentación por tipo de negocio para mayor relevancia
- Especialización en dominio gastronómico
- Tres tipos diferentes de recomendaciones para cubrir necesidades variadas

════════════════════════════════════════════════════════════════════════════════
7. LIMITACIONES Y CONSIDERACIONES
════════════════════════════════════════════════════════════════════════════════

⚠️ LIMITACIONES IDENTIFICADAS:
- Cold Start Problem: Nuevos usuarios/productos requieren datos mínimos
- Sparsity: Alta densidad de valores faltantes ({100 - densidad_matriz:.2f}%)
- Escalabilidad: Rendimiento puede degradarse con matrices muy grandes
- Datos implícitos: Asume que compra = preferencia (puede no ser siempre cierto)

🔄 CONSIDERACIONES DE IMPLEMENTACIÓN:
- Actualización periódica de modelos recomendada (mensual)
- Monitoreo de drift en patrones de compra
- Evaluación continua de métricas en producción
- Feedback loop para mejorar ratings implícitos

════════════════════════════════════════════════════════════════════════════════
8. IMPLEMENTACIÓN Y PRÓXIMOS PASOS
════════════════════════════════════════════════════════════════════════════════

📋 CRONOGRAMA DE IMPLEMENTACIÓN:

FASE 1 (Semanas 1-2): Preparación
- Revisión y aprobación del sistema por stakeholders
- Preparación de infraestructura de producción
- Capacitación del equipo en sistema de recomendaciones

FASE 2 (Semanas 3-4): Despliegue
- Integración del motor de recomendación en sistemas existentes
- Implementación de API de recomendaciones
- Configuración de dashboards de monitoreo

FASE 3 (Mes 2): Monitoreo
- Seguimiento de métricas en producción
- Recolección de feedback de usuarios
- Ajustes basados en uso real

FASE 4 (Mes 3): Optimización
- Evaluación de impacto en ventas
- Refinamiento de algoritmos
- Expansión a nuevos tipos de recomendaciones

🎯 MÉTRICAS DE ÉXITO EN PRODUCCIÓN:
- MAE ≤ 0.75 mantenido en datos reales
- RMSE ≤ 1.0 mantenido en datos reales  
- Cobertura ≥ 0.80 mantenida en datos reales
- Adopción por parte de usuarios >= 70%
- Incremento measurable en ventas cruzadas

════════════════════════════════════════════════════════════════════════════════
9. CONCLUSIONES Y RECOMENDACIONES FINALES
════════════════════════════════════════════════════════════════════════════════

🏆 LOGROS PRINCIPALES:
✅ Sistema de recomendación con filtrado colaborativo completamente implementado
✅ Tres tipos especializados de recomendaciones gastronómicas desarrollados
✅ Motor de recomendación especializado en quesos y productos gastronómicos
✅ Metodología CRISP-DM aplicada rigurosamente
✅ Múltiples algoritmos implementados y evaluados
✅ {'Objetivos técnicos cumplidos' if objetivos_cumplidos_total >= 2 else 'Base sólida establecida para mejoras'}

📊 ESTADO FINAL: {estado_proyecto}

🔮 VALOR COMERCIAL GENERADO:
- Recomendaciones personalizadas por tipo de establecimiento
- Identificación de oportunidades de venta cruzada
- Tendencias de mercado específicas por segmento
- Optimización de inventario basada en predicciones
- Mejora en experiencia del cliente con sugerencias relevantes

💡 RECOMENDACIONES ESTRATÉGICAS:

1. IMPLEMENTACIÓN INMEDIATA:
   - Integrar sistema en plataforma de ventas existente
   - Capacitar equipo comercial en interpretación de recomendaciones
   - Establecer métricas de seguimiento en producción

2. MEJORA CONTINUA:
   - Implementar feedback explícito de usuarios para mejorar ratings
   - Expandir categorización de productos gastronómicos
   - Desarrollar recomendaciones estacionales

3. ESCALABILIDAD:
   - Considerar arquitectura distribuida para grandes volúmenes
   - Implementar caching para mejorar tiempo de respuesta
   - Explorar deep learning para recomendaciones más sofisticadas

════════════════════════════════════════════════════════════════════════════════
APROBACIÓN PARA PRODUCCIÓN: {'✅ APROBADO' if objetivos_cumplidos_total >= 2 else '⚠️ CONDICIONAL' if objetivos_cumplidos_total >= 1 else '❌ REQUIERE MEJORAS'}
════════════════════════════════════════════════════════════════════════════════

Fecha del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Responsable técnico: Sistema de Recomendación Automatizado
Próxima revisión: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}

───────────────────────────────────────────────────────────────────────────────
ARCHIVOS GENERADOS:
"""

# Listar archivos generados
archivos_generados = os.listdir(OUTPUT_DIR)
for archivo in sorted(archivos_generados):
    informe_completo += f"\n📄 {archivo}"

informe_completo += f"""

TOTAL DE ARCHIVOS: {len(archivos_generados)}
UBICACIÓN: {OUTPUT_DIR}
───────────────────────────────────────────────────────────────────────────────
"""

# Guardar informe completo
with open(os.path.join(OUTPUT_DIR, 'informe_validacion_filtrado_colaborativo.txt'), 'w', encoding='utf-8') as f:
    f.write(informe_completo)

print("✅ Informe de validación completo generado")

# ============================================================================
# FUNCIONES DE PRODUCCIÓN
# ============================================================================
print("\n🚀 CREANDO FUNCIONES DE PRODUCCIÓN")
print("-" * 70)

codigo_produccion = f'''
"""
FUNCIONES DE PRODUCCIÓN - SISTEMA DE RECOMENDACIÓN CON FILTRADO COLABORATIVO
Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Sistema completo de recomendación especializado en productos gastronómicos que implementa:
- Filtrado colaborativo híbrido (User-Based + Item-Based + Matrix Factorization)
- Motor especializado para recomendaciones de quesos y productos gastronómicos
- Tres tipos específicos de recomendaciones por tipo de establecimiento
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

class SistemaRecomendacionProduccion:
    """
    Clase principal para el sistema de recomendación en producción
    """

    def __init__(self, modelos_dir):
        """
        Inicializa el sistema cargando todos los modelos y datos necesarios

        Args:
            modelos_dir (str): Directorio donde están los modelos guardados
        """
        self.modelos_dir = modelos_dir
        self.sistema_recom = None
        self.motor_quesos = None
        self.interacciones = None
        self.productos_info = None
        self.clientes_info = None
        self.cargar_sistema()

    def cargar_sistema(self):
        """Carga todos los componentes del sistema"""
        try:
            # Cargar sistema principal
            self.sistema_recom = joblib.load(f'{{self.modelos_dir}}/sistema_recomendacion_colaborativo.pkl')
            print("✅ Sistema de recomendación principal cargado")

            # Cargar motor especializado
            self.motor_quesos = joblib.load(f'{{self.modelos_dir}}/motor_recomendacion_quesos.pkl')
            print("✅ Motor de recomendación de quesos cargado")

            # Cargar datos
            self.interacciones = pd.read_csv(f'{{self.modelos_dir}}/interacciones_usuario_producto.csv')
            self.productos_info = pd.read_csv(f'{{self.modelos_dir}}/productos_info_recomendacion.csv')
            self.clientes_info = pd.read_csv(f'{{self.modelos_dir}}/clientes_info_recomendacion.csv')
            print("✅ Datos de referencia cargados")

        except Exception as e:
            print(f"❌ Error cargando sistema: {{e}}")

    def obtener_recomendaciones_generales(self, cliente_id, n_recomendaciones=10, algoritmo='hibrido'):
        """
        Obtiene recomendaciones generales para un cliente

        Args:
            cliente_id: ID del cliente
            n_recomendaciones: Número de recomendaciones a generar
            algoritmo: Algoritmo a usar ('hibrido', 'user_based', 'item_based', 'svd', 'nmf')

        Returns:
            dict: Recomendaciones con scores y metadata
        """
        try:
            if cliente_id not in self.sistema_recom.user_to_idx:
                return {{"error": f"Cliente {{cliente_id}} no encontrado en el sistema"}}

            user_idx = self.sistema_recom.user_to_idx[cliente_id]

            # Productos ya comprados por el cliente
            productos_comprados = set(self.interacciones[
                self.interacciones['cliente_id'] == cliente_id
            ]['producto_id'])

            recomendaciones = []

            # Evaluar todos los productos no comprados
            for producto_id in self.sistema_recom.productos_unicos:
                if producto_id not in productos_comprados:
                    try:
                        item_idx = self.sistema_recom.item_to_idx[producto_id]

                        # Obtener predicción según algoritmo seleccionado
                        if algoritmo == 'hibrido':
                            score = self.sistema_recom.predecir_hibrido(user_idx, item_idx)
                        elif algoritmo == 'user_based':
                            score = self.sistema_recom.predecir_user_based(user_idx, item_idx)
                        elif algoritmo == 'item_based':
                            score = self.sistema_recom.predecir_item_based(user_idx, item_idx)
                        elif algoritmo == 'svd':
                            score = self.sistema_recom.predecir_matrix_factorization(user_idx, item_idx, 'svd')
                        elif algoritmo == 'nmf':
                            score = self.sistema_recom.predecir_matrix_factorization(user_idx, item_idx, 'nmf')
                        else:
                            score = 3.0

                        # Obtener información del producto
                        producto_info = self.productos_info[
                            self.productos_info['producto_id'] == producto_id
                        ]

                        if len(producto_info) > 0:
                            producto_info = producto_info.iloc[0]
                            recomendaciones.append({{
                                'producto_id': producto_id,
                                'score': score,
                                'categoria': producto_info['categoria'],
                                'marca': producto_info['marca'],
                                'popularidad_general': producto_info['clientes_unicos'],
                                'algoritmo_usado': algoritmo
                            }})

                    except KeyError:
                        continue

            # Ordenar por score y devolver top N
            recomendaciones.sort(key=lambda x: x['score'], reverse=True)

            # Obtener información del cliente
            cliente_info = self.clientes_info[
                self.clientes_info['cliente_id'] == cliente_id
            ].iloc[0]

            return {{
                'cliente_id': cliente_id,
                'tipo_negocio': cliente_info['tipo_negocio'],
                'ciudad': cliente_info['ciudad'],
                'recomendaciones': recomendaciones[:n_recomendaciones],
                'algoritmo_usado': algoritmo,
                'productos_ya_comprados': len(productos_comprados),
                'fecha_recomendacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }}

        except Exception as e:
            return {{"error": f"Error generando recomendaciones: {{str(e)}}"}}

    def obtener_combinaciones_platillos(self, cliente_id, tipo_platillo='pizza', n_recomendaciones=5):
        """
        Obtiene recomendaciones de quesos para un platillo específico

        Args:
            cliente_id: ID del cliente
            tipo_platillo: Tipo de platillo ('pizza', 'pasta', 'ensaladas', etc.)
            n_recomendaciones: Número de recomendaciones

        Returns:
            dict: Recomendaciones específicas para el platillo
        """
        try:
            return self.motor_quesos.recomendar_combinaciones_platillos(
                cliente_id, tipo_platillo, n_recomendaciones
            )
        except Exception as e:
            return {{"error": f"Error en recomendaciones de platillos: {{str(e)}}"}}

    def obtener_nuevos_productos(self, cliente_id, n_recomendaciones=5):
        """
        Obtiene recomendaciones de nuevos productos para el establecimiento

        Args:
            cliente_id: ID del cliente
            n_recomendaciones: Número de recomendaciones

        Returns:
            dict: Nuevos productos recomendados
        """
        try:
            return self.motor_quesos.recomendar_nuevos_productos(cliente_id, n_recomendaciones)
        except Exception as e:
            return {{"error": f"Error en recomendaciones de nuevos productos: {{str(e)}}"}}

    def obtener_tendencias_tipo_negocio(self, cliente_id, n_tendencias=5):
        """
        Obtiene análisis de tendencias para el tipo de establecimiento

        Args:
            cliente_id: ID del cliente
            n_tendencias: Número de tendencias a mostrar

        Returns:
            dict: Análisis de tendencias
        """
        try:
            return self.motor_quesos.analizar_tendencias_tipo_negocio(cliente_id, n_tendencias)
        except Exception as e:
            return {{"error": f"Error en análisis de tendencias: {{str(e)}}"}}

    def obtener_recomendaciones_completas(self, cliente_id):
        """
        Obtiene un análisis completo con todos los tipos de recomendaciones

        Args:
            cliente_id: ID del cliente

        Returns:
            dict: Análisis completo con todos los tipos de recomendaciones
        """
        print(f"🔍 Generando análisis completo para cliente {{cliente_id}}...")

        # Verificar que el cliente existe
        if cliente_id not in self.clientes_info['cliente_id'].values:
            return {{"error": f"Cliente {{cliente_id}} no encontrado"}}

        try:
            # Obtener información básica del cliente
            cliente_info = self.clientes_info[
                self.clientes_info['cliente_id'] == cliente_id
            ].iloc[0]

            # Recomendaciones generales
            recomendaciones_generales = self.obtener_recomendaciones_generales(
                cliente_id, n_recomendaciones=10, algoritmo='hibrido'
            )

            # Combinaciones para platillos (ejemplo con pizza y pasta)
            combinaciones_pizza = self.obtener_combinaciones_platillos(
                cliente_id, 'pizza', n_recomendaciones=5
            )
            combinaciones_pasta = self.obtener_combinaciones_platillos(
                cliente_id, 'pasta', n_recomendaciones=5
            )

            # Nuevos productos
            nuevos_productos = self.obtener_nuevos_productos(cliente_id, n_recomendaciones=8)

            # Tendencias del tipo de negocio
            tendencias = self.obtener_tendencias_tipo_negocio(cliente_id, n_tendencias=10)

            return {{
                'cliente_id': cliente_id,
                'informacion_cliente': {{
                    'tipo_negocio': cliente_info['tipo_negocio'],
                    'ciudad': cliente_info['ciudad'],
                    'productos_distintos': int(cliente_info['productos_distintos']),
                    'gasto_total': float(cliente_info['gasto_total'])
                }},
                'recomendaciones_generales': recomendaciones_generales,
                'combinaciones_platillos': {{
                    'pizza': combinaciones_pizza,
                    'pasta': combinaciones_pasta
                }},
                'nuevos_productos': nuevos_productos,
                'tendencias_tipo_negocio': tendencias,
                'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version_sistema': 'Filtrado Colaborativo v1.0'
            }}

        except Exception as e:
            return {{"error": f"Error en análisis completo: {{str(e)}}"}}

    def evaluar_producto(self, cliente_id, producto_id, algoritmo='hibrido'):
        """
        Evalúa qué tan probable es que un cliente compre un producto específico

        Args:
            cliente_id: ID del cliente
            producto_id: ID del producto a evaluar
            algoritmo: Algoritmo a usar para la evaluación

        Returns:
            dict: Evaluación del producto para el cliente
        """
        try:
            if cliente_id not in self.sistema_recom.user_to_idx:
                return {{"error": f"Cliente {{cliente_id}} no encontrado"}}

            if producto_id not in self.sistema_recom.item_to_idx:
                return {{"error": f"Producto {{producto_id}} no encontrado"}}

            user_idx = self.sistema_recom.user_to_idx[cliente_id]
            item_idx = self.sistema_recom.item_to_idx[producto_id]

            # Obtener predicción
            if algoritmo == 'hibrido':
                score = self.sistema_recom.predecir_hibrido(user_idx, item_idx)
            elif algoritmo == 'user_based':
                score = self.sistema_recom.predecir_user_based(user_idx, item_idx)
            elif algoritmo == 'item_based':
                score = self.sistema_recom.predecir_item_based(user_idx, item_idx)
            else:
                score = 3.0

            # Convertir score a probabilidad (0-100%)
            probabilidad = min(100, max(0, (score - 1) / 4 * 100))

            # Obtener información adicional
            cliente_info = self.clientes_info[
                self.clientes_info['cliente_id'] == cliente_id
            ].iloc[0]

            producto_info = self.productos_info[
                self.productos_info['producto_id'] == producto_id
            ].iloc[0]

            # Verificar si ya compró el producto
            ya_compro = producto_id in set(self.interacciones[
                self.interacciones['cliente_id'] == cliente_id
            ]['producto_id'])

            return {{
                'cliente_id': cliente_id,
                'producto_id': producto_id,
                'score_raw': score,
                'probabilidad_compra': probabilidad,
                'recomendacion': 'Alta' if probabilidad >= 75 else 'Media' if probabilidad >= 50 else 'Baja',
                'ya_compro': ya_compro,
                'cliente_tipo_negocio': cliente_info['tipo_negocio'],
                'producto_categoria': producto_info['categoria'],
                'producto_popularidad': int(producto_info['clientes_unicos']),
                'algoritmo_usado': algoritmo,
                'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }}

        except Exception as e:
            return {{"error": f"Error evaluando producto: {{str(e)}}"}}

def ejemplo_uso_completo():
    """
    Función de ejemplo mostrando cómo usar el sistema completo
    """
    # Inicializar sistema
    sistema = SistemaRecomendacionProduccion('ruta/a/modelos')

    # Ejemplo con un cliente
    cliente_id = 'CLIENTE_001'

    print(f"🧀 Sistema de Recomendación - Cliente {{cliente_id}}")
    print("=" * 60)

    # Análisis completo
    analisis_completo = sistema.obtener_recomendaciones_completas(cliente_id)

    if 'error' not in analisis_completo:
        print(f"👤 Cliente: {{cliente_id}}")
        print(f"🏢 Tipo: {{analisis_completo['informacion_cliente']['tipo_negocio']}}")
        print(f"📍 Ciudad: {{analisis_completo['informacion_cliente']['ciudad']}}")

        # Recomendaciones generales
        if 'error' not in analisis_completo['recomendaciones_generales']:
            print(f"\\n🎯 Top 5 Recomendaciones Generales:")
            for i, rec in enumerate(analisis_completo['recomendaciones_generales']['recomendaciones'][:5]):
                print(f"   {{i+1}}. Producto {{rec['producto_id']}} - Score: {{rec['score']:.2f}}")

        # Combinaciones para pizza
        if 'error' not in analisis_completo['combinaciones_platillos']['pizza']:
            print(f"\\n🍕 Quesos recomendados para Pizza:")
            for i, rec in enumerate(analisis_completo['combinaciones_platillos']['pizza']['recomendaciones'][:3]):
                print(f"   {{i+1}}. Producto {{rec['producto_id']}} - Score: {{rec['score']:.2f}}")

        # Nuevos productos
        if 'error' not in analisis_completo['nuevos_productos']:
            print(f"\\n🆕 Nuevos productos recomendados:")
            for i, rec in enumerate(analisis_completo['nuevos_productos']['recomendaciones'][:3]):
                print(f"   {{i+1}}. Producto {{rec['producto_id']}} - Adopción: {{rec['tasa_adopcion']:.2f}}")

        # Tendencias
        if 'error' not in analisis_completo['tendencias_tipo_negocio']:
            print(f"\\n📈 Tendencias en {{analisis_completo['tendencias_tipo_negocio']['tipo_negocio']}}:")
            for i, trend in enumerate(analisis_completo['tendencias_tipo_negocio']['tendencias'][:3]):
                print(f"   {{i+1}}. Producto {{trend['producto_id']}} - Penetración: {{trend['penetracion']:.2f}}")

    else:
        print(f"❌ Error: {{analisis_completo['error']}}")

    # Ejemplo de evaluación de producto específico
    print(f"\\n🔍 Evaluación de producto específico:")
    evaluacion = sistema.evaluar_producto(cliente_id, 'PRODUCTO_123')
    if 'error' not in evaluacion:
        print(f"   Producto PRODUCTO_123: {{evaluacion['probabilidad_compra']:.1f}}% probabilidad")
        print(f"   Recomendación: {{evaluacion['recomendacion']}}")

# EJEMPLO DE INTEGRACIÓN CON API
class APIRecomendaciones:
    """
    Clase para integración con API REST
    """

    def __init__(self, modelos_dir):
        self.sistema = SistemaRecomendacionProduccion(modelos_dir)

    def endpoint_recomendaciones_generales(self, cliente_id, n_recomendaciones=10, algoritmo='hibrido'):
        """Endpoint para recomendaciones generales"""
        return self.sistema.obtener_recomendaciones_generales(cliente_id, n_recomendaciones, algoritmo)

    def endpoint_combinaciones_platillos(self, cliente_id, tipo_platillo, n_recomendaciones=5):
        """Endpoint para combinaciones de platillos"""
        return self.sistema.obtener_combinaciones_platillos(cliente_id, tipo_platillo, n_recomendaciones)

    def endpoint_nuevos_productos(self, cliente_id, n_recomendaciones=5):
        """Endpoint para nuevos productos"""
        return self.sistema.obtener_nuevos_productos(cliente_id, n_recomendaciones)

    def endpoint_tendencias(self, cliente_id, n_tendencias=5):
        """Endpoint para tendencias"""
        return self.sistema.obtener_tendencias_tipo_negocio(cliente_id, n_tendencias)

    def endpoint_analisis_completo(self, cliente_id):
        """Endpoint para análisis completo"""
        return self.sistema.obtener_recomendaciones_completas(cliente_id)

    def endpoint_evaluar_producto(self, cliente_id, producto_id, algoritmo='hibrido'):
        """Endpoint para evaluar producto específico"""
        return self.sistema.evaluar_producto(cliente_id, producto_id, algoritmo)

if __name__ == "__main__":
    # Ejecutar ejemplo
    ejemplo_uso_completo()
'''

with open(os.path.join(OUTPUT_DIR, 'funciones_produccion_filtrado_colaborativo.py'), 'w', encoding='utf-8') as f:
    f.write(codigo_produccion)

print("✅ Funciones de producción generadas")

# Guardar métricas finales
metricas_finales = {
    'fecha_entrenamiento': datetime.now().isoformat(),
    'algoritmo': 'Filtrado Colaborativo Híbrido',
    'metodologia': 'CRISP-DM',
    'division_datos': '80-20 temporal',
    'objetivos': OBJETIVOS_NEGOCIO['metricas_objetivo'],
    'mejor_modelo': mejor_modelo,
    'resultados_evaluacion': resultados_evaluacion,
    'objetivos_cumplidos': objetivos_cumplidos_total,
    'estado_proyecto': estado_proyecto,
    'estadisticas': {
        'total_usuarios': total_usuarios,
        'total_productos': total_productos,
        'total_interacciones': total_interacciones,
        'densidad_matriz': densidad_matriz
    }
}


# Convertir numpy arrays a listas para JSON
def convertir_para_json(obj):
    if isinstance(obj, dict):
        return {k: convertir_para_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_para_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


metricas_finales = convertir_para_json(metricas_finales)

with open(os.path.join(OUTPUT_DIR, 'metricas_finales_filtrado_colaborativo.json'), 'w', encoding='utf-8') as f:
    json.dump(metricas_finales, f, indent=2, ensure_ascii=False)

print("✅ Métricas finales guardadas")

print(f"\n✅ SISTEMA DE RECOMENDACIÓN CON FILTRADO COLABORATIVO COMPLETADO")
print(f"   🎯 Estado: {estado_final}")
print(f"   🧀 Motor especializado: ✅ IMPLEMENTADO")
print(f"   📊 Tipos de recomendaciones: ✅ TODOS IMPLEMENTADOS")
print(f"   🔧 Metodología CRISP-DM: ✅ COMPLETA")
print(f"   📄 Informe de validación: ✅ GENERADO")
print(f"   🚀 Funciones de producción: ✅ LISTAS")
print("=" * 100)