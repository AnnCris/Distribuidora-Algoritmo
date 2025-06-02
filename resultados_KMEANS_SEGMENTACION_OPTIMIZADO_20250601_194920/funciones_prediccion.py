
# FUNCIONES DE PREDICCIÓN K-MEANS - DISTRIBUIDORA
# Generado automáticamente el 2025-06-01 19:49:27

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Cargar modelos y transformadores
def cargar_modelo_segmentacion(path_dir):
    """Carga el modelo de segmentación entrenado"""
    modelo = {}

    modelo['kmeans'] = joblib.load(f'{path_dir}/modelo_kmeans.pkl')
    modelo['scaler'] = joblib.load(f'{path_dir}/scaler.pkl')
    modelo['perfiles'] = pd.read_csv(f'{path_dir}/perfiles_clusters.csv', index_col=0)

    # Cargar encoders
    for encoder_name in ['ciudad', 'tipo_negocio', 'categoria_preferida', 'marca_preferida', 'turno_preferido']:
        try:
            modelo[f'encoder_{encoder_name}'] = joblib.load(f'{path_dir}/encoder_{encoder_name}.pkl')
        except:
            pass

    return modelo

def predecir_segmento_cliente(cliente_data, modelo):
    """
    Predice el segmento de un nuevo cliente

    cliente_data debe contener:
    - frecuencia: número de compras
    - recencia_dias: días desde última compra  
    - valor_total: valor total de compras
    - ticket_promedio: ticket promedio
    - tipo_negocio: tipo de negocio
    - ciudad: ciudad del cliente
    """

    try:
        # Crear características básicas
        features = [
            cliente_data.get('frecuencia', 0),
            cliente_data.get('recencia_dias', 30),
            cliente_data.get('valor_total', 1000),
            cliente_data.get('ticket_promedio', 500),
            cliente_data.get('frecuencia_mensual', 1),
            cliente_data.get('intensidad_compra', 500),
            cliente_data.get('variabilidad_ticket', 0.1),
            cliente_data.get('antigüedad_meses', 6),
            cliente_data.get('rango_ticket', 200),
            cliente_data.get('num_categorias', 2),
            cliente_data.get('num_marcas', 2),
            cliente_data.get('diversidad_categorias', 1),
            cliente_data.get('precio_promedio', 25),
            cliente_data.get('especializacion', 5),
            cliente_data.get('tendencia_gasto', 0),
            cliente_data.get('regularidad_compras', 0.5),
            cliente_data.get('intervalo_promedio_dias', 30),
            0,  # tipo_negocio_encoded (por defecto)
            0   # categoria_pref_encoded (por defecto)
        ]

        # Normalizar características
        X = np.array(features).reshape(1, -1)
        X_scaled = modelo['scaler'].transform(X)

        # Predecir cluster
        cluster = modelo['kmeans'].predict(X_scaled)[0]

        # Obtener información del perfil
        perfil = modelo['perfiles'].iloc[cluster]

        return {
            'cluster': int(cluster),
            'grupo': perfil.get('grupo_objetivo', cluster),
            'nombre_grupo': perfil.get('nombre', f'Grupo {cluster + 1}'),
            'descripcion': perfil.get('descripcion', 'Segmento identificado'),
            'probabilidades': modelo['kmeans'].transform(X_scaled)[0].tolist(),
            'recomendacion': obtener_recomendacion_por_grupo(perfil.get('grupo_objetivo', cluster))
        }

    except Exception as e:
        return {'error': f'Error en predicción: {str(e)}'}

def obtener_recomendacion_por_grupo(grupo):
    """Devuelve recomendaciones específicas por grupo"""

    recomendaciones = {
        0: {
            'estrategia': 'Retención Premium',
            'acciones': ['Ofertas VIP', 'Atención prioritaria', 'Productos exclusivos'],
            'frecuencia_contacto': 'Semanal'
        },
        1: {
            'estrategia': 'Fidelización Especializada', 
            'acciones': ['Descuentos por volumen', 'Capacitación', 'Catálogo especializado'],
            'frecuencia_contacto': 'Quincenal'
        },
        2: {
            'estrategia': 'Crecimiento de Volumen',
            'acciones': ['Crédito comercial', 'Entregas programadas', 'Promociones'],
            'frecuencia_contacto': 'Mensual'
        },
        3: {
            'estrategia': 'Desarrollo y Acompañamiento',
            'acciones': ['Capacitación', 'Facilidades de pago', 'Soporte técnico'],
            'frecuencia_contacto': 'Semanal'
        },
        4: {
            'estrategia': 'Reactivación',
            'acciones': ['Promociones estacionales', 'Recordatorios', 'Ofertas limitadas'],
            'frecuencia_contacto': 'Mensual'
        }
    }

    return recomendaciones.get(grupo, recomendaciones[3])

def analizar_cartera_clientes(df_clientes, modelo):
    """Analiza una cartera completa de clientes"""

    resultados = []
    for _, cliente in df_clientes.iterrows():
        prediccion = predecir_segmento_cliente(cliente.to_dict(), modelo)
        prediccion['cliente_id'] = cliente.get('cliente_id', 'N/A')
        resultados.append(prediccion)

    return pd.DataFrame(resultados)

# EJEMPLO DE USO:
# modelo = cargar_modelo_segmentacion('ruta/a/resultados')
# nuevo_cliente = {
#     'frecuencia': 15,
#     'recencia_dias': 25,
#     'valor_total': 45000,
#     'ticket_promedio': 3000,
#     'tipo_negocio': 'PIZZERIA'
# }
# resultado = predecir_segmento_cliente(nuevo_cliente, modelo)
# print(resultado)
