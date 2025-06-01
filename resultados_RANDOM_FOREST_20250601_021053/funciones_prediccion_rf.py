
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Cargar modelos y datos
def cargar_modelos(path_dir):
    """Carga todos los modelos y datos necesarios"""
    modelos = {}

    try:
        modelos['rf_proxima'] = joblib.load(f'{path_dir}/modelo_rf_proxima_compra.pkl')
        modelos['features_proxima'] = joblib.load(f'{path_dir}/features_proxima_compra.pkl')
    except:
        modelos['rf_proxima'] = None

    modelos['rf_productos'] = joblib.load(f'{path_dir}/modelo_rf_productos.pkl')
    modelos['features_productos'] = joblib.load(f'{path_dir}/features_productos.pkl')

    try:
        modelos['rf_cambios'] = joblib.load(f'{path_dir}/modelo_rf_cambios_patron.pkl')
        modelos['features_cambios'] = joblib.load(f'{path_dir}/features_cambios_patron.pkl')
    except:
        modelos['rf_cambios'] = None

    # Cargar encoders
    modelos['le_ciudad'] = joblib.load(f'{path_dir}/le_ciudad_rf.pkl')
    modelos['le_tipo_negocio'] = joblib.load(f'{path_dir}/le_tipo_negocio_rf.pkl')
    modelos['le_turno'] = joblib.load(f'{path_dir}/le_turno_rf.pkl')

    # Cargar datos históricos
    modelos['cliente_historico'] = pd.read_csv(f'{path_dir}/cliente_historico_rf.csv')
    modelos['producto_stats'] = pd.read_csv(f'{path_dir}/producto_stats_rf.csv')

    return modelos

def predecir_proxima_compra(cliente_id, modelos):
    """Predice cuándo será la próxima compra de un cliente"""
    if modelos['rf_proxima'] is None:
        return {"error": "Modelo no disponible"}

    try:
        cliente_data = modelos['cliente_historico'][
            modelos['cliente_historico']['cliente_id'] == cliente_id
        ].iloc[0]

        # Crear features
        features_dict = {
            'total_compras': cliente_data['total_compras'],
            'gasto_promedio': cliente_data['gasto_promedio'],
            'gasto_total': cliente_data['gasto_total'],
            'frecuencia_compra': cliente_data['frecuencia_compra'],
            'recencia': cliente_data['recencia'],
            'categorias_distintas': cliente_data['categorias_distintas'],
            'cantidad_promedio': cliente_data['cantidad_promedio'],
            'mes': datetime.now().month,
            'trimestre': (datetime.now().month - 1) // 3 + 1,
            'dia_semana': datetime.now().weekday(),
            'tipo_negocio_encoded': 0,  # Default
            'ciudad_encoded': 0,  # Default
            'tendencia_gasto': cliente_data['tendencia_gasto']
        }

        # Crear DataFrame con las features en el orden correcto
        X = pd.DataFrame([features_dict])[modelos['features_proxima']]

        # Predecir
        dias_predichos = modelos['rf_proxima'].predict(X)[0]
        fecha_proxima = datetime.now() + timedelta(days=int(dias_predichos))

        return {
            'cliente_id': cliente_id,
            'dias_hasta_proxima': int(dias_predichos),
            'fecha_proxima_estimada': fecha_proxima.strftime('%Y-%m-%d'),
            'confianza': 'Alta' if dias_predichos <= 30 else 'Media' if dias_predichos <= 60 else 'Baja'
        }

    except Exception as e:
        return {"error": f"Error en predicción: {str(e)}"}

def recomendar_productos(cliente_id, modelos, top_n=5):
    """Recomienda productos para un cliente"""
    try:
        cliente_data = modelos['cliente_historico'][
            modelos['cliente_historico']['cliente_id'] == cliente_id
        ].iloc[0]

        recomendaciones = []

        for _, producto in modelos['producto_stats'].iterrows():
            features_dict = {
                'cliente_total_compras': cliente_data['total_compras'],
                'cliente_gasto_promedio': cliente_data['gasto_promedio'],
                'cliente_categorias': cliente_data['categorias_distintas'],
                'cliente_recencia': cliente_data['recencia'],
                'cliente_frecuencia': cliente_data['frecuencia_compra'],
                'cliente_tendencia': cliente_data['tendencia_gasto'],
                'producto_popularidad': producto['producto_popularidad'],
                'producto_ventas_promedio': producto['producto_ventas_promedio'],
                'producto_clientes': producto['producto_clientes_unicos']
            }

            X = pd.DataFrame([features_dict])[modelos['features_productos']]
            probabilidad = modelos['rf_productos'].predict_proba(X)[0][1]

            recomendaciones.append({
                'producto_id': producto['producto_id'],
                'probabilidad_compra': probabilidad * 100,
                'popularidad': producto['producto_popularidad']
            })

        # Ordenar y devolver top N
        recomendaciones.sort(key=lambda x: x['probabilidad_compra'], reverse=True)
        return recomendaciones[:top_n]

    except Exception as e:
        return {"error": f"Error en recomendación: {str(e)}"}

def detectar_cambio_patron(cliente_id, modelos):
    """Detecta si un cliente está cambiando sus patrones"""
    if modelos['rf_cambios'] is None:
        return {"error": "Modelo no disponible"}

    try:
        cliente_data = modelos['cliente_historico'][
            modelos['cliente_historico']['cliente_id'] == cliente_id
        ].iloc[0]

        features_dict = {
            'total_compras': cliente_data['total_compras'],
            'gasto_promedio': cliente_data['gasto_promedio'],
            'gasto_total': cliente_data['gasto_total'],
            'recencia': cliente_data['recencia'],
            'dias_como_cliente': cliente_data['dias_como_cliente'],
            'categorias_distintas': cliente_data['categorias_distintas'],
            'frecuencia_compra': cliente_data['frecuencia_compra'],
            'tendencia_gasto': cliente_data['tendencia_gasto']
        }

        X = pd.DataFrame([features_dict])[modelos['features_cambios']]

        probabilidad_cambio = modelos['rf_cambios'].predict_proba(X)[0][1] * 100
        cambio_predicho = modelos['rf_cambios'].predict(X)[0]

        return {
            'cliente_id': cliente_id,
            'cambio_detectado': bool(cambio_predicho),
            'probabilidad_cambio': probabilidad_cambio,
            'nivel_riesgo': 'Alto' if probabilidad_cambio > 70 else 'Medio' if probabilidad_cambio > 40 else 'Bajo',
            'recomendacion': 'Acción inmediata' if probabilidad_cambio > 70 else 'Monitorear' if probabilidad_cambio > 40 else 'Normal'
        }

    except Exception as e:
        return {"error": f"Error en detección: {str(e)}"}
