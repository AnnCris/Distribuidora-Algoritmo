
"""
FUNCIONES DE PREDICCIÓN RANDOM FOREST OPTIMIZADO - PRODUCCIÓN
Generado automáticamente el 2025-06-02 10:46:19

Sistema predictivo completo basado en Random Forest OPTIMIZADO que implementa:
- Predicción de próxima fecha de compra (con feature selection)
- Recomendación de productos (con SMOTE y balanceado)
- Detección de cambios en patrones de consumo (con Grid Search optimizado)

OPTIMIZACIONES INCLUIDAS:
- Feature Selection automática con Mutual Information
- Balanceado de datos con SMOTE
- Hiperparámetros optimizados
- Class Weight Balanced
- Validación cruzada exhaustiva
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class RandomForestPredictorOptimizado:
    """
    Clase principal para predicciones usando modelos Random Forest OPTIMIZADOS
    """

    def __init__(self, modelos_dir):
        """
        Inicializa el predictor cargando todos los modelos optimizados y datos necesarios

        Args:
            modelos_dir (str): Directorio donde están los modelos optimizados guardados
        """
        self.modelos_dir = modelos_dir
        self.modelos = {}
        self.encoders = {}
        self.datos = {}
        self.selectores = {}
        self.cargar_modelos()

    def cargar_modelos(self):
        """Carga todos los modelos optimizados, encoders, selectores y datos necesarios"""
        try:
            # Cargar modelos Random Forest optimizados
            try:
                self.modelos['modelo1'] = joblib.load(f'{self.modelos_dir}/modelo_rf_proxima_compra_optimizado.pkl')
                self.modelos['features_m1'] = joblib.load(f'{self.modelos_dir}/features_modelo1_optimizado.pkl')
                self.selectores['selector_m1'] = joblib.load(f'{self.modelos_dir}/selector_modelo1.pkl')
                print("✅ Modelo 1 Optimizado (Próxima Compra) cargado")
            except:
                print("⚠️ Modelo 1 Optimizado no disponible")

            try:
                self.modelos['modelo2'] = joblib.load(f'{self.modelos_dir}/modelo_rf_productos_optimizado.pkl')
                self.modelos['features_m2'] = joblib.load(f'{self.modelos_dir}/features_modelo2_optimizado.pkl')
                self.selectores['selector_m2'] = joblib.load(f'{self.modelos_dir}/selector_modelo2.pkl')
                print("✅ Modelo 2 Optimizado (Productos) cargado")
            except:
                print("⚠️ Modelo 2 Optimizado no disponible")

            try:
                self.modelos['modelo3'] = joblib.load(f'{self.modelos_dir}/modelo_rf_cambios_patron_optimizado.pkl')
                self.modelos['features_m3'] = joblib.load(f'{self.modelos_dir}/features_modelo3_optimizado.pkl')
                self.selectores['selector_m3'] = joblib.load(f'{self.modelos_dir}/selector_modelo3.pkl')
                print("✅ Modelo 3 Optimizado (Cambios Patrón) cargado")
            except:
                print("⚠️ Modelo 3 Optimizado no disponible")

            # Cargar encoders optimizados
            encoders_disponibles = ['ciudad', 'tipo_negocio', 'turno_preferido', 'producto_categoria', 'producto_marca']
            for encoder_name in encoders_disponibles:
                try:
                    self.encoders[encoder_name] = joblib.load(f'{self.modelos_dir}/encoder_{encoder_name}_rf_optimizado.pkl')
                except:
                    pass
            print(f"✅ {len(self.encoders)} encoders optimizados cargados")

            # Cargar datos de referencia optimizados
            try:
                self.datos['clientes'] = pd.read_csv(f'{self.modelos_dir}/cliente_metricas_completas_rf_optimizado.csv')
                print(f"✅ Datos optimizados de {len(self.datos['clientes'])} clientes cargados")
            except:
                print("⚠️ Datos optimizados de clientes no disponibles")

            try:
                self.datos['productos'] = pd.read_csv(f'{self.modelos_dir}/producto_metricas_rf_optimizado.csv')
                print(f"✅ Datos optimizados de {len(self.datos['productos'])} productos cargados")
            except:
                print("⚠️ Datos optimizados de productos no disponibles")

        except Exception as e:
            print(f"❌ Error cargando modelos optimizados: {e}")

    def aplicar_feature_selection(self, X, modelo_num):
        """
        Aplica la selección de features optimizada según el modelo
        """
        try:
            if f'selector_m{modelo_num}' in self.selectores:
                selector = self.selectores[f'selector_m{modelo_num}']
                X_selected = selector.transform(X)
                return X_selected
            else:
                return X
        except Exception as e:
            print(f"⚠️ Error en feature selection: {e}")
            return X

    def predecir_proxima_compra_optimizado(self, cliente_id):
        """
        Predice cuándo será la próxima compra de un cliente usando modelo optimizado

        Args:
            cliente_id: ID del cliente

        Returns:
            dict: Predicción optimizada con fecha estimada y confianza
        """
        if 'modelo1' not in self.modelos:
            return {"error": "Modelo 1 Optimizado no disponible"}

        try:
            cliente_data = self.datos['clientes'][self.datos['clientes']['cliente_id'] == cliente_id]

            if len(cliente_data) == 0:
                return {"error": f"Cliente {cliente_id} no encontrado en la base de datos optimizada"}

            cliente_data = cliente_data.iloc[0]

            # Preparar features según el modelo optimizado entrenado
            features_dict = {
                'total_compras_historicas': cliente_data.get('total_compras', 0),
                'gasto_total_historico': cliente_data.get('gasto_total_productos', 0),
                'gasto_promedio_historico': cliente_data.get('gasto_promedio_productos', 0),
                'recencia_desde_anterior': cliente_data.get('recencia_dias', 30),
                'intervalo_promedio_historico': cliente_data.get('intervalo_promedio_dias', 30),
                'variabilidad_intervalos': cliente_data.get('regularidad_compras', 0.5) * 10,
                'mes': datetime.now().month,
                'trimestre': (datetime.now().month - 1) // 3 + 1,
                'dia_semana': datetime.now().weekday(),
                'es_fin_semana': 1 if datetime.now().weekday() >= 5 else 0,
                'semana_mes': datetime.now().day // 7 + 1,
                'tipo_negocio_encoded': cliente_data.get('tipo_negocio_encoded', 0),
                'ciudad_encoded': cliente_data.get('ciudad_encoded', 0),
                'categorias_distintas': cliente_data.get('categorias_distintas', 1),
                'productos_distintos': cliente_data.get('productos_distintos', 1),
                'tendencia_gasto': cliente_data.get('tendencia_gasto', 0),
                'regularidad_compras': cliente_data.get('regularidad_compras', 0.5)
            }

            # Crear DataFrame con todas las features originales
            features_originales = [
                'total_compras_historicas', 'gasto_total_historico', 'gasto_promedio_historico',
                'recencia_desde_anterior', 'intervalo_promedio_historico', 'variabilidad_intervalos',
                'mes', 'trimestre', 'dia_semana', 'es_fin_semana', 'semana_mes',
                'tipo_negocio_encoded', 'ciudad_encoded', 'categorias_distintas', 'productos_distintos',
                'tendencia_gasto', 'regularidad_compras'
            ]

            X_original = pd.DataFrame([features_dict])
            X_original = X_original[features_originales]

            # Aplicar feature selection optimizada
            X_selected = self.aplicar_feature_selection(X_original, 1)

            # Predecir días hasta próxima compra
            dias_predichos = self.modelos['modelo1'].predict(X_selected)[0]
            fecha_proxima = datetime.now() + timedelta(days=int(dias_predichos))

            # Calcular nivel de confianza mejorado
            total_compras = cliente_data.get('total_compras', 0)
            recencia = cliente_data.get('recencia_dias', 999)

            if total_compras >= 15 and recencia <= 45:
                confianza = "Muy Alta"
            elif total_compras >= 10 and recencia <= 60:
                confianza = "Alta"
            elif total_compras >= 5 and recencia <= 120:
                confianza = "Media"
            else:
                confianza = "Baja"

            return {
                'cliente_id': cliente_id,
                'dias_hasta_proxima': int(dias_predichos),
                'fecha_proxima_estimada': fecha_proxima.strftime('%Y-%m-%d'),
                'confianza': confianza,
                'total_compras_historicas': int(total_compras),
                'recencia_dias': int(recencia),
                'modelo_version': 'Random Forest Optimizado v2.0',
                'features_utilizadas': len(self.modelos['features_m1']),
                'fecha_prediccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {"error": f"Error en predicción optimizada: {str(e)}"}

    def recomendar_productos_optimizado(self, cliente_id, top_n=5):
        """
        Recomienda productos para un cliente basado en modelo optimizado

        Args:
            cliente_id: ID del cliente
            top_n: Número de productos a recomendar

        Returns:
            list: Lista de productos recomendados con probabilidades optimizadas
        """
        if 'modelo2' not in self.modelos:
            return {"error": "Modelo 2 Optimizado no disponible"}

        try:
            cliente_data = self.datos['clientes'][self.datos['clientes']['cliente_id'] == cliente_id]

            if len(cliente_data) == 0:
                return {"error": f"Cliente {cliente_id} no encontrado"}

            cliente_data = cliente_data.iloc[0]
            recomendaciones = []

            # Evaluar cada producto con modelo optimizado
            for _, producto in self.datos['productos'].iterrows():
                # Preparar features cliente-producto
                features_dict = {
                    'cliente_total_compras': cliente_data.get('total_compras', 0),
                    'cliente_gasto_promedio': cliente_data.get('gasto_promedio_productos', 0),
                    'cliente_ticket_promedio': cliente_data.get('ticket_promedio_ventas', 0),
                    'cliente_recencia': cliente_data.get('recencia_dias', 30),
                    'cliente_frecuencia_mensual': cliente_data.get('frecuencia_mensual', 1),
                    'cliente_categorias_distintas': cliente_data.get('categorias_distintas', 1),
                    'cliente_productos_distintos': cliente_data.get('productos_distintos', 1),
                    'cliente_diversidad_productos': cliente_data.get('diversidad_productos', 0.5),
                    'cliente_lealtad_marca': cliente_data.get('lealtad_marca', 0.5),
                    'cliente_tendencia_gasto': cliente_data.get('tendencia_gasto', 0),
                    'cliente_regularidad_compras': cliente_data.get('regularidad_compras', 0.5),
                    'producto_popularidad': producto.get('producto_popularidad', 0.01),
                    'producto_penetracion': producto.get('producto_penetracion', 0.01),
                    'producto_precio_promedio': producto.get('producto_precio_promedio', 100),
                    'producto_ventas_promedio': producto.get('producto_ventas_promedio', 50),
                    'producto_clientes_unicos': producto.get('producto_clientes_unicos', 1),
                    'producto_frecuencia_cliente': producto.get('producto_frecuencia_cliente', 1),
                    'producto_dias_mercado': producto.get('producto_dias_mercado', 30),
                    'producto_recencia': producto.get('producto_recencia', 30),
                    'tipo_negocio_encoded': cliente_data.get('tipo_negocio_encoded', 0),
                    'ciudad_encoded': cliente_data.get('ciudad_encoded', 0)
                }

                # Crear DataFrame y aplicar feature selection optimizada
                features_originales = [
                    'cliente_total_compras', 'cliente_gasto_promedio', 'cliente_ticket_promedio',
                    'cliente_recencia', 'cliente_frecuencia_mensual', 'cliente_categorias_distintas',
                    'cliente_productos_distintos', 'cliente_diversidad_productos', 'cliente_lealtad_marca',
                    'cliente_tendencia_gasto', 'cliente_regularidad_compras',
                    'producto_popularidad', 'producto_penetracion', 'producto_precio_promedio',
                    'producto_ventas_promedio', 'producto_clientes_unicos', 'producto_frecuencia_cliente',
                    'producto_dias_mercado', 'producto_recencia',
                    'tipo_negocio_encoded', 'ciudad_encoded'
                ]

                X_original = pd.DataFrame([features_dict])
                X_original = X_original[features_originales]

                # Aplicar feature selection optimizada
                X_selected = self.aplicar_feature_selection(X_original, 2)

                # Predecir con modelo optimizado
                probabilidad = self.modelos['modelo2'].predict_proba(X_selected)[0][1] * 100

                recomendaciones.append({
                    'producto_id': producto['producto_id'],
                    'probabilidad_compra': probabilidad,
                    'categoria': producto.get('producto_categoria', 'N/A'),
                    'marca': producto.get('producto_marca', 'N/A'),
                    'popularidad': producto.get('producto_popularidad', 0) * 100,
                    'precio_promedio': producto.get('producto_precio_promedio', 0)
                })

            # Ordenar por probabilidad y devolver top N
            recomendaciones.sort(key=lambda x: x['probabilidad_compra'], reverse=True)

            return {
                'cliente_id': cliente_id,
                'recomendaciones': recomendaciones[:top_n],
                'total_productos_evaluados': len(recomendaciones),
                'modelo_version': 'Random Forest Optimizado v2.0',
                'optimizaciones': 'SMOTE + Feature Selection + Class Weight',
                'fecha_recomendacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {"error": f"Error en recomendación optimizada: {str(e)}"}

    def detectar_cambio_patron_optimizado(self, cliente_id):
        """
        Detecta si un cliente está cambiando sus patrones usando modelo optimizado

        Args:
            cliente_id: ID del cliente

        Returns:
            dict: Análisis optimizado de cambio de patrón
        """
        if 'modelo3' not in self.modelos:
            return {"error": "Modelo 3 Optimizado no disponible"}

        try:
            cliente_data = self.datos['clientes'][self.datos['clientes']['cliente_id'] == cliente_id]

            if len(cliente_data) == 0:
                return {"error": f"Cliente {cliente_id} no encontrado"}

            cliente_data = cliente_data.iloc[0]

            # Preparar features extendidas para detección optimizada
            features_dict = {
                'total_compras': cliente_data.get('total_compras', 0),
                'gasto_total': cliente_data.get('gasto_total_productos', 0),
                'gasto_promedio': cliente_data.get('gasto_promedio_productos', 0),
                'ticket_promedio': cliente_data.get('ticket_promedio_ventas', 0),
                'recencia_dias': cliente_data.get('recencia_dias', 30),
                'antiguedad_meses': cliente_data.get('antiguedad_meses', 1),
                'frecuencia_mensual': cliente_data.get('frecuencia_mensual', 1),
                'categorias_distintas': cliente_data.get('categorias_distintas', 1),
                'productos_distintos': cliente_data.get('productos_distintos', 1),
                'diversidad_productos': cliente_data.get('diversidad_productos', 0.5),
                'lealtad_marca': cliente_data.get('lealtad_marca', 0.5),
                'variabilidad_gasto': cliente_data.get('variabilidad_gasto', 0.5),
                'tendencia_gasto': cliente_data.get('tendencia_gasto', 0),
                'tendencia_cantidad': cliente_data.get('tendencia_cantidad', 0),
                'regularidad_compras': cliente_data.get('regularidad_compras', 0.5),
                'concentracion_estacional': cliente_data.get('concentracion_estacional', 0.5),
                'tipo_negocio_encoded': cliente_data.get('tipo_negocio_encoded', 0),
                'ciudad_encoded': cliente_data.get('ciudad_encoded', 0),
                # Features adicionales optimizadas
                'max_cambio': abs(cliente_data.get('tendencia_gasto', 0)) + abs(cliente_data.get('tendencia_cantidad', 0)),
                'cambio_promedio': (abs(cliente_data.get('tendencia_gasto', 0)) + abs(cliente_data.get('tendencia_cantidad', 0))) / 2,
                'cambios_p1_p2': abs(cliente_data.get('tendencia_gasto', 0)) * 0.7,
                'cambios_p2_p3': abs(cliente_data.get('tendencia_cantidad', 0)) * 0.7
            }

            # Crear DataFrame y aplicar feature selection optimizada
            features_originales = [
                'total_compras', 'gasto_total', 'gasto_promedio', 'ticket_promedio',
                'recencia_dias', 'antiguedad_meses', 'frecuencia_mensual',
                'categorias_distintas', 'productos_distintos', 'diversidad_productos',
                'lealtad_marca', 'variabilidad_gasto', 'tendencia_gasto', 'tendencia_cantidad',
                'regularidad_compras', 'concentracion_estacional',
                'tipo_negocio_encoded', 'ciudad_encoded',
                'max_cambio', 'cambio_promedio', 'cambios_p1_p2', 'cambios_p2_p3'
            ]

            X_original = pd.DataFrame([features_dict])
            X_original = X_original[features_originales]

            # Aplicar feature selection optimizada
            X_selected = self.aplicar_feature_selection(X_original, 3)

            # Predecir con modelo optimizado
            probabilidad_cambio = self.modelos['modelo3'].predict_proba(X_selected)[0][1] * 100
            cambio_detectado = self.modelos['modelo3'].predict(X_selected)[0]

            # Determinar nivel de riesgo optimizado
            if probabilidad_cambio >= 85:
                nivel_riesgo = "Crítico"
                recomendacion = "Contacto inmediato del gerente comercial"
                urgencia = "CRÍTICA"
            elif probabilidad_cambio >= 70:
                nivel_riesgo = "Muy Alto"
                recomendacion = "Acción comercial en las próximas 24 horas"
                urgencia = "MUY ALTA"
            elif probabilidad_cambio >= 55:
                nivel_riesgo = "Alto"
                recomendacion = "Contacto comercial en las próximas 48 horas"
                urgencia = "ALTA"
            elif probabilidad_cambio >= 40:
                nivel_riesgo = "Medio"
                recomendacion = "Monitoreo cercano y contacto en la próxima semana"
                urgencia = "MEDIA"
            elif probabilidad_cambio >= 25:
                nivel_riesgo = "Bajo"
                recomendacion = "Incluir en próxima campaña de retención"
                urgencia = "BAJA"
            else:
                nivel_riesgo = "Muy Bajo"
                recomendacion = "Mantener estrategia actual"
                urgencia = "NORMAL"

            return {
                'cliente_id': cliente_id,
                'cambio_detectado': bool(cambio_detectado),
                'probabilidad_cambio': probabilidad_cambio,
                'nivel_riesgo': nivel_riesgo,
                'urgencia': urgencia,
                'recomendacion': recomendacion,
                'modelo_version': 'Random Forest Optimizado v2.0',
                'optimizaciones': 'Grid Search + SMOTE + Features Adicionales',
                'metricas_cliente': {
                    'total_compras': int(cliente_data.get('total_compras', 0)),
                    'recencia_dias': int(cliente_data.get('recencia_dias', 0)),
                    'tendencia_gasto': float(cliente_data.get('tendencia_gasto', 0)),
                    'regularidad_compras': float(cliente_data.get('regularidad_compras', 0))
                },
                'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {"error": f"Error en detección optimizada: {str(e)}"}

    def analisis_completo_cliente_optimizado(self, cliente_id):
        """
        Realiza un análisis completo optimizado del cliente usando los 3 modelos

        Args:
            cliente_id: ID del cliente

        Returns:
            dict: Análisis completo optimizado con todas las predicciones
        """
        print(f"🔍 Analizando cliente {cliente_id} con modelos optimizados...")

        # Obtener información básica del cliente
        cliente_info = self.datos['clientes'][self.datos['clientes']['cliente_id'] == cliente_id]
        if len(cliente_info) == 0:
            return {"error": f"Cliente {cliente_id} no encontrado"}

        cliente_info = cliente_info.iloc[0]

        # Realizar todas las predicciones optimizadas
        proxima_compra = self.predecir_proxima_compra_optimizado(cliente_id)
        productos_recomendados = self.recomendar_productos_optimizado(cliente_id, top_n=5)
        cambio_patron = self.detectar_cambio_patron_optimizado(cliente_id)

        return {
            'cliente_id': cliente_id,
            'informacion_basica': {
                'total_compras': int(cliente_info.get('total_compras', 0)),
                'gasto_total': float(cliente_info.get('gasto_total_productos', 0)),
                'gasto_promedio': float(cliente_info.get('gasto_promedio_productos', 0)),
                'recencia_dias': int(cliente_info.get('recencia_dias', 0)),
                'tipo_negocio': cliente_info.get('tipo_negocio', 'N/A'),
                'ciudad': cliente_info.get('ciudad', 'N/A')
            },
            'prediccion_proxima_compra': proxima_compra,
            'productos_recomendados': productos_recomendados,
            'deteccion_cambio_patron': cambio_patron,
            'version_sistema': 'Random Forest Optimizado v2.0',
            'optimizaciones_aplicadas': [
                'Feature Selection automática',
                'Balanceado SMOTE',
                'Hiperparámetros optimizados',
                'Class Weight Balanced',
                'Grid Search exhaustivo'
            ],
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def ejemplo_uso_optimizado():
    """
    Función de ejemplo mostrando cómo usar el predictor optimizado
    """
    # Inicializar predictor optimizado
    predictor = RandomForestPredictorOptimizado('ruta/a/modelos/optimizados')

    # Ejemplo de uso con un cliente
    cliente_id = 'CLIENTE_001'

    print(f"📊 Análisis completo OPTIMIZADO para cliente {cliente_id}")
    print("=" * 70)

    # Análisis completo optimizado
    resultado = predictor.analisis_completo_cliente_optimizado(cliente_id)

    if 'error' not in resultado:
        print(f"👤 Cliente: {cliente_id}")
        print(f"🏢 Tipo: {resultado['informacion_basica']['tipo_negocio']}")
        print(f"📍 Ciudad: {resultado['informacion_basica']['ciudad']}")
        print(f"🛒 Total compras: {resultado['informacion_basica']['total_compras']}")
        print(f"💰 Gasto total: Bs. {resultado['informacion_basica']['gasto_total']:,.2f}")
        print(f"🚀 Sistema: {resultado['version_sistema']}")

        # Próxima compra optimizada
        if 'error' not in resultado['prediccion_proxima_compra']:
            pc = resultado['prediccion_proxima_compra']
            print(f"\n📅 Próxima compra estimada: {pc['fecha_proxima_estimada']} ({pc['dias_hasta_proxima']} días)")
            print(f"   Confianza OPTIMIZADA: {pc['confianza']} ({pc['features_utilizadas']} features)")

        # Productos recomendados optimizados
        if 'error' not in resultado['productos_recomendados']:
            print(f"\n🛒 Top 3 productos recomendados (OPTIMIZADO):")
            for i, prod in enumerate(resultado['productos_recomendados']['recomendaciones'][:3]):
                print(f"   {i+1}. Producto {prod['producto_id']} - {prod['probabilidad_compra']:.1f}% prob. OPTIMIZADA")

        # Cambio de patrón optimizado
        if 'error' not in resultado['deteccion_cambio_patron']:
            cp = resultado['deteccion_cambio_patron']
            print(f"\n📈 Detección OPTIMIZADA: {cp['probabilidad_cambio']:.1f}% - Riesgo {cp['nivel_riesgo']}")
            print(f"   Recomendación: {cp['recomendacion']}")
            print(f"   Urgencia: {cp['urgencia']}")

        print(f"\n🚀 Optimizaciones aplicadas:")
        for opt in resultado['optimizaciones_aplicadas']:
            print(f"   • {opt}")

    else:
        print(f"❌ Error: {resultado['error']}")

# EJEMPLO DE INTEGRACIÓN CON API OPTIMIZADA
class RandomForestAPIOptimizada:
    """
    Clase para integración con API REST optimizada
    """

    def __init__(self, modelos_dir):
        self.predictor = RandomForestPredictorOptimizado(modelos_dir)

    def endpoint_proxima_compra_optimizado(self, cliente_id):
        """Endpoint optimizado para predicción de próxima compra"""
        return self.predictor.predecir_proxima_compra_optimizado(cliente_id)

    def endpoint_productos_optimizado(self, cliente_id, top_n=5):
        """Endpoint optimizado para recomendación de productos"""
        return self.predictor.recomendar_productos_optimizado(cliente_id, top_n)

    def endpoint_cambio_patron_optimizado(self, cliente_id):
        """Endpoint optimizado para detección de cambios"""
        return self.predictor.detectar_cambio_patron_optimizado(cliente_id)

    def endpoint_analisis_completo_optimizado(self, cliente_id):
        """Endpoint optimizado para análisis completo"""
        return self.predictor.analisis_completo_cliente_optimizado(cliente_id)

if __name__ == "__main__":
    # Ejecutar ejemplo optimizado
    ejemplo_uso_optimizado()
