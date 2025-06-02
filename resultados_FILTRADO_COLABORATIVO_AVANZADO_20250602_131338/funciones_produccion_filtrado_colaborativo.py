
"""
FUNCIONES DE PRODUCCIÓN - SISTEMA DE RECOMENDACIÓN CON FILTRADO COLABORATIVO
Generado automáticamente el 2025-06-02 13:13:41

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
            self.sistema_recom = joblib.load(f'{self.modelos_dir}/sistema_recomendacion_colaborativo.pkl')
            print("✅ Sistema de recomendación principal cargado")

            # Cargar motor especializado
            self.motor_quesos = joblib.load(f'{self.modelos_dir}/motor_recomendacion_quesos.pkl')
            print("✅ Motor de recomendación de quesos cargado")

            # Cargar datos
            self.interacciones = pd.read_csv(f'{self.modelos_dir}/interacciones_usuario_producto.csv')
            self.productos_info = pd.read_csv(f'{self.modelos_dir}/productos_info_recomendacion.csv')
            self.clientes_info = pd.read_csv(f'{self.modelos_dir}/clientes_info_recomendacion.csv')
            print("✅ Datos de referencia cargados")

        except Exception as e:
            print(f"❌ Error cargando sistema: {e}")

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
                return {"error": f"Cliente {cliente_id} no encontrado en el sistema"}

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
                            recomendaciones.append({
                                'producto_id': producto_id,
                                'score': score,
                                'categoria': producto_info['categoria'],
                                'marca': producto_info['marca'],
                                'popularidad_general': producto_info['clientes_unicos'],
                                'algoritmo_usado': algoritmo
                            })

                    except KeyError:
                        continue

            # Ordenar por score y devolver top N
            recomendaciones.sort(key=lambda x: x['score'], reverse=True)

            # Obtener información del cliente
            cliente_info = self.clientes_info[
                self.clientes_info['cliente_id'] == cliente_id
            ].iloc[0]

            return {
                'cliente_id': cliente_id,
                'tipo_negocio': cliente_info['tipo_negocio'],
                'ciudad': cliente_info['ciudad'],
                'recomendaciones': recomendaciones[:n_recomendaciones],
                'algoritmo_usado': algoritmo,
                'productos_ya_comprados': len(productos_comprados),
                'fecha_recomendacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {"error": f"Error generando recomendaciones: {str(e)}"}

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
            return {"error": f"Error en recomendaciones de platillos: {str(e)}"}

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
            return {"error": f"Error en recomendaciones de nuevos productos: {str(e)}"}

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
            return {"error": f"Error en análisis de tendencias: {str(e)}"}

    def obtener_recomendaciones_completas(self, cliente_id):
        """
        Obtiene un análisis completo con todos los tipos de recomendaciones

        Args:
            cliente_id: ID del cliente

        Returns:
            dict: Análisis completo con todos los tipos de recomendaciones
        """
        print(f"🔍 Generando análisis completo para cliente {cliente_id}...")

        # Verificar que el cliente existe
        if cliente_id not in self.clientes_info['cliente_id'].values:
            return {"error": f"Cliente {cliente_id} no encontrado"}

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

            return {
                'cliente_id': cliente_id,
                'informacion_cliente': {
                    'tipo_negocio': cliente_info['tipo_negocio'],
                    'ciudad': cliente_info['ciudad'],
                    'productos_distintos': int(cliente_info['productos_distintos']),
                    'gasto_total': float(cliente_info['gasto_total'])
                },
                'recomendaciones_generales': recomendaciones_generales,
                'combinaciones_platillos': {
                    'pizza': combinaciones_pizza,
                    'pasta': combinaciones_pasta
                },
                'nuevos_productos': nuevos_productos,
                'tendencias_tipo_negocio': tendencias,
                'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version_sistema': 'Filtrado Colaborativo v1.0'
            }

        except Exception as e:
            return {"error": f"Error en análisis completo: {str(e)}"}

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
                return {"error": f"Cliente {cliente_id} no encontrado"}

            if producto_id not in self.sistema_recom.item_to_idx:
                return {"error": f"Producto {producto_id} no encontrado"}

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

            return {
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
            }

        except Exception as e:
            return {"error": f"Error evaluando producto: {str(e)}"}

def ejemplo_uso_completo():
    """
    Función de ejemplo mostrando cómo usar el sistema completo
    """
    # Inicializar sistema
    sistema = SistemaRecomendacionProduccion('ruta/a/modelos')

    # Ejemplo con un cliente
    cliente_id = 'CLIENTE_001'

    print(f"🧀 Sistema de Recomendación - Cliente {cliente_id}")
    print("=" * 60)

    # Análisis completo
    analisis_completo = sistema.obtener_recomendaciones_completas(cliente_id)

    if 'error' not in analisis_completo:
        print(f"👤 Cliente: {cliente_id}")
        print(f"🏢 Tipo: {analisis_completo['informacion_cliente']['tipo_negocio']}")
        print(f"📍 Ciudad: {analisis_completo['informacion_cliente']['ciudad']}")

        # Recomendaciones generales
        if 'error' not in analisis_completo['recomendaciones_generales']:
            print(f"\n🎯 Top 5 Recomendaciones Generales:")
            for i, rec in enumerate(analisis_completo['recomendaciones_generales']['recomendaciones'][:5]):
                print(f"   {i+1}. Producto {rec['producto_id']} - Score: {rec['score']:.2f}")

        # Combinaciones para pizza
        if 'error' not in analisis_completo['combinaciones_platillos']['pizza']:
            print(f"\n🍕 Quesos recomendados para Pizza:")
            for i, rec in enumerate(analisis_completo['combinaciones_platillos']['pizza']['recomendaciones'][:3]):
                print(f"   {i+1}. Producto {rec['producto_id']} - Score: {rec['score']:.2f}")

        # Nuevos productos
        if 'error' not in analisis_completo['nuevos_productos']:
            print(f"\n🆕 Nuevos productos recomendados:")
            for i, rec in enumerate(analisis_completo['nuevos_productos']['recomendaciones'][:3]):
                print(f"   {i+1}. Producto {rec['producto_id']} - Adopción: {rec['tasa_adopcion']:.2f}")

        # Tendencias
        if 'error' not in analisis_completo['tendencias_tipo_negocio']:
            print(f"\n📈 Tendencias en {analisis_completo['tendencias_tipo_negocio']['tipo_negocio']}:")
            for i, trend in enumerate(analisis_completo['tendencias_tipo_negocio']['tendencias'][:3]):
                print(f"   {i+1}. Producto {trend['producto_id']} - Penetración: {trend['penetracion']:.2f}")

    else:
        print(f"❌ Error: {analisis_completo['error']}")

    # Ejemplo de evaluación de producto específico
    print(f"\n🔍 Evaluación de producto específico:")
    evaluacion = sistema.evaluar_producto(cliente_id, 'PRODUCTO_123')
    if 'error' not in evaluacion:
        print(f"   Producto PRODUCTO_123: {evaluacion['probabilidad_compra']:.1f}% probabilidad")
        print(f"   Recomendación: {evaluacion['recomendacion']}")

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
