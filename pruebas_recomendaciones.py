"""
SCRIPT DE PRUEBAS Y DIAGNÓSTICO - SISTEMA DE RECOMENDACIONES
===========================================================

Este script te ayuda a:
1. Verificar qué datos tienes disponibles
2. Probar recomendaciones específicas
3. Diagnosticar problemas
4. Generar ejemplos prácticos
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


class DiagnosticoRecomendaciones:
    """
    Clase para diagnosticar y probar el sistema de recomendaciones
    """

    def __init__(self, directorio_resultados):
        """
        Inicializa el diagnóstico cargando los datos generados

        Args:
            directorio_resultados: Directorio donde están los resultados del sistema
        """
        self.directorio = directorio_resultados
        self.interacciones = None
        self.productos_info = None
        self.clientes_info = None
        self.cargar_datos()

    def cargar_datos(self):
        """Carga los datos generados por el sistema"""
        try:
            self.interacciones = pd.read_csv(os.path.join(self.directorio, 'interacciones_usuario_producto.csv'))
            self.productos_info = pd.read_csv(os.path.join(self.directorio, 'productos_info_recomendacion.csv'))
            self.clientes_info = pd.read_csv(os.path.join(self.directorio, 'clientes_info_recomendacion.csv'))
            print("✅ Datos cargados exitosamente")
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")

    def mostrar_resumen_datos(self):
        """Muestra un resumen completo de los datos disponibles"""
        print("\n" + "=" * 60)
        print("📊 RESUMEN COMPLETO DE DATOS DISPONIBLES")
        print("=" * 60)

        if self.interacciones is not None:
            print(f"\n🔗 INTERACCIONES USUARIO-PRODUCTO:")
            print(f"   • Total interacciones: {len(self.interacciones):,}")
            print(f"   • Usuarios únicos: {self.interacciones['cliente_id'].nunique():,}")
            print(f"   • Productos únicos: {self.interacciones['producto_id'].nunique():,}")
            print(f"   • Rating promedio: {self.interacciones['rating'].mean():.2f}")
            print(
                f"   • Rango de ratings: {self.interacciones['rating'].min():.2f} - {self.interacciones['rating'].max():.2f}")

            print(f"\n👥 TOP 5 USUARIOS MÁS ACTIVOS:")
            top_usuarios = self.interacciones['cliente_id'].value_counts().head()
            for i, (cliente, compras) in enumerate(top_usuarios.items()):
                print(f"   {i + 1}. Cliente {cliente}: {compras} productos comprados")

            print(f"\n🛒 TOP 5 PRODUCTOS MÁS POPULARES:")
            top_productos = self.interacciones['producto_id'].value_counts().head()
            for i, (producto, clientes) in enumerate(top_productos.items()):
                print(f"   {i + 1}. Producto {producto}: {clientes} clientes lo compraron")

        if self.clientes_info is not None:
            print(f"\n🏢 TIPOS DE NEGOCIO DISPONIBLES:")
            tipos_negocio = self.clientes_info['tipo_negocio'].value_counts()
            for tipo, cantidad in tipos_negocio.items():
                print(f"   • {tipo}: {cantidad} establecimientos")

            print(f"\n📍 CIUDADES DISPONIBLES:")
            ciudades = self.clientes_info['ciudad'].value_counts().head()
            for ciudad, cantidad in ciudades.items():
                print(f"   • {ciudad}: {cantidad} establecimientos")

        if self.productos_info is not None:
            print(f"\n📦 INFORMACIÓN DE PRODUCTOS:")
            print(f"   • Total productos: {len(self.productos_info):,}")
            if 'categoria' in self.productos_info.columns:
                print(f"   • Categorías únicas: {self.productos_info['categoria'].nunique()}")
                print(f"   • Top categorías:")
                top_categorias = self.productos_info['categoria'].value_counts().head()
                for categoria, cantidad in top_categorias.items():
                    print(f"     - {categoria}: {cantidad} productos")

            if 'marca' in self.productos_info.columns:
                print(f"   • Marcas únicas: {self.productos_info['marca'].nunique()}")

    def buscar_clientes_activos(self, min_productos=3):
        """
        Encuentra clientes con suficientes productos para generar recomendaciones

        Args:
            min_productos: Número mínimo de productos comprados
        """
        if self.interacciones is None:
            print("❌ No hay datos de interacciones cargados")
            return []

        clientes_activos = self.interacciones['cliente_id'].value_counts()
        clientes_validos = clientes_activos[clientes_activos >= min_productos]

        print(f"\n🔍 CLIENTES CON ≥{min_productos} PRODUCTOS COMPRADOS:")
        print(f"   • Total clientes válidos: {len(clientes_validos)}")

        if len(clientes_validos) > 0:
            print(f"   • Top 10 clientes activos:")
            for i, (cliente, productos) in enumerate(clientes_validos.head(10).items()):
                tipo_negocio = "N/A"
                if self.clientes_info is not None:
                    cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente]
                    if len(cliente_info) > 0:
                        tipo_negocio = cliente_info.iloc[0]['tipo_negocio']

                print(f"     {i + 1}. Cliente {cliente}: {productos} productos ({tipo_negocio})")

            return clientes_validos.index.tolist()
        else:
            print("   ⚠️ No se encontraron clientes con suficientes productos")
            return []

    def analizar_productos_por_categoria(self):
        """Analiza los productos disponibles por categoría"""
        if self.productos_info is None:
            print("❌ No hay información de productos cargada")
            return

        print(f"\n📦 ANÁLISIS DETALLADO DE PRODUCTOS:")

        # Mostrar todas las categorías disponibles
        if 'categoria' in self.productos_info.columns:
            print(f"\n🏷️ TODAS LAS CATEGORÍAS DISPONIBLES:")
            categorias = self.productos_info['categoria'].value_counts()
            for categoria, cantidad in categorias.items():
                print(f"   • {categoria}: {cantidad} productos")

                # Mostrar algunos productos de ejemplo de cada categoría
                productos_categoria = self.productos_info[
                    self.productos_info['categoria'] == categoria
                    ]['producto_id'].head(3).tolist()
                print(f"     Ejemplos: {', '.join(map(str, productos_categoria))}")

        # Buscar productos relacionados con quesos
        print(f"\n🧀 BÚSQUEDA DE PRODUCTOS RELACIONADOS CON QUESOS:")
        palabras_queso = ['queso', 'cheese', 'mozzarella', 'parmesano', 'cheddar', 'gouda', 'brie']

        productos_queso = []
        for palabra in palabras_queso:
            if 'categoria' in self.productos_info.columns:
                coincidencias_cat = self.productos_info[
                    self.productos_info['categoria'].str.contains(palabra, case=False, na=False)
                ]
                productos_queso.extend(coincidencias_cat['producto_id'].tolist())

            if 'marca' in self.productos_info.columns:
                coincidencias_marca = self.productos_info[
                    self.productos_info['marca'].str.contains(palabra, case=False, na=False)
                ]
                productos_queso.extend(coincidencias_marca['producto_id'].tolist())

        productos_queso = list(set(productos_queso))

        if productos_queso:
            print(f"   ✅ Encontrados {len(productos_queso)} productos relacionados con quesos:")
            for producto in productos_queso[:10]:  # Mostrar primeros 10
                print(f"     • Producto {producto}")
        else:
            print(f"   ⚠️ No se encontraron productos claramente relacionados con quesos")
            print(f"   💡 El sistema usará categorías genéricas para las recomendaciones")

    def probar_recomendacion_manual(self, cliente_id, tipo_recomendacion='general'):
        """
        Prueba manualmente una recomendación para un cliente específico

        Args:
            cliente_id: ID del cliente a probar
            tipo_recomendacion: 'general', 'pizza', 'nuevos', 'tendencias'
        """
        if self.interacciones is None:
            print("❌ No hay datos cargados")
            return

        print(f"\n🧪 PRUEBA MANUAL DE RECOMENDACIÓN")
        print(f"   Cliente: {cliente_id}")
        print(f"   Tipo: {tipo_recomendacion}")
        print("-" * 50)

        # Verificar que el cliente existe
        if cliente_id not in self.interacciones['cliente_id'].values:
            print(f"❌ Cliente {cliente_id} no encontrado en las interacciones")
            print(f"   Clientes disponibles: {self.interacciones['cliente_id'].unique()[:10].tolist()}...")
            return

        # Mostrar información del cliente
        cliente_interacciones = self.interacciones[self.interacciones['cliente_id'] == cliente_id]
        productos_comprados = cliente_interacciones['producto_id'].tolist()
        rating_promedio = cliente_interacciones['rating'].mean()

        print(f"📊 INFORMACIÓN DEL CLIENTE {cliente_id}:")
        print(f"   • Productos comprados: {len(productos_comprados)}")
        print(f"   • Rating promedio: {rating_promedio:.2f}")
        print(f"   • Productos: {productos_comprados}")

        if self.clientes_info is not None:
            cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente_id]
            if len(cliente_info) > 0:
                cliente_info = cliente_info.iloc[0]
                print(f"   • Tipo de negocio: {cliente_info['tipo_negocio']}")
                print(f"   • Ciudad: {cliente_info['ciudad']}")

        # Generar recomendaciones manuales simples
        if tipo_recomendacion == 'general':
            self._generar_recomendaciones_generales_manual(cliente_id, productos_comprados)
        elif tipo_recomendacion == 'pizza':
            self._generar_recomendaciones_pizza_manual(cliente_id, productos_comprados)
        elif tipo_recomendacion == 'nuevos':
            self._generar_recomendaciones_nuevos_manual(cliente_id)
        elif tipo_recomendacion == 'tendencias':
            self._generar_recomendaciones_tendencias_manual(cliente_id)

    def _generar_recomendaciones_generales_manual(self, cliente_id, productos_comprados):
        """Genera recomendaciones generales de forma manual"""
        print(f"\n🎯 RECOMENDACIONES GENERALES MANUAL:")

        # Productos más populares que el cliente no ha comprado
        todos_productos = self.interacciones['producto_id'].value_counts()
        productos_no_comprados = [p for p in todos_productos.index if p not in productos_comprados]

        if productos_no_comprados:
            print(f"   📈 Top 5 productos populares no comprados:")
            for i, producto in enumerate(productos_no_comprados[:5]):
                popularidad = todos_productos[producto]
                print(f"     {i + 1}. Producto {producto} (comprado por {popularidad} clientes)")
        else:
            print(f"   ⚠️ El cliente ya compró todos los productos disponibles")

    def _generar_recomendaciones_pizza_manual(self, cliente_id, productos_comprados):
        """Genera recomendaciones para pizza de forma manual"""
        print(f"\n🍕 RECOMENDACIONES PARA PIZZA MANUAL:")

        # Buscar productos que otros clientes compran junto con los productos del cliente
        productos_relacionados = {}

        for producto_comprado in productos_comprados:
            # Encontrar otros clientes que compraron este producto
            otros_clientes = self.interacciones[
                self.interacciones['producto_id'] == producto_comprado
                ]['cliente_id'].unique()

            # Ver qué otros productos compran esos clientes
            for otro_cliente in otros_clientes:
                if otro_cliente != cliente_id:
                    productos_otro = self.interacciones[
                        self.interacciones['cliente_id'] == otro_cliente
                        ]['producto_id'].tolist()

                    for producto in productos_otro:
                        if producto not in productos_comprados:
                            if producto not in productos_relacionados:
                                productos_relacionados[producto] = 0
                            productos_relacionados[producto] += 1

        if productos_relacionados:
            # Ordenar por frecuencia
            productos_recomendados = sorted(productos_relacionados.items(),
                                            key=lambda x: x[1], reverse=True)

            print(f"   🧀 Productos que otros clientes similares compraron:")
            for i, (producto, frecuencia) in enumerate(productos_recomendados[:5]):
                print(f"     {i + 1}. Producto {producto} (frecuencia: {frecuencia})")
        else:
            print(f"   ⚠️ No se encontraron productos relacionados suficientes")

    def _generar_recomendaciones_nuevos_manual(self, cliente_id):
        """Genera recomendaciones de nuevos productos manual"""
        print(f"\n🆕 RECOMENDACIONES DE NUEVOS PRODUCTOS MANUAL:")

        # Encontrar clientes del mismo tipo de negocio
        if self.clientes_info is not None:
            cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente_id]
            if len(cliente_info) > 0:
                tipo_negocio = cliente_info.iloc[0]['tipo_negocio']

                clientes_mismo_tipo = self.clientes_info[
                    (self.clientes_info['tipo_negocio'] == tipo_negocio) &
                    (self.clientes_info['cliente_id'] != cliente_id)
                    ]['cliente_id'].tolist()

                if clientes_mismo_tipo:
                    print(f"   🏢 Analizando {len(clientes_mismo_tipo)} establecimientos de tipo '{tipo_negocio}'")

                    # Productos populares en este tipo de negocio
                    interacciones_tipo = self.interacciones[
                        self.interacciones['cliente_id'].isin(clientes_mismo_tipo)
                    ]

                    productos_tipo = interacciones_tipo['producto_id'].value_counts()
                    productos_cliente = set(self.interacciones[
                                                self.interacciones['cliente_id'] == cliente_id
                                                ]['producto_id'])

                    productos_nuevos = [p for p in productos_tipo.index if p not in productos_cliente]

                    if productos_nuevos:
                        print(f"   📦 Productos populares en tu tipo de negocio que no tienes:")
                        for i, producto in enumerate(productos_nuevos[:5]):
                            clientes_que_compran = productos_tipo[producto]
                            adopcion = clientes_que_compran / len(clientes_mismo_tipo) * 100
                            print(f"     {i + 1}. Producto {producto} (adopción: {adopcion:.1f}%)")
                    else:
                        print(f"   ✅ Ya tienes todos los productos populares de tu tipo de negocio")
                else:
                    print(f"   ⚠️ No hay otros clientes del mismo tipo de negocio")
        else:
            print(f"   ⚠️ No hay información de tipos de negocio disponible")

    def _generar_recomendaciones_tendencias_manual(self, cliente_id):
        """Genera análisis de tendencias manual"""
        print(f"\n📈 ANÁLISIS DE TENDENCIAS MANUAL:")

        if self.clientes_info is not None:
            cliente_info = self.clientes_info[self.clientes_info['cliente_id'] == cliente_id]
            if len(cliente_info) > 0:
                tipo_negocio = cliente_info.iloc[0]['tipo_negocio']

                # Productos más populares por tipo de negocio
                clientes_mismo_tipo = self.clientes_info[
                    self.clientes_info['tipo_negocio'] == tipo_negocio
                    ]['cliente_id'].tolist()

                interacciones_tipo = self.interacciones[
                    self.interacciones['cliente_id'].isin(clientes_mismo_tipo)
                ]

                # Calcular métricas de tendencia
                tendencias = interacciones_tipo.groupby('producto_id').agg({
                    'cliente_id': 'nunique',
                    'rating': 'mean',
                    'cantidad_total': 'sum'
                }).reset_index()

                tendencias.columns = ['producto_id', 'clientes_unicos', 'rating_promedio', 'cantidad_total']
                tendencias['penetracion'] = tendencias['clientes_unicos'] / len(clientes_mismo_tipo)

                # Ordenar por penetración
                tendencias = tendencias.sort_values('penetracion', ascending=False)

                print(f"   📊 Tendencias en tipo '{tipo_negocio}' ({len(clientes_mismo_tipo)} establecimientos):")
                for i, row in tendencias.head(5).iterrows():
                    print(f"     {i + 1}. Producto {row['producto_id']}:")
                    print(f"        • Penetración: {row['penetracion']:.1%}")
                    print(f"        • Rating promedio: {row['rating_promedio']:.2f}")
                    print(f"        • Clientes que lo usan: {row['clientes_unicos']}")


def main():
    """Función principal para ejecutar las pruebas"""
    print("🧪 SISTEMA DE PRUEBAS DE RECOMENDACIONES")
    print("=" * 50)

    # Buscar el directorio de resultados más reciente
    directorios_resultados = [d for d in os.listdir('.') if d.startswith('resultados_FILTRADO_COLABORATIVO')]

    if not directorios_resultados:
        print("❌ No se encontraron directorios de resultados")
        print("   Asegúrate de haber ejecutado el sistema de recomendaciones primero")
        return

    # Usar el directorio más reciente
    directorio_reciente = sorted(directorios_resultados)[-1]
    print(f"📁 Usando directorio: {directorio_reciente}")

    # Inicializar diagnóstico
    diagnostico = DiagnosticoRecomendaciones(directorio_reciente)

    # Ejecutar diagnósticos
    diagnostico.mostrar_resumen_datos()

    # Buscar clientes activos
    clientes_activos = diagnostico.buscar_clientes_activos(min_productos=2)

    # Analizar productos
    diagnostico.analizar_productos_por_categoria()

    # Probar recomendaciones con clientes activos
    if clientes_activos:
        print(f"\n🧪 PROBANDO RECOMENDACIONES CON CLIENTES ACTIVOS:")

        # Probar con los primeros 3 clientes más activos
        for i, cliente in enumerate(clientes_activos[:3]):
            print(f"\n--- CLIENTE {i + 1}: {cliente} ---")

            # Probar diferentes tipos de recomendaciones
            diagnostico.probar_recomendacion_manual(cliente, 'general')
            diagnostico.probar_recomendacion_manual(cliente, 'pizza')
            diagnostico.probar_recomendacion_manual(cliente, 'nuevos')
            diagnostico.probar_recomendacion_manual(cliente, 'tendencias')

            if i == 0:  # Solo mostrar ejemplo detallado del primer cliente
                print(f"\n" + "=" * 60)
                print(f"💡 EJEMPLO PRÁCTICO DE USO:")
                print(f"=" * 60)
                print(f"# Para usar este cliente en el sistema real:")
                print(f"cliente_id = '{cliente}'")
                print(f"")
                print(f"# Recomendaciones generales:")
                print(f"recomendaciones = sistema.obtener_recomendaciones_generales(cliente_id, n_recomendaciones=10)")
                print(f"")
                print(f"# Combinaciones para pizza:")
                print(f"pizza_recs = sistema.obtener_combinaciones_platillos(cliente_id, 'pizza', 5)")
                print(f"")
                print(f"# Nuevos productos:")
                print(f"nuevos = sistema.obtener_nuevos_productos(cliente_id, 5)")
                print(f"")
                print(f"# Análisis completo:")
                print(f"analisis = sistema.obtener_recomendaciones_completas(cliente_id)")

    print(f"\n✅ DIAGNÓSTICO COMPLETADO")
    print(f"📝 RESUMEN:")
    print(f"   • Usa los clientes activos identificados para probar recomendaciones")
    print(f"   • El sistema funciona mejor con clientes que tienen ≥3 productos")
    print(f"   • Si no hay productos de quesos específicos, el sistema usa categorías genéricas")


if __name__ == "__main__":
    main()