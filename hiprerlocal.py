#!/usr/bin/env python3
"""
SISTEMA DE RUTAS DE DISTRIBUCIÓN CON CLUSTERING
Análisis hiperlocal de ubicaciones de clientes y optimización de rutas

Reglas de distribución:
- Lunes y Jueves: LA PAZ
- Martes y Viernes: EL ALTO
- Pizzerías: siempre en la TARDE
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
import os
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

class SistemaRutasDistribucion:
    def __init__(self, archivo_excel="dataset inicial.xlsx"):
        self.archivo_excel = archivo_excel
        self.df_clientes = None
        self.coordenadas_clientes = None
        self.clusters_rutas = None

        # Reglas de distribución
        self.reglas_distribucion = {
            'LA PAZ': ['Lunes', 'Jueves'],
            'EL ALTO': ['Martes', 'Viernes'],
            'PIZZERIA_TURNO': 'TARDE'
        }

        # Coordenadas base para Bolivia (La Paz y El Alto)
        self.coordenadas_base = {
            'LA PAZ': {'lat': -16.5000, 'lon': -68.1193},
            'EL ALTO': {'lat': -16.5040, 'lon': -68.1644},
            'COPACABANA': {'lat': -16.1661, 'lon': -69.0864},
            'CARANAVI': {'lat': -15.8380, 'lon': -67.5690}
        }

        print("🚛 SISTEMA DE RUTAS DE DISTRIBUCIÓN INICIALIZADO")
        print("="*60)

    def cargar_datos_excel(self):
        """Carga y procesa los datos del Excel"""
        try:
            print("📊 Cargando datos del Excel...")

            # Leer el Excel
            df = pd.read_excel(self.archivo_excel)

            # Procesar datos de clientes
            clientes_data = []
            for index, row in df.iterrows():
                if pd.notna(row.iloc[1]) and pd.notna(row.iloc[2]):  # Si tiene cliente y tipo de negocio
                    cliente = {
                        'numero': row.iloc[0] if pd.notna(row.iloc[0]) else index,
                        'nombre': str(row.iloc[1]).strip(),
                        'tipo_negocio': str(row.iloc[2]).strip(),
                        'ciudad': str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else 'Sin especificar',
                        'lugar_entrega': str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else 'Sin especificar',
                        'turno': str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else 'Sin especificar'
                    }
                    clientes_data.append(cliente)

            self.df_clientes = pd.DataFrame(clientes_data)

            # Limpiar y estandarizar ciudades
            self.df_clientes['ciudad_limpia'] = self.df_clientes['ciudad'].str.upper().str.strip()
            self.df_clientes['ciudad_limpia'] = self.df_clientes['ciudad_limpia'].replace({
                'El ALTO': 'EL ALTO',
                'LA PAZ': 'LA PAZ'
            })

            print(f"✅ {len(self.df_clientes)} clientes cargados")
            print(f"📍 Ciudades: {self.df_clientes['ciudad_limpia'].value_counts().to_dict()}")
            print(f"🏢 Tipos de negocio: {self.df_clientes['tipo_negocio'].nunique()} únicos")

            return True

        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False

    def generar_coordenadas_hiperlocales(self):
        """Genera coordenadas hiperlocales para cada cliente"""
        print("\n🗺️ Generando coordenadas hiperlocales...")

        coordenadas = []

        for _, cliente in self.df_clientes.iterrows():
            ciudad = cliente['ciudad_limpia']
            lugar_entrega = cliente['lugar_entrega']

            # Obtener coordenadas base de la ciudad
            if ciudad in self.coordenadas_base:
                base_lat = self.coordenadas_base[ciudad]['lat']
                base_lon = self.coordenadas_base[ciudad]['lon']

                # Generar variación hiperlocal basada en el lugar de entrega
                # Usar hash del lugar para generar posiciones consistentes
                hash_lugar = hash(lugar_entrega) % 10000

                # Crear dispersión realista (aprox 5-15 km de radio)
                radio_km = np.random.uniform(2, 12)
                angulo = (hash_lugar / 10000) * 2 * np.pi

                # Convertir a coordenadas (aproximación)
                lat_offset = (radio_km / 111) * np.cos(angulo)  # 1 grado lat ≈ 111 km
                lon_offset = (radio_km / (111 * np.cos(np.radians(base_lat)))) * np.sin(angulo)

                lat_final = base_lat + lat_offset
                lon_final = base_lon + lon_offset

                coordenadas.append({
                    'numero': cliente['numero'],
                    'nombre': cliente['nombre'],
                    'tipo_negocio': cliente['tipo_negocio'],
                    'ciudad': ciudad,
                    'lugar_entrega': lugar_entrega,
                    'turno': cliente['turno'],
                    'lat': lat_final,
                    'lon': lon_final,
                    'dia_distribucion': self._asignar_dia_distribucion(ciudad, cliente['tipo_negocio'], cliente['turno'])
                })
            else:
                print(f"⚠️ Ciudad {ciudad} no encontrada en coordenadas base")

        self.coordenadas_clientes = pd.DataFrame(coordenadas)

        print(f"✅ {len(self.coordenadas_clientes)} coordenadas generadas")
        print("📍 Distribución por ciudad:")
        for ciudad, count in self.coordenadas_clientes['ciudad'].value_counts().items():
            print(f"   {ciudad}: {count} clientes")

        return self.coordenadas_clientes

    def _asignar_dia_distribucion(self, ciudad, tipo_negocio, turno):
        """Asigna día de distribución según reglas de negocio"""
        dias_ciudad = []

        # Asignar días según ciudad
        if ciudad == 'LA PAZ':
            dias_ciudad = ['Lunes', 'Jueves']
        elif ciudad == 'EL ALTO':
            dias_ciudad = ['Martes', 'Viernes']
        else:
            # Otras ciudades en días específicos
            dias_ciudad = ['Miércoles', 'Sábado']

        # Determinar turno
        turno_asignado = 'TARDE'
        if 'PIZZERIA' in tipo_negocio.upper():
            turno_asignado = 'TARDE'  # Pizzerías siempre en tarde
        elif turno == 'MAÑANA':
            turno_asignado = 'MAÑANA'

        # Seleccionar día (alternando para balancear carga)
        dia_seleccionado = dias_ciudad[hash(str(ciudad + tipo_negocio)) % len(dias_ciudad)]

        return f"{dia_seleccionado} - {turno_asignado}"

    def realizar_clustering_geografico(self, n_clusters_por_ciudad=3):
        """Realiza clustering geográfico por ciudad"""
        print(f"\n🎯 Realizando clustering geográfico ({n_clusters_por_ciudad} clusters por ciudad)...")

        if self.coordenadas_clientes is None:
            print("❌ Primero generar coordenadas")
            return False

        clusters_finales = []

        for ciudad in self.coordenadas_clientes['ciudad'].unique():
            print(f"\n📍 Procesando {ciudad}...")

            clientes_ciudad = self.coordenadas_clientes[
                self.coordenadas_clientes['ciudad'] == ciudad
                ].copy()

            if len(clientes_ciudad) < n_clusters_por_ciudad:
                print(f"   ⚠️ Solo {len(clientes_ciudad)} clientes, usando 1 cluster")
                clientes_ciudad['cluster'] = 0
                clientes_ciudad['cluster_global'] = f"{ciudad}_Ruta_1"
            else:
                # Preparar datos para clustering
                X = clientes_ciudad[['lat', 'lon']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Aplicar K-means
                kmeans = KMeans(n_clusters=n_clusters_por_ciudad, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                clientes_ciudad['cluster'] = clusters
                clientes_ciudad['cluster_global'] = [
                    f"{ciudad}_Ruta_{cluster+1}" for cluster in clusters
                ]

                print(f"   ✅ {n_clusters_por_ciudad} clusters creados")
                for cluster_id in range(n_clusters_por_ciudad):
                    count = sum(clusters == cluster_id)
                    print(f"      Ruta {cluster_id+1}: {count} clientes")

            clusters_finales.append(clientes_ciudad)

        self.clusters_rutas = pd.concat(clusters_finales, ignore_index=True)

        print(f"\n✅ Clustering completado:")
        print(f"   📊 Total rutas: {self.clusters_rutas['cluster_global'].nunique()}")
        print(f"   🚛 Promedio clientes por ruta: {len(self.clusters_rutas) / self.clusters_rutas['cluster_global'].nunique():.1f}")

        return True

    def crear_mapa_interactivo(self):
        """Crea mapa interactivo con rutas de distribución"""
        print("\n🗺️ Creando mapa interactivo...")

        if self.clusters_rutas is None:
            print("❌ Primero realizar clustering")
            return None

        # Crear mapa centrado en La Paz
        centro_lat = self.coordenadas_base['LA PAZ']['lat']
        centro_lon = self.coordenadas_base['LA PAZ']['lon']

        mapa = folium.Map(
            location=[centro_lat, centro_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )

        # Colores para diferentes rutas
        colores_rutas = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9F43', '#D63031', '#74B9FF', '#00B894', '#FDCB6E'
        ]

        # Colores por día de distribución
        colores_dias = {
            'Lunes': '#FF6B6B',
            'Martes': '#4ECDC4',
            'Miércoles': '#45B7D1',
            'Jueves': '#96CEB4',
            'Viernes': '#FECA57',
            'Sábado': '#FF9F43'
        }

        # Agregar marcadores por ruta
        rutas_unicas = self.clusters_rutas['cluster_global'].unique()

        for i, ruta in enumerate(rutas_unicas):
            clientes_ruta = self.clusters_rutas[self.clusters_rutas['cluster_global'] == ruta]
            color_ruta = colores_rutas[i % len(colores_rutas)]

            # Obtener día de distribución
            dia_ejemplo = clientes_ruta['dia_distribucion'].iloc[0].split(' - ')[0]
            color_dia = colores_dias.get(dia_ejemplo, '#95A5A6')

            # Crear grupo de marcadores para esta ruta
            grupo_ruta = folium.FeatureGroup(name=f"{ruta} ({len(clientes_ruta)} clientes)")

            for _, cliente in clientes_ruta.iterrows():
                # Icono según tipo de negocio
                if 'PIZZERIA' in cliente['tipo_negocio']:
                    icono = 'cutlery'
                    color_icono = 'red'
                elif 'FRIAL' in cliente['tipo_negocio']:
                    icono = 'shopping-cart'
                    color_icono = 'blue'
                elif 'RESTAURANTE' in cliente['tipo_negocio']:
                    icono = 'cutlery'
                    color_icono = 'green'
                else:
                    icono = 'home'
                    color_icono = 'gray'

                # Crear popup con información
                popup_text = f"""
                <b>{cliente['nombre']}</b><br>
                🏢 {cliente['tipo_negocio']}<br>
                📍 {cliente['lugar_entrega']}<br>
                🗓️ {cliente['dia_distribucion']}<br>
                🚛 {ruta}
                """

                folium.Marker(
                    location=[cliente['lat'], cliente['lon']],
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=f"{cliente['nombre']} - {cliente['tipo_negocio']}",
                    icon=folium.Icon(
                        color=color_icono,
                        icon=icono,
                        prefix='fa'
                    )
                ).add_to(grupo_ruta)

            # Agregar líneas conectando la ruta
            if len(clientes_ruta) > 1:
                coordenadas_ruta = [[row['lat'], row['lon']] for _, row in clientes_ruta.iterrows()]

                folium.PolyLine(
                    locations=coordenadas_ruta,
                    color=color_ruta,
                    weight=3,
                    opacity=0.7,
                    popup=f"Ruta: {ruta}"
                ).add_to(grupo_ruta)

            mapa.add_child(grupo_ruta)

        # Agregar centros de distribución
        for ciudad, coords in self.coordenadas_base.items():
            if ciudad in ['LA PAZ', 'EL ALTO']:
                folium.Marker(
                    location=[coords['lat'], coords['lon']],
                    popup=f"Centro de Distribución - {ciudad}",
                    tooltip=f"Centro: {ciudad}",
                    icon=folium.Icon(
                        color='darkred',
                        icon='warehouse',
                        prefix='fa'
                    )
                ).add_to(mapa)

        # Control de capas
        folium.LayerControl().add_to(mapa)

        # Agregar leyenda
        leyenda_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 300px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>🚛 Leyenda de Rutas</h4>
        <p><i class="fa fa-cutlery" style="color:red"></i> Pizzerías</p>
        <p><i class="fa fa-shopping-cart" style="color:blue"></i> Friales</p>
        <p><i class="fa fa-cutlery" style="color:green"></i> Restaurantes</p>
        <p><i class="fa fa-home" style="color:gray"></i> Otros</p>
        <p><i class="fa fa-warehouse" style="color:darkred"></i> Centro Distribución</p>
        <br>
        <h5>📅 Días de Entrega:</h5>
        <p style="color:#FF6B6B">● Lunes (La Paz)</p>
        <p style="color:#4ECDC4">● Martes (El Alto)</p>
        <p style="color:#96CEB4">● Jueves (La Paz)</p>
        <p style="color:#FECA57">● Viernes (El Alto)</p>
        </div>
        '''
        mapa.get_root().html.add_child(folium.Element(leyenda_html))

        # Guardar mapa
        archivo_mapa = 'mapa_rutas_distribucion.html'
        mapa.save(archivo_mapa)

        print(f"✅ Mapa guardado como: {archivo_mapa}")
        print(f"📊 {len(rutas_unicas)} rutas visualizadas")

        return mapa

    def generar_horarios_distribucion(self):
        """Genera horarios detallados de distribución"""
        print("\n📅 Generando horarios de distribución...")

        if self.clusters_rutas is None:
            print("❌ Primero realizar clustering")
            return None

        horarios = {}

        # Agrupar por día
        for dia in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']:
            clientes_dia = self.clusters_rutas[
                self.clusters_rutas['dia_distribucion'].str.contains(dia)
            ]

            if len(clientes_dia) > 0:
                horarios[dia] = {
                    'mañana': [],
                    'tarde': []
                }

                # Separar por turno
                for turno in ['MAÑANA', 'TARDE']:
                    clientes_turno = clientes_dia[
                        clientes_dia['dia_distribucion'].str.contains(turno)
                    ]

                    if len(clientes_turno) > 0:
                        # Agrupar por ruta
                        for ruta in clientes_turno['cluster_global'].unique():
                            clientes_ruta = clientes_turno[
                                clientes_turno['cluster_global'] == ruta
                                ]

                            # Calcular tiempo estimado (5 min por cliente + 15 min viaje entre puntos)
                            tiempo_estimado = len(clientes_ruta) * 5 + (len(clientes_ruta) - 1) * 15

                            horarios[dia][turno.lower()].append({
                                'ruta': ruta,
                                'clientes': len(clientes_ruta),
                                'tiempo_estimado_min': tiempo_estimado,
                                'ciudad': clientes_ruta['ciudad'].iloc[0],
                                'lista_clientes': clientes_ruta[['nombre', 'tipo_negocio', 'lugar_entrega']].to_dict('records')
                            })

        # Mostrar horarios
        print("\n📋 HORARIOS DE DISTRIBUCIÓN:")
        print("="*60)

        for dia, turnos in horarios.items():
            if any(turnos.values()):
                print(f"\n📅 {dia.upper()}")
                print("-" * 30)

                for turno, rutas in turnos.items():
                    if rutas:
                        print(f"\n🕐 {turno.upper()}:")
                        hora_inicio = "08:00" if turno == "mañana" else "14:00"

                        for i, ruta_info in enumerate(rutas):
                            print(f"   🚛 {ruta_info['ruta']} ({ruta_info['ciudad']})")
                            print(f"      👥 {ruta_info['clientes']} clientes")
                            print(f"      ⏱️ {ruta_info['tiempo_estimado_min']} minutos estimados")
                            print(f"      🕐 Inicio sugerido: {hora_inicio}")

                            # Calcular siguiente hora
                            hora_inicio = self._calcular_siguiente_hora(hora_inicio, ruta_info['tiempo_estimado_min'])

        return horarios

    def _calcular_siguiente_hora(self, hora_actual, minutos_agregar):
        """Calcula la siguiente hora agregando minutos"""
        from datetime import datetime, timedelta

        hora_obj = datetime.strptime(hora_actual, "%H:%M")
        nueva_hora = hora_obj + timedelta(minutes=minutos_agregar + 30)  # +30 min buffer
        return nueva_hora.strftime("%H:%M")

    def generar_reporte_estadisticas(self):
        """Genera reporte estadístico completo"""
        print("\n📊 GENERANDO REPORTE ESTADÍSTICO...")

        if self.clusters_rutas is None:
            print("❌ Primero realizar clustering")
            return None

        # Estadísticas generales
        total_clientes = len(self.clusters_rutas)
        total_rutas = self.clusters_rutas['cluster_global'].nunique()

        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   👥 Total clientes: {total_clientes}")
        print(f"   🚛 Total rutas: {total_rutas}")
        print(f"   📍 Ciudades atendidas: {self.clusters_rutas['ciudad'].nunique()}")

        # Por ciudad
        print(f"\n🏙️ DISTRIBUCIÓN POR CIUDAD:")
        for ciudad, count in self.clusters_rutas['ciudad'].value_counts().items():
            porcentaje = (count / total_clientes) * 100
            print(f"   {ciudad}: {count} clientes ({porcentaje:.1f}%)")

        # Por tipo de negocio
        print(f"\n🏢 DISTRIBUCIÓN POR TIPO DE NEGOCIO:")
        for tipo, count in self.clusters_rutas['tipo_negocio'].value_counts().head(5).items():
            porcentaje = (count / total_clientes) * 100
            print(f"   {tipo}: {count} clientes ({porcentaje:.1f}%)")

        # Por día de distribución
        print(f"\n📅 DISTRIBUCIÓN POR DÍA:")
        dias_dist = self.clusters_rutas['dia_distribucion'].str.split(' - ').str[0].value_counts()
        for dia, count in dias_dist.items():
            porcentaje = (count / total_clientes) * 100
            print(f"   {dia}: {count} clientes ({porcentaje:.1f}%)")

        # Eficiencia de rutas
        print(f"\n⚡ EFICIENCIA DE RUTAS:")
        clientes_por_ruta = self.clusters_rutas.groupby('cluster_global').size()
        print(f"   📊 Promedio clientes por ruta: {clientes_por_ruta.mean():.1f}")
        print(f"   📈 Máximo clientes en una ruta: {clientes_por_ruta.max()}")
        print(f"   📉 Mínimo clientes en una ruta: {clientes_por_ruta.min()}")

        return {
            'total_clientes': total_clientes,
            'total_rutas': total_rutas,
            'por_ciudad': self.clusters_rutas['ciudad'].value_counts().to_dict(),
            'por_tipo_negocio': self.clusters_rutas['tipo_negocio'].value_counts().to_dict(),
            'por_dia': dias_dist.to_dict(),
            'clientes_por_ruta': clientes_por_ruta.to_dict()
        }

    def exportar_rutas_excel(self, nombre_archivo="rutas_optimizadas.xlsx"):
        """Exporta las rutas optimizadas a Excel"""
        print(f"\n📤 Exportando rutas a {nombre_archivo}...")

        if self.clusters_rutas is None:
            print("❌ Primero realizar clustering")
            return False

        try:
            with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
                # Hoja 1: Todas las rutas
                self.clusters_rutas.to_excel(writer, sheet_name='Rutas_Completas', index=False)

                # Hoja 2: Resumen por ruta
                resumen_rutas = self.clusters_rutas.groupby('cluster_global').agg({
                    'nombre': 'count',
                    'ciudad': 'first',
                    'dia_distribucion': 'first',
                    'tipo_negocio': lambda x: ', '.join(x.unique()[:3])
                }).rename(columns={
                    'nombre': 'total_clientes',
                    'tipo_negocio': 'tipos_negocio'
                })
                resumen_rutas.to_excel(writer, sheet_name='Resumen_Rutas')

                # Hoja 3: Por día
                for dia in ['Lunes', 'Martes', 'Jueves', 'Viernes']:
                    clientes_dia = self.clusters_rutas[
                        self.clusters_rutas['dia_distribucion'].str.contains(dia)
                    ]
                    if len(clientes_dia) > 0:
                        clientes_dia.to_excel(writer, sheet_name=f'Ruta_{dia}', index=False)

            print(f"✅ Rutas exportadas exitosamente")
            return True

        except Exception as e:
            print(f"❌ Error exportando: {e}")
            return False

    def ejecutar_analisis_completo(self):
        """Ejecuta el análisis completo del sistema"""
        print("🚛 INICIANDO ANÁLISIS COMPLETO DE RUTAS")
        print("="*50)

        # 1. Cargar datos
        if not self.cargar_datos_excel():
            return False

        # 2. Generar coordenadas
        if self.generar_coordenadas_hiperlocales() is None:
            return False

        # 3. Realizar clustering
        if not self.realizar_clustering_geografico(n_clusters_por_ciudad=3):
            return False

        # 4. Crear mapa
        mapa = self.crear_mapa_interactivo()

        # 5. Generar horarios
        horarios = self.generar_horarios_distribucion()

        # 6. Estadísticas
        stats = self.generar_reporte_estadisticas()

        # 7. Exportar a Excel
        self.exportar_rutas_excel()

        print("\n🎉 ANÁLISIS COMPLETO FINALIZADO")
        print("="*40)
        print("📄 Archivos generados:")
        print("   🗺️ mapa_rutas_distribucion.html")
        print("   📊 rutas_optimizadas.xlsx")
        print("\n💡 Próximos pasos:")
        print("   1. Abrir el mapa HTML en tu navegador")
        print("   2. Revisar las rutas en el Excel")
        print("   3. Ajustar horarios según capacidad de vehículos")

        return True

def main():
    """Función principal del sistema"""
    print("🚛 SISTEMA DE RUTAS DE DISTRIBUCIÓN CON CLUSTERING")
    print("="*60)
    print("📋 REGLAS DE DISTRIBUCIÓN:")
    print("   🔵 Lunes y Jueves: LA PAZ")
    print("   🟢 Martes y Viernes: EL ALTO")
    print("   🔴 Pizzerías: siempre en la TARDE")
    print("="*60)

    # Verificar si existe el archivo Excel
    archivo_excel = "dataset inicial.xlsx"
    if not os.path.exists(archivo_excel):
        print(f"❌ Archivo {archivo_excel} no encontrado")
        print("💡 Asegúrate de tener el archivo Excel en la misma carpeta")
        return

    # Crear sistema
    sistema = SistemaRutasDistribucion(archivo_excel)

    # Ejecutar análisis completo
    sistema.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()