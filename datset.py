import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("🔧 GENERADOR DE DATASET MEJORADO - USANDO TODOS LOS CLIENTES DEL EXCEL")
print("=" * 80)

# ============================================================================
# PASO 1: CARGAR TODOS LOS CLIENTES DEL EXCEL
# ============================================================================
print("\n📊 PASO 1: CARGANDO CLIENTES DEL EXCEL")
print("-" * 50)

try:
    # Leer el archivo Excel
    import openpyxl

    df_excel = pd.read_excel('dataset inicial.xlsx')
    print("✓ Archivo Excel cargado exitosamente")
    print(f"✓ Registros encontrados: {len(df_excel)}")

    # Mostrar primeras filas para entender estructura
    print("\nPrimeras 5 filas del Excel:")
    print(df_excel.head())

    # Mostrar columnas disponibles
    print(f"\nColumnas disponibles: {list(df_excel.columns)}")

except Exception as e:
    print(f"❌ Error al cargar Excel: {e}")
    print("Usando datos de ejemplo...")
    # Datos de fallback si no se puede cargar el Excel
    df_excel = pd.DataFrame({
        'CLIENTE': ['SANDER', 'LURDES', 'GLADYS', 'ELVIRA', 'ELY', 'MARIA LUNA', 'ROSALIA'],
        'TIPO NEGOCIO': ['PIZZERIA', 'PIZZERIA', 'PIZZERIA', 'PIZZERIA', 'PIZZERIA', 'PIZZERIA', 'PIZZERIA'],
        'CIUDAD/LOCALIDAD': ['El ALTO', 'El ALTO', 'El ALTO', 'El ALTO', 'El ALTO', 'El ALTO', 'El ALTO'],
        'LUGAR ENTREGA': ['APACHETA', 'PUENTE VELA', 'ESTRUCTURANTE', 'SENKATA', 'INGAVI', 'KENKO', 'HORIZONTES'],
        'TURNO': ['TARDE', 'TARDE', 'TARDE', 'TARDE', 'TARDE', 'TARDE', 'TARDE']
    })

# ============================================================================
# PASO 2: PROCESAR Y LIMPIAR DATOS DE CLIENTES
# ============================================================================
print("\n🧹 PASO 2: PROCESANDO DATOS DE CLIENTES")
print("-" * 50)


def procesar_clientes_excel(df):
    """Procesa y limpia los datos de clientes del Excel"""

    # Identificar las columnas correctas (manejo de diferentes nombres posibles)
    columnas_mapeo = {}

    for col in df.columns:
        col_upper = str(col).upper().strip()
        if 'CLIENTE' in col_upper or 'NOMBRE' in col_upper:
            columnas_mapeo['cliente'] = col
        elif 'TIPO' in col_upper and 'NEGOCIO' in col_upper:
            columnas_mapeo['tipo_negocio'] = col
        elif 'CIUDAD' in col_upper or 'LOCALIDAD' in col_upper:
            columnas_mapeo['ciudad'] = col
        elif 'LUGAR' in col_upper and 'ENTREGA' in col_upper:
            columnas_mapeo['lugar_entrega'] = col
        elif 'TURNO' in col_upper:
            columnas_mapeo['turno'] = col

    print(f"Columnas identificadas: {columnas_mapeo}")

    # Crear lista de clientes procesada
    clientes_procesados = []

    for index, row in df.iterrows():
        try:
            cliente = {
                "id": index + 1,  # ID secuencial
                "nombre": str(row.get(columnas_mapeo.get('cliente', 'CLIENTE'), f'CLIENTE_{index + 1}')).strip(),
                "tipo_negocio": str(row.get(columnas_mapeo.get('tipo_negocio', 'TIPO NEGOCIO'), 'FRIAL')).strip(),
                "ciudad": str(row.get(columnas_mapeo.get('ciudad', 'CIUDAD/LOCALIDAD'), 'El ALTO')).strip(),
                "lugar_entrega": str(row.get(columnas_mapeo.get('lugar_entrega', 'LUGAR ENTREGA'), 'CENTRO')).strip(),
                "turno": str(row.get(columnas_mapeo.get('turno', 'TURNO'), 'MAÑANA')).strip()
            }

            # Limpiar datos
            if cliente["nombre"] and cliente["nombre"] != 'nan' and len(cliente["nombre"]) > 0:
                # Mantener el tipo de negocio original pero normalizarlo
                tipo_negocio = cliente["tipo_negocio"].upper().strip()

                # Debugging: mostrar tipo de negocio original
                if index < 10:  # Solo mostrar primeros 10 para debug
                    print(f"  Cliente {cliente['nombre']}: tipo_negocio = '{tipo_negocio}'")

                # Normalizar tipo de negocio SIN cambiar pizzerías
                if tipo_negocio == 'PIZZERIA':
                    cliente["tipo_negocio"] = "PIZZERIA"
                elif 'SALCHI' in tipo_negocio:
                    cliente["tipo_negocio"] = "SALCHIPAPERIA"
                elif 'HAMBURG' in tipo_negocio:
                    cliente["tipo_negocio"] = "HAMBURGUESERIA"
                elif 'MERCADO' in tipo_negocio:
                    cliente["tipo_negocio"] = "PUESTO DE MERCADO"
                elif 'FRIAL' in tipo_negocio:
                    cliente["tipo_negocio"] = "FRIAL"
                elif 'TIENDA' in tipo_negocio:
                    cliente["tipo_negocio"] = "TIENDA"
                elif 'MINIMARKET' in tipo_negocio:
                    cliente["tipo_negocio"] = "MINIMARKET"
                elif 'RESTAURANTE' in tipo_negocio:
                    cliente["tipo_negocio"] = "RESTAURANTE"
                else:
                    cliente["tipo_negocio"] = tipo_negocio  # Mantener original si no coincide

                # Normalizar ciudad
                ciudad = cliente["ciudad"].upper()
                if 'ALTO' in ciudad:
                    cliente["ciudad"] = "El ALTO"
                elif 'PAZ' in ciudad:
                    cliente["ciudad"] = "LA PAZ"
                elif 'COPACABANA' in ciudad:
                    cliente["ciudad"] = "COPACABANA"
                elif 'CARANAVI' in ciudad:
                    cliente["ciudad"] = "CARANAVI"
                else:
                    cliente["ciudad"] = "El ALTO"  # Valor por defecto

                # Normalizar turno
                turno = cliente["turno"].upper()
                if 'TARDE' in turno:
                    cliente["turno"] = "TARDE"
                else:
                    cliente["turno"] = "MAÑANA"

                clientes_procesados.append(cliente)

        except Exception as e:
            print(f"⚠ Error procesando fila {index}: {e}")
            continue

    return clientes_procesados


# Procesar clientes
clientes_excel = procesar_clientes_excel(df_excel)
print(f"✓ Clientes procesados del Excel: {len(clientes_excel)}")

# Mostrar algunos ejemplos
print("\nEjemplos de clientes procesados:")
for i, cliente in enumerate(clientes_excel[:5]):
    print(f"  {i + 1}. {cliente['nombre']} - {cliente['tipo_negocio']} - {cliente['ciudad']}")

# ============================================================================
# PASO 3: AGREGAR PIZZERÍAS FALTANTES (NO EXPANDIR)
# ============================================================================
print(f"\n🍕 PASO 3: AGREGANDO PIZZERÍAS FALTANTES")
print("-" * 50)


def agregar_pizzerias_faltantes(clientes_base):
    """Agrega solo las pizzerías que faltan del dataset original"""

    print(f"Clientes actuales: {len(clientes_base)}")

    # Contar pizzerías existentes CON DEBUGGING
    pizzerias_existentes = []
    for c in clientes_base:
        print(f"  Cliente: {c['nombre']} - Tipo: '{c['tipo_negocio']}'")
        if "PIZZERIA" in c["tipo_negocio"].upper():
            pizzerias_existentes.append(c)

    print(f"🍕 Pizzerías encontradas en Excel: {len(pizzerias_existentes)}")
    for p in pizzerias_existentes:
        print(f"  - {p['nombre']} ({p['tipo_negocio']})")

    # Si ya hay suficientes pizzerías, no agregar más
    if len(pizzerias_existentes) >= 5:
        print(f"✓ Ya hay suficientes pizzerías ({len(pizzerias_existentes)}), no se agregarán más")
        return clientes_base

    # Pizzerías del dataset original que pueden estar faltando
    pizzerias_originales = [
        {"nombre": "SANDER", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "APACHETA",
         "turno": "TARDE"},
        {"nombre": "LURDES", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "PUENTE VELA",
         "turno": "TARDE"},
        {"nombre": "GLADYS", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "ESTRUCTURANTE",
         "turno": "TARDE"},
        {"nombre": "ELVIRA", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "SENKATA",
         "turno": "TARDE"},
        {"nombre": "ELY", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "INGAVI", "turno": "TARDE"},
        {"nombre": "MARIA LUNA", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "KENKO",
         "turno": "TARDE"},
        {"nombre": "ROSALIA", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "HORIZONTES",
         "turno": "TARDE"},
        {"nombre": "SEGUNDINA", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "HORNITOS",
         "turno": "MAÑANA"},
        {"nombre": "SANTOS", "tipo_negocio": "PIZZERIA", "ciudad": "El ALTO", "lugar_entrega": "16 DE JULIO",
         "turno": "MAÑANA"},
    ]

    # Verificar qué pizzerías faltan
    nombres_existentes = [c["nombre"].upper().strip() for c in clientes_base]
    pizzerias_a_agregar = []

    next_id = max([c["id"] for c in clientes_base]) + 1

    print(f"Verificando pizzerías faltantes...")
    for pizzeria in pizzerias_originales:
        # Buscar si ya existe (comparación flexible)
        nombre_buscar = pizzeria["nombre"].upper().strip()
        existe = False

        for nombre_existente in nombres_existentes:
            if nombre_buscar == nombre_existente or nombre_buscar in nombre_existente:
                existe = True
                print(f"  ✓ {nombre_buscar} ya existe como {nombre_existente}")
                break

        if not existe:
            nueva_pizzeria = {
                "id": next_id,
                "nombre": pizzeria["nombre"],
                "tipo_negocio": pizzeria["tipo_negocio"],
                "ciudad": pizzeria["ciudad"],
                "lugar_entrega": pizzeria["lugar_entrega"],
                "turno": pizzeria["turno"]
            }
            pizzerias_a_agregar.append(nueva_pizzeria)
            next_id += 1
            print(f"  + Agregando pizzería faltante: {pizzeria['nombre']}")

    if pizzerias_a_agregar:
        clientes_finales = clientes_base + pizzerias_a_agregar
        print(f"✓ Pizzerías agregadas: {len(pizzerias_a_agregar)}")
    else:
        clientes_finales = clientes_base
        print(f"✓ No se necesita agregar pizzerías adicionales")

    print(f"✓ Total clientes finales: {len(clientes_finales)}")
    return clientes_finales


# Agregar solo las pizzerías faltantes (no expandir)
clientes_finales = agregar_pizzerias_faltantes(clientes_excel)

# Mostrar distribución final
print(f"\nDistribución final por tipo de negocio:")
tipos_distribucion = {}
for cliente in clientes_finales:
    tipo = cliente["tipo_negocio"]
    tipos_distribucion[tipo] = tipos_distribucion.get(tipo, 0) + 1

for tipo, cantidad in sorted(tipos_distribucion.items()):
    print(f"  - {tipo}: {cantidad} clientes")

# ============================================================================
# PASO 4: DEFINIR PRODUCTOS (MANTENER LOS EXISTENTES)
# ============================================================================
print(f"\n🛍️ PASO 4: DEFINIENDO PRODUCTOS")
print("-" * 50)

productos = [
    {"id": 1, "nombre": "SAN JAVIER", "marca": "SAN JAVIER", "categoria": "BOLITA", "peso": "420GR", "precio": 21.50},
    {"id": 2, "nombre": "HOLANDESA BOLITA", "marca": "HOLANDESA", "categoria": "BOLITA", "peso": "500GR",
     "precio": 25.00},
    {"id": 3, "nombre": "HOLANDESA BARRA", "marca": "HOLANDESA", "categoria": "BARRA", "peso": "3.500KG",
     "precio": 120.00},
    {"id": 4, "nombre": "HOLANDESA CHEDDAR", "marca": "HOLANDESA", "categoria": "BARRA", "peso": "3.500KG",
     "precio": 130.00},
    {"id": 5, "nombre": "CRISTY BOLITA", "marca": "CRISTY", "categoria": "BOLITA", "peso": "420GR", "precio": 20.00},
    {"id": 6, "nombre": "CRISTY BARRA", "marca": "CRISTY", "categoria": "BARRA", "peso": "3.500KG", "precio": 115.00},
    {"id": 7, "nombre": "CRISTY LAMINADO", "marca": "CRISTY", "categoria": "LAMINADO", "peso": "160GR",
     "precio": 12.00},
    {"id": 8, "nombre": "CRISTY LAMINADO 500", "marca": "CRISTY", "categoria": "LAMINADO", "peso": "500GR",
     "precio": 35.00},
    {"id": 9, "nombre": "CRISTY CHEDDAR", "marca": "CRISTY", "categoria": "LAMINADO", "peso": "160GR", "precio": 14.00},
    {"id": 10, "nombre": "CRISTY CHEDDAR 500", "marca": "CRISTY", "categoria": "LAMINADO", "peso": "500GR",
     "precio": 40.00},
    {"id": 11, "nombre": "LA RIBERA", "marca": "LA RIBERA", "categoria": "BOLITA", "peso": "420GR", "precio": 20.00},
    {"id": 12, "nombre": "CHIQUITANO", "marca": "CHIQUITANO", "categoria": "BOLITA", "peso": "420GR", "precio": 18.50},
    {"id": 13, "nombre": "LA MARAVILLA", "marca": "LA MARAVILLA", "categoria": "BARRA", "peso": "3.500KG",
     "precio": 110.00},
    {"id": 14, "nombre": "SANTA ROSA", "marca": "SANTA ROSA", "categoria": "BARRA", "peso": "3.500KG",
     "precio": 105.00},
    {"id": 15, "nombre": "QUE QUESITOS", "marca": "QUE QUESITOS", "categoria": "BOLITA", "peso": "420GR",
     "precio": 19.00},
    {"id": 16, "nombre": "EL CORRALITO", "marca": "CORRALITO", "categoria": "BARRA", "peso": "3.500KG",
     "precio": 125.00},
    {"id": 17, "nombre": "SAN JAVIER BLOQUE", "marca": "SAN JAVIER", "categoria": "QUESO ARTESANAL", "peso": "8KG",
     "precio": 250.00},
    {"id": 18, "nombre": "CRISTY BARRA LAMINADA", "marca": "CRISTY", "categoria": "BARRA LAMINADA", "peso": "3.5KG",
     "precio": 120.00},
    {"id": 19, "nombre": "CHIQUITANO PVC", "marca": "CHIQUITANO", "categoria": "BOLITA", "peso": "420GR",
     "precio": 19.00},
    {"id": 20, "nombre": "CRISTY BARRA LAMINADA PVC", "marca": "CRISTY", "categoria": "BARRA LAMINADA", "peso": "3.5KG",
     "precio": 118.00}
]

print(f"✓ Productos definidos: {len(productos)}")

# ============================================================================
# PASO 5: GENERAR DATASET DE VENTAS MEJORADO
# ============================================================================
print(f"\n💰 PASO 5: GENERANDO DATASET DE VENTAS")
print("-" * 50)

# Definir preferencias de productos según tipo de negocio
preferencias = {
    "PIZZERIA": ["BOLITA", "BARRA"],
    "SALCHIPAPERIA": ["LAMINADO", "BARRA LAMINADA"],
    "HAMBURGUESERIA": ["LAMINADO", "BARRA LAMINADA"],
    "PUESTO DE MERCADO": ["BOLITA", "BARRA", "LAMINADO", "QUESO ARTESANAL"],
    "FRIAL": ["BOLITA", "BARRA", "LAMINADO", "QUESO ARTESANAL"],
    "TIENDA": ["BOLITA", "BARRA", "LAMINADO", "QUESO ARTESANAL"],
    "MINIMARKET": ["BOLITA", "BARRA", "LAMINADO", "QUESO ARTESANAL"],
    "RESTAURANTE": ["LAMINADO", "BARRA", "BOLITA"],
}


def generar_fecha_aleatoria():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    return random_date.strftime("%Y-%m-%d")


def es_producto_preferido(producto_categoria, tipo_negocio):
    return producto_categoria in preferencias.get(tipo_negocio, [])


def generar_cantidad(tipo_negocio, categoria, fecha):
    fecha_obj = datetime.strptime(fecha, "%Y-%m-%d")

    # Temporadas altas
    temporadas_altas = [
        (datetime(2023, 12, 15), datetime(2023, 12, 31)),
        (datetime(2023, 7, 1), datetime(2023, 8, 31)),
        (datetime(2024, 12, 15), datetime(2024, 12, 31)),
        (datetime(2024, 7, 1), datetime(2024, 8, 31)),
        (datetime(2023, 2, 15), datetime(2023, 2, 28)),
        (datetime(2024, 2, 10), datetime(2024, 2, 25)),
    ]

    es_temporada_alta = any(inicio <= fecha_obj <= fin for inicio, fin in temporadas_altas)
    factor_temporada = 1.5 if es_temporada_alta else 1.0

    if "PIZZERIA" in tipo_negocio:
        if categoria == "BOLITA":
            base = random.randint(30, 100)
            return int(base * factor_temporada)
        elif categoria == "BARRA":
            return random.randint(1, 3)
    elif "SALCHIPAPERIA" in tipo_negocio or "HAMBURGUESERIA" in tipo_negocio:
        if categoria == "LAMINADO":
            return random.randint(5, 15)
        elif categoria == "BARRA LAMINADA":
            return random.randint(1, 2)
    elif any(palabra in tipo_negocio for palabra in ["MERCADO", "FRIAL", "TIENDA", "MINIMARKET"]):
        if categoria == "BOLITA":
            return random.randint(5, 30)
        elif categoria == "BARRA":
            return random.randint(1, 3)
        elif categoria == "LAMINADO":
            return random.randint(3, 12)
        elif categoria == "QUESO ARTESANAL":
            return random.randint(1, 2)
    elif "RESTAURANTE" in tipo_negocio:
        if categoria == "LAMINADO":
            return random.randint(5, 15)
        elif categoria == "BARRA":
            return random.randint(1, 3)
        elif categoria == "BOLITA":
            return random.randint(5, 20)

    return random.randint(1, 5)


def generar_venta(venta_id):
    # Seleccionar cliente con bias hacia algunos clientes (distribución más realista)
    if random.random() < 0.6:  # 60% de ventas van a primeros 80% de clientes
        cliente = random.choice(clientes_finales[:int(len(clientes_finales) * 0.8)])
    else:  # 40% de ventas van a todos los clientes
        cliente = random.choice(clientes_finales)

    fecha = generar_fecha_aleatoria()

    # Determinar número de productos
    if any(palabra in cliente["tipo_negocio"] for palabra in ["MERCADO", "FRIAL", "TIENDA"]):
        num_productos = random.randint(2, 5)
    elif "PIZZERIA" in cliente["tipo_negocio"]:
        num_productos = random.randint(1, 3)
    else:
        num_productos = random.randint(1, 3)

    # Filtrar productos preferidos
    productos_preferidos = []
    for producto in productos:
        if es_producto_preferido(producto["categoria"], cliente["tipo_negocio"]):
            productos_preferidos.append(producto)

    if not productos_preferidos:
        productos_preferidos = productos

    productos_seleccionados = random.sample(productos_preferidos, min(num_productos, len(productos_preferidos)))

    # Generar detalles de venta
    detalles_venta = []
    total_venta = 0

    for producto in productos_seleccionados:
        cantidad = generar_cantidad(cliente["tipo_negocio"], producto["categoria"], fecha)
        precio_unitario = producto["precio"]
        subtotal = cantidad * precio_unitario
        total_venta += subtotal

        detalles_venta.append({
            "venta_id": venta_id,
            "producto_id": producto["id"],
            "producto_nombre": producto["nombre"],
            "producto_marca": producto["marca"],
            "producto_categoria": producto["categoria"],
            "producto_peso": producto["peso"],
            "cantidad": cantidad,
            "precio_unitario": precio_unitario,
            "subtotal": subtotal
        })

    # Descuento
    descuento = 0
    if total_venta > 1000:
        descuento = total_venta * 0.05

    total_con_descuento = total_venta - descuento

    # Crear registro de venta
    venta = {
        "venta_id": venta_id,
        "cliente_id": cliente["id"],
        "cliente_nombre": cliente["nombre"],
        "tipo_negocio": cliente["tipo_negocio"],
        "ciudad": cliente["ciudad"],
        "lugar_entrega": cliente["lugar_entrega"],
        "turno": cliente["turno"],
        "fecha": fecha,
        "total_bruto": total_venta,
        "descuento": descuento,
        "total_neto": total_con_descuento
    }

    return venta, detalles_venta


# Generar dataset con los clientes reales del Excel
num_ventas = 3000  # Suficientes ventas para los ~114-120 clientes
print(f"Generando {num_ventas} ventas...")

ventas = []
detalles = []

for i in range(1, num_ventas + 1):
    if i % 500 == 0:
        print(f"  Generadas {i} ventas...")

    venta, detalle_venta = generar_venta(i)
    ventas.append(venta)
    detalles.extend(detalle_venta)

# Convertir a DataFrames
df_ventas = pd.DataFrame(ventas)
df_detalles = pd.DataFrame(detalles)

# Ordenar por fecha
df_ventas = df_ventas.sort_values(by="fecha")

# ============================================================================
# PASO 6: VALIDAR Y GUARDAR DATASET
# ============================================================================
print(f"\n💾 PASO 6: VALIDANDO Y GUARDANDO DATASET")
print("-" * 50)

# Validaciones
print("Validando dataset generado...")

clientes_unicos = df_ventas['cliente_id'].nunique()
ventas_por_cliente = len(df_ventas) / clientes_unicos
productos_unicos = df_detalles['producto_id'].nunique()

print(f"✓ Total ventas generadas: {len(df_ventas)}")
print(f"✓ Total detalles generados: {len(df_detalles)}")
print(f"✓ Clientes únicos: {clientes_unicos}")
print(f"✓ Ventas por cliente (promedio): {ventas_por_cliente:.1f}")
print(f"✓ Productos únicos vendidos: {productos_unicos}")

# Mostrar distribución por tipo de negocio
print(f"\nDistribución de ventas por tipo de negocio:")
distribucion_ventas = df_ventas['tipo_negocio'].value_counts()
for tipo, cantidad in distribucion_ventas.items():
    print(f"  - {tipo}: {cantidad} ventas ({cantidad / len(df_ventas) * 100:.1f}%)")

# Verificar calidad de datos
print(f"\nVerificaciones de calidad:")
print(f"  ✓ Valores nulos en ventas: {df_ventas.isnull().sum().sum()}")
print(f"  ✓ Valores nulos en detalles: {df_detalles.isnull().sum().sum()}")
print(f"  ✓ Fechas válidas: {pd.to_datetime(df_ventas['fecha'], errors='coerce').notna().sum()}/{len(df_ventas)}")

# Guardar archivos
df_ventas.to_csv("ventas.csv", index=False)
df_detalles.to_csv("detalles_ventas.csv", index=False)

# Guardar también versión con sufijo para comparación
df_ventas.to_csv("ventas_mejorado.csv", index=False)
df_detalles.to_csv("detalles_ventas_mejorado.csv", index=False)

print(f"\n✅ Archivos guardados exitosamente:")
print(f"  - ventas.csv")
print(f"  - detalles_ventas.csv")
print(f"  - ventas_mejorado.csv (copia de respaldo)")
print(f"  - detalles_ventas_mejorado.csv (copia de respaldo)")

# ============================================================================
# PASO 7: MOSTRAR EJEMPLOS Y ESTADÍSTICAS
# ============================================================================
print(f"\n📊 PASO 7: EJEMPLOS Y ESTADÍSTICAS FINALES")
print("-" * 50)

print("Ejemplo de registros de ventas:")
print(df_ventas.head())

print("\nEjemplo de detalles de ventas:")
print(df_detalles.head())

print(f"\nEstadísticas de ventas por cliente:")
ventas_stats = df_ventas.groupby('cliente_id').size().describe()
print(ventas_stats)

print(f"\nRango de fechas: {df_ventas['fecha'].min()} a {df_ventas['fecha'].max()}")

print(f"\n🎉 ¡DATASET MEJORADO GENERADO EXITOSAMENTE!")
print(f"   📊 {clientes_unicos} clientes únicos (vs 21 anterior)")
print(f"   💰 {len(df_ventas)} ventas generadas")
print(f"   🛍️ {len(df_detalles)} productos vendidos")
print(f"   📈 {ventas_por_cliente:.1f} ventas por cliente promedio")
print("=" * 80)