# extract_gee_osm_features.py
# Requisitos:
#   pip install earthengine-api pandas
# Autenticación GEE (una vez): `earthengine authenticate` en tu terminal
# Uso: python extraer_datos_nuevos.py

import ee
import pandas as pd
import sys
import os
import json
import argparse
from datetime import datetime, timedelta, timezone
import time

# Cargar variables de GEE.env
if os.path.exists('GEE.env'):
    with open('GEE.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip()


def init_earthengine(service_key_path=None, project=None):
    """Inicializa la librería `ee`.
    - Si `service_key_path` apunta a un JSON de cuenta de servicio, extrae el `client_email`
      y usa `ServiceAccountCredentials(..., key_path)` pasando `project`.
    - Si no hay key, intenta la inicialización interactiva con `ee.Initialize(project=...)`.
    """
    try:
        # Preferencia: cuenta de servicio vía parámetro o variable de entorno
        key_path = service_key_path or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if key_path and os.path.exists(key_path):
            with open(key_path, 'r', encoding='utf-8') as fh:
                key = json.load(fh)
            client_email = key.get('client_email')
            if not client_email:
                raise ValueError('El JSON de credenciales no contiene `client_email`.')
            creds = ee.ServiceAccountCredentials(client_email, key_path)
            ee.Initialize(credentials=creds, project=project)
            print(f'Earth Engine inicializado con cuenta de servicio {client_email} y project={project}')
            return
        # Fallback: intentar inicialización interactiva (usa credentials del usuario previamente autenticado)
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print('Earth Engine inicializado (usuario interactivo).')
    except Exception as e:
        print('ERROR inicializando Earth Engine:', e)
        print('Solución rápida: ejecuta `earthengine authenticate` en tu terminal o configura la variable')
        print('de entorno GOOGLE_APPLICATION_CREDENTIALS con la ruta al JSON de la cuenta de servicio.')
        raise

# Parse CLI args early so we can initialize ee with project or key
parser = argparse.ArgumentParser(description='Extraer SOLO las 5 nuevas features prioritarias desde GEE')
# input/output opcionales: por defecto usa water_quality_training_dataset.csv
parser.add_argument('input_csv', nargs='?', default='water_quality_training_dataset.csv', help='CSV de muestras: id,latitude,longitude,sample_date (default: submission_with_values.csv)')
parser.add_argument('output_csv', nargs='?', default='submission_gee_features.csv', help='CSV de salida con features (default: submission_gee_features.csv)')
parser.add_argument('--service-key', default=None, help='Ruta al JSON de cuenta de servicio (opcional)')
parser.add_argument('--project', default=os.getenv('GEE_PROJECT_ID'), help='GCP project id para inicializar Earth Engine (cargado de GEE.env)')
parser.add_argument('--drive-folder', default='GEE_exports', help='Carpeta en Google Drive donde exportar (default: GEE_exports)')
args_cli = parser.parse_args(sys.argv[1:])

# Inicializa Earth Engine (soporta --service-key y --project)
init_earthengine(service_key_path=args_cli.service_key, project=args_cli.project)

# === SOLO FUNCIONES PRIORITARIAS ===

# --- Vectorized extraction (SOLO NUEVAS FEATURES) ---
def vectorized_extract(input_csv, output_csv, export_to_drive=False, drive_folder='GEE_exports'):
    """Extracción vectorizada server-side SOLO de las 5 nuevas features prioritarias.
    - Si número de puntos > 500 y export_to_drive=True, exporta resultados a Google Drive.
    """
    df_raw = pd.read_csv(input_csv)
    if {'Latitude','Longitude','Sample Date'}.issubset(set(df_raw.columns)):
        df = df_raw.rename(columns={'Latitude':'latitude','Longitude':'longitude','Sample Date':'sample_date'}).copy()
        df['sample_date'] = pd.to_datetime(df['sample_date'], dayfirst=True, errors='coerce')
    else:
        # intentar columnas minúsculas
        df = df_raw.rename(columns={c: c.lower() for c in df_raw.columns})
        if not {'id','latitude','longitude','sample_date'}.issubset(set(df.columns)):
            raise ValueError('CSV no contiene columnas esperadas para vectorized_extract')
        df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df)+1))

    # construir FeatureCollection de puntos
    feats = []
    for _, r in df.iterrows():
        geom = ee.Geometry.Point([float(r['longitude']), float(r['latitude'])])
        sd = None
        if not pd.isna(r['sample_date']):
            try:
                sd = pd.to_datetime(r['sample_date']).strftime('%d/%m/%Y')
            except Exception:
                sd = str(r['sample_date'])
        f = ee.Feature(geom, {'id': int(r['id']), 'sample_date': sd, 'latitude': float(r['latitude']), 'longitude': float(r['longitude'])})
        feats.append(f)
    fc = ee.FeatureCollection(feats)

    # ventanas temporales
    end = ee.Date(datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    start_90 = end.advance(-90, 'day')
    start_30 = end.advance(-30, 'day')

    # === SOLO NUEVAS FEATURES PRIORITARIAS (para vectorized) ===
    # 1. Soil Moisture (SMAP - 30 días)
    smap_col = ee.ImageCollection('NASA/SMAP/SPL4SMGP/008').filterDate(start_30, end)
    smap_mean_img = smap_col.select('sm_surface').mean()
    
    # 2. Land Surface Temperature (MODIS - 30 días) - Convertir a Celsius directamente
    lst_col = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(start_30, end).select('LST_Day_1km')
    lst_mean_img = lst_col.mean().multiply(0.02).subtract(273.15)  # Convertir a Celsius
    lst_max_img = lst_col.max().multiply(0.02).subtract(273.15)    # Convertir a Celsius
    
    # 3. EVI y LAI (MODIS - 90 días)
    evi_col = ee.ImageCollection('MODIS/061/MOD13A2').filterDate(start_90, end).select('EVI')
    evi_stat_img = evi_col.reduce(ee.Reducer.percentile([50]).combine(ee.Reducer.stdDev(), '', True))
    lai_col = ee.ImageCollection('MODIS/061/MCD15A3H').filterDate(start_90, end).select('Lai')
    lai_mean_img = lai_col.mean().multiply(0.1)  # scale factor
    
    # 4. Water Occurrence (JRC - estático)
    gsw_img = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    water_occ_img = gsw_img.select('occurrence')
    water_seas_img = gsw_img.select('seasonality')
    
    # 5. Population Density (WorldPop - estático)
    pop_img = ee.ImageCollection('WorldPop/GP/100m/pop').mosaic().select('population')

    merged_fc = None
    # para cada buffer, hacer buffer sobre la colección y reducir (solo 1000m)
    buffered = fc.map(lambda f: f.buffer(1000))
    
    # === REDUCCIÓN VECTORIZADA DE LAS 5 NUEVAS FEATURES ===
    # Soil Moisture
    smap_reduced = smap_mean_img.reduceRegions(collection=buffered, reducer=ee.Reducer.mean(), scale=10000)
    # LST mean y max
    lst_mean_reduced = lst_mean_img.reduceRegions(collection=buffered, reducer=ee.Reducer.mean(), scale=1000)
    lst_max_reduced = lst_max_img.reduceRegions(collection=buffered, reducer=ee.Reducer.max(), scale=1000)
    # EVI/LAI
    evi_reduced = evi_stat_img.reduceRegions(collection=buffered, reducer=ee.Reducer.first(), scale=500)
    lai_reduced = lai_mean_img.reduceRegions(collection=buffered, reducer=ee.Reducer.mean(), scale=500)

    # === Extracción de features globales (buffer único 5km): Water Occurrence y Population ===
    # Se extrae solo una vez porque son estáticas y usan buffer grande
    buffered_5km = fc.map(lambda f: f.buffer(5000))
    water_occ_reduced = water_occ_img.reduceRegions(collection=buffered_5km, reducer=ee.Reducer.mean(), scale=30)
    water_seas_reduced = water_seas_img.reduceRegions(collection=buffered_5km, reducer=ee.Reducer.mean(), scale=30)
    pop_reduced = pop_img.reduceRegions(collection=buffered_5km, reducer=ee.Reducer.sum(), scale=100)
    
    # SIEMPRE exportar a Drive usando tareas de Earth Engine
    ts = int(time.time())
    prefix = f'gee_extract_nuevas_features_{ts}'
    
    # === RENOMBRAR PROPIEDADES EN CADA FEATURECOLLECTION REDUCIDA ===
    # Esto evita conflictos de nombres al hacer los joins
    smap_final = smap_reduced.map(lambda f: f.set({
        'soil_moisture_mean_1000m': f.get('mean'),
        'Latitude': f.get('latitude'),
        'Longitude': f.get('longitude'),
        'Sample_Date': f.get('sample_date')
    }))
    
    lst_mean_final = lst_mean_reduced.map(lambda f: f.set('lst_mean_1000m', f.get('mean')))
    lst_max_final = lst_max_reduced.map(lambda f: f.set('lst_max_1000m', f.get('max')))
    
    evi_final = evi_reduced.map(lambda f: f.set({
        'evi_p50_1000m': f.get('EVI_p50'),
        'evi_std_1000m': f.get('EVI_stdDev')
    }))
    
    lai_final = lai_reduced.map(lambda f: f.set('lai_mean_1000m', f.get('mean')))
    
    water_occ_final = water_occ_reduced.map(lambda f: f.set('water_occurrence_5000m', f.get('mean')))
    water_seas_final = water_seas_reduced.map(lambda f: f.set('water_seasonality_5000m', f.get('mean')))
    pop_final = pop_reduced.map(lambda f: f.set('population_sum_5000m', f.get('sum')))
    
    # === COMBINAR USANDO JOINS SECUENCIALES ===
    id_filter = ee.Filter.equals(leftField='id', rightField='id')
    
    def simple_join(primary, secondary):
        return ee.Join.inner().apply(primary, secondary, id_filter).map(
            lambda pair: ee.Feature(pair.get('primary')).copyProperties(
                ee.Feature(pair.get('secondary')),
                exclude=['system:index', 'id', '.geo', 'latitude', 'longitude', 'sample_date']
            )
        )
    
    # Joins secuenciales comenzando con smap_final (tiene todas las columnas base)
    c1 = simple_join(smap_final, lst_mean_final)
    c2 = simple_join(c1, lst_max_final)
    c3 = simple_join(c2, evi_final)
    c4 = simple_join(c3, lai_final)
    c5 = simple_join(c4, water_occ_final)
    c6 = simple_join(c5, water_seas_final)
    combined_final = simple_join(c6, pop_final)
    
    # Exportar UN SOLO ARCHIVO CON TODAS LAS COLUMNAS
    task = ee.batch.Export.table.toDrive(
        collection=combined_final,
        description=prefix + '_all_features',
        folder=drive_folder,
        fileNamePrefix=prefix + '_all_features',
        fileFormat='CSV',
        selectors=['id', 'Latitude', 'Longitude', 'Sample_Date', 
                  'soil_moisture_mean_1000m', 'lst_mean_1000m', 'lst_max_1000m',
                  'evi_p50_1000m', 'evi_std_1000m', 'lai_mean_1000m',
                  'water_occurrence_5000m', 'water_seasonality_5000m', 'population_sum_5000m']
    )
    task.start()
    
    print(f'✅ Export task started to Drive: {prefix}_all_features.csv')
    print(f'📊 Columns: id, Latitude, Longitude, Sample_Date + 9 new features')
    print(f'🔗 Check progress: https://code.earthengine.google.com/tasks')
    print(f'\n📌 Cuando termine la tarea:')
    print(f'   1. Descarga el CSV de tu Google Drive')
    print(f'   2. Usa fix_date_format.py para corregir formato de fechas')
    print(f'   3. Usa add_new_features.py para fusionar con {input_csv}')
    return None


if __name__ == '__main__':
    # Siempre usar vectorized_extract con tareas de Earth Engine
    vectorized_extract(args_cli.input_csv, args_cli.output_csv, 
                      export_to_drive=True, drive_folder=args_cli.drive_folder)