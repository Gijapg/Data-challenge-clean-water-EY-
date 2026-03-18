import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

N_SPATIAL_CLUSTERS = 10

PREDICTION_MODE = "top_n"  # "all" or "top_n"
ENSEMBLE_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232]

def spatial_features(df, kmeans_model=None, n_clusters=N_SPATIAL_CLUSTERS):

    coords = df[['Latitude', 'Longitude']].values
    
    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
        kmeans_model.fit(coords)
        print(f"K-means fitted on {len(coords)} locations")
        
        kmeans_model.lat_range = coords[:, 0].max() - coords[:, 0].min()
        kmeans_model.lon_range = coords[:, 1].max() - coords[:, 1].min()
        kmeans_model.lat_min = coords[:, 0].min()
        kmeans_model.lon_min = coords[:, 1].min()
    
    centroids = kmeans_model.cluster_centers_
    
    all_dists = np.zeros((len(coords), n_clusters))
    for i, centroid in enumerate(centroids):
        dist = np.sqrt((coords[:, 0] - centroid[0])**2 + (coords[:, 1] - centroid[1])**2)
        all_dists[:, i] = dist
    
    sorted_dists = np.sort(all_dists, axis=1)
    
    df['nearest_centroid_dist'] = sorted_dists[:, 0]
    df['second_nearest_dist'] = sorted_dists[:, 1]
    df['third_nearest_dist'] = sorted_dists[:, 2]
    df['avg_nearest_3_dist'] = sorted_dists[:, :3].mean(axis=1)
    
    df['inv_nearest_dist'] = 1.0 / (sorted_dists[:, 0] + 0.05)
    df['inv_second_nearest'] = 1.0 / (sorted_dists[:, 1] + 0.05)
    
    df['spatial_isolation'] = sorted_dists[:, 0] / (sorted_dists[:, 1] + 0.01)

    df['lat_normalized'] = (coords[:, 0] - kmeans_model.lat_min) / kmeans_model.lat_range
    df['lon_normalized'] = (coords[:, 1] - kmeans_model.lon_min) / kmeans_model.lon_range
    
    spatial_features = [
        'nearest_centroid_dist', 'second_nearest_dist', 'third_nearest_dist', 'avg_nearest_3_dist',
        'inv_nearest_dist', 'inv_second_nearest', 'spatial_isolation',
        'lat_normalized', 'lon_normalized'
    ]
    
    print(f"Added {len(spatial_features)} GLOBAL spatial features")
    
    return df, kmeans_model, spatial_features

Water_Quality_df = pd.read_csv("water_quality_training_dataset.csv")
print(f"Training samples: {len(Water_Quality_df)}")
print(Water_Quality_df.head())

landsat_train = pd.read_csv("landsat_features_training.csv")
landsat_train['NDMI'] = landsat_train['NDMI'].astype(float)
landsat_train['MNDWI'] = landsat_train['MNDWI'].astype(float)
print(f"Landsat features: {landsat_train.shape}")
print(landsat_train.head(3))

climate_train = pd.read_csv("climate_features_training.csv")
print(f"Climate features: {climate_train.shape}")
print(f"Variables: {[col for col in climate_train.columns if col not in ['Latitude', 'Longitude', 'Sample Date']]}")

if 'tmax' in climate_train.columns:
    climate_train['tmax'] = climate_train['tmax'] / 10.0
if 'tmin' in climate_train.columns:
    climate_train['tmin'] = climate_train['tmin'] / 10.0
if 'tmax' in climate_train.columns and 'tmin' in climate_train.columns:
    climate_train['temp_range'] = climate_train['tmax'] - climate_train['tmin']
    climate_train['temp_mean'] = (climate_train['tmax'] + climate_train['tmin']) / 2

print(climate_train.head(3))

wq_data = pd.concat([Water_Quality_df, landsat_train, climate_train], axis=1)
wq_data = wq_data.loc[:, ~wq_data.columns.duplicated()]
print(f"Combined shape: {wq_data.shape}")
missing_total = wq_data.isna().sum().sum()
total_values = wq_data.shape[0] * wq_data.shape[1]
print(f"Missing values: {missing_total} ({missing_total/total_values*100:.2f}%)")
print(f"\n Missing values per column:")
missing_cols = wq_data.isna().sum()
missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
for col, count in missing_cols.items():
    print(f"   {col}: {count} ({count/len(wq_data)*100:.1f}%)")

wq_data['Sample Date'] = pd.to_datetime(wq_data['Sample Date'], format='%d-%m-%Y')
wq_data['month'] = wq_data['Sample Date'].dt.month
wq_data['day_of_year'] = wq_data['Sample Date'].dt.dayofyear
wq_data['year'] = wq_data['Sample Date'].dt.year

def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3

wq_data['season'] = wq_data['month'].apply(get_season)
wq_data['month_sin'] = np.sin(2 * np.pi * wq_data['month'] / 12)
wq_data['month_cos'] = np.cos(2 * np.pi * wq_data['month'] / 12)
wq_data['day_sin'] = np.sin(2 * np.pi * wq_data['day_of_year'] / 365)
wq_data['day_cos'] = np.cos(2 * np.pi * wq_data['day_of_year'] / 365)

print("Created temporal features")

print("Climate temporal feature engineering...")

wq_data['ppt_log'] = np.log1p(wq_data['ppt'].clip(lower=0))
wq_data['soil_log'] = np.log1p(wq_data['soil'].clip(lower=0))
wq_data['ppt_soil_ratio'] = wq_data['ppt'] / (wq_data['soil'] + 0.01)
wq_data['temp_moisture_interaction'] = wq_data['temp_mean'] * wq_data['soil']
wq_data['ppt_temp_interaction'] = wq_data['ppt'] * wq_data['temp_mean']
wq_data['seasonal_ppt'] = wq_data['ppt'] * wq_data['season']
wq_data['seasonal_temp'] = wq_data['temp_mean'] * wq_data['season']

wq_data_sorted = wq_data.sort_values(['Latitude', 'Longitude', 'Sample Date'])
wq_data_sorted['ppt_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['ppt'].shift(1)
wq_data_sorted['temp_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['temp_mean'].shift(1)
wq_data_sorted['soil_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['soil'].shift(1)
wq_data = wq_data_sorted.sort_index()

print("Created climate temporal features")

print("Advanced climate features (pet, vpd, q, aet, def)...")

wq_data['water_balance'] = wq_data['pet'] - wq_data['aet']
wq_data['water_stress'] = wq_data['def'] / (wq_data['ppt'] + 0.01)
wq_data['runoff_fraction'] = wq_data['q'] / (wq_data['ppt'] + 0.01)
wq_data['water_efficiency'] = wq_data['aet'] / (wq_data['pet'] + 0.01)

wq_data['phosphorus_transport_proxy'] = wq_data['q'] * wq_data['ppt']
wq_data['erosion_risk'] = wq_data['q'] / (wq_data['soil'] + 0.01)
wq_data['nutrient_dilution'] = wq_data['q'] / (wq_data['def'] + 0.01)

wq_data['vpd_stress'] = wq_data['vpd'] / (wq_data['temp_mean'] + 0.01)
wq_data['deficit_intensity'] = wq_data['def'] * wq_data['vpd']
wq_data['evaporative_demand'] = wq_data['pet'] * wq_data['vpd']

wq_data_sorted = wq_data.sort_values(['Latitude', 'Longitude', 'Sample Date'])
wq_data_sorted['q_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['q'].shift(1)
wq_data_sorted['q_lag14'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['q'].shift(2)
wq_data_sorted['vpd_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['vpd'].shift(1)
wq_data_sorted['def_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['def'].shift(1)
wq_data_sorted['pet_lag7'] = wq_data_sorted.groupby(['Latitude', 'Longitude'])['pet'].shift(1)
wq_data = wq_data_sorted.sort_index()

wq_data['runoff_vegetation'] = wq_data['q'] * wq_data['NDMI']
wq_data['erosion_water'] = wq_data['q'] * wq_data['MNDWI']
wq_data['concentration_effect'] = wq_data['def'] * wq_data['MNDWI']

wq_data['water_balance_ratio'] = (wq_data['pet'] - wq_data['aet']) / (wq_data['ppt'] + 0.01)
wq_data['effective_runoff'] = wq_data['q'] / (wq_data['ppt'] + wq_data['soil'] + 0.01)
wq_data['compound_stress'] = (wq_data['vpd'] * wq_data['def']) / (wq_data['soil'] + 0.01)

wq_data['pet_log'] = np.log1p(wq_data['pet'].clip(lower=0))
wq_data['q_log'] = np.log1p(wq_data['q'].clip(lower=0))
wq_data['vpd_log'] = np.log1p(wq_data['vpd'].clip(lower=0))

wq_data['def_log'] = np.log1p(wq_data['def'].clip(lower=0))
print(f"   Log transforms: 4 | Seasonal: 3")

print(f"   Climate-spectral: 3 | Advanced ratios: 3")

wq_data['seasonal_runoff'] = wq_data['q'] * wq_data['season']
print(f"   Water balance: 4 | Nutrient transport (DRP): 3")

wq_data['seasonal_deficit'] = wq_data['def'] * wq_data['season']
print("Created 35+ advanced climate features")

wq_data['seasonal_pet'] = wq_data['pet'] * wq_data['season']

print("Spectral feature engineering...")

wq_data['nir_swir22_ratio'] = wq_data['nir'] / (wq_data['swir22'] + 0.0001)
wq_data['green_swir16_ratio'] = wq_data['green'] / (wq_data['swir16'] + 0.0001)
wq_data['nir_green_ratio'] = wq_data['nir'] / (wq_data['green'] + 0.0001)
wq_data['NDMI_MNDWI_product'] = wq_data['NDMI'] * wq_data['MNDWI']
wq_data['NDMI_squared'] = wq_data['NDMI'] ** 2
wq_data['MNDWI_squared'] = wq_data['MNDWI'] ** 2
wq_data['nir_squared'] = wq_data['nir'] ** 2
wq_data['turbidity_proxy'] = wq_data['green'] / (wq_data['nir'] + 0.0001)
wq_data['water_signature'] = (wq_data['MNDWI'] + wq_data['NDMI']) / 2
wq_data['nir_green_distance'] = np.sqrt((wq_data['nir'] - wq_data['green'])**2)
wq_data['swir_distance'] = np.sqrt((wq_data['swir16'] - wq_data['swir22'])**2)

print("Created spectral engineered features")

print("Climate-spectral interactions...")

wq_data['NDMI_ppt_interaction'] = wq_data['NDMI'] * wq_data['ppt']
wq_data['MNDWI_soil_interaction'] = wq_data['MNDWI'] * wq_data['soil']
wq_data['nir_ppt_interaction'] = wq_data['nir'] * wq_data['ppt']
wq_data['temp_MNDWI_interaction'] = wq_data['temp_mean'] * wq_data['MNDWI']
wq_data['water_stress_index'] = wq_data['MNDWI'] / (wq_data['temp_mean'] + 10)
wq_data['vegetation_moisture'] = wq_data['NDMI'] * wq_data['soil']

print("Created climate-spectral interactions")

print("\n" + "="*70)
print(f"CREATING K-MEANS SPATIAL FEATURES (n_clusters={N_SPATIAL_CLUSTERS})")
print("="*70)

wq_data, kmeans_model, spatial_feature_names = spatial_features(wq_data, kmeans_model=None, n_clusters=N_SPATIAL_CLUSTERS)

print("="*70)

print("\n Creating river groups for GroupKFold...")

wq_data['river_group'] = (
    wq_data['Latitude'].round(2).astype(str) + '_' + 
    wq_data['Longitude'].round(2).astype(str)
)
n_groups = wq_data['river_group'].nunique()
print(f"Created {n_groups} river location groups")
print("GroupKFold will ensure same river not in both train/val")

print("Applying PCA to spectral features...")

spectral_features = ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']
spectral_df = wq_data[spectral_features].fillna(wq_data[spectral_features].median())

pca_spectral = PCA(n_components=0.95, random_state=42)
spectral_pca = pca_spectral.fit_transform(spectral_df)

for i in range(spectral_pca.shape[1]):
    wq_data[f'spectral_PC{i+1}'] = spectral_pca[:, i]

print(f"Created {spectral_pca.shape[1]} PCA components ({pca_spectral.explained_variance_ratio_.sum()*100:.1f}% variance)")

X = wq_data.drop(columns=['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus',
                          'Latitude', 'Longitude', 'Sample Date', 'river_group'], errors='ignore')
y_TA = wq_data['Total Alkalinity']
y_EC = wq_data['Electrical Conductance']
y_DRP = wq_data['Dissolved Reactive Phosphorus']
river_groups = wq_data['river_group']

print(f"Features shape: {X.shape}")
print(f"Targets: TA, EC, DRP")
print(f"Groups: {len(river_groups.unique())} unique river locations")

def train_with_groupkfold(X, y, river_groups, param_name="Parameter"):
    print(f"\n{'='*70}")
    print(f"Training: {param_name}")
    print(f"{'='*70}")
    
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=river_groups), 1):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        train_medians = X_train_fold.median()
        X_train_imp = X_train_fold.fillna(train_medians)
        X_val_imp = X_val_fold.fillna(train_medians)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=20,
            min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train_fold)
        
        y_pred = model.predict(X_val_scaled)
        r2_fold = r2_score(y_val_fold, y_pred)
        cv_scores.append(r2_fold)
        print(f"  Fold {fold}/5: R² = {r2_fold:.4f}")

    print(f"GroupKFold CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    train_medians = X.median()
    X_imputed = X.fillna(train_medians)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    models_final = []
    for seed in ENSEMBLE_SEEDS:
        model_final = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=20,
            min_samples_leaf=10, max_features='sqrt', random_state=seed, n_jobs=-1
        )
        model_final.fit(X_scaled, y)
        models_final.append(model_final)

    print(f"Final ensemble trained: {len(models_final)} models")

    return models_final, scaler, train_medians, np.mean(cv_scores)

print("\n" + "="*70)
print("PHASE 1: Training with ALL features")
print("="*70)

model_TA_all, scaler_TA_all, medians_TA_all, cv_TA_all = train_with_groupkfold(X, y_TA, river_groups, "Total Alkalinity")
model_EC_all, scaler_EC_all, medians_EC_all, cv_EC_all = train_with_groupkfold(X, y_EC, river_groups, "Electrical Conductance")
model_DRP_all, scaler_DRP_all, medians_DRP_all, cv_DRP_all = train_with_groupkfold(X, y_DRP, river_groups, "Dissolved Reactive Phosphorus")

print("\nAll models trained with ALL features")

print("\n" + "="*70)
print("PHASE 2: Feature importance analysis per target")
print("="*70)

feature_names = X.columns.tolist()

importances_TA = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.mean([m.feature_importances_ for m in model_TA_all], axis=0)
}).sort_values('Importance', ascending=False)

importances_EC = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.mean([m.feature_importances_ for m in model_EC_all], axis=0)
}).sort_values('Importance', ascending=False)

importances_DRP = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.mean([m.feature_importances_ for m in model_DRP_all], axis=0)
}).sort_values('Importance', ascending=False)

print("\nTop 15 features for Total Alkalinity:")
print(importances_TA.head(15).to_string(index=False))

print("\nTop 15 features for Electrical Conductance:")
print(importances_EC.head(15).to_string(index=False))

print("\nTop 15 features for Dissolved Reactive Phosphorus:")
print(importances_DRP.head(15).to_string(index=False))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.barh(range(30), importances_TA.head(30)['Importance'].values)
plt.yticks(range(30), importances_TA.head(30)['Feature'].values)
plt.xlabel('Importance')
plt.title('Total Alkalinity - Top 10')
plt.gca().invert_yaxis()

plt.subplot(1, 3, 2)
plt.barh(range(30), importances_EC.head(30)['Importance'].values)
plt.yticks(range(30), importances_EC.head(30)['Feature'].values)
plt.xlabel('Importance')
plt.title('Electrical Conductance - Top 10')
plt.gca().invert_yaxis()

plt.subplot(1, 3, 3)
plt.barh(range(30), importances_DRP.head(30)['Importance'].values)
plt.yticks(range(30), importances_DRP.head(30)['Feature'].values)
plt.xlabel('Importance')
plt.title('Dissolved Reactive Phosphorus - Top 10')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
print("\nFeature importance plot saved: feature_importance_comparison.png")
plt.close()

print("\n" + "="*70)
print("ANÁLISIS: ¿Cuántas features seleccionar?")
print("="*70)

thresholds = [0.80, 0.85, 0.90, 0.95, 0.98]
analysis_results = []

for threshold in thresholds:
    n_TA = (importances_TA.sort_values('Importance', ascending=False)['Importance'].cumsum() / 
            importances_TA['Importance'].sum() <= threshold).sum()
    n_EC = (importances_EC.sort_values('Importance', ascending=False)['Importance'].cumsum() / 
            importances_EC['Importance'].sum() <= threshold).sum()
    n_DRP = (importances_DRP.sort_values('Importance', ascending=False)['Importance'].cumsum() / 
             importances_DRP['Importance'].sum() <= threshold).sum()
    
    avg_features = (n_TA + n_EC + n_DRP) / 3
    samples_per_feature = 1860 / avg_features 
    
    analysis_results.append({
        'Threshold': f'{threshold*100}%',
        'TA': n_TA,
        'EC': n_EC,
        'DRP': n_DRP,
        'Avg': f'{avg_features:.0f}',
        'Samples/Feature': f'{samples_per_feature:.1f}',
        'Status': 'Óptimo' if 40 <= samples_per_feature <= 60 else 
                  ('Bueno' if 30 <= samples_per_feature else 'Riesgo')
    })

import pandas as pd
df_analysis = pd.DataFrame(analysis_results)
print("\nComparación de umbrales de importancia:")
print(df_analysis.to_string(index=False))

print("\nInterpretación:")
print("   - Samples/Feature > 40: Generalización excelente")
print("   - Samples/Feature 30-40: Balance bueno")
print("   - Samples/Feature < 30: Riesgo de overfitting")
print(f"\nDataset: 9,319 samples | GroupKFold 5-fold: ~1,860 samples/fold")

print(f"\nRECORDATORIO: Es 'importancia acumulada'")
print(f"   • DISMINUIR threshold (0.90 → 0.85) = MENOS features")
print(f"   • AUMENTAR threshold (0.90 → 0.95) = MÁS features")
print(f"\nRecomendación: Usa 0.85 o 0.80 si tienes >60 features por modelo")

n_top_features = 45  

top_features_TA = importances_TA.head(n_top_features)['Feature'].tolist()
top_features_EC = importances_EC.head(n_top_features)['Feature'].tolist()
top_features_DRP = importances_DRP.head(n_top_features)['Feature'].tolist()

print(f"\nFixed feature selection: Top {n_top_features} features per target")
print(f"   TA features: {len(top_features_TA)}")
print(f"   EC features: {len(top_features_EC)}")
print(f"   DRP features: {len(top_features_DRP)}")

samples_per_feature_fold = 1860 / n_top_features 

print(f"\nQuality check:")
print(f"   Samples/feature per fold: {samples_per_feature_fold:.1f}")
if samples_per_feature_fold >= 40:
    print(f"   EXCELENTE - Baja probabilidad de overfitting")
elif samples_per_feature_fold >= 30:
    print(f"   BUENO - Balance aceptable")
else:
    print(f"   RIESGO - Considerar reducir n_top_features")

print(f"\nPara ajustar cambiar n_top_features (línea 1 de esta celda)")


print("PHASE 3: Retraining with target-specific features")
print("="*70)

X_TA_opt = X[top_features_TA]
X_EC_opt = X[top_features_EC]
X_DRP_opt = X[top_features_DRP]

model_TA_opt, scaler_TA_opt, medians_TA_opt, cv_TA_opt = train_with_groupkfold(
    X_TA_opt, y_TA, river_groups, "Total Alkalinity (Optimized)"
)

model_EC_opt, scaler_EC_opt, medians_EC_opt, cv_EC_opt = train_with_groupkfold(
    X_EC_opt, y_EC, river_groups, "Electrical Conductance (Optimized)"
)

model_DRP_opt, scaler_DRP_opt, medians_DRP_opt, cv_DRP_opt = train_with_groupkfold(
    X_DRP_opt, y_DRP, river_groups, "Dissolved Reactive Phosphorus (Optimized)"
)

print("\nOptimized models trained")

comparison = pd.DataFrame({
    'Target': ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus'],
    'All_Features_CV_R2': [cv_TA_all, cv_EC_all, cv_DRP_all],
    'Top_Features_CV_R2': [cv_TA_opt, cv_EC_opt, cv_DRP_opt],
    'Improvement': [cv_TA_opt - cv_TA_all, cv_EC_opt - cv_EC_all, cv_DRP_opt - cv_DRP_all],
    'Num_Features_Used': [len(top_features_TA), len(top_features_EC), len(top_features_DRP)]
})

print("\n" + "="*70)
print("MODEL COMPARISON: All Features vs Target-Specific")
print("="*70)
print(comparison)

avg_improvement = comparison['Improvement'].mean()
print(f"\n{'OK' if avg_improvement >= -0.01 else 'WARNING'} Average improvement: {avg_improvement:+.4f}")

print(f"\nKey Observations:")
print(f"   1) All features model has {X.shape[1]} features")
print(f"   2) More features = more signal but risk of overfitting to regions")
print(f"   3) GroupKFold tests on UNSEEN rivers, so lower CV is expected")
print(f"\nWhy is DRP R² so low (~0.06) vs TA/EC (~0.30)?")
print(f"   - DRP = Dissolved Reactive Phosphorus (point-source pollution)")
print(f"   - Depends on local runoff, agricultural practices, sewage")
print(f"   - Satellites capture WATER SURFACE, not phosphorus sources")
print(f"   - TA/EC depend on geology/climate which satellites DO capture")
print(f"   - Would need: soil chemistry maps, farm locations, rainfall events")

if PREDICTION_MODE == "top_n":
    print("\nUsing top_n optimized models for final predictions")
    models_final = {
        'TA': (model_TA_opt, scaler_TA_opt, medians_TA_opt, top_features_TA),
        'EC': (model_EC_opt, scaler_EC_opt, medians_EC_opt, top_features_EC),
        'DRP': (model_DRP_opt, scaler_DRP_opt, medians_DRP_opt, top_features_DRP)
    }
    selected_cv_TA, selected_cv_EC, selected_cv_DRP = cv_TA_opt, cv_EC_opt, cv_DRP_opt
else:
    print("\nUsing all-features models for final predictions")
    all_features = X.columns.tolist()
    models_final = {
        'TA': (model_TA_all, scaler_TA_all, medians_TA_all, all_features),
        'EC': (model_EC_all, scaler_EC_all, medians_EC_all, all_features),
        'DRP': (model_DRP_all, scaler_DRP_all, medians_DRP_all, all_features)
    }
    selected_cv_TA, selected_cv_EC, selected_cv_DRP = cv_TA_all, cv_EC_all, cv_DRP_all

print("\n" + "="*70)
print("Loading validation data")
print("="*70)

test_template = pd.read_csv("submission_template.csv")
landsat_val = pd.read_csv("landsat_features_validation.csv")
climate_val = pd.read_csv("climate_features_validation.csv")

landsat_val['NDMI'] = landsat_val['NDMI'].astype(float)
landsat_val['MNDWI'] = landsat_val['MNDWI'].astype(float)

if 'tmax' in climate_val.columns:
    climate_val['tmax'] = climate_val['tmax'] / 10.0
if 'tmin' in climate_val.columns:
    climate_val['tmin'] = climate_val['tmin'] / 10.0
if 'tmax' in climate_val.columns and 'tmin' in climate_val.columns:
    climate_val['temp_range'] = climate_val['tmax'] - climate_val['tmin']
    climate_val['temp_mean'] = (climate_val['tmax'] + climate_val['tmin']) / 2

print(f"Validation data loaded: {len(test_template)} samples")

val_data = pd.concat([test_template[['Latitude', 'Longitude', 'Sample Date']], 
                      landsat_val, climate_val], axis=1)
val_data = val_data.loc[:, ~val_data.columns.duplicated()]

print(f"Validation combined shape: {val_data.shape}")

print("Creating temporal features for validation...")

val_data['Sample Date'] = pd.to_datetime(val_data['Sample Date'], format='%d-%m-%Y')
val_data['month'] = val_data['Sample Date'].dt.month
val_data['day_of_year'] = val_data['Sample Date'].dt.dayofyear
val_data['year'] = val_data['Sample Date'].dt.year
val_data['season'] = val_data['month'].apply(get_season)
val_data['month_sin'] = np.sin(2 * np.pi * val_data['month'] / 12)
val_data['month_cos'] = np.cos(2 * np.pi * val_data['month'] / 12)
val_data['day_sin'] = np.sin(2 * np.pi * val_data['day_of_year'] / 365)
val_data['day_cos'] = np.cos(2 * np.pi * val_data['day_of_year'] / 365)

print("Temporal features created")

print("Climate temporal features for validation...")

val_data['ppt_log'] = np.log1p(val_data['ppt'].clip(lower=0))
val_data['soil_log'] = np.log1p(val_data['soil'].clip(lower=0))
val_data['ppt_soil_ratio'] = val_data['ppt'] / (val_data['soil'] + 0.01)
val_data['temp_moisture_interaction'] = val_data['temp_mean'] * val_data['soil']
val_data['ppt_temp_interaction'] = val_data['ppt'] * val_data['temp_mean']
val_data['seasonal_ppt'] = val_data['ppt'] * val_data['season']
val_data['seasonal_temp'] = val_data['temp_mean'] * val_data['season']

val_data_sorted = val_data.sort_values(['Latitude', 'Longitude', 'Sample Date'])
val_data_sorted['ppt_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['ppt'].shift(1)
val_data_sorted['temp_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['temp_mean'].shift(1)
val_data_sorted['soil_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['soil'].shift(1)
val_data = val_data_sorted.sort_index()

print("Climate temporal features created")

print("Advanced climate features for validation (pet, vpd, q, aet, def)...")

val_data['water_balance'] = val_data['pet'] - val_data['aet']
val_data['water_stress'] = val_data['def'] / (val_data['ppt'] + 0.01)
val_data['runoff_fraction'] = val_data['q'] / (val_data['ppt'] + 0.01)
val_data['water_efficiency'] = val_data['aet'] / (val_data['pet'] + 0.01)

val_data['phosphorus_transport_proxy'] = val_data['q'] * val_data['ppt']
val_data['erosion_risk'] = val_data['q'] / (val_data['soil'] + 0.01)
val_data['nutrient_dilution'] = val_data['q'] / (val_data['def'] + 0.01)

val_data['vpd_stress'] = val_data['vpd'] / (val_data['temp_mean'] + 0.01)
val_data['deficit_intensity'] = val_data['def'] * val_data['vpd']
val_data['evaporative_demand'] = val_data['pet'] * val_data['vpd']

val_data_sorted = val_data.sort_values(['Latitude', 'Longitude', 'Sample Date'])
val_data_sorted['q_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['q'].shift(1)
val_data_sorted['q_lag14'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['q'].shift(2)
val_data_sorted['vpd_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['vpd'].shift(1)
val_data_sorted['def_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['def'].shift(1)
val_data_sorted['pet_lag7'] = val_data_sorted.groupby(['Latitude', 'Longitude'])['pet'].shift(1)
val_data = val_data_sorted.sort_index()

val_data['runoff_vegetation'] = val_data['q'] * val_data['NDMI']
val_data['erosion_water'] = val_data['q'] * val_data['MNDWI']
val_data['concentration_effect'] = val_data['def'] * val_data['MNDWI']

val_data['water_balance_ratio'] = (val_data['pet'] - val_data['aet']) / (val_data['ppt'] + 0.01)
val_data['effective_runoff'] = val_data['q'] / (val_data['ppt'] + val_data['soil'] + 0.01)
val_data['compound_stress'] = (val_data['vpd'] * val_data['def']) / (val_data['soil'] + 0.01)

val_data['pet_log'] = np.log1p(val_data['pet'].clip(lower=0))
print("Advanced climate features created for validation")

val_data['q_log'] = np.log1p(val_data['q'].clip(lower=0))

val_data['vpd_log'] = np.log1p(val_data['vpd'].clip(lower=0))
val_data['seasonal_pet'] = val_data['pet'] * val_data['season']

val_data['def_log'] = np.log1p(val_data['def'].clip(lower=0))
val_data['seasonal_deficit'] = val_data['def'] * val_data['season']

val_data['seasonal_runoff'] = val_data['q'] * val_data['season']

print("Spectral features for validation...")

val_data['nir_swir22_ratio'] = val_data['nir'] / (val_data['swir22'] + 0.0001)
val_data['green_swir16_ratio'] = val_data['green'] / (val_data['swir16'] + 0.0001)
val_data['nir_green_ratio'] = val_data['nir'] / (val_data['green'] + 0.0001)
val_data['NDMI_MNDWI_product'] = val_data['NDMI'] * val_data['MNDWI']
val_data['NDMI_squared'] = val_data['NDMI'] ** 2
val_data['MNDWI_squared'] = val_data['MNDWI'] ** 2
val_data['nir_squared'] = val_data['nir'] ** 2
val_data['turbidity_proxy'] = val_data['green'] / (val_data['nir'] + 0.0001)
val_data['water_signature'] = (val_data['MNDWI'] + val_data['NDMI']) / 2
val_data['nir_green_distance'] = np.sqrt((val_data['nir'] - val_data['green'])**2)
val_data['swir_distance'] = np.sqrt((val_data['swir16'] - val_data['swir22'])**2)

print("Spectral features created")

print("Climate-spectral interactions for validation...")

val_data['NDMI_ppt_interaction'] = val_data['NDMI'] * val_data['ppt']
val_data['MNDWI_soil_interaction'] = val_data['MNDWI'] * val_data['soil']
val_data['nir_ppt_interaction'] = val_data['nir'] * val_data['ppt']
val_data['temp_MNDWI_interaction'] = val_data['temp_mean'] * val_data['MNDWI']
val_data['water_stress_index'] = val_data['MNDWI'] / (val_data['temp_mean'] + 10)
val_data['vegetation_moisture'] = val_data['NDMI'] * val_data['soil']

print("Climate-spectral interactions created")

print("\nAdding K-means spatial features to validation data...")
val_data, _, _ = spatial_features(val_data, kmeans_model=kmeans_model, n_clusters=N_SPATIAL_CLUSTERS)

test_nearest_dist = val_data['nearest_centroid_dist']
print(f"\nTest set spatial analysis:")
print(f"   Mean dist to nearest centroid: {test_nearest_dist.mean():.2f}° (~{test_nearest_dist.mean()*111:.0f} km)")
print(f"   Median: {test_nearest_dist.median():.2f}°")
print(f"   Max: {test_nearest_dist.max():.2f}°")
print(f"   Min: {test_nearest_dist.min():.2f}°")

print("\nApplying PCA to validation spectral features...")

spectral_df_val = val_data[spectral_features].fillna(val_data[spectral_features].median())
spectral_pca_val = pca_spectral.transform(spectral_df_val)

for i in range(spectral_pca_val.shape[1]):
    val_data[f'spectral_PC{i+1}'] = spectral_pca_val[:, i]

print(f"Applied {spectral_pca_val.shape[1]} PCA components to validation")

print("\n" + "="*70)
print("Generating predictions")
print("="*70)

val_data_clean = val_data.drop(columns=['Latitude', 'Longitude', 'Sample Date'], errors='ignore')

model_TA, scaler_TA, medians_TA, features_TA = models_final['TA']
model_EC, scaler_EC, medians_EC, features_EC = models_final['EC']
model_DRP, scaler_DRP, medians_DRP, features_DRP = models_final['DRP']

X_val_TA = val_data_clean[features_TA].fillna(medians_TA)
X_val_EC = val_data_clean[features_EC].fillna(medians_EC)
X_val_DRP = val_data_clean[features_DRP].fillna(medians_DRP)

X_val_TA_scaled = scaler_TA.transform(X_val_TA)
X_val_EC_scaled = scaler_EC.transform(X_val_EC)
X_val_DRP_scaled = scaler_DRP.transform(X_val_DRP)

def predict_ensemble(models, X_scaled):
    preds = np.column_stack([m.predict(X_scaled) for m in models])
    return preds.mean(axis=1)

pred_TA = predict_ensemble(model_TA, X_val_TA_scaled)
pred_EC = predict_ensemble(model_EC, X_val_EC_scaled)
pred_DRP = predict_ensemble(model_DRP, X_val_DRP_scaled)

print("Predictions generated for all targets")

submission_df = pd.DataFrame({
    'Longitude': test_template['Longitude'].values,
    'Latitude': test_template['Latitude'].values,
    'Sample Date': test_template['Sample Date'].values,
    'Total Alkalinity': pred_TA,
    'Electrical Conductance': pred_EC,
    'Dissolved Reactive Phosphorus': pred_DRP
})

print("\n" + "="*70)
print("Submission Preview")
print("="*70)
print(submission_df.head(10))
print("\n")
print(submission_df.describe())

output_filename = "submission.csv"
submission_df.to_csv(output_filename, index=False)
print(f"\nSubmission file saved: {output_filename}")
print(f"Shape: {submission_df.shape}")
print(f"Format matches submission_template.csv")
print("\nModel Summary:")
print(f"   - Prediction mode: {PREDICTION_MODE}")
print(f"   - Ensemble size: {len(ENSEMBLE_SEEDS)}")
print(f"   - TA: {len(features_TA)} features, CV R² ref: {selected_cv_TA:.4f}")
print(f"   - EC: {len(features_EC)} features, CV R² ref: {selected_cv_EC:.4f}")
print(f"   - DRP: {len(features_DRP)} features, CV R² ref: {selected_cv_DRP:.4f}")
print(f"   - GroupKFold ensured no river leakage")
print(f"   - PCA applied to spectral features")
