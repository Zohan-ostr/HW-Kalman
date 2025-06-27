"""
kf_map.py — улучшенная визуализация фильтра Калмана
• Цвет маршрута зависит от скорости
• Области неопределенности на графике и карте
• Полная синхронизация стилей визуализации
"""

import pandas as pd
import numpy as np
import folium
from branca.colormap import LinearColormap
import webbrowser, os, warnings

# -----------------------------------------------------------------------------
# 1. Константы и функции преобразования координат
# -----------------------------------------------------------------------------
EARTH_RADIUS = 6_371_000  # м
CMAP_NAME = 'viridis'  # цветовая схема для скоростей

def latlon_to_xy(lat, lon, lat0, lon0):
    lat0_rad = np.radians(lat0)
    x = EARTH_RADIUS * np.radians(lon - lon0) * np.cos(lat0_rad)
    y = EARTH_RADIUS * np.radians(lat - lat0)
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + np.degrees(y / EARTH_RADIUS)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS * np.cos(np.radians(lat0))))
    return lat, lon

# -----------------------------------------------------------------------------
# 2. Загрузка и подготовка данных
# -----------------------------------------------------------------------------
def load_and_prepare_data():
    df = pd.read_csv('filtered_data1.csv', parse_dates=['Device Time'])
    res = pd.read_csv('kf_results.csv', index_col='Device Time', parse_dates=True)
    
    # Проверка обязательных колонок
    required = ['Latitude', 'Longitude', 'Bearing', 'Speed (OBD)(km/h)']
    if not all(col in df.columns for col in required):
        missing = [col for col in required if col not in df.columns]
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Начальная точка для проекции
    lat0, lon0 = df.loc[0, ['Latitude', 'Longitude']]
    
    # Преобразование координат
    df[['x_gps', 'y_gps']] = df.apply(
        lambda r: latlon_to_xy(r['Latitude'], r['Longitude'], lat0, lon0),
        axis=1, result_type='expand'
    )
    
    # Скорости
    df['speed_obd'] = df['Speed (OBD)(km/h)'] * 1000/3600  # км/ч → м/с
    res['speed_kf'] = np.hypot(res['vx'], res['vy'])  # скорость из фильтра
    
    # Дисперсии
    if {'Pxx', 'Pyy'}.issubset(res.columns):
        res['sigma_x'] = np.sqrt(res['Pxx'])
        res['sigma_y'] = np.sqrt(res['Pyy'])
    else:
        warnings.warn('Нет данных о дисперсиях Pxx/Pyy - области неопределенности не будут построены')
        res['sigma_x'] = res['sigma_y'] = 0.0
    
    # Дисперсия скорости
    if {'Pxx', 'Pyy'}.issubset(res.columns):
        v_norm = res['speed_kf'].replace(0, np.nan)
        var_v = (res['vx']**2 * res['Pxx'] + res['vy']**2 * res['Pyy']) / v_norm.clip(lower=1e-6)**2
        res['sigma_v'] = np.sqrt(var_v).fillna(0.0)
    else:
        res['sigma_v'] = 0.0
    
    return df, res, lat0, lon0

df, res, lat0, lon0 = load_and_prepare_data()

# -----------------------------------------------------------------------------
# 3. Интерактивная карта Folium с цветом по скорости и областями неопределенности
# -----------------------------------------------------------------------------
def create_interactive_map(df, res, lat0, lon0):
    # Преобразуем координаты Калмана
    lat_kf, lon_kf = xy_to_latlon(res['x'].values, res['y'].values, lat0, lon0)
    
    # Создаем карту
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles='OpenStreetMap')
    
    # Цветовая шкала для скорости (в км/ч для удобства)
    speed_min = min(df['Speed (OBD)(km/h)'].min(), res['speed_kf'].min() * 3.6)
    speed_max = max(df['Speed (OBD)(km/h)'].max(), res['speed_kf'].max() * 3.6)
    colormap = LinearColormap(['blue', 'green', 'yellow', 'red'],
                            vmin=speed_min, vmax=speed_max,
                            caption='Скорость (км/ч)')
    m.add_child(colormap)
    
    # 1. GPS трек с цветом по скорости
    gps_fg = folium.FeatureGroup(name='GPS трек (цвет по скорости)', show=True)
    for i in range(len(df)-1):
        speed = (df['Speed (OBD)(km/h)'].iloc[i] + df['Speed (OBD)(km/h)'].iloc[i+1]) / 2
        folium.PolyLine(
            locations=df[['Latitude', 'Longitude']].iloc[i:i+2].values,
            color=colormap(speed),
            weight=5,
            opacity=0.8
        ).add_to(gps_fg)
    gps_fg.add_to(m)
    
    # 2. Трек фильтра Калмана
    kf_fg = folium.FeatureGroup(name='Фильтр Калмана', show=True)
    folium.PolyLine(
        list(zip(lat_kf, lon_kf)),
        color='red',
        weight=4,
        opacity=1
    ).add_to(kf_fg)
    kf_fg.add_to(m)
    
    # 3. Области неопределенности
    if 'Pxx' in res.columns and 'Pyy' in res.columns:
        uncertainty_fg = folium.FeatureGroup(name='Область неопределенности ±3σ', show=True)
        
        for i in range(0, len(lat_kf)):  # шаг 5 для производительности
            # Преобразуем радиус в градусы (приблизительно)
            radius_m = 3 * np.sqrt(res['Pxx'].iloc[i]) + 3 * np.sqrt(res['Pvx'].iloc[i])
            
            folium.Circle(
                location=[lat_kf[i], lon_kf[i]],
                radius=radius_m,
                color='purple',
                fill=True,
                fill_color='purple',
                fill_opacity=0.1,
                weight=0.1
            ).add_to(uncertainty_fg)
        
        uncertainty_fg.add_to(m)
    
    # Маркеры начала и конца
    folium.Marker([lat_kf[0], lon_kf[0]], 
                  popup='Начало', 
                  icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([lat_kf[-1], lon_kf[-1]], 
                  popup='Конец', 
                  icon=folium.Icon(color='red')).add_to(m)
    
    # Управление слоями
    folium.LayerControl().add_to(m)
    
    # Сохраняем и открываем карту
    output_html = 'kalman_filter_map.html'
    m.save(output_html)
    webbrowser.open('file://' + os.path.realpath(output_html))
    print(f'Интерактивная карта сохранена в: {output_html}')

create_interactive_map(df, res, lat0, lon0)