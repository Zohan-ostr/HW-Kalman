import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ── вспомогательная функция ────────────────────────────────────────────────────
def latlon_to_xy(lat, lon, lat0, lon0):
    """Перевод широты/долготы → локальные X‑Y (метры) через простую проекцию."""
    R = 6_371_000
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = R * dlon * np.cos(lat0_rad)
    y = R * dlat
    return x, y


# ── 1. GPS‑лог ────────────────────────────────────────────────────────────────
gps = pd.read_csv(
    "filtered_data1.csv",
    parse_dates=["Device Time"]
)

# «нулевая» точка отсчёта (первое измерение)
lat0, lon0 = gps.loc[0, ["Latitude", "Longitude"]]

# переводим все точки в метры
gps[["x", "y"]] = gps.apply(
    lambda r: latlon_to_xy(r["Latitude"], r["Longitude"], lat0, lon0),
    axis=1, result_type="expand"
)

# ── 2. Результаты фильтра Калмана ─────────────────────────────────────────────
kf = pd.read_csv(
    "kf_results.csv",
    index_col="Device Time",
    parse_dates=True
)
# kf['x'], kf['y'] уже в метрах (вы писали их в фильтре)

# ── 3. Визуализация траектории ───────────────────────────────────────────────
plt.figure(figsize=(8, 8))

plt.plot(gps["x"], gps["y"],
         "--", lw=1.5, label="GPS (raw)")

plt.plot(kf["x"], kf["y"],
         "-",  lw=2.0, label="Kalman filter")

plt.scatter(gps["x"].iloc[0],  gps["y"].iloc[0],
            c="green", marker="o", s=70, label="Start")
plt.scatter(gps["x"].iloc[-1], gps["y"].iloc[-1],
            c="red", marker="s", s=70, label="End")

plt.title("Vehicle trajectory", fontsize=14)
plt.xlabel("X, m"); plt.ylabel("Y, m")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Загрузка и расчёт скоростей ---
df = pd.read_csv("filtered_data1.csv", parse_dates=['Device Time'])
df['speed_gps'] = df['Speed (GPS)(km/h)'] * 1000 / 3600
df['speed_obd'] = df['Speed (OBD)(km/h)'] * 1000 / 3600

kf2 = pd.read_csv('kf_results.csv', index_col='Device Time', parse_dates=['Device Time'])
kf2['speed_kf'] = np.hypot(kf2['vx'], kf2['vy'])

# --- Фильтрация GPS: только ненулевые (не NaN) значения ---
df_gps = df[df['speed_gps'].notna()]

# --- Построение графика ---
plt.figure(figsize=(10, 4))

# GPS: только там, где speed_gps не NaN
plt.plot(df_gps['Device Time'], df_gps['speed_gps'],
         linestyle='-', linewidth=1.5, markersize=4,
         label='GPS speed (only non-NaN)')

# OBD и KF — весь ряд
plt.plot(df['Device Time'], df['speed_obd'],
         linestyle='--', linewidth=1.5, label='OBD speed')
plt.plot(kf2.index, kf2['speed_kf'],
         linestyle='-.', linewidth=2.0, label='KF speed')

plt.title('Сравнение скоростей: GPS vs OBD vs модуль KF', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Speed (m/s)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()



df_kf = pd.read_csv('kf_results.csv', index_col='Device Time', parse_dates=['Device Time'])
# df_kf — ваш df с колонками 'vx','vy','Pvx','Pvy' и индексом Device Time
df_kf['speed_kf'] = np.hypot(df_kf['vx'], df_kf['vy'])

# 1) Оценка σ_speed через линейную аппроксимацию погрешности:
#    ∂s/∂vx = vx/s, ∂s/∂vy = vy/s, var(s)=J·P·Jᵀ
vx = df_kf['vx']
vy = df_kf['vy']
s  = df_kf['speed_kf']
pv = df_kf['Pvx']
pw = df_kf['Pvy']

df_kf['sigma_speed'] = np.sqrt((vx/s)**2 * pv + (vy/s)**2 * pw)

# 2) 3σ-границы
df_kf['upper3'] = df_kf['speed_kf'] + 3*df_kf['sigma_speed']
df_kf['lower3'] = df_kf['speed_kf'] - 3*df_kf['sigma_speed']

# 3) Рисуем основной график + банду
plt.figure(figsize=(10,4))
plt.plot(df_kf.index, df_kf['speed_kf'], '-', label='KF speed')
plt.fill_between(df_kf.index, df_kf['lower3'], df_kf['upper3'],
                 alpha=0.3, label='±3σ interval')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Speed (m/s)')
plt.grid(True)
plt.tight_layout()
plt.show()