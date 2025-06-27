from __future__ import annotations

import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Параметры и константы
# -----------------------------------------------------------------------------
DATA_FILE = Path("data_ready1.csv")            # сырой файл
READY_CSV = Path("filtered_data1.csv")   # промежуточный результат
KF_OUT    = Path("kf_results.csv")       # оценки фильтра

# Шум модели и сенсоров  (можно менять через аргументы CLI)
SIGMA_A_DEFAULT   = 1    #   — интенсивность шума
SIGMA_GPS_DEFAULT = 5.0    #     — точность GPS‑позиции
SIGMA_OBD_SPEED_DEFAULT = 0.3    #   — точность скорости OBD 
SIGMA_GPS_SPEED_DEFAULT = 3

# Порог «дыр» во входных данных
DROPOUT_THRESHOLD = 0.2    # c
SIM_DT            = 0.1    # c  — шаг при симуляции predict‑only

# Флаги для искусственного «отключения» сенсоров
SIM_GPS_DROPOUT: bool = False
SIM_OBD_DROPOUT: bool = False

# Окно, в которое «глушим» сенсор (пример)
GPS_OFF_INTERVAL  = (pd.Timestamp.min, pd.Timestamp.min)  # заполнится CLI
OBD_OFF_INTERVAL  = (pd.Timestamp.min, pd.Timestamp.min)

# -----------------------------------------------------------------------------
# 2. Утилитарные функции перевода координат
# -----------------------------------------------------------------------------
EARTH_RADIUS = 6_371_000  # м радиус Земли

def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Перевод широты/долготы в локальные метры (приближение сферы)."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = EARTH_RADIUS * dlon * np.cos(np.radians(lat0))
    y = EARTH_RADIUS * dlat
    return x, y

# -----------------------------------------------------------------------------
# 3. Фильтр Калмана (4‑мерный)
# -----------------------------------------------------------------------------
class KalmanFilter:
    """X = [x, y, vx, vy]ᵀ — состояние: координаты + скорости."""

    def __init__(self, sigma_a: float, sigma_gps: float, sigma_obd_speed: float, sigma_gps_speed: float):
        self.sigma_a   = sigma_a
        self.sigma_gps = sigma_gps
        self.sigma_obd_speed = sigma_obd_speed
        self.sigma_gps_speed = sigma_gps_speed

        self.x = np.zeros(4)            # начальное состояние
        self.P = np.eye(4) * 1e3        # большая неопределённость

    # ---------- прогноз ----------
    def predict(self, dt: float):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)

        q = self.sigma_a ** 2
        Q = np.diag([q * dt ** 2 / 2,
                     q * dt ** 2 / 2,
                     q * dt,
                     q * dt])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ---------- общее обновление ----------
    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        y = z - H @ self.x                  # инновация
        S = H @ self.P @ H.T + R            # ковариация инновации
        K = self.P @ H.T @ np.linalg.inv(S) # коэффициент Калмана

        self.x += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    # ---------- GPS‑позиция ----------
    def update_gps(self, pos_xy: np.ndarray):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.diag([self.sigma_gps ** 2, self.sigma_gps ** 2])
        self._update(pos_xy, H, R)

    # ---------- OBD‑скорость (вектор) ----------
    def update_obd_speed(self, vx: float, vy: float):
        H = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        R = np.diag([self.sigma_obd_speed ** 2, self.sigma_obd_speed ** 2])
        self._update(np.array([vx, vy]), H, R)

        # ---------- GPS‑скорость (вектор) ----------
    def update_gps_speed(self, vx: float, vy: float):
        H = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        R = np.diag([self.sigma_gps_speed ** 2, self.sigma_gps_speed ** 2])
        self._update(np.array([vx, vy]), H, R)

# -----------------------------------------------------------------------------
# 4. Подготовка данных
# -----------------------------------------------------------------------------

def prepare_dataframe() -> pd.DataFrame:
    """Читает *data_ready1.csv* и возвращает предобработанный DataFrame."""
    cols_needed = [
        "Device Time",
        "Longitude", "Latitude",
        "Speed (GPS)(km/h)", "Bearing", "Speed (OBD)(km/h)",
    ]

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    df = df.loc[:, cols_needed]

    # Время устройства
    df["Device Time"] = pd.to_datetime(df["Device Time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Device Time"]).set_index("Device Time").sort_index()

    # Разница по времени для анализа пропаданий (удобно, но не критично)
    df["dt"] = df.index.to_series().diff().dt.total_seconds().fillna(1.0)

    # Принудительное обнуление GPS-скорости, если скорость по OBD равна нулю
    df.loc[df["Speed (OBD)(km/h)"] == 0, "Speed (GPS)(km/h)"] = 0

    df.to_csv(READY_CSV)
    return df

# -----------------------------------------------------------------------------
# 5. Вспомогательные функции для «глушения» сенсоров
# -----------------------------------------------------------------------------

def _interval_str_to_pair(arg: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """CLI helper: "2025‑05‑01T12:00,2025‑05‑01T12:30" → (start, end)."""
    if not arg:
        return (pd.Timestamp.min, pd.Timestamp.min)
    start_str, end_str = arg.split(",")
    return pd.to_datetime(start_str), pd.to_datetime(end_str)


def sensor_available(ts: pd.Timestamp, dropout_flag: bool,
                     off_interval: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    """Возвращает True, если сенсор *доступен* (учитывает симуляцию dropout)."""
    if not dropout_flag:
        return True
    start, end = off_interval
    #if (start <= ts <= end):
        #print("ДОНТ ВРОКРАШВАЫГ ((((((((((((((((((()))))))))))))))))))")
    #if not (start <= ts <= end):
        #print("ВРОКРАШВАЫГ !!!!!!!!!!!!!!!!!!!!!!!!!!")
    return not (start <= ts <= end)

# -----------------------------------------------------------------------------
# 6. Главный цикл обработки
# -----------------------------------------------------------------------------

def run_filter(df: pd.DataFrame, sigma_a: float, sigma_gps: float, sigma_obd_speed: float, sigma_gps_speed: float):
    # Пре‑проекция GPS‑координат в локальную систему (метры)
    lat0, lon0 = df.iloc[0][["Latitude", "Longitude"]]
    df[["x_gps", "y_gps"]] = df.apply(
        lambda r: latlon_to_xy(r["Latitude"], r["Longitude"], lat0, lon0),
        axis=1, result_type="expand",
    )

    kf = KalmanFilter(sigma_a, sigma_gps, sigma_obd_speed, sigma_gps_speed)

    state_hist: list[np.ndarray] = []
    var_hist:   list[np.ndarray] = []
    index_hist:   list[np.ndarray] = []

    t_cur = df.index[0]

    for ts, row in df.iterrows():
        dt_total = (ts - t_cur).total_seconds()

        # --- 6.1 Заполняем большие «дыры» ---
        while dt_total > DROPOUT_THRESHOLD:
            step = min(SIM_DT, dt_total)
            kf.predict(step)                 # прогноз
            t_cur += timedelta(seconds=step) # двигаем «текущий» момент

            # сохраняем прогноз
            state_hist.append(kf.x.copy())
            var_hist.append(np.diag(kf.P).copy())
            index_hist.append(t_cur)

            dt_total -= step                 # ещё осталось?

        # ---------- 6.2 Обновление на измерение ----------
        kf.predict(dt_total)                 # маленький шаг до самого ts
        t_cur = ts

        # ---------- 6.2 Формирование наблюдений ----------
        gps_ok = pd.notna(row["Latitude"]) #jkdbfaskjbfdsafjkbl GPS SPEED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        gps_speed_ok = pd.notna(row["Speed (GPS)(km/h)"]) and sensor_available(ts, SIM_GPS_DROPOUT, GPS_OFF_INTERVAL)
        obd_ok = pd.notna(row["Speed (OBD)(km/h)"]) and sensor_available(ts, SIM_OBD_DROPOUT, OBD_OFF_INTERVAL)

        # bearing → радианы (если NaN, пусть будет 0 — в этом случае скорость тоже NaN) КАК ИЗМЕРЯЕТЬСЯ БИПРИНГ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        th = np.radians(row["Bearing"]) if not np.isnan(row["Bearing"]) else np.nan
        v_obd = row["Speed (OBD)(km/h)"] * (1000 / 3600)  # м/с
        vx, vy = v_obd * np.sin(th), v_obd * np.cos(th)

        # ---------- 6.3 Приоритет обновлений ----------
        if gps_ok:
            kf.update_gps(np.array([row["x_gps"], row["y_gps"]]))
        if gps_speed_ok:
            kf.update_gps_speed(vx, vy)
            #print('GPS alive')
        elif obd_ok: #else
            kf.update_obd_speed(vx, vy)
            #print('OBD alive')

        state_hist.append(kf.x.copy())
        var_hist.append(np.diag(kf.P).copy())
        index_hist.append(t_cur)


    # ---------- 6.4 Сохранение результатов ----------
    state_cols = ["x", "y", "vx", "vy"]
    var_cols   = ["Pxx", "Pyy", "Pvx", "Pvy"]
    res_df = pd.concat([
        pd.DataFrame(state_hist, columns=state_cols, index=index_hist),
        pd.DataFrame(var_hist,   columns=var_cols,   index=index_hist),
    ], axis=1).sort_index()
    res_df.index.name = "Device Time"

    res_df.to_csv(KF_OUT)
    print(f"➜ Результаты сохранены в kf_results") #{KF_OUT.relative_to(Path.cwd())}

# -----------------------------------------------------------------------------
# 7. Точка входа + CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="4‑D Kalman filter with dropout handling")
    parser.add_argument("--sigma_a",   type=float, default=SIGMA_A_DEFAULT,   help="σ_acc (м/с²)")
    parser.add_argument("--sigma_gps", type=float, default=SIGMA_GPS_DEFAULT, help="σ_GPS (м)")
    parser.add_argument("--sigma_obd_speed", type=float, default=SIGMA_OBD_SPEED_DEFAULT, help="σ_OBD (м/с)")
    parser.add_argument("--sigma_gps_speed", type=float, default=SIGMA_GPS_SPEED_DEFAULT, help="σ_GPS (м/с)")
    parser.add_argument("--gps_off",   type=str,   default="", help="Интервал 'T1,T2' отключения GPS (UTC)")
    parser.add_argument("--obd_off",   type=str,   default="", help="Интервал 'T1,T2' отключения OBD (UTC)")

    args = parser.parse_args()

    global SIM_GPS_DROPOUT, SIM_OBD_DROPOUT, GPS_OFF_INTERVAL, OBD_OFF_INTERVAL
    if args.gps_off:
        SIM_GPS_DROPOUT = True
        GPS_OFF_INTERVAL = _interval_str_to_pair(args.gps_off)
    if args.obd_off:
        SIM_OBD_DROPOUT = True
        OBD_OFF_INTERVAL = _interval_str_to_pair(args.obd_off)

    df = prepare_dataframe()
    run_filter(df, args.sigma_a, args.sigma_gps, args.sigma_obd_speed, args.sigma_gps_speed)

if __name__ == "__main__":
    main()
