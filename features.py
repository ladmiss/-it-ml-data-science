from __future__ import annotations

import pandas as pd

from config import MIN_TEST_SAMPLES, MIN_TRAIN_SAMPLES, TARGET_COLUMN, TEST_RATIO

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "roll_mean_3",
    "roll_mean_7",
    "roll_std_7",
    "day_of_week",
    "is_weekend",
]


def create_feature_frame(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        raise ValueError("серия пуста нельзя создать признаки")

    # сначала выравниваем ряд по дням чтобы лаги считались правильно
    ts = series.copy().astype(float)
    ts.index = pd.to_datetime(ts.index)
    ts = ts.groupby(level=0).mean().sort_index()

    # заполняем пропуски чтобы не ломать обучение на дырках в истории
    full_index = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    ts = ts.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()

    df = pd.DataFrame({TARGET_COLUMN: ts})
    # лаги показывают что было совсем недавно
    shifted = df[TARGET_COLUMN].shift(1)

    df["lag_1"] = shifted
    df["lag_2"] = df[TARGET_COLUMN].shift(2)
    df["lag_3"] = df[TARGET_COLUMN].shift(3)
    df["lag_7"] = df[TARGET_COLUMN].shift(7)
    # rolling признаки дают модели более спокойный фон ряда
    df["roll_mean_3"] = shifted.rolling(window=3).mean()
    df["roll_mean_7"] = shifted.rolling(window=7).mean()
    df["roll_std_7"] = shifted.rolling(window=7).std()
    # день недели помогает поймать простую календарную сезонность
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["roll_std_7"] = df["roll_std_7"].fillna(0.0)
    df = df.dropna()
    return df


def split_train_test(
    feature_df: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    min_train: int = MIN_TRAIN_SAMPLES,
    min_test: int = MIN_TEST_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_df.empty:
        raise ValueError("пустой датафрейм нельзя делить на train test")

    n_samples = len(feature_df)

    # для коротких рядов делаем мягкое разбиение чтобы test не стал пустым
    adaptive_min_test = min_test if n_samples >= 30 else 3
    adaptive_min_train = min_train if n_samples >= 30 else 8

    # делим строго по времени без перемешивания
    test_size = max(int(round(n_samples * test_ratio)), adaptive_min_test)
    test_size = min(test_size, max(1, n_samples - adaptive_min_train))
    train_size = n_samples - test_size

    if train_size < adaptive_min_train or test_size < 1:
        raise ValueError(
            "недостаточно данных для обучения\n"
            f"получено n={n_samples} train={train_size} test={test_size}\n"
            f"нужно минимум train={adaptive_min_train} test={adaptive_min_test}"
        )

    train_df = feature_df.iloc[:train_size].copy()
    test_df = feature_df.iloc[train_size:].copy()
    return train_df, test_df


def build_feature_row_from_history(
    history_series: pd.Series, forecast_date: pd.Timestamp
) -> pd.DataFrame:
    if history_series.empty:
        raise ValueError("история пуста нельзя построить признаки")

    # для прогноза берем только уже известные значения
    ts = history_series.copy().astype(float)
    ts.index = pd.to_datetime(ts.index)
    ts = ts.groupby(level=0).mean().sort_index()
    full_index = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    ts = ts.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()

    if len(ts) < 7:
        raise ValueError("для прогноза нужно минимум 7 наблюдений")

    # это грубая оценка разброса на последней неделе
    roll_std_7 = ts.iloc[-7:].std()
    if pd.isna(roll_std_7):
        roll_std_7 = 0.0

    row = pd.DataFrame(
        [
            {
                "lag_1": float(ts.iloc[-1]),
                "lag_2": float(ts.iloc[-2]),
                "lag_3": float(ts.iloc[-3]),
                "lag_7": float(ts.iloc[-7]),
                "roll_mean_3": float(ts.iloc[-3:].mean()),
                "roll_mean_7": float(ts.iloc[-7:].mean()),
                "roll_std_7": float(roll_std_7),
                "day_of_week": int(pd.to_datetime(forecast_date).dayofweek),
                "is_weekend": int(pd.to_datetime(forecast_date).dayofweek >= 5),
            }
        ],
        index=[pd.to_datetime(forecast_date)],
    )
    return row[FEATURE_COLUMNS]

