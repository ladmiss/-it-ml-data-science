from __future__ import annotations

import argparse
from datetime import date, datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import (
    DATE_COLUMN,
    DEFAULT_HISTORY_DAYS,
    RF_PARAMS,
    TARGET_COLUMN,
    TOP_DIRECTIONS_COUNT,
    default_date_range,
    ensure_project_dirs,
    model_artifact_path,
)
from data_loader import (
    get_top_directions,
    load_processed_salary_data,
    prepare_and_save_salary_dataset,
)
from features import FEATURE_COLUMNS, create_feature_frame, split_train_test
from utils import build_metrics_table, calculate_metrics, choose_best_model


def _parse_date(date_str: str) -> date:
    return pd.to_datetime(date_str, format="%Y-%m-%d", errors="raise").date()


def _prepare_direction_series(processed_df: pd.DataFrame, direction_name: str) -> pd.Series:
    if processed_df.empty:
        raise ValueError("processed данные пусты")
    if "direction" not in processed_df.columns:
        raise ValueError("в processed данных нет колонки direction")
    if DATE_COLUMN not in processed_df.columns or TARGET_COLUMN not in processed_df.columns:
        raise ValueError(f"в данных нет колонок {DATE_COLUMN} и {TARGET_COLUMN}")

    # берем только один вариант чтобы модель училась отдельно по направлению
    data = processed_df[processed_df["direction"] == direction_name].copy()
    if data.empty:
        raise ValueError(f"нет данных по направлению {direction_name}")

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).sort_values(DATE_COLUMN)
    data = data.drop_duplicates(subset=[DATE_COLUMN], keep="last")

    series = data.set_index(DATE_COLUMN)[TARGET_COLUMN].astype(float)
    # выравниваем временной ряд по дням чтобы обучение было стабильным
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()
    series.name = direction_name
    return series


def train_direction_model(
    processed_df: pd.DataFrame,
    direction_name: str,
    save_artifact: bool = True,
) -> dict:
    series = _prepare_direction_series(processed_df, direction_name)
    feature_df = create_feature_frame(series)
    if feature_df.empty:
        raise ValueError(f"после генерации признаков для {direction_name} не осталось строк")

    # делим строго по времени без рандома потому что это временной ряд
    train_df, test_df = split_train_test(feature_df)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(**RF_PARAMS),
    }

    predictions: dict[str, pd.Series] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}

    # пробуем две модели и смотрим какая реально лучше на test
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = pd.Series(model.predict(x_test), index=x_test.index, name="predicted").clip(lower=0.0)
        predictions[model_name] = y_pred
        metrics_by_model[model_name] = calculate_metrics(y_test.to_numpy(), y_pred.to_numpy())

    best_model_name = choose_best_model(metrics_by_model)
    best_model = models[best_model_name]
    best_metrics = metrics_by_model[best_model_name]

    test_result_df = pd.DataFrame(
        {
            "date": y_test.index,
            "actual": y_test.values,
            "predicted": predictions[best_model_name].values,
        }
    )

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "direction_name": direction_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics_by_model,
        "best_metrics": best_metrics,
        "rmse_for_range": float(best_metrics["RMSE"]),
        "test": {
            "dates": test_result_df["date"].astype(str).tolist(),
            "actual": test_result_df["actual"].astype(float).tolist(),
            "predicted": test_result_df["predicted"].astype(float).tolist(),
        },
        "last_actual_salary": float(series.iloc[-1]),
        "last_history_date": str(series.index.max().date()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    path = model_artifact_path(direction_name)
    if save_artifact:
        ensure_project_dirs()
        # сохраняем не только модель но и все что нужно для прогноза
        joblib.dump(artifact, path)

    return {
        "direction_name": direction_name,
        "model_name": best_model_name,
        "metrics": metrics_by_model,
        "best_metrics": best_metrics,
        "test_df": test_result_df,
        "artifact_path": path,
    }


def load_model_artifact(direction_name: str) -> dict:
    path = model_artifact_path(direction_name)
    if not path.exists():
        raise FileNotFoundError(f"модель для направления {direction_name} не найдена: {path}")
    return joblib.load(path)


def train_top_directions(
    processed_df: pd.DataFrame,
    top_n: int = TOP_DIRECTIONS_COUNT,
) -> dict[str, dict]:
    directions = get_top_directions(processed_df, top_n=top_n)
    results: dict[str, dict] = {}
    for direction in directions:
        results[direction] = train_direction_model(processed_df, direction, save_artifact=True)
    return results


def _load_or_refresh_processed_data(
    refresh_data: bool,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if refresh_data:
        try:
            processed_df, messages = prepare_and_save_salary_dataset(
                start_date=start_date,
                end_date=end_date,
                use_fallback=True,
            )
            for message in messages:
                print(f"[data] {message}")
            return processed_df
        except Exception as exc:
            print(f"[warn] не получилось обновить данные через api: {exc}")
            print("[warn] использую последние processed данные с диска")
            return load_processed_salary_data()

    return load_processed_salary_data()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="обучение моделей прогноза зарплат по направлениям ml ds"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="глубина истории в днях если не заданы start date и end date",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="дата начала периода в формате yyyy-mm-dd",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="дата конца периода в формате yyyy-mm-dd",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_DIRECTIONS_COUNT,
        help="сколько направлений обучать по популярности",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="не обновлять api использовать только локальные processed данные",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    if args.start_date and args.end_date:
        start_date = _parse_date(args.start_date)
        end_date = _parse_date(args.end_date)
    elif args.start_date or args.end_date:
        raise ValueError("нужно указать обе даты и start date и end date")
    else:
        start_date, end_date = default_date_range(args.days)

    print("=" * 80)
    print("прогноз зарплат в it ml ds этап обучения")
    print(f"период данных: {start_date} -> {end_date}")
    print("=" * 80)

    processed_df = _load_or_refresh_processed_data(
        refresh_data=not args.no_refresh,
        start_date=start_date,
        end_date=end_date,
    )

    top_directions = get_top_directions(processed_df, top_n=args.top_n)
    if not top_directions:
        raise RuntimeError("не удалось определить популярные направления для обучения")

    trained_count = 0
    for direction in top_directions:
        try:
            # обучение идет отдельно для каждого направления чтобы модели не путали ряд
            result = train_direction_model(processed_df, direction, save_artifact=True)
            metrics_table = build_metrics_table(result["metrics"])
            print("-" * 80)
            print(f"направление: {direction}")
            print(f"лучшая модель: {result['model_name']}")
            print(metrics_table.to_string(index=False))
            print(f"артефакт сохранён: {result['artifact_path']}")
            trained_count += 1
        except Exception as exc:
            print(f"[error] не удалось обучить {direction}: {exc}")

    print("-" * 80)
    if trained_count == 0:
        raise RuntimeError("не удалось обучить ни одну модель")
    print(f"обучение завершено успешно обучено: {trained_count}/{len(top_directions)}")


if __name__ == "__main__":
    main()

