from __future__ import annotations

import argparse

import pandas as pd

from config import DATE_COLUMN, FORECAST_HORIZON_DAYS, TARGET_COLUMN
from data_loader import get_top_directions, load_processed_salary_data
from features import build_feature_row_from_history
from train import load_model_artifact


def _prepare_direction_history(processed_df: pd.DataFrame, direction_name: str) -> pd.Series:
    if processed_df.empty:
        raise ValueError("processed данные пусты")
    if "direction" not in processed_df.columns:
        raise ValueError("в processed данных нет колонки direction")
    if DATE_COLUMN not in processed_df.columns or TARGET_COLUMN not in processed_df.columns:
        raise ValueError(f"в processed данных нет {DATE_COLUMN} или {TARGET_COLUMN}")

    data = processed_df[processed_df["direction"] == direction_name].copy()
    if data.empty:
        raise ValueError(f"нет данных для направления {direction_name}")

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).sort_values(DATE_COLUMN)
    data = data.drop_duplicates(subset=[DATE_COLUMN], keep="last")

    series = data.set_index(DATE_COLUMN)[TARGET_COLUMN].astype(float)
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()
    series.name = direction_name
    return series


def recursive_forecast(
    model,
    history_series: pd.Series,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    if history_series.empty:
        raise ValueError("история ряда пуста")
    if horizon < 1:
        raise ValueError("горизонт прогноза должен быть больше нуля")

    history = history_series.copy().astype(float)
    history.index = pd.to_datetime(history.index)
    history = history.sort_index()

    rows: list[dict] = []
    for _ in range(horizon):
        next_date = history.index.max() + pd.Timedelta(days=1)
        feature_row = build_feature_row_from_history(history, next_date)
        pred = float(model.predict(feature_row)[0])
        pred = max(0.0, pred)
        history.loc[next_date] = pred
        rows.append({"date": next_date, "prediction": pred})

    return pd.DataFrame(rows)


def forecast_direction(
    direction_name: str,
    horizon: int = FORECAST_HORIZON_DAYS,
    processed_df: pd.DataFrame | None = None,
) -> dict:
    data = processed_df.copy() if processed_df is not None else load_processed_salary_data()
    history_series = _prepare_direction_history(data, direction_name)

    artifact = load_model_artifact(direction_name)
    model = artifact.get("model")
    if model is None:
        raise ValueError(f"артефакт модели {direction_name} повреждён")

    forecast_df = recursive_forecast(model=model, history_series=history_series, horizon=horizon)
    rmse = float(artifact.get("rmse_for_range", 0.0))
    forecast_df["lower"] = (forecast_df["prediction"] - rmse).clip(lower=0.0)
    forecast_df["upper"] = forecast_df["prediction"] + rmse

    test_info = artifact.get("test", {})
    if test_info:
        test_df = pd.DataFrame(
            {
                "date": pd.to_datetime(test_info.get("dates", []), errors="coerce"),
                "actual": pd.to_numeric(test_info.get("actual", []), errors="coerce"),
                "predicted": pd.to_numeric(test_info.get("predicted", []), errors="coerce"),
            }
        ).dropna(subset=["date", "actual", "predicted"])
    else:
        test_df = pd.DataFrame(columns=["date", "actual", "predicted"])

    return {
        "direction_name": direction_name,
        "history_series": history_series,
        "forecast_df": forecast_df,
        "test_df": test_df,
        "artifact": artifact,
        "rmse_for_range": rmse,
    }


def forecast_top_directions(
    processed_df: pd.DataFrame,
    top_n: int = 5,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> dict[str, dict]:
    directions = get_top_directions(processed_df, top_n=top_n)
    results: dict[str, dict] = {}
    for direction in directions:
        results[direction] = forecast_direction(
            direction_name=direction,
            horizon=horizon,
            processed_df=processed_df,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="прогноз зарплаты по одному направлению")
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        help="название направления например data scientist",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=FORECAST_HORIZON_DAYS,
        help="горизонт прогноза в днях",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = forecast_direction(direction_name=args.direction, horizon=args.horizon, processed_df=None)
    forecast_df = result["forecast_df"].copy()
    forecast_df["date"] = forecast_df["date"].dt.strftime("%Y-%m-%d")
    for col in ("prediction", "lower", "upper"):
        forecast_df[col] = forecast_df[col].round(0)

    print("=" * 80)
    print(f"прогноз зарплаты для направления: {args.direction}")
    print(forecast_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
