from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}


def choose_best_model(metrics_by_model: dict[str, dict[str, float]]) -> str:
    if not metrics_by_model:
        raise ValueError("словарь метрик пуст")
    return min(
        metrics_by_model,
        key=lambda name: (metrics_by_model[name]["RMSE"], metrics_by_model[name]["MAE"]),
    )


def build_metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "модель": model_name,
                "MAE": round(metrics["MAE"], 2),
                "RMSE": round(metrics["RMSE"], 2),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def _safe_plotly_import():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "для графиков нужен plotly установите зависимости pip install -r requirements.txt"
        ) from exc
    return go


def create_direction_forecast_figure(
    direction_name: str,
    history_series: pd.Series,
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
):
    go = _safe_plotly_import()

    history = history_series.copy().astype(float).sort_index()
    history_tail = history.tail(120)

    forecast = forecast_df.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_tail.index,
            y=history_tail.values,
            mode="lines",
            name="история",
            line={"color": "#1f77b4", "width": 2},
        )
    )

    if test_df is not None and not test_df.empty:
        test = test_df.copy()
        test["date"] = pd.to_datetime(test["date"])
        fig.add_trace(
            go.Scatter(
                x=test["date"],
                y=test["actual"],
                mode="lines",
                name="факт test",
                line={"color": "#2ca02c", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test["date"],
                y=test["predicted"],
                mode="lines",
                name="прогноз модели test",
                line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
            )
        )

    # полоса неопределенности по rmse
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.15)",
            name="диапазон прогноза +- rmse",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["prediction"],
            mode="lines+markers",
            name="прогноз",
            line={"color": "#d62728", "width": 3},
            marker={"size": 7},
        )
    )

    fig.update_layout(
        title=f"{direction_name} история и прогноз зарплаты",
        xaxis_title="дата",
        yaxis_title="оценка зарплаты руб",
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def create_popularity_figure(popularity_df: pd.DataFrame):
    go = _safe_plotly_import()
    if popularity_df.empty:
        raise ValueError("пустая таблица популярности")

    data = popularity_df.copy().sort_values("vacancies_total", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["vacancies_total"],
            y=data["direction"],
            orientation="h",
            marker={"color": "#1f77b4"},
            text=data["vacancies_total"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="топ 5 популярных направлений по числу вакансий",
        xaxis_title="количество вакансий",
        yaxis_title="направление",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def _trend_label(delta_pct: float) -> str:
    if delta_pct > 10:
        return "сильный рост"
    if delta_pct > 3:
        return "умеренный рост"
    if delta_pct < -10:
        return "сильное снижение"
    if delta_pct < -3:
        return "умеренное снижение"
    return "стабильный уровень"


def generate_direction_recommendation(direction_name: str, delta_pct: float, avg_salary: float) -> str:
    trend = _trend_label(delta_pct)
    rounded = int(round(avg_salary, 0))
    if delta_pct > 3:
        return (
            f"по направлению {direction_name} ожидается {trend} "
            f"ориентир по зарплате около {rounded:,} руб в месяц "
            "рекомендация усилить профильные навыки и активнее откликаться"
        ).replace(",", " ")
    if delta_pct < -3:
        return (
            f"по направлению {direction_name} прогнозируется {trend} "
            f"ориентир около {rounded:,} руб в месяц "
            "рекомендация расширить стек и смотреть смежные роли"
        ).replace(",", " ")
    return (
        f"по направлению {direction_name} ожидается {trend} "
        f"ориентир около {rounded:,} руб в месяц "
        "рекомендация сфокусироваться на резюме портфолио и точечных откликах"
    ).replace(",", " ")


def build_detailed_report(
    all_results: dict[str, dict],
    popularity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if not all_results:
        raise ValueError("нет результатов для отчета")

    rows = []
    text_lines: list[str] = []

    popularity_map = {
        row["direction"]: int(row["vacancies_total"])
        for row in popularity_df.to_dict(orient="records")
    }

    for direction_name, result in all_results.items():
        history = result["history_series"]
        forecast = result["forecast_df"].copy()
        rmse = float(result.get("rmse_for_range", 0.0))

        recent_mean = float(history.iloc[-14:].mean())
        future_mean = float(forecast["prediction"].mean())
        delta_pct = (future_mean - recent_mean) / max(recent_mean, 1.0) * 100

        plus_minus = max(rmse, future_mean * 0.03)
        salary_low = max(0.0, future_mean - plus_minus)
        salary_high = future_mean + plus_minus

        rows.append(
            {
                "направление": direction_name,
                "вакансий в периоде": popularity_map.get(direction_name, 0),
                "текущая средняя 14д руб": round(recent_mean),
                "прогноз средний руб": round(future_mean),
                "диапазон прогноза руб": f"{round(salary_low):,} - {round(salary_high):,}".replace(",", " "),
                "изменение %": round(delta_pct, 1),
                "тренд": _trend_label(delta_pct),
            }
        )

        text_lines.append(generate_direction_recommendation(direction_name, delta_pct, future_mean))

    report_df = pd.DataFrame(rows).sort_values("вакансий в периоде", ascending=False).reset_index(drop=True)
    return report_df, text_lines
