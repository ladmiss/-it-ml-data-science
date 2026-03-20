from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_HISTORY_DAYS,
    FORECAST_HORIZON_DAYS,
    TOP_DIRECTIONS_COUNT,
    default_date_range,
    ensure_project_dirs,
)
from data_loader import (
    COUNT_COLUMN,
    get_top_directions,
    load_processed_salary_data,
    prepare_and_save_salary_dataset,
)
from predict import forecast_direction
from train import train_direction_model
from utils import (
    build_detailed_report,
    create_direction_forecast_figure,
    create_popularity_figure,
)


st.set_page_config(page_title="прогноз зарплат в it ml ds", layout="wide")
ensure_project_dirs()


def init_state() -> None:
    defaults = {
        "auto_result": None,
        "last_error": None,
        "last_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_full_auto_pipeline(history_days: int, horizon_days: int) -> dict:
    start_date, end_date = default_date_range(history_days)
    messages: list[str] = []

    try:
        processed_df, load_messages = prepare_and_save_salary_dataset(
            start_date=start_date,
            end_date=end_date,
            use_fallback=True,
        )
        messages.extend(load_messages)
    except Exception as exc:
        messages.append(f"api недоступен беру локальные данные: {exc}")
        processed_df = load_processed_salary_data()

    top_directions = get_top_directions(processed_df, top_n=TOP_DIRECTIONS_COUNT)
    if not top_directions:
        raise RuntimeError("не удалось определить топ популярных направлений")

    popularity_df = (
        processed_df.groupby("direction", as_index=False)[COUNT_COLUMN]
        .sum()
        .rename(columns={COUNT_COLUMN: "vacancies_total"})
        .sort_values("vacancies_total", ascending=False)
        .head(TOP_DIRECTIONS_COUNT)
        .reset_index(drop=True)
    )

    all_results: dict[str, dict] = {}
    for direction in top_directions:
        try:
            result = forecast_direction(
                direction_name=direction,
                horizon=horizon_days,
                processed_df=processed_df,
            )
        except FileNotFoundError:
            train_direction_model(processed_df=processed_df, direction_name=direction, save_artifact=True)
            result = forecast_direction(
                direction_name=direction,
                horizon=horizon_days,
                processed_df=processed_df,
            )
        all_results[direction] = result

    report_df, recommendation_lines = build_detailed_report(
        all_results=all_results,
        popularity_df=popularity_df,
    )

    return {
        "processed_df": processed_df,
        "top_directions": top_directions,
        "all_results": all_results,
        "report_df": report_df,
        "recommendation_lines": recommendation_lines,
        "popularity_df": popularity_df,
        "messages": messages,
        "start_date": start_date,
        "end_date": end_date,
        "created_at": datetime.now(),
    }


init_state()

st.title("автопрогноз зарплат в it")
st.subheader("ml и data science система сама собирает данные и делает отчет")

with st.expander("что здесь происходит", expanded=True):
    st.write(
        """
        я сделал полностью автоматический режим
        ничего нажимать по шагам не надо
        приложение само берет вакансии
        само отбирает топ 5 направлений
        само учит модели
        и сразу выдает подробный отчет и рекомендации
        """
    )

with st.sidebar:
    st.markdown("### настройки")
    history_days = st.slider(
        "глубина истории дней",
        min_value=60,
        max_value=365,
        value=DEFAULT_HISTORY_DAYS,
        step=10,
    )
    horizon_days = st.slider(
        "горизонт прогноза дней",
        min_value=7,
        max_value=45,
        value=FORECAST_HORIZON_DAYS,
        step=1,
    )
    force_refresh = st.button("обновить автоотчет", use_container_width=True)


need_refresh = force_refresh or st.session_state["auto_result"] is None
if need_refresh:
    with st.spinner("делаю автоанализ подождите немного"):
        try:
            st.session_state["auto_result"] = run_full_auto_pipeline(
                history_days=history_days,
                horizon_days=horizon_days,
            )
            st.session_state["last_error"] = None
            st.session_state["last_messages"] = st.session_state["auto_result"]["messages"]
        except Exception as exc:
            st.session_state["last_error"] = str(exc)


if st.session_state["last_error"]:
    st.error(f"ошибка автоанализа: {st.session_state['last_error']}")
    st.stop()

result = st.session_state["auto_result"]
if result is None:
    st.warning("отчет пока не сформирован")
    st.stop()

period_text = f"{result['start_date']} — {result['end_date']}"
generated_at = result["created_at"].strftime("%d.%m.%Y %H:%M:%S")
st.caption(f"период анализа: {period_text}   отчет обновлен: {generated_at}")

if st.session_state["last_messages"]:
    with st.expander("журнал загрузки"):
        for line in st.session_state["last_messages"]:
            st.write(f"- {line}")

st.markdown("## подробный отчет")
st.dataframe(result["report_df"], use_container_width=True)

st.markdown("## топ 5 популярных направлений")
try:
    popularity_fig = create_popularity_figure(result["popularity_df"])
    st.plotly_chart(popularity_fig, use_container_width=True)
except Exception as exc:
    st.error(f"не удалось построить график популярности: {exc}")

st.markdown("## рекомендации")
for line in result["recommendation_lines"]:
    st.write(f"- {line}")

st.markdown("## графики по каждому направлению")
for direction in result["top_directions"]:
    direction_result = result["all_results"][direction]
    st.markdown(f"### {direction}")

    forecast_table = direction_result["forecast_df"].copy()
    forecast_table["date"] = pd.to_datetime(forecast_table["date"]).dt.strftime("%d.%m.%Y")
    for col in ("prediction", "lower", "upper"):
        forecast_table[col] = forecast_table[col].round(0).astype(int)
    forecast_table = forecast_table.rename(
        columns={
            "date": "дата",
            "prediction": "прогноз руб",
            "lower": "нижняя граница руб",
            "upper": "верхняя граница руб",
        }
    )

    try:
        fig = create_direction_forecast_figure(
            direction_name=direction,
            history_series=direction_result["history_series"],
            test_df=direction_result.get("test_df"),
            forecast_df=direction_result["forecast_df"],
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"ошибка графика для {direction}: {exc}")

    st.dataframe(
        forecast_table[
            ["дата", "прогноз руб", "нижняя граница руб", "верхняя граница руб"]
        ],
        use_container_width=True,
    )

st.markdown("---")
st.caption("источник вакансий hh api")
