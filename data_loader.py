from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from config import (
    COUNT_COLUMN,
    CURRENCY_TO_RUB,
    DATE_COLUMN,
    DIRECTION_QUERIES,
    HH_API_URL,
    HH_AREA,
    HH_PAGES_PER_QUERY,
    HH_PER_PAGE,
    MAX_SALARY_RUB,
    MIN_SALARY_RUB,
    PROCESSED_SALARY_PATH,
    RAW_VACANCIES_PATH,
    REQUEST_TIMEOUT_SECONDS,
    REQUEST_USER_AGENT,
    TARGET_COLUMN,
    TOP_DIRECTIONS_COUNT,
    ensure_project_dirs,
)


def _to_iso_start(value: date) -> str:
    return f"{value.isoformat()}T00:00:00"


def _to_iso_end(value: date) -> str:
    return f"{value.isoformat()}T23:59:59"


def _safe_datetime(value: str | None) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def _salary_to_rub(salary_obj: dict | None) -> float | None:
    if not salary_obj:
        return None

    salary_from = salary_obj.get("from")
    salary_to = salary_obj.get("to")
    currency = salary_obj.get("currency")
    gross = salary_obj.get("gross")

    if salary_from is None and salary_to is None:
        return None

    if salary_from is None:
        raw_salary = float(salary_to)
    elif salary_to is None:
        raw_salary = float(salary_from)
    else:
        raw_salary = (float(salary_from) + float(salary_to)) / 2.0

    rate = CURRENCY_TO_RUB.get(str(currency).upper())
    if rate is None:
        return None

    salary_rub = raw_salary * rate
    if gross is True:
        salary_rub *= 0.87

    return float(salary_rub)


def fetch_direction_vacancies(
    direction_name: str,
    query_text: str,
    start_date: date,
    end_date: date,
    pages_limit: int = HH_PAGES_PER_QUERY,
) -> pd.DataFrame:
    headers = {"User-Agent": REQUEST_USER_AGENT}
    all_rows: list[dict] = []

    for page in range(pages_limit):
        params = {
            "text": query_text,
            "search_field": "name",
            "area": HH_AREA,
            "per_page": HH_PER_PAGE,
            "page": page,
            "date_from": _to_iso_start(start_date),
            "date_to": _to_iso_end(end_date),
            "only_with_salary": False,
            "order_by": "publication_time",
        }
        try:
            response = requests.get(
                HH_API_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"таймаут hh api для направления {direction_name}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"ошибка hh api для направления {direction_name}: {exc}"
            ) from exc

        payload = response.json()
        items = payload.get("items", [])
        if not items:
            break

        for item in items:
            published_ts = _safe_datetime(item.get("published_at"))
            if pd.isna(published_ts):
                continue

            all_rows.append(
                {
                    "direction": direction_name,
                    "vacancy_id": item.get("id"),
                    "vacancy_name": item.get("name"),
                    "published_at": published_ts,
                    DATE_COLUMN: published_ts.normalize(),
                    TARGET_COLUMN: _salary_to_rub(item.get("salary")),
                    "currency": (item.get("salary") or {}).get("currency"),
                    "employer": (item.get("employer") or {}).get("name"),
                    "area_name": (item.get("area") or {}).get("name"),
                    "vacancy_url": item.get("alternate_url"),
                }
            )

        pages_total = int(payload.get("pages", 0))
        if page + 1 >= pages_total:
            break

    return pd.DataFrame(all_rows)


def save_raw_vacancies(df: pd.DataFrame) -> None:
    ensure_project_dirs()
    df_to_save = df.copy()
    if "published_at" in df_to_save.columns:
        df_to_save["published_at"] = pd.to_datetime(df_to_save["published_at"], errors="coerce")
    if DATE_COLUMN in df_to_save.columns:
        df_to_save[DATE_COLUMN] = pd.to_datetime(df_to_save[DATE_COLUMN], errors="coerce")
    df_to_save.to_csv(RAW_VACANCIES_PATH, index=False, date_format="%Y-%m-%d")


def load_raw_vacancies() -> pd.DataFrame:
    if not RAW_VACANCIES_PATH.exists():
        raise FileNotFoundError(f"сырой файл вакансий не найден: {RAW_VACANCIES_PATH}")

    df = pd.read_csv(RAW_VACANCIES_PATH)
    if df.empty:
        raise ValueError("файл сырых вакансий пуст")
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"в сырых данных нет колонки {DATE_COLUMN}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df.get(TARGET_COLUMN), errors="coerce")
    return df.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)


def download_all_vacancies(
    start_date: date,
    end_date: date,
    use_fallback: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    ensure_project_dirs()
    messages: list[str] = []
    all_frames: list[pd.DataFrame] = []
    fallback_df: pd.DataFrame | None = None

    if use_fallback and RAW_VACANCIES_PATH.exists():
        try:
            fallback_df = load_raw_vacancies()
        except Exception:
            fallback_df = None

    errors: list[str] = []
    for direction_name, query_text in DIRECTION_QUERIES.items():
        try:
            direction_df = fetch_direction_vacancies(
                direction_name=direction_name,
                query_text=query_text,
                start_date=start_date,
                end_date=end_date,
            )
            if direction_df.empty:
                raise RuntimeError("api вернул пустой набор")
            all_frames.append(direction_df)
            messages.append(f"{direction_name}: загружено {len(direction_df)} вакансий из api")
        except Exception as exc:
            if fallback_df is not None:
                local_part = fallback_df[fallback_df["direction"] == direction_name].copy()
                local_part = local_part[
                    (local_part[DATE_COLUMN] >= pd.to_datetime(start_date))
                    & (local_part[DATE_COLUMN] <= pd.to_datetime(end_date))
                ]
                if not local_part.empty:
                    all_frames.append(local_part)
                    messages.append(
                        f"{direction_name}: использован локальный fallback {len(local_part)} строк"
                    )
                else:
                    errors.append(f"{direction_name}: api ошибка и fallback пуст {exc}")
            else:
                errors.append(f"{direction_name}: {exc}")

    if not all_frames:
        details = "\n".join(f"- {line}" for line in errors) if errors else "причина не уточнена"
        raise RuntimeError(
            "не удалось получить данные ни по одному направлению\n"
            f"{details}\n"
            "проверьте интернет или наличие файла data/raw/it_salary_vacancies_raw.csv"
        )

    raw_df = pd.concat(all_frames, ignore_index=True)
    raw_df = raw_df.drop_duplicates(subset=["vacancy_id", "direction"], keep="last")
    raw_df[DATE_COLUMN] = pd.to_datetime(raw_df[DATE_COLUMN], errors="coerce")
    raw_df = raw_df.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
    save_raw_vacancies(raw_df)

    if errors:
        messages.append("часть направлений была загружена через fallback")
        messages.extend([f"- {line}" for line in errors])

    messages.append(f"сырые данные сохранены: {RAW_VACANCIES_PATH}")
    return raw_df, messages


def build_processed_salary_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {"direction", DATE_COLUMN, TARGET_COLUMN}
    if raw_df.empty:
        raise ValueError("сырые данные пустые")
    if not required.issubset(set(raw_df.columns)):
        missing = required.difference(set(raw_df.columns))
        raise ValueError(f"в сырых данных не хватает колонок: {', '.join(sorted(missing))}")

    data = raw_df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN])

    # считаем все вакансии даже если в них нет зарплаты
    counts_daily = (
        data.groupby(["direction", DATE_COLUMN], as_index=False)
        .size()
        .rename(columns={"size": COUNT_COLUMN})
    )

    salaries = data.dropna(subset=[TARGET_COLUMN]).copy()
    salaries = salaries[
        (salaries[TARGET_COLUMN] >= MIN_SALARY_RUB) & (salaries[TARGET_COLUMN] <= MAX_SALARY_RUB)
    ]
    salary_daily = (
        salaries.groupby(["direction", DATE_COLUMN], as_index=False)[TARGET_COLUMN]
        .median()
        .sort_values(["direction", DATE_COLUMN])
    )

    merged = counts_daily.merge(
        salary_daily,
        on=["direction", DATE_COLUMN],
        how="left",
    )

    parts: list[pd.DataFrame] = []
    for direction_name, group in merged.groupby("direction"):
        part = group.copy().sort_values(DATE_COLUMN).reset_index(drop=True)
        full_dates = pd.date_range(part[DATE_COLUMN].min(), part[DATE_COLUMN].max(), freq="D")
        frame = pd.DataFrame({DATE_COLUMN: full_dates})
        frame["direction"] = direction_name
        frame = frame.merge(part, on=["direction", DATE_COLUMN], how="left")
        frame[COUNT_COLUMN] = frame[COUNT_COLUMN].fillna(0).astype(int)

        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].interpolate(limit_direction="both")
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].ffill().bfill()

        if frame[TARGET_COLUMN].isna().all():
            frame[TARGET_COLUMN] = 120_000.0

        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].clip(lower=MIN_SALARY_RUB, upper=MAX_SALARY_RUB)
        parts.append(frame)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["direction", DATE_COLUMN]).reset_index(drop=True)
    return result


def save_processed_salary_data(df: pd.DataFrame) -> None:
    ensure_project_dirs()
    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data.to_csv(PROCESSED_SALARY_PATH, index=False, date_format="%Y-%m-%d")


def load_processed_salary_data() -> pd.DataFrame:
    if not PROCESSED_SALARY_PATH.exists():
        raise FileNotFoundError(
            f"файл обработанных данных не найден: {PROCESSED_SALARY_PATH}"
        )
    df = pd.read_csv(PROCESSED_SALARY_PATH)
    if df.empty:
        raise ValueError("файл обработанных данных пуст")
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"в processed данных нет колонки {DATE_COLUMN}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df.get(TARGET_COLUMN), errors="coerce")
    df[COUNT_COLUMN] = pd.to_numeric(df.get(COUNT_COLUMN), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).sort_values(["direction", DATE_COLUMN])
    return df.reset_index(drop=True)


def prepare_and_save_salary_dataset(
    start_date: date,
    end_date: date,
    use_fallback: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    raw_df, messages = download_all_vacancies(
        start_date=start_date,
        end_date=end_date,
        use_fallback=use_fallback,
    )
    processed_df = build_processed_salary_dataset(raw_df)
    save_processed_salary_data(processed_df)
    messages.append(f"обработанные данные сохранены: {PROCESSED_SALARY_PATH}")
    return processed_df, messages


def get_top_directions(
    processed_df: pd.DataFrame,
    top_n: int = TOP_DIRECTIONS_COUNT,
) -> list[str]:
    if processed_df.empty:
        raise ValueError("нельзя выбрать топ направлений processed данные пустые")
    if "direction" not in processed_df.columns or COUNT_COLUMN not in processed_df.columns:
        raise ValueError("в processed данных нет колонок direction vacancies_count")

    popularity = (
        processed_df.groupby("direction", as_index=False)[COUNT_COLUMN]
        .sum()
        .sort_values(COUNT_COLUMN, ascending=False)
    )
    return popularity.head(top_n)["direction"].tolist()
