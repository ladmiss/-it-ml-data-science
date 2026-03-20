from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_VACANCIES_PATH = RAW_DATA_DIR / "it_salary_vacancies_raw.csv"
PROCESSED_SALARY_PATH = PROCESSED_DATA_DIR / "it_salary_processed_daily.csv"

# направления для автофильтра
DIRECTION_QUERIES = {
    "Data Scientist": "data scientist OR дата сайентист",
    "Machine Learning Engineer": "machine learning engineer OR ml engineer",
    "Data Engineer": "data engineer",
    "Data Analyst": "data analyst OR аналитик данных",
    "MLOps Engineer": "mlops OR MLOps engineer",
    "BI Analyst": "bi analyst OR business intelligence",
    "NLP Engineer": "nlp engineer OR natural language processing",
    "Computer Vision Engineer": "computer vision engineer",
}

TOP_DIRECTIONS_COUNT = 5
FORECAST_HORIZON_DAYS = 30
DEFAULT_HISTORY_DAYS = 120

HH_API_URL = "https://api.hh.ru/vacancies"
HH_AREA = 113
HH_PER_PAGE = 100
HH_PAGES_PER_QUERY = 5
REQUEST_TIMEOUT_SECONDS = 25
REQUEST_USER_AGENT = "ITSalaryForecastStudentProject/1.0"

DATE_COLUMN = "date"
TARGET_COLUMN = "salary_rub"
COUNT_COLUMN = "vacancies_count"

CURRENCY_TO_RUB = {
    "RUR": 1.0,
    "RUB": 1.0,
    "KZT": 0.20,
    "BYN": 28.0,
    "USD": 95.0,
    "EUR": 103.0,
}

MIN_SALARY_RUB = 35_000
MAX_SALARY_RUB = 1_200_000

TEST_RATIO = 0.2
MIN_TRAIN_SAMPLES = 8
MIN_TEST_SAMPLES = 3

RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def default_date_range(history_days: int = DEFAULT_HISTORY_DAYS) -> tuple[date, date]:
    safe_days = max(30, int(history_days))
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=safe_days - 1)
    return start_date, end_date


def direction_code(direction_name: str) -> str:
    normalized = direction_name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_") or "direction"


def model_artifact_path(direction_name: str) -> Path:
    return MODELS_DIR / f"{direction_code(direction_name)}_salary_model.joblib"
