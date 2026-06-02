from __future__ import annotations

from datetime import date, datetime

import pandas as pd


FIRST_SEASON_MONTHS = {2, 3, 4, 5, 6, 7}


def season_phase_from_code(code: str) -> str:
    text = str(code)
    if "_" not in text:
        return text.replace("_", " ").strip().title()
    _, phase = text.split("_", 1)
    return f"{phase.replace('_', ' ').strip().title()} season"


def season_year_from_code(code: str) -> int:
    text = str(code)
    year_text = text.split("_", 1)[0]
    try:
        return int(year_text)
    except ValueError:
        return -1


def current_season_phase(today: date | datetime | None = None) -> str:
    current_day = today.date() if isinstance(today, datetime) else today or date.today()
    return "First season" if current_day.month in FIRST_SEASON_MONTHS else "Second season"


def season_phase_options(district_df: pd.DataFrame) -> list[str]:
    options = {season_phase_from_code(code) for code in district_df["season"].dropna().astype(str)}
    return sorted(options)


def latest_season_code_for_phase(district_df: pd.DataFrame, season_phase: str) -> str:
    matching_codes = [
        code
        for code in district_df["season"].dropna().astype(str).unique()
        if season_phase_from_code(code) == season_phase
    ]
    if not matching_codes:
        raise ValueError(f"No season code available for phase={season_phase}")
    return max(matching_codes, key=season_year_from_code)