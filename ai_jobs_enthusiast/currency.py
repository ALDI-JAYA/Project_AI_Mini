import pandas as pd

from .config import DEFAULT_USD_TO_IDR_RATE


def add_salary_idr(
    df: pd.DataFrame,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    """Add an Indonesian Rupiah salary column while keeping the original USD data."""
    result = df.copy()
    result["Average_Salary_IDR"] = (
        result["Average_Salary"] * exchange_rate
    ).round(0).astype("Int64")
    return result


def format_rupiah(value: float | int) -> str:
    return f"Rp{value:,.0f}".replace(",", ".")
