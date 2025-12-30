import pandas as pd
import numpy as np
from typing import List

def fill_median(df: pd.DataFrame, col_name: str) -> pd.DataFrame:

    median_val = df[col_name].median()
    df[col_name] = df[col_name].fillna(median_val)
    return df

def fill_mode(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    mode_val = df[col_name].mode()[0] if not df[col_name].mode().empty else None
    if mode_val is not None:
        df[col_name] = df[col_name].fillna(mode_val)
    return df

def check_name_type_suite(df: pd.DataFrame, pattern: str) -> tuple:
    matching_cols = [col for col in df.columns if pattern in col.upper()]
    num_cols = [col for col in matching_cols if pd.api.types.is_numeric_dtype(df[col])]
    str_cols = [col for col in matching_cols if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
    return num_cols, str_cols

