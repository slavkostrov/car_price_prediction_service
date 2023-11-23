import re

import numpy as np
import pandas as pd


def check_float(x):
    try:
        float(x)
    except ValueError:
        return False
    else:
        return True


def cast_to_float(s: pd.Series) -> pd.Series:
    s = s.apply(
        lambda value: None
        if pd.isna(value) or not value.strip() or not check_float(value)
        else value
    )
    return s.astype(float)


def split_torque(s: pd.Series) -> tuple[list[float], list[float]]:
    """Разделяет колонку torque на две."""
    torque_nm = []
    torque_max_rpm = []
    for value in s.values:
        if pd.isna(value):
            torque_nm.append(value)
            torque_max_rpm.append(value)
            continue

        value = str(value).lower()
        value = value.replace(",", "")
        numbers = re.findall("[+]?(?:\d*\.*\d+)", value)

        max_rpm_value = None
        nm_value = None

        if len(numbers) == 1:
            if "nm" not in value:
                raise RuntimeError(f"Incorrect value: {value}")
            else:
                nm_value = numbers[0]
        elif len(numbers) == 2:
            nm_value = numbers[0]
            max_rpm_value = numbers[-1]
        elif len(numbers) == 3:
            nm_value = numbers[0]
            max_rpm_value = numbers[-1]
        else:
            value = ""

        if "rpm" in value:
            max_rpm_value = numbers[-1]

        # переводим kgm в nm
        if nm_value and "kgm" in value:
            nm_value = float(nm_value) * 9.81

        torque_nm.append(nm_value)
        torque_max_rpm.append(max_rpm_value)

    return torque_nm, torque_max_rpm


def fix_columns(
    df: pd.DataFrame, inplace: bool = False, save_base_torque: bool = False
) -> pd.DataFrame:
    """Исправляет значения в столбцах, убирая единицы измерения и приводя к float."""
    if not inplace:
        df = df.copy()

    df["engine"] = cast_to_float(df["engine"].str.lower().str.replace("cc", ""))
    df["max_power"] = cast_to_float(df["max_power"].str.lower().str.replace("bhp", ""))

    mileage_coef = (
        df["mileage"].str.lower().str.contains("km/kg").map({True: 1.33, False: 1.0})
    )
    df["mileage"] = cast_to_float(
        df["mileage"].str.lower().str.replace("kmpl", "").str.replace("km/kg", "")
    )
    df["mileage"] = df["mileage"] * mileage_coef

    if save_base_torque:
        df["torque_base"] = df["torque"]
    df["torque"], df["max_torque_rpm"] = split_torque(df["torque"])
    df["torque"] = df["torque"].astype(float)
    df["max_torque_rpm"] = df["max_torque_rpm"].astype(float)

    return df
