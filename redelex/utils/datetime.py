import warnings

import numpy as np
import pandas as pd
from relbench.base import Database

TIMESTAMP_MIN = np.datetime64(pd.Timestamp.min.ceil("D"))
"""The earliest whole-day timestamp representable as ``datetime64[ns]``."""

TIMESTAMP_MAX = np.datetime64(pd.Timestamp.max.floor("D"))
"""The latest whole-day timestamp representable as ``datetime64[ns]``."""

_UNIX_TIME_DIVISORS = {
    np.dtype("datetime64[s]"): 1,
    np.dtype("datetime64[ms]"): 10**3,
    np.dtype("datetime64[us]"): 10**6,
    np.dtype("datetime64[ns]"): 10**9,
}

NAT_UNIX_TIME = np.iinfo(np.int64).min // 10**9
"""UNIX time (in seconds) used for missing timestamps (``NaT``).

The minimum representable value is used so that in temporal neighbor sampling
rows without a timestamp count as the earliest possible time, i.e. they are
always available.
"""


def to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp (in seconds).

    Missing values (``NaT``) are mapped to :data:`NAT_UNIX_TIME` (the earliest
    representable time) and a warning is emitted.
    """
    divisor = _UNIX_TIME_DIVISORS.get(ser.dtype)
    if divisor is None:
        raise ValueError(f"Expected a datetime64 series, got dtype {ser.dtype}.")

    unix_time = ser.astype("int64").to_numpy() // divisor

    na_mask = ser.isna().to_numpy()
    if na_mask.any():
        warnings.warn(
            f"Series '{ser.name}' contains {int(na_mask.sum())} missing timestamps; "
            "mapping them to the earliest representable time.",
            stacklevel=2,
        )
        unix_time[na_mask] = NAT_UNIX_TIME

    return unix_time


def convert_timedelta(db: Database):
    """Converts timedelta columns to datetime columns (in place), anchored at
    1900-01-01."""

    for table in db.table_dict.values():
        timedeltas = table.df.select_dtypes(include=["timedelta"])
        if not timedeltas.empty:
            timedeltas = pd.Timestamp("1900-01-01") + timedeltas
            table.df[timedeltas.columns] = timedeltas


__all__ = [
    "to_unix_time",
    "convert_timedelta",
    "TIMESTAMP_MIN",
    "TIMESTAMP_MAX",
    "NAT_UNIX_TIME",
]
