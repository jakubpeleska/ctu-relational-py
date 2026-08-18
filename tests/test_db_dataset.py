import numpy as np
import pandas as pd
import pytest
from relbench.base import Database

from redelex.datasets import db_dataset as db_dataset_module

from .helpers import SqliteTestDataset


def test_make_db_basic_structure(sqlite_db_url):
    db = SqliteTestDataset(sqlite_db_url).make_db()

    assert set(db.table_dict) == {"customer", "product", "orders", "events"}
    for table in db.table_dict.values():
        assert table.pkey_col == "__PK__"
        np.testing.assert_array_equal(
            table.df["__PK__"].to_numpy(), np.arange(len(table.df))
        )

    customer = db.table_dict["customer"]
    assert pd.api.types.is_datetime64_any_dtype(customer.df["signup_date"])
    assert customer.df["age"].dtype == pd.Int32Dtype()

    orders = db.table_dict["orders"]
    assert orders.time_col == "ordered_at"
    assert pd.api.types.is_datetime64_any_dtype(orders.df["ordered_at"])
    assert orders.fkey_col_to_pkey_table == {
        "FK_customer_cust_id": "customer",
        "FK_product_region_code": "product",
    }


def test_composite_fk_reindexed(sqlite_db_url):
    db = SqliteTestDataset(sqlite_db_url).make_db()

    fk = db.table_dict["orders"].df["FK_product_region_code"]
    # product rows in insertion order: (EU,1)->0, (EU,2)->1, (US,1)->2
    expected = [0.0, 1.0, 2.0, 0.0, np.nan, 1.0]
    np.testing.assert_array_equal(fk.to_numpy(dtype=float), expected)


def test_dangling_fk_becomes_nan(sqlite_db_url):
    db = SqliteTestDataset(sqlite_db_url).make_db()

    fk = db.table_dict["orders"].df["FK_customer_cust_id"]
    assert pd.isna(fk.iloc[3])  # cust_id=99 has no matching customer
    assert fk.drop(index=3).notna().all()


def test_original_single_column_keys_dropped(sqlite_db_url):
    """Regression: original FK columns were never dropped because a SQLAlchemy
    Table object was compared against string table names."""
    db = SqliteTestDataset(sqlite_db_url).make_db()

    assert "cust_id" not in db.table_dict["orders"].df.columns
    assert "id" not in db.table_dict["customer"].df.columns
    assert "cust_name" not in db.table_dict["events"].df.columns
    # Compound keys are kept by default.
    assert {"region", "code"}.issubset(db.table_dict["orders"].df.columns)
    assert {"region", "code"}.issubset(db.table_dict["product"].df.columns)


def test_keep_original_keys_flags(sqlite_db_url):
    db = SqliteTestDataset(sqlite_db_url, keep_original_keys=True).make_db()
    assert "cust_id" in db.table_dict["orders"].df.columns
    assert "id" in db.table_dict["customer"].df.columns

    db = SqliteTestDataset(sqlite_db_url, keep_original_compound_keys=False).make_db()
    assert not {"region", "code"} & set(db.table_dict["orders"].df.columns)
    assert not {"region", "code"} & set(db.table_dict["product"].df.columns)


def test_reindex_fk_nonunique_ref_no_row_explosion(sqlite_db_url):
    """Regression: a FK onto a non-unique column multiplied rows in the merge
    and misaligned all following foreign keys."""
    with pytest.warns(UserWarning, match="not unique"):
        db = SqliteTestDataset(sqlite_db_url).make_db()

    events = db.table_dict["events"].df
    assert len(events) == 4

    fk = events["FK_customer_cust_name"]
    assert fk.iloc[0] == 0  # "Alice" -> first matching customer row
    assert fk.iloc[1] == 2  # "Bob"
    assert fk.iloc[2] == 5  # "Eve"
    assert pd.isna(fk.iloc[3])  # NULL name stays dangling


def test_customize_db_is_applied(sqlite_db_url):
    class DropEventsDataset(SqliteTestDataset):
        def customize_db(self, db: Database) -> Database:
            del db.table_dict["events"]
            return db

    db = DropEventsDataset(sqlite_db_url).make_db()
    assert "events" not in db.table_dict


def test_customize_db_may_drop_key_columns(sqlite_db_url):
    """customize_db dropping key columns itself must not break the automatic
    key removal afterwards."""

    class DropColumnDataset(SqliteTestDataset):
        def customize_db(self, db: Database) -> Database:
            db.table_dict["orders"].df.drop(columns=["cust_id"], inplace=True)
            return db

    db = DropColumnDataset(sqlite_db_url).make_db()
    assert "cust_id" not in db.table_dict["orders"].df.columns


def test_customize_db_errors_propagate(sqlite_db_url):
    """Regression: NotImplementedError raised inside a subclass's customize_db
    was silently swallowed."""

    class BoomDataset(SqliteTestDataset):
        def customize_db(self, db: Database) -> Database:
            raise NotImplementedError("some unimplemented internal helper")

    with pytest.raises(NotImplementedError, match="internal helper"):
        BoomDataset(sqlite_db_url).make_db()

    class ValueBoomDataset(SqliteTestDataset):
        def customize_db(self, db: Database) -> Database:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        ValueBoomDataset(sqlite_db_url).make_db()


def test_connection_closed(sqlite_db_url, monkeypatch):
    created = []
    real_get_connection = db_dataset_module.get_db_connection

    def spy(url):
        con = real_get_connection(url)
        created.append(con)
        return con

    monkeypatch.setattr(db_dataset_module, "get_db_connection", spy)

    SqliteTestDataset(sqlite_db_url).make_db()
    assert len(created) == 1 and created[0].closed

    # The connection must also be closed when the download fails mid-way.
    created.clear()

    def boom(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(db_dataset_module.pd, "read_sql_query", boom)
    with pytest.raises(RuntimeError, match="download failed"):
        SqliteTestDataset(sqlite_db_url).make_db()
    assert len(created) == 1 and created[0].closed
