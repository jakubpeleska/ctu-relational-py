import numpy as np
import pandas as pd
from relbench.base import Database, Table

from redelex.db import DBInspector, RelbenchDBInterface, RemoteDBInterface
from redelex.db.utils import get_db_connection, get_db_url

from .helpers import InMemoryDataset


def test_dbinterface_get_tables(sqlite_db_url):
    """Regression: get_tables() called the table_names property as a function."""
    with RemoteDBInterface(sqlite_db_url) as iface:
        tables = iface.get_tables()

    assert set(tables) == {"customer", "product", "orders", "events"}
    assert all(isinstance(df, pd.DataFrame) for df in tables.values())
    assert len(tables["customer"]) == 6


def test_remote_interface_lifecycle(sqlite_db_url):
    iface = RemoteDBInterface(sqlite_db_url)
    iface.close()  # close before connect is a no-op

    with iface:
        assert iface.connection is not None
        count = iface.sql("SELECT COUNT(*) AS n FROM orders")["n"].iloc[0]
        assert count == 6
    assert iface.connection is None
    iface.close()  # double close is a no-op


def test_inspector_metadata(sqlite_db_url):
    con = get_db_connection(sqlite_db_url)
    try:
        inspector = DBInspector(con)
        assert inspector.get_tables() == {"customer", "product", "orders", "events"}
        assert inspector.get_primary_key("product") == {"region", "code"}

        fks = inspector.get_foreign_keys("orders")
        constrained = {tuple(fk.src_columns) for fk in fks}
        assert constrained == {("cust_id",), ("region", "code")}

        columns = inspector.get_columns("customer")
        assert set(columns) == {"id", "name", "age", "signup_date"}
    finally:
        con.close()


def test_remote_interface_get_relbench_db(sqlite_db_url):
    with RemoteDBInterface(sqlite_db_url) as iface:
        db = iface.get_relbench_db(time_col_dict={"orders": "ordered_at"})

    orders = db.table_dict["orders"]
    assert orders.time_col == "ordered_at"
    assert orders.pkey_col == "__PK__"
    assert orders.fkey_col_to_pkey_table == {
        "FK_customer_cust_id": "customer",
        "FK_product_region_code": "product",
    }
    # Row alignment survives the non-unique events FK (shared reindex helper).
    assert len(db.table_dict["events"].df) == 4


def _mini_dataset() -> InMemoryDataset:
    a = pd.DataFrame({"a_id": np.arange(3), "x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"b_id": np.arange(4), "a_ref": [0, 1, 2, 0]})
    db = Database(
        table_dict={
            "a": Table(df=a, fkey_col_to_pkey_table={}, pkey_col="a_id", time_col=None),
            "b": Table(
                df=b, fkey_col_to_pkey_table={"a_ref": "a"}, pkey_col="b_id", time_col=None
            ),
        }
    )
    return InMemoryDataset(db)


def test_relbench_interface_roundtrip():
    iface = RelbenchDBInterface(_mini_dataset())
    with iface:
        assert set(iface.table_names) == {"a", "b"}
        assert iface.sql("SELECT COUNT(*) AS n FROM b")["n"].iloc[0] == 4
        assert iface.get_primary_key("b") == ["b_id"]

        # Regression: ref_columns used the *source* table's pkey.
        fks = iface.get_foreign_keys("b")
        assert len(fks) == 1
        assert fks[0].src_columns == ["a_ref"]
        assert fks[0].ref_table == "a"
        assert fks[0].ref_columns == ["a_id"]

        schema = iface.get_schema()
        assert set(schema.table_schemas) == {"a", "b"}
    assert iface.db is None


def test_relbench_interfaces_are_isolated():
    """Regression: registrations on duckdb's shared default connection made two
    interfaces with the same table names clobber each other."""
    iface1 = RelbenchDBInterface(_mini_dataset())
    iface2 = RelbenchDBInterface(_mini_dataset())
    iface1.connect()
    iface2.connect()

    iface2.close()
    # iface1 must still be able to query its own registrations.
    assert iface1.sql("SELECT COUNT(*) AS n FROM a")["n"].iloc[0] == 3
    iface1.close()


def test_get_db_url_format():
    url = get_db_url("mariadb", "pymysql", "user", "pw", "host", 3306, "db")
    assert url == "mariadb+pymysql://user:pw@host:3306/db"
