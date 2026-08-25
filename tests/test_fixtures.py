"""Smoke tests for the shared fixtures themselves."""

import pandas as pd
import sqlalchemy as sa
from torch_frame import stype

from .helpers import DANGLING_VISIT_ROW, N_USERS, N_VISITS, TEST_TIMESTAMP


def test_sqlite_fixture_schema(sqlite_db_url):
    engine = sa.create_engine(sqlite_db_url)
    inspector = sa.inspect(engine)

    assert set(inspector.get_table_names()) == {"customer", "product", "orders", "events"}

    fks = inspector.get_foreign_keys("orders")
    constrained = {tuple(fk["constrained_columns"]) for fk in fks}
    assert ("cust_id",) in constrained
    assert ("region", "code") in constrained

    pk = inspector.get_pk_constraint("product")
    assert pk["constrained_columns"] == ["region", "code"]

    with engine.connect() as conn:
        names = conn.execute(sa.text("SELECT name FROM customer")).scalars().all()
    assert names.count("Alice") == 2
    engine.dispose()


def test_synthetic_dataset_builds(synthetic_dataset):
    db = synthetic_dataset.get_db(upto_test_timestamp=False)

    users = db.table_dict["users"]
    visits = db.table_dict["visits"]
    assert len(users.df) == N_USERS
    assert len(visits.df) == N_VISITS

    # The dangling foreign key was mapped to a missing value by relbench.
    assert pd.isna(visits.df["FK_users"].iloc[DANGLING_VISIT_ROW])
    assert visits.df["FK_users"].dropna().between(0, N_USERS - 1).all()

    assert db.min_timestamp == pd.Timestamp("2019-01-01")

    truncated = synthetic_dataset.get_db()
    assert truncated.table_dict["users"].df["reg_date"].max() <= TEST_TIMESTAMP


def test_all_stype_tensor_frame(all_stype_tf):
    assert all_stype_tf.num_rows == 20
    assert stype.numerical in all_stype_tf.stypes
    assert stype.categorical in all_stype_tf.stypes
    assert stype.multicategorical in all_stype_tf.stypes
    assert stype.timestamp in all_stype_tf.stypes


def test_fake_text_embedder(text_embedder):
    emb = text_embedder(["hello", "world"])
    assert emb.shape == (2, text_embedder.embedding_dim)
