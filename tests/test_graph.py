import numpy as np
import pandas as pd
import pytest
import torch
from relbench.base import Database, Table
from torch_frame import stype

from redelex.data import make_pkey_fkey_graph
from redelex.data.graph import remove_pkey_fkey

from .helpers import DANGLING_VISIT_ROW, N_USERS, N_VISITS


def test_make_pkey_fkey_graph_structure(hetero_graph):
    data, col_stats_dict = hetero_graph

    assert set(data.node_types) == {"users", "visits"}
    assert data["users"].tf.num_rows == N_USERS
    assert data["visits"].tf.num_rows == N_VISITS
    assert set(col_stats_dict) == {"users", "visits"}

    assert ("visits", "f2p_FK_users", "users") in data.edge_types
    assert ("users", "rev_f2p_FK_users", "visits") in data.edge_types

    data.validate()


def test_graph_time_attribute(hetero_graph):
    data, _ = hetero_graph
    assert data["users"].time.shape == (N_USERS,)
    assert data["visits"].time.shape == (N_VISITS,)
    assert data["users"].time.dtype == torch.int64


def test_graph_dangling_fk_edges_filtered(hetero_graph):
    data, _ = hetero_graph
    edge_index = data["visits", "f2p_FK_users", "users"].edge_index

    assert edge_index.size(1) == N_VISITS - 1  # one dangling FK
    assert (edge_index[1] < N_USERS).all()
    assert DANGLING_VISIT_ROW not in edge_index[0].tolist()


def test_graph_key_only_table_gets_const_feature(text_embedder):
    users = pd.DataFrame({"__PK__": np.arange(3), "score": [0.1, 0.2, 0.3]})
    links = pd.DataFrame({"__PK__": np.arange(4), "FK_users": [0, 1, 2, 0]})
    db = Database(
        table_dict={
            "users": Table(
                df=users, fkey_col_to_pkey_table={}, pkey_col="__PK__", time_col=None
            ),
            "links": Table(
                df=links,
                fkey_col_to_pkey_table={"FK_users": "users"},
                pkey_col="__PK__",
                time_col=None,
            ),
        }
    )
    col_to_stype = {"users": {"score": stype.numerical}, "links": {}}

    data, _ = make_pkey_fkey_graph(db, col_to_stype, text_embedder=text_embedder)

    assert data["links"].tf.num_rows == 4
    assert data["links"].tf.col_names_dict[stype.numerical] == ["__const__"]
    assert data["links", "f2p_FK_users", "users"].edge_index.size(1) == 4


def test_remove_pkey_fkey():
    df = pd.DataFrame({"pk": [0], "fk": [0], "feat": [1.0]})
    table = Table(
        df=df, fkey_col_to_pkey_table={"fk": "other"}, pkey_col="pk", time_col=None
    )
    col_to_stype = {"pk": None, "fk": None, "feat": stype.numerical}

    remove_pkey_fkey(col_to_stype, table)
    assert col_to_stype == {"feat": stype.numerical}


def test_make_pkey_fkey_graph_rejects_non_consecutive_pkey(text_embedder):
    users = pd.DataFrame({"__PK__": [0, 2, 3], "score": [0.1, 0.2, 0.3]})
    db = Database(
        table_dict={
            "users": Table(
                df=users, fkey_col_to_pkey_table={}, pkey_col="__PK__", time_col=None
            )
        }
    )
    with pytest.raises(ValueError, match="consecutive"):
        make_pkey_fkey_graph(
            db, {"users": {"score": stype.numerical}}, text_embedder=text_embedder
        )
