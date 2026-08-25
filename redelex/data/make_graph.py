import itertools
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from redelex.utils import to_unix_time


class MakeGraph:
    def __init__(
        self,
        db: Database,
        col_to_stype_dict: Dict[str, Dict[str, stype]],
        text_embedder_cfg: Optional[TextEmbedderConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        self.db = db
        self.col_to_stype_dict = col_to_stype_dict
        self.text_embedder_cfg = text_embedder_cfg
        self.cache_dir = cache_dir
        self.data: HeteroData = HeteroData()
        self.col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]] = dict()
        self.cannot_delete: Dict[str, bool] = dict()
        self.__dataInit()
        assert self.data is not None and self.col_stats_dict is not None

    def __dataInit(self):
        self.data = HeteroData()
        self.col_stats_dict = dict()
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __remove_pkey_fkey(self, col_to_stype: Dict[str, Any], table: Table):
        r"""Remove pkey, fkey columns since they will not be used as input feature."""
        if table.pkey_col is not None and table.pkey_col in col_to_stype:
            col_to_stype.pop(table.pkey_col)
        for fkey in table.fkey_col_to_pkey_table:
            if fkey in col_to_stype:
                col_to_stype.pop(fkey)

    def __make_dataset(self, table_name: str, table: Table) -> Dataset:
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = self.col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input feature.
        self.__remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None
            if self.cache_dir is None
            else os.path.join(self.cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=self.text_embedder_cfg,
        ).materialize(path=path)
        return dataset

    def __tensor_frame_to_edge_attr(self, tensor_frame: Any) -> Optional[Tensor]:
        feat_dict = getattr(tensor_frame, "feat_dict", None)
        if feat_dict is None:
            return None

        tensor_list = []
        for _, feat in sorted(feat_dict.items(), key=lambda item: str(item[0])):
            if isinstance(feat, Tensor):
                values = feat
            else:
                feat_values = getattr(feat, "values", None)
                values = feat_values if isinstance(feat_values, Tensor) else None

            if values is None:
                continue

            if values.dim() == 1:
                values = values.unsqueeze(-1)
            elif values.dim() > 2:
                values = values.reshape(values.size(0), -1)

            tensor_list.append(values.float())

        if len(tensor_list) == 0:
            return None

        # Ensure all tensors are on the same device before concatenation
        device = tensor_list[0].device
        tensor_list = [t.to(device) for t in tensor_list]
        edge_attr = torch.cat(tensor_list, dim=-1)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        return edge_attr

    def __create_edge(
        self, table_1, table_2, df, table_name: str, dataset: Dataset, with_edge_attr: bool
    ):
        fkey_name, pkey_table_name = table_1
        fkey_name_2, pkey_table_name_2 = table_2

        pkey_index = df[fkey_name]
        pkey_index_2 = df[fkey_name_2]
        # Filter out dangling fkeys
        mask = (~pkey_index.isna()) & (~pkey_index_2.isna())
        pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
        pkey_index_2 = torch.from_numpy(pkey_index_2[mask].astype(int).values)

        # Include the link table name in relation labels so edges from
        # different link tables never collide in HeteroData.
        relation_label = f"p2p_{table_name}_{fkey_name}_{fkey_name_2}"
        rev_relation_label = f"rev_p2p_{table_name}_{fkey_name}_{fkey_name_2}"

        # fkey1 -> fkey2
        edge_index_1 = torch.stack([pkey_index, pkey_index_2], dim=0)
        edge_type_1 = (pkey_table_name, relation_label, pkey_table_name_2)
        self.data[edge_type_1].edge_index = edge_index_1

        # fkey2 -> fkey1
        edge_index_2 = torch.stack([pkey_index_2, pkey_index], dim=0)
        edge_type_2 = (pkey_table_name_2, rev_relation_label, pkey_table_name)
        self.data[edge_type_2].edge_index = edge_index_2

        # Adds edge attributes instead of throwing data away
        if with_edge_attr:
            tf = dataset.tensor_frame[torch.from_numpy(mask.values)]
            edge_attr = self.__tensor_frame_to_edge_attr(tf)
            if edge_attr is not None:
                assert edge_attr.size(0) == edge_index_1.size(1)
                self.data[edge_type_1].edge_attr = edge_attr
                self.data[edge_type_2].edge_attr = edge_attr

    def pkey_fkey_structure_to_graph(
        self, df, table_name: str, table: Table, dataset: Dataset
    ):  # Default method to create graph with pkey-fkey edges
        # Add table node features:
        self.data[table_name].tf = dataset.tensor_frame
        self.col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            self.data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            if pkey_table_name not in self.db.table_dict:
                print(
                    f"[WARN] Skipping edge {table_name}.{fkey_name} -> {pkey_table_name}: "
                    "target table not found in db.table_dict"
                )
                continue

            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(self.db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            self.data[edge_type].edge_index = sort_edge_index(edge_index)  # type: ignore

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            self.data[edge_type].edge_index = sort_edge_index(edge_index)

    def hub_structure_to_graph(
        self,
        df,
        table_name: str,
        table: Table,
        dataset: Dataset,
        hubStrategy: str = "default",
    ):
        match hubStrategy:
            case "keep_attributes":
                with_edge_attr, keep_table = True, False
            case "keep_table":
                with_edge_attr, keep_table = False, True
            case _:
                with_edge_attr, keep_table = False, False

        if self.cannot_delete[table_name]:
            keep_table, with_edge_attr = True, False

        table_pairs = list(
            itertools.combinations(list(table.fkey_col_to_pkey_table.items()), 2)
        )

        col_stats = dataset.col_stats
        self.col_stats_dict[table_name] = col_stats

        for table_1, table_2 in table_pairs:
            self.__create_edge(table_1, table_2, df, table_name, dataset, with_edge_attr)

        if keep_table:
            self.pkey_fkey_structure_to_graph(df, table_name, table, dataset)

    def bridge_structure_to_graph(
        self,
        df,
        table_name: str,
        table: Table,
        dataset: Dataset,
        bridgeStrategy: str = "default",
    ):
        match bridgeStrategy:
            case "keep_attributes":
                with_edge_attr, keep_table = True, False
            case "keep_table":
                with_edge_attr, keep_table = False, True
            case _:
                with_edge_attr, keep_table = False, False

        if self.cannot_delete[table_name]:
            keep_table, with_edge_attr = True, False

        col_stats = dataset.col_stats
        self.col_stats_dict[table_name] = col_stats

        table_1, table_2 = table.fkey_col_to_pkey_table.items()
        self.__create_edge(table_1, table_2, df, table_name, dataset, with_edge_attr)

        if keep_table:
            self.pkey_fkey_structure_to_graph(df, table_name, table, dataset)

    def main(
        self,
        process_bridge: bool = False,
        bridgeStrategy: str = "default",
        process_hub: bool = False,
        hubStrategy: str = "default_combinations",
    ) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
        r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
        foreign key relationships, together with the column stats of each table.

        Args:
            db: A database object containing a set of tables.
            col_to_stype_dict: Column to stype for
                each table.
            text_embedder_cfg: Text embedder config.
            cache_dir: A directory for storing materialized tensor
                frames. If specified, we will either cache the file or use the
                cached file. If not specified, we will not use cached file and
                re-process everything from scratch without saving the cache.

        Returns:
            HeteroData: The heterogeneous :class:`PyG` object with
                :class:`TensorFrame` feature.
        """
        self.cannot_delete = dict()
        for table_name in self.db.table_dict:
            self.cannot_delete[table_name] = False

        for _, table in self.db.table_dict.items():
            for t in table.fkey_col_to_pkey_table.values():
                self.cannot_delete[t] = True

        for table_name, table in self.db.table_dict.items():
            df = table.df
            dataset = self.__make_dataset(table_name, table)

            if process_hub and len(table.fkey_col_to_pkey_table.items()) >= 3:
                self.hub_structure_to_graph(df, table_name, table, dataset, hubStrategy)

            elif process_bridge and len(table.fkey_col_to_pkey_table.items()) == 2:
                self.bridge_structure_to_graph(
                    df, table_name, table, dataset, bridgeStrategy
                )

            else:
                self.pkey_fkey_structure_to_graph(df, table_name, table, dataset)

        self.data.validate()

        return self.data, self.col_stats_dict


def make_pkey_fkey_graph_custom(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    process_bridge: bool = False,
    bridgeStrategy: str = "default",
    process_hub: bool = False,
    hubStrategy: str = "default_combinations",
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.
    Uses the custom MakeGraph class with configurable bridge and hub processing.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for each table.
        text_embedder_cfg: Text embedder config.
        cache_dir: A directory for storing materialized tensor frames.
        process_bridge: Whether to process bridge tables (tables with exactly 2 foreign keys).
        bridgeStrategy: Strategy for processing bridge tables ("default", "keep_attributes", "keep_table").
        process_hub: Whether to process hub tables (tables with 3+ foreign keys).
        hubStrategy: Strategy for processing hub tables ("default_combinations", "keep_attributes", "keep_table").

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with :class:`TensorFrame` feature.
    """
    graph_maker = MakeGraph(
        db=db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=cache_dir,
    )

    return graph_maker.main(
        process_bridge=process_bridge,
        bridgeStrategy=bridgeStrategy,
        process_hub=process_hub,
        hubStrategy=hubStrategy,
    )
