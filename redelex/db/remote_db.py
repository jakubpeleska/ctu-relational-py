import warnings
from functools import cached_property

import pandas as pd
import sqlalchemy as sa
from relbench.base import Database, Table
from tqdm import tqdm

from .foreign_key import ForeignKey
from .interface import DBInterface
from .schema import DBSchema, TableSchema
from .utils import (
    SQL_DATE_MAP,
    SQL_DATE_TYPES,
    get_db_connection,
    reindex_fk,
    resolve_column_dtype,
)


class RemoteDBInterface(DBInterface):
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.connection = None
        self.inspector = None
        self.remote_md = None

        super().__init__()

    @cached_property
    def table_names(self) -> list[str]:
        return self.inspector.get_table_names()

    def connect(self):
        self.connection = get_db_connection(self.connection_url)
        self.inspector = sa.inspect(self.connection.engine)
        self.remote_md = sa.MetaData()
        self.remote_md.reflect(bind=self.inspector.engine)

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection.engine.dispose()
            self.connection = None
            self.inspector = None
            self.remote_md = None

    def get_primary_key(self, table_name: str) -> list[str]:
        return self.inspector.get_pk_constraint(table_name).get("constrained_columns", [])

    def get_foreign_keys(self, table_name: str) -> list[ForeignKey]:
        return [
            ForeignKey(
                src_columns=fk["constrained_columns"],
                ref_table=fk["referred_table"],
                ref_columns=fk["referred_columns"],
            )
            for fk in self.inspector.get_foreign_keys(table_name)
        ]

    def get_schema(self) -> DBSchema:
        table_schemas = {}
        for tname in self.table_names:
            pk = self.get_primary_key(tname)
            fks = self.get_foreign_keys(tname)

            sql_table = sa.Table(tname, self.remote_md)

            type_dict: dict[str, str] = {}
            for c in sql_table.columns:
                try:
                    sql_type = type(c.type.as_generic())
                except NotImplementedError:
                    sql_type = type(c.type)

                type_dict[c.name] = sql_type.__name__

            table_schemas[tname] = TableSchema(
                name=tname, pk=pk, fks=fks, type_dict=type_dict
            )
        return DBSchema(table_schemas=table_schemas)

    def sql(self, query: str) -> "pd.DataFrame":
        return pd.read_sql(query, self.connection)

    def get_table(self, table_name: str) -> "pd.DataFrame":
        sql_table = sa.Table(table_name, self.remote_md)

        dtypes: dict[str, str] = {}
        sql_types_dict: dict[str, sa.types.TypeEngine] = {}

        for c in sql_table.columns:
            dtype, sql_type = resolve_column_dtype(c)

            if dtype is not None:
                dtypes[c.name] = dtype
                sql_types_dict[c.name] = sql_type
            else:
                warnings.warn(
                    f"Unknown data type {c.type} in {table_name}.{c.name}; "
                    "the column is skipped.",
                    stacklevel=2,
                )

        statement = sa.select(sql_table.columns)
        query = statement.compile(self.connection.engine)
        df = pd.read_sql_query(str(query), con=self.connection, dtype=dtypes)

        for col, sql_type in sql_types_dict.items():
            if sql_type in SQL_DATE_TYPES:
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                    warnings.warn(
                        f"Could not convert {table_name}.{col} to datetime: {e}",
                        stacklevel=2,
                    )

            if SQL_DATE_MAP.get(sql_type, None) is not None:
                try:
                    df[col] = df[col].astype(SQL_DATE_MAP[sql_type], errors="raise")
                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                    warnings.warn(
                        f"Could not convert {table_name}.{col} to "
                        f"{SQL_DATE_MAP[sql_type]}: {e}",
                        stacklevel=2,
                    )

        return df

    def get_relbench_db(self, time_col_dict: dict[str, str] = None) -> Database:
        if time_col_dict is None:
            time_col_dict = {}
        df_dict: dict[str, pd.DataFrame] = {}

        fk_dict: dict[str, list[ForeignKey]] = {
            tname: self.get_foreign_keys(tname) for tname in self.table_names
        }

        for tname in (pbar := tqdm(self.table_names)):
            pbar.set_postfix_str(tname)
            df = self.get_table(tname)
            df.index.name = "__PK__"
            # Re-index to remove composite keys.
            df.reset_index(inplace=True)

            df_dict[tname] = df

        table_dict: dict[str, Table] = {}

        for tname in self.table_names:
            fkey_col_to_pkey_table: dict[str, str] = {}

            for fk in fk_dict[tname]:
                # Re-index to remove composite keys.
                fk_col, fk_name = reindex_fk(
                    df_dict, tname, fk.src_columns, fk.ref_table, fk.ref_columns
                )

                fkey_col_to_pkey_table[fk_name] = fk.ref_table
                # All original columns are preserved.
                df_dict[tname][fk_name] = fk_col

            time_col = time_col_dict.get(tname)
            if time_col is not None:
                if time_col not in df_dict[tname].columns:
                    raise ValueError(
                        f"Configured time column '{time_col}' not found in table '{tname}'."
                    )
                try:
                    df_dict[tname][time_col] = pd.to_datetime(df_dict[tname][time_col])
                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                    warnings.warn(
                        f"Could not convert {tname}.{time_col} to datetime: {e}",
                        stacklevel=2,
                    )

            table_dict[tname] = Table(
                df=df_dict[tname],
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col="__PK__",
                time_col=time_col,
            )

        db = Database(table_dict)

        return db
