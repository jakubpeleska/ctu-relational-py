from typing import Dict, List, Optional

from tqdm.std import tqdm

import pandas as pd

from sqlalchemy import (
    types as sqltypes,
    Connection,
    create_engine,
    MetaData,
    Table as SQLTable,
    select,
)

from relbench.base import Dataset, Database, Table

from ctu_relational.db import DBInspector, ForeignKeyDef

__ALL__ = ["DBDataset"]


class DBDataset(Dataset):
    """
    A dataset that is created from a remote relational database.

    Attributes:
        remote_url (str): The URL for connecting to the remote database.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        remote_url: Optional[str] = None,
        dialect: Optional[str] = None,
        driver: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        time_col_dict: Optional[Dict[str, str]] = None,
        keep_original_keys: bool = False,
    ):
        """Create a database dataset object.

        Args:
            cache_dir (str, optional): The directory to cache the dataset. Defaults to None.
            remote_url (str, optional): The URL for connecting to the remote database in SQLAlchemy format. \
                For more information, see https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls. \
                If not defined, all following parameters have to be specified. Defaults to None.
            dialect (str, optional): The dialect for the database connection. Defaults to None.
            driver (str, optional): The driver for the database connection. Defaults to None.
            user (str, optional): The username for the database connection. Defaults to None.
            password (str, optional): The password for the database connection. Defaults to None.
            host (str, optional): The host address of the remote database. Defaults to None.
            port (str, optional): The port number for the database connection. Defaults to None.
            database (str, optional): The name of the database. Defaults to None.
            time_col_dict (Dict[str, str], optional): A dictionary mapping table names to time columns. Defaults to None.
            keep_original_keys (bool, optional): Whether to keep original primary and foreign keys \
                after duplication during re-indexing. This is useful when the keys contain information \
                beyond just their relationship to other rows. Defaults to False.
        """

        self.remote_url = (
            remote_url
            if remote_url is not None
            else self.get_url(dialect, driver, user, password, host, port, database)
        )

        self.time_col_dict = time_col_dict if time_col_dict is not None else {}

        self.keep_original_keys = keep_original_keys

        super().__init__(cache_dir)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(remote_url={self.remote_url})"

    @classmethod
    def get_url(
        cls,
        dialect: str,
        driver: str,
        user: str,
        password: str,
        host: str,
        port: str,
        database: str,
    ) -> str:
        """
        Returns the URL for connecting to the remote database in format used by SQLAlchemy.
        For more information, see https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls.

        Args:
            dialect (str): The dialect for the database connection.
            driver (str): The driver for the database connection.
            user (str): The username for the database connection.
            password (str): The password for the database connection.
            host (str): The host address of the remote database.
            port (str): The port number for the database connection.
            database (str): The name of the database.

        Returns:
            str: The URL for connecting to the remote database.
        """
        return f"{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}"

    @classmethod
    def create_remote_connection(cls, remote_url: str) -> Connection:
        """
        Create a new SQLAlchemy Connection instance to the remote database.
        Don't forget to close the Connection after you are done using it!

        Args:
            remote_url (str): The URL for connecting to the remote database.
                Format is dialect+driver://username:password@host:port/database

        Returns:
            Connection: The SQLAlchemy Connection instance to the remote database.
        """
        return Connection(create_engine(remote_url))

    def get_scheme(self) -> Dict[str, Dict[str, sqltypes.TypeEngine]]:
        """Get the type scheme of the remote database.

        Returns:
            Dict[str, Dict[str, TypeEngine]]: A dictionary mapping table names to column names and their types.
        """
        remote_con = self.create_remote_connection(self.remote_url)

        inspector = DBInspector(remote_con)

        remote_md = MetaData()
        remote_md.reflect(bind=inspector.engine)

        table_names = inspector.get_tables()

        table_sql__types = {}

        for t_name in table_names:

            sql_table = SQLTable(t_name, remote_md)

            table_sql__types[t_name] = {}

            for c in sql_table.columns:
                try:
                    sql_type = type(c.type.as_generic())
                except NotImplementedError:
                    sql_type = None

                table_sql__types[t_name][c.name] = sql_type

        remote_con.close()

        return table_sql__types

    def make_db(self) -> Database:
        """
        Create a Database instance from the remote database.

        Returns:
            Database: The Database instance.
        """
        remote_con = self.create_remote_connection(self.remote_url)

        inspector = DBInspector(remote_con)

        remote_md = MetaData()
        remote_md.reflect(bind=inspector.engine)

        table_names = inspector.get_tables()

        df_dict: Dict[str, pd.DataFrame] = {}
        fk_dict: Dict[str, List[ForeignKeyDef]] = {}

        for t_name in tqdm(table_names, desc="Downloading tables"):

            sql_table = SQLTable(t_name, remote_md)

            dtypes: Dict[str, str] = {}
            sql_types_dict: Dict[str, sqltypes.TypeEngine] = {}

            for c in sql_table.columns:
                try:
                    sql_type = type(c.type.as_generic())
                except NotImplementedError:
                    sql_type = None

                dtype = SQL_TO_PANDAS.get(sql_type, None)
                if dtype is not None:
                    dtypes[c.name] = dtype
                    sql_types_dict[c.name] = sql_type
                else:
                    print(f"Unknown data type {c.type}")

            statement = select(sql_table.columns)
            query = statement.compile(remote_con.engine)
            df = pd.read_sql_query(str(query), con=remote_con, dtype=dtypes)

            for col, sql_type in sql_types_dict.items():
                if sql_type in DATE_TYPES:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except pd.errors.OutOfBoundsDatetime:
                        pass
                        # print(f"Out of bounds datetime in {t_name}.{col}", file=sys.stderr)

            # Create index column used as artificial primary key
            df.index.name = "__PK__"
            df.reset_index(inplace=True)

            df_dict[t_name] = df

            fk_dict[t_name] = inspector.get_foreign_keys(t_name)

        table_dict: Dict[str, Table] = {}

        # Re-index keys as RelBench do not support composite keys.
        # Also this way all original columns are preserved.
        for t_name in table_names:
            fkey_col_to_pkey_table: Dict[str, str] = {}

            for fk in fk_dict[t_name]:
                fk_col, fk_name = self._reindex_fk(
                    df_dict, t_name, fk.src_columns, fk.ref_table, fk.ref_columns
                )

                fkey_col_to_pkey_table[fk_name] = fk.ref_table
                df_dict[t_name][fk_name] = fk_col

            table_dict[t_name] = Table(
                df=df_dict[t_name],
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col="__PK__",
                time_col=self.time_col_dict.get(t_name, None),
            )

        # Remove original primary and foreign keys

        for t_name in table_names:
            sql_table = SQLTable(t_name, remote_md)
            table = table_dict[t_name]

            print(t_name)
            print("PK:", [c.name for c in sql_table.primary_key.columns])
            print(
                "FKs:",
                {
                    fk.referred_table.name: [c.name for c in fk.columns]
                    for fk in sql_table.foreign_key_constraints
                },
            )
            if not self.keep_original_keys:
                table.df.drop(
                    # Drop primary key columns
                    columns={c.name for c in sql_table.primary_key.columns}.union(
                        # Drop foreign key columns
                        {
                            c.name
                            for fk in sql_table.foreign_key_constraints
                            for c in fk.columns
                        }
                    ),
                    inplace=True,
                )

        remote_con.close()

        return Database(table_dict)

    def _reindex_fk(
        self,
        df_dict: Dict[str, pd.DataFrame],
        src_table: str,
        src_columns: List[str],
        ref_table: str,
        ref_columns: List[str],
    ):
        fk_name = f"FK_{ref_table}_" + "_".join(src_columns)

        df_src = df_dict[src_table][src_columns]
        df_ref = df_dict[ref_table]

        fk_col = df_src.merge(
            df_ref,
            how="left",
            left_on=src_columns,
            right_on=ref_columns,
        )["__PK__"]
        # fk_col = pd.merge(
        #     left=df_src,
        #     right=df_ref,
        #     how="inner",
        #     left_on=src_columns,
        #     right_on=ref_columns,
        # )["__PK__"]

        return fk_col, fk_name


DATE_TYPES = (sqltypes.Date, sqltypes.DateTime)

SQL_TO_PANDAS = {
    sqltypes.BigInteger: pd.Int64Dtype(),
    sqltypes.Boolean: pd.BooleanDtype(),
    sqltypes.Date: "object",
    sqltypes.DateTime: "object",
    sqltypes.Double: pd.Float64Dtype(),
    sqltypes.Enum: pd.CategoricalDtype(),
    sqltypes.Float: pd.Float64Dtype(),
    sqltypes.Integer: pd.Int32Dtype(),
    sqltypes.Interval: "object",
    sqltypes.LargeBinary: "object",
    sqltypes.Numeric: pd.Float64Dtype(),
    sqltypes.SmallInteger: pd.Int16Dtype(),
    sqltypes.String: "string",
    sqltypes.Text: "string",
    sqltypes.Time: "object",
    sqltypes.Unicode: "string",
    sqltypes.UnicodeText: "string",
    sqltypes.Uuid: "object",
}
