from typing import Dict, List, Optional, Tuple

import pandas as pd

from sqlalchemy import types
from sqlalchemy.engine import Connection, create_engine
from sqlalchemy.schema import MetaData, Table as SQLTable
from sqlalchemy.sql import select

from relbench.base import Dataset, Database, Table

from ctu_relational.db import DBInspector


class DBDataset(Dataset):
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
    ):
        """
        Initialize a DBDataset instance.

        Args:
            cache_dir (str, optional): The directory to cache the dataset. Defaults to None.
            remote_url (str, optional):
                The URL for connecting to the remote database in SQLAlchemy format.
                For more information, see https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls.
                If not defined, all following parameters have to be specified. Defaults to None.
            dialect (str, optional): The dialect for the database connection. Defaults to None.
            driver (str, optional): The driver for the database connection. Defaults to None.
            user (str, optional): The username for the database connection. Defaults to None.
            password (str, optional): The password for the database connection. Defaults to None.
            host (str, optional): The host address of the remote database. Defaults to None.
            port (str, optional): The port number for the database connection. Defaults to None.
            database (str, optional): The name of the database. Defaults to None.
        """
        if remote_url is not None:
            self.remote_url = remote_url
        else:
            self.remote_url = self.get_url(
                dialect, driver, user, password, host, port, database
            )
        super().__init__(cache_dir)

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
        pk_dict: Dict[str, List[str]] = {}
        fk_dict: Dict[str, List[Tuple[str, List[str]]]] = {}

        for t_name in table_names:

            sql_table = SQLTable(t_name, remote_md)

            dtypes: Dict[str, str] = {}

            for c in sql_table.columns:
                dtype = SQL_TO_PANDAS.get(type(c.type.as_generic()), None)
                if dtype is not None:
                    dtypes[c.name] = dtype
                else:
                    print(f"Unknown data type {c.type}")

            statement = select(sql_table.columns)
            query = statement.compile(remote_con.engine)
            df = pd.read_sql_query(str(query), con=remote_con, dtype=dtypes)

            # Create index column used as artificial primary key
            df.index.name = "__PK__"
            df.reset_index(inplace=True)

            df_dict[t_name] = df

            pk_dict[t_name] = list(inspector.get_primary_key(t_name))
            fk_dict[t_name] = [
                (fk.ref_table, fk.src_columns) for fk in inspector.get_foreign_keys(t_name)
            ]

        # Close the connection as we have all the data we need
        remote_con.close()

        table_dict: Dict[str, Table] = {}

        # Re-index keys as RelBench do not support composite keys.
        # Also this way all original columns are preserved.
        for t_name in table_names:
            fkey_col_to_pkey_table: Dict[str, str] = {}

            for ref_table, fk_cols in fk_dict[t_name]:
                fk_name = f"FK_{ref_table}_" + "_".join(fk_cols)
                fkey_col_to_pkey_table[fk_name] = ref_table

                df_pk = df_dict[t_name][fk_cols]
                df_fk = df_dict[ref_table].set_index(pk_dict[ref_table])

                df_dict[t_name][fk_name] = pd.merge(
                    df_pk,
                    df_fk,
                    how="left",
                    left_on=fk_cols,
                    right_index=True,
                )["__PK__"]

            table_dict[t_name] = Table(
                df=df_dict[t_name],
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col="__PK__",
            )

        return Database(table_dict)


SQL_TO_PANDAS = {
    types.BigInteger: pd.Int64Dtype(),
    types.Boolean: pd.BooleanDtype(),
    types.Date: "object",
    types.DateTime: "object",
    types.Double: pd.Float64Dtype(),
    types.Enum: pd.CategoricalDtype(),
    types.Float: pd.Float64Dtype(),
    types.Integer: pd.Int32Dtype(),
    types.Interval: "object",
    types.LargeBinary: "object",
    types.Numeric: pd.Float64Dtype(),
    types.SmallInteger: pd.Int16Dtype(),
    types.String: "string",
    types.Text: "string",
    types.Time: "object",
    types.Unicode: "string",
    types.UnicodeText: "string",
    types.Uuid: "object",
}
