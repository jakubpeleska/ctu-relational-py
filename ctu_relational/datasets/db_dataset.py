from typing import Dict, List, Optional, Tuple

from tqdm.std import tqdm

import pandas as pd

from sqlalchemy import types, Connection, create_engine, MetaData, Table as SQLTable, select

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
        val_timestamp: Optional[pd.Timestamp] = None,
        test_timestamp: Optional[pd.Timestamp] = None,
    ):
        """Create a database dataset object.

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
            time_col_dict (Dict[str, str], optional): A dictionary mapping table names to time columns. Defaults to None.
            val_timestamp (pd.Timestamp, optional): The timestamp for the validation split. Defaults to None.
            test_timestamp (pd.Timestamp, optional): The timestamp for the test split. Defaults to None.
        """

        self.remote_url = (
            remote_url
            if remote_url is not None
            else self.get_url(dialect, driver, user, password, host, port, database)
        )

        self.time_col_dict = time_col_dict if time_col_dict is not None else {}

        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp

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

            for c in sql_table.columns:
                try:
                    sql_type = type(c.type.as_generic())
                except NotImplementedError:
                    sql_type = None
                if sql_type in DATE_TYPES and t_name not in self.time_col_dict:
                    self.time_col_dict[t_name] = c.name
                dtype = SQL_TO_PANDAS.get(sql_type, None)
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

            fk_dict[t_name] = inspector.get_foreign_keys(t_name)

        # Close the connection as we have all the data we need
        remote_con.close()

        table_dict: Dict[str, Table] = {}

        # Re-index keys as RelBench do not support composite keys.
        # Also this way all original columns are preserved.
        for t_name in table_names:
            fkey_col_to_pkey_table: Dict[str, str] = {}

            for fk in fk_dict[t_name]:
                fk_name = f"FK_{fk.ref_table}_" + "_".join(fk.src_columns)
                fkey_col_to_pkey_table[fk_name] = fk.ref_table

                df_src = df_dict[t_name][fk.src_columns]
                df_ref = df_dict[fk.ref_table]

                out = pd.merge(
                    left=df_src,
                    right=df_ref,
                    how="inner",
                    left_on=fk.src_columns,
                    right_on=fk.ref_columns,
                )
                df_dict[t_name][fk_name] = out["__PK__"]

            table_dict[t_name] = Table(
                df=df_dict[t_name],
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col="__PK__",
                time_col=self.time_col_dict.get(t_name, None),
            )

        return Database(table_dict)


DATE_TYPES = (types.Date, types.DateTime)

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
