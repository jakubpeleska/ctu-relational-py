from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from attrs import define, field

import sqlalchemy
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.types import TypeEngine

__ALL__ = [
    "DBInspectorInterface",
    "DBInspector",
    "CachedDBInspector",
]


@define()
class ForeignKeyDef:
    """
    Represents one foreign key.
    """

    columns: List[str] = field(converter=list)
    """
    The referencing columns (in this table)
    """

    ref_table: str
    """
    The referenced table name
    """

    ref_columns: List[str] = field(converter=list)
    """
    The referenced columns (in the referenced table)
    """


class DBInspectorInterface(ABC):
    """The interface for :py:class:`DBInspector`."""

    @property
    @abstractmethod
    def connection(self) -> Connection:
        """The underlying SQLAlchemy `Connection`."""
        ...

    @property
    @abstractmethod
    def engine(self) -> Engine:
        """The underlying SQLAlchemy `Engine`."""
        ...

    @abstractmethod
    def get_tables(self) -> Set[str]:
        """Get all tables in the databases as a set."""
        ...

    @abstractmethod
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        """Get all columns in a table as a dictionary of string -> SQLAlchemy type."""
        ...

    @abstractmethod
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        """Get all (table, column) pairs in the database as a set."""
        ...

    @abstractmethod
    def get_primary_key(self, table: str) -> Set[str]:
        """Get the primary key of the given table as a set of column names."""
        ...

    @abstractmethod
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        """
        Get all foreign key constraints of the given table as a dictionary, where the key is a subset of table columns,
        and the value is SQLAlchemy's `ForeignKeyDef` instance.
        """
        ...


class DBInspector(DBInspectorInterface):
    """
    A simplified helper class that allows to retrieve select basic information about a database,
    its tables, columns, and values.
    """

    def __init__(
        self,
        connection: Connection,
    ):
        """
        :field connection: The database connection - instance of SQLAlchemy's `Connection` class.
        :field table_filter: A :py:class:`db_transformer.helpers.collections.set_filter.SetFilter` instance or a callable that filters a set of values. \
All values that remain are the tables that the inspector will be aware of; excluded ones will be ignored.
        """
        self._connection = connection
        self._inspect = sqlalchemy.inspect(self._connection.engine)

    @property
    def connection(self) -> Connection:
        return self._connection

    @property
    def engine(self) -> Engine:
        return self._connection.engine

    def get_tables(self) -> Set[str]:
        out = set(self._inspect.get_table_names())

        return out

    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        out = {col["name"]: col["type"] for col in self._inspect.get_columns(table)}
        return out

    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        out = set()

        for tbl in self.get_tables():
            out |= {(tbl, col) for col in self.get_columns(tbl).keys()}

        return out

    def get_primary_key(self, table: str) -> Set[str]:
        return set(self._inspect.get_pk_constraint(table)["constrained_columns"])

    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        return {
            frozenset(fk["constrained_columns"]): ForeignKeyDef(
                columns=fk["constrained_columns"],
                ref_table=fk["referred_table"],
                ref_columns=fk["referred_columns"],
            )
            for fk in self._inspect.get_foreign_keys(table)
        }


class CachedDBInspector(DBInspectorInterface):
    """:py:class:`DBInspector`, but that caches its results instead of repeated SQL calls/retrievals."""

    def __init__(self, delegate: DBInspectorInterface):
        """:field delegate: The :py:class:`DBInspectorInterface` instance to cache the results of. :py:class:`CachedDBInspector` will delegate the function calls to this instance."""
        if isinstance(delegate, CachedDBInspector):
            raise TypeError("DatabaseWrapper is already cached.")

        self._delegate = delegate

    @property
    def connection(self) -> Connection:
        return self._delegate.connection

    @property
    def engine(self) -> Engine:
        return self._delegate.engine

    @lru_cache(maxsize=None)
    def get_tables(self) -> Set[str]:
        return self._delegate.get_tables()

    @lru_cache(maxsize=None)
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        return self._delegate.get_columns(table)

    @lru_cache(maxsize=None)
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        return self._delegate.get_table_column_pairs()

    @lru_cache(maxsize=None)
    def get_primary_key(self, table: str) -> Set[str]:
        return self._delegate.get_primary_key(table)

    @lru_cache(maxsize=None)
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        return self._delegate.get_foreign_keys(table)
