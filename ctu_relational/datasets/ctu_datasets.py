import datetime
from typing import Dict, Literal, Optional, Union

import pandas as pd

from relbench.base import Database

from .db_dataset import DBDataset

__ALL__ = [
    "Accidents",
    "Airline",
    "Expenditures",
    "Employee",
    "LegalActs",
    "Seznam",
    "TPCC",
    "TPCD",
    "TPCDS",
    "TPCH",
    "VOC",
    "Walmart",
]

# fmt: off
CTUDatabaseName = Literal[
    'Accidents', 'AdventureWorks2014', 'Airline', 'Atherosclerosis', 'AustralianFootball', 
    'Basketball_men', 'Basketball_women', 'Biodegradability', 'Bupa', 'Carcinogenesis', 
    'ccs', 'CDESchools', 'Chess', 'CiteSeer', 'classicmodels', 'ConsumerExpenditures',
    'CORA', 'Countries', 'CraftBeer', 'Credit', 'cs', 'Dallas', 'DCG', 'Dunur', 'Elti',
    'employee', 'ErgastF1', 'Facebook', 'financial', 'FNHK', 'ftp', 'geneea', 'genes',
    'GOSales', 'Grants', 'Hepatitis_std', 'Hockey', 'imdb_ijs', 'KRK', 'lahman_2014',
    'legalActs', 'Mesh', 'medical', 'Mondial', 'Mooney_Family', 'imdb_MovieLens',
    'MuskSmall', 'mutagenesis', 'nations', 'NBA', 'NCAA', 'northwind', 'Pima',
    'PremierLeague', 'PTE', 'PubMed_Diabetes', 'pubs', 'Pyrimidine', 'restbase',
    'sakila', 'SalesDB', 'Same_gen', 'SAP', 'SAT', 'Seznam', 'SFScores', 'Shakespeare',
    'stats', 'Student_loan', 'Toxicology', 'tpcc', 'tpcd', 'tpcds', 'tpch', 'trains',
    'Triazine', 'university', 'UTube', 'UW_std', 'VisualGenome', 'voc', 'Walmart','WebKP',
    'world'
]
# fmt: on


class CTUDataset(DBDataset):
    def __init__(
        self,
        database: CTUDatabaseName,
        cache_dir: Optional[str] = None,
        time_col_dict: Optional[Dict[str, str]] = None,
        keep_original_keys: bool = False,
    ):
        """Create a database dataset object.

        Args:
            database (CTUDatabaseName): The name of the database.
            cache_dir (str, optional): The directory to cache the dataset. Defaults to None.
            time_col_dict (Dict[str, str], optional): A dictionary mapping table names to time columns. Defaults to None.
            keep_original_keys (bool, optional): Whether to keep original primary and foreign keys \
                after duplication during re-indexing. This is useful when the keys contain information \
                beyond just their relationship to other rows. Defaults to False.
        """
        super().__init__(
            cache_dir=cache_dir,
            dialect="mariadb",
            driver="mysqlconnector",
            user="guest",
            password="ctu-relational",
            host="relational.fel.cvut.cz",
            port=3306,
            database=database,
            time_col_dict=time_col_dict,
            keep_original_keys=keep_original_keys,
        )


class Accidents(CTUDataset):
    val_timestamp = pd.Timestamp("2003-01-01")
    test_timestamp = pd.Timestamp("2005-01-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Accidents",
            cache_dir=cache_dir,
            time_col_dict={"nesreca": "cas_nesreca"},
            keep_original_keys=False,
        )


class Airline(CTUDataset):
    val_timestamp = pd.Timestamp("2016-01-18")
    test_timestamp = pd.Timestamp("2016-01-25")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Airline",
            cache_dir=cache_dir,
            time_col_dict={"On_Time_On_Time_Performance_2016_1": "FlightDate"},
            keep_original_keys=False,
        )


class Expenditures(CTUDataset):
    """
    The Consumer Expenditure Survey (CE) collects data on expenditures, income, \
    and demographics in the United States. The public-use microdata (PUMD) files \
    provide this information for individual respondents without any information that \
    could identify respondents. PUMD files include adjustments for information that is \
    missing because respondents were unwilling or unable to provide it. The files also \
    have been adjusted to reduce the likelihood of identifying respondents, either \
    directly or through inference. The task is to predict, whether the expenditure
    is a gift or not. Household ids change from year to year - this is a property of the \
    data source.
    
    Original source: www.bls.gov
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "ConsumerExpenditures",
            cache_dir=cache_dir,
            keep_original_keys=False,
        )


class Employee(CTUDataset):
    val_timestamp = pd.Timestamp("1998-02-01")
    test_timestamp = pd.Timestamp("2000-05-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "employee",
            cache_dir=cache_dir,
            time_col_dict={
                "dept_emp": "from_date",
                "dept_manager": "from_date",
                "employees": "hire_date",
                "salaries": "from_date",
                "titles": "from_date",
            },
            keep_original_keys=True,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        # Drop redundant key columns
        db.table_dict["titles"].df.drop(columns=["emp_no"], inplace=True)
        db.table_dict["dept_manager"].df.drop(columns=["emp_no", "dept_no"], inplace=True)
        db.table_dict["departments"].df.drop(columns=["dept_no"], inplace=True)
        db.table_dict["salaries"].df.drop(columns=["emp_no"], inplace=True)
        db.table_dict["dept_emp"].df.drop(columns=["emp_no", "dept_no"], inplace=True)
        db.table_dict["employees"].df.drop(columns=["emp_no"], inplace=True)

        # Convert date columns to datetime and handle NaT values
        db.table_dict["titles"].df["to_date"] = pd.to_datetime(
            db.table_dict["titles"].df["to_date"], errors="coerce"
        )
        db.table_dict["dept_manager"].df["to_date"] = pd.to_datetime(
            db.table_dict["dept_manager"].df["to_date"], errors="coerce"
        )
        db.table_dict["salaries"].df["to_date"] = pd.to_datetime(
            db.table_dict["salaries"].df["to_date"], errors="coerce"
        )
        db.table_dict["dept_emp"].df["to_date"] = pd.to_datetime(
            db.table_dict["dept_emp"].df["to_date"], errors="coerce"
        )

        return db


class LegalActs(CTUDataset):
    """
    Bulgarian court decision metadata.
    """

    val_timestamp = pd.Timestamp("2011-08-01")
    test_timestamp = pd.Timestamp("2012-03-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "legalActs",
            cache_dir=cache_dir,
            time_col_dict={
                "legalacts": "StartDate",
            },
            keep_original_keys=False,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        # Remove scrape fix table
        db.table_dict.pop("scrapefix")

        return db


# class Sales(CTUDataset):
#     def __init__(self, cache_dir: Optional[str] = None):
#         super().__init__(
#             "SalesDB",
#             cache_dir=cache_dir,
#             keep_original_keys=False,
#         )


class Seznam(CTUDataset):
    val_timestamp = pd.Timestamp("2014-12-01")
    test_timestamp = pd.Timestamp("2015-05-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Seznam",
            cache_dir=cache_dir,
            time_col_dict={
                "dobito": "month_year_datum_transakce",
                "probehnuto": "month_year_datum_transakce",
                "probehnuto_mimo_penezenku": "Month/Year",
            },
            keep_original_keys=False,
        )


class TPCC(CTUDataset):
    val_timestamp = pd.Timestamp("1996-01-01")
    test_timestamp = pd.Timestamp("1997-05-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpcc",
            cache_dir=cache_dir,
            time_col_dict={
                "dss_lineitem": "l_shipdate",
                "dss_order": "o_orderdate",
            },
            keep_original_keys=True,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        return db


class TPCD(CTUDataset):
    val_timestamp = pd.Timestamp("1996-01-01")
    test_timestamp = pd.Timestamp("1997-05-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpcd",
            cache_dir=cache_dir,
            time_col_dict={
                "dss_lineitem": "l_shipdate",
                "dss_order": "o_orderdate",
            },
            keep_original_keys=True,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        # Add missing foreign keys
        fk_col, fk_name = self._reindex_fk(
            {tn: t.df for tn, t in db.table_dict.items()},
            "dss_partsupp",
            ["ps_partkey"],
            "dss_part",
            ["p_partkey"],
        )
        db.table_dict["dss_partsupp"].fkey_col_to_pkey_table[fk_name] = "dss_part"
        db.table_dict["dss_partsupp"].df[fk_name] = fk_col

        # Drop redundant key columns
        db.table_dict["dss_lineitem"].df.drop(columns=["l_orderkey"], inplace=True)
        db.table_dict["dss_part"].df.drop(columns=["p_partkey"], inplace=True)
        db.table_dict["dss_partsupp"].df.drop(
            columns=["ps_partkey", "ps_suppkey"], inplace=True
        )
        db.table_dict["dss_region"].df.drop(columns=["r_regionkey"], inplace=True)
        db.table_dict["dss_nation"].df.drop(
            columns=["n_nationkey", "n_regionkey"], inplace=True
        )
        db.table_dict["dss_customer"].df.drop(
            columns=["c_custkey", "c_nationkey"], inplace=True
        )
        db.table_dict["dss_supplier"].df.drop(
            columns=["s_suppkey", "s_nationkey"], inplace=True
        )
        db.table_dict["dss_order"].df.drop(
            columns=["o_orderkey", "o_custkey"], inplace=True
        )

        return db


class TPCDS(CTUDataset):
    val_timestamp = pd.Timestamp("1999-05-01")
    test_timestamp = pd.Timestamp("2001-05-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpcds",
            cache_dir=cache_dir,
            time_col_dict={
                "store": "s_rec_start_date",
                "customer": "c_first_sales_date_sk",
                "web_page": "wp_creation_date_sk",
                "inventory": "inv_date_sk",
                "catalog_page": "cp_start_date_sk",
                "promotion": "p_start_date_sk",
                "web_site": "web_open_date_sk",
                "item": "i_rec_start_date",
                "store_returns": "sr_returned_date_sk",
                "catalog_sales": "cs_ship_date_sk",
                "call_center": "cc_open_date_sk",
                "web_sales": "ws_ship_date_sk",
            },
            keep_original_keys=False,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        # Replace date and time tables with simple datetime columns
        time_df = db.table_dict["time_dim"].df
        date_df = db.table_dict["date_dim"].df

        def to_time(x):
            return pd.Timedelta(
                hours=x["t_hour"], minutes=x["t_minute"], seconds=x["t_second"]
            )

        time_df["time_delta"] = db.table_dict["time_dim"].df.apply(to_time, axis=1)

        date_fk_dict = {
            tn: [
                fk for fk, fk_tn in t.fkey_col_to_pkey_table.items() if fk_tn == "date_dim"
            ]
            for tn, t in db.table_dict.items()
        }

        time_fk_dict = {
            "store_returns": "FK_time_dim_sr_return_time_sk",
            "catalog_returns": "FK_time_dim_cr_returned_time_sk",
            "web_returns": "FK_time_dim_wr_returned_time_sk",
            "store_sales": "FK_time_dim_ss_sold_time_sk",
            "catalog_sales": "FK_time_dim_cs_sold_time_sk",
            "web_sales": "FK_time_dim_ws_sold_time_sk",
        }

        for t_name, fks in date_fk_dict.items():
            for fk in fks:
                print(db.table_dict[t_name].df)
                print(date_df)
                db.table_dict[t_name].df[fk.removeprefix(f"FK_date_dim_")] = db.table_dict[
                    t_name
                ].df.join(date_df["d_date"], on=fk, how="left")["d_date"]
                db.table_dict[t_name].df.drop(columns=[fk], inplace=True)
                db.table_dict[t_name].fkey_col_to_pkey_table.pop(fk)

        for t_name, fk in time_fk_dict.items():
            db.table_dict[t_name].df[
                fk.removeprefix(f"FK_time_dim_").replace("time", "date")
            ] = db.table_dict[t_name].df.join(time_df["time_delta"], on=fk, how="left")[
                "time_delta"
            ]
            db.table_dict[t_name].df.drop(columns=[fk], inplace=True)
            db.table_dict[t_name].fkey_col_to_pkey_table.pop(fk)

        db.table_dict.pop("time_dim")
        db.table_dict.pop("date_dim")

        return db


class TPCH(CTUDataset):
    val_timestamp = pd.Timestamp("1996-01-01")
    test_timestamp = pd.Timestamp("1997-05-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpch",
            cache_dir=cache_dir,
            time_col_dict={"lineitem": "l_shipdate", "orders": "o_orderdate"},
            keep_original_keys=True,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        # Drop redundant key columns
        db.table_dict["lineitem"].df.drop(
            columns=["l_orderkey", "l_partkey", "l_suppkey"], inplace=True
        )
        db.table_dict["part"].df.drop(columns=["p_partkey"], inplace=True)
        db.table_dict["partsupp"].df.drop(
            columns=["ps_partkey", "ps_suppkey"], inplace=True
        )
        db.table_dict["region"].df.drop(columns=["r_regionkey"], inplace=True)
        db.table_dict["nation"].df.drop(
            columns=["n_nationkey", "n_regionkey"], inplace=True
        )
        db.table_dict["customer"].df.drop(
            columns=["c_custkey", "c_nationkey"], inplace=True
        )
        db.table_dict["supplier"].df.drop(
            columns=["s_suppkey", "s_nationkey"], inplace=True
        )
        db.table_dict["orders"].df.drop(columns=["o_orderkey", "o_custkey"], inplace=True)

        return db


class VOC(CTUDataset):
    # Offset to avoid issues with pandas datetime
    YEAR_OFFSET = 100

    val_timestamp = pd.Timestamp("1837-01-01")
    test_timestamp = pd.Timestamp("1867-01-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "voc",
            cache_dir=cache_dir,
            time_col_dict={"voyages": "departure_date"},
            keep_original_keys=False,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        voyages_df = db.table_dict["voyages"].df

        def conv(x: Union[datetime.date, None]):
            if x is None:
                return pd.NaT
            # Ignore dates before 1678 to avoid issues with pandas datetime
            if x.year + self.YEAR_OFFSET < 1678:
                return pd.NaT
            # Add offset to year to avoid issues with pandas datetime
            return pd.Timestamp(year=x.year + self.YEAR_OFFSET, month=x.month, day=x.day)

        voyages_df["departure_date"] = voyages_df["departure_date"].apply(conv)
        voyages_df["arrival_date"] = voyages_df["arrival_date"].apply(conv)
        voyages_df["cape_departure"] = voyages_df["cape_departure"].apply(conv)
        voyages_df["cape_arrival"] = voyages_df["cape_arrival"].apply(conv)

        db.table_dict["voyages"].df = voyages_df

        return db


class Walmart(CTUDataset):
    TIME_ORIGIN = pd.Timestamp("1970-01-01")

    val_timestamp = pd.Timestamp("2013-09-01")
    test_timestamp = pd.Timestamp("2014-04-01")

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Walmart",
            cache_dir=cache_dir,
            time_col_dict={"weather": "date"},
            keep_original_keys=True,
        )

    def make_db(self) -> Database:
        db = super().make_db()

        # Convert sunrise and sunset to datetime
        weather_df = db.table_dict["weather"].df
        weather_df["sunrise"] = pd.to_datetime(self.TIME_ORIGIN + weather_df["sunrise"])
        weather_df["sunset"] = pd.to_datetime(self.TIME_ORIGIN + weather_df["sunset"])
        db.table_dict["weather"].df = weather_df

        return db
