from dataclasses import dataclass
from typing import Callable, Literal, Dict, Optional, Tuple, Union

from db_transformer.data.dataset_defaults.utils import TaskType
from db_transformer.db.distinct_cnt_retrieval import DBDistinctCounter
from db_transformer.db.schema_autodetect import BuiltinDBDistinctCounter
from db_transformer.schema import Schema, ForeignKeyDef


@dataclass
class CTUDatasetDefault:
    target_table: str
    target_column: str
    target_id: str
    task: TaskType
    timestamp_column: Optional[str] = None
    schema_fixer: Optional[Callable[[Schema], None]] = None
    db_distinct_counter: Union[DBDistinctCounter, BuiltinDBDistinctCounter] = "db_distinct"
    force_collation: Optional[str] = None

    @property
    def target(self) -> Tuple[str, str]:
        return self.target_table, self.target_column


# fmt: off
CTUDatasetName = Literal[
    'Accidents', 'AdventureWorks2014', 'Airline', 'Atherosclerosis', 'AustralianFootball', 
    'Basketball_men', 'Basketball_women', 'Biodegradability', 'Bupa', 'Carcinogenesis', 
    'ccs', 'CDESchools', 'Chess', 'CiteSeer', 'classicmodels', 'ConsumerExpenditures',
    'CORA', 'Countries', 'CraftBeer', 'Credit', 'cs', 'Dallas', 'DCG', 'Dunur', 'Elti',
    'employee', 'ErgastF1', 'Facebook', 'financial', 'FNHK', 'ftp', 'geneea', 'genes',
    'GOSales', 'Grants', 'Hepatitis_std', 'Hockey', 'imdb_ijs', 'KRK', 'lahman_2014',
    'legalActs', 'Mesh', 'medical', 'Mondial', 'Mooney_Family', 'imdb_MovieLens',
    'MuskSmall', 'mutagenesis', 'nations', 'NBA', 'NCAA', 'northwind', 'Pima',
    'PremierLeague', 'PTC', 'PTE', 'PubMed_Diabetes', 'pubs', 'Pyrimidine', 'restbase',
    'sakila', 'SalesDB', 'Same_gen', 'SAP', 'SAT', 'Seznam', 'SFScores', 'Shakespeare',
    'stats', 'Student_loan', 'Toxicology', 'tpcc', 'tpcd', 'tpcds', 'tpch', 'trains',
    'Triazine', 'university', 'UTube', 'UW_std', 'VisualGenome', 'voc', 'Walmart','WebKP',
    'world'
]
# fmt: on


CTU_REPOSITORY_DEFAULTS: Dict[CTUDatasetName, CTUDatasetDefault] = {
    "Accidents": CTUDatasetDefault(
        target_table="nesreca",
        target_column="klas_nesreca",
        target_id="id_nesreca",
        timestamp_column="cas_nesreca",
        task=TaskType.CLASSIFICATION,
    ),
    "AdventureWorks2014": CTUDatasetDefault(
        target_table="SalesOrderHeader",
        target_column="TotalDue",
        target_id="CustomerID",
        timestamp_column="OrderDate",
        task=TaskType.REGRESSION,
    ),
    "Airline": CTUDatasetDefault(
        target_table="On_Time_On_Time_Performance_2016_1",
        target_column="ArrDel15",
        target_id="?",
        timestamp_column="FlightDate",
        task=TaskType.CLASSIFICATION,
    ),
    # TODO: Review the change in target
    "Atherosclerosis": CTUDatasetDefault(
        target_table="Entry",
        target_column="STAV",
        target_id="ICO",
        # timestamp_column="ROKUMR",
        task=TaskType.CLASSIFICATION,
    ),
    "AustralianFootball": CTUDatasetDefault(
        target_table="matches",
        target_column="target",
        target_id="mid",
        timestamp_column="start_dt",
        task=TaskType.CLASSIFICATION,
    ),
    "Basketball_men": CTUDatasetDefault(
        target_table="teams",
        target_column="rank",
        target_id="tmID, year",
        timestamp_column="year",
        task=TaskType.REGRESSION,
    ),
    "Basketball_women": CTUDatasetDefault(
        target_table="teams",
        target_column="playoff",
        target_id="tmID, year",
        timestamp_column="year",
        task=TaskType.CLASSIFICATION,
    ),
    "Biodegradability": CTUDatasetDefault(
        target_table="molecule",
        target_column="activity",
        target_id="molecule_id",
        task=TaskType.REGRESSION,
    ),
    "Bupa": CTUDatasetDefault(
        target_table="bupa",
        target_column="arg2",
        target_id="arg1",
        task=TaskType.CLASSIFICATION,
    ),
    "Carcinogenesis": CTUDatasetDefault(
        target_table="canc",
        target_column="class",
        target_id="drug_id",
        task=TaskType.CLASSIFICATION,
    ),
    "ccs": CTUDatasetDefault(
        target_table="transactions_1k",
        target_column="Price",
        target_id="TransactionID",
        timestamp_column="Date",
        task=TaskType.REGRESSION,
    ),
    "CDESchools": CTUDatasetDefault(
        target_table="satscores",
        target_column="PctGE1500",
        target_id="cds",
        task=TaskType.REGRESSION,
    ),
    "Chess": CTUDatasetDefault(
        target_table="game",
        target_column="game_result",
        target_id="game_id",
        timestamp_column="event_date",
        task=TaskType.CLASSIFICATION,
    ),
    "CiteSeer": CTUDatasetDefault(
        target_table="paper",
        target_column="class_label",
        target_id="paper_id",
        task=TaskType.CLASSIFICATION,
        schema_fixer=lambda schema: schema.cites.foreign_keys.extend(
            [
                ForeignKeyDef(["cited_paper_id"], "paper", ["paper_id"]),
                ForeignKeyDef(["citing_paper_id"], "paper", ["paper_id"]),
            ]
        ),
    ),
    "classicmodels": CTUDatasetDefault(
        target_table="payments",
        target_column="amount",
        target_id="checkNumber",
        timestamp_column="paymentDate",
        task=TaskType.REGRESSION,
    ),
    "ConsumerExpenditures": CTUDatasetDefault(
        target_table="EXPENDITURES",
        target_column="GIFT",
        target_id="EXPENDITURE_ID",
        task=TaskType.REGRESSION,
    ),
    "CORA": CTUDatasetDefault(
        target_table="paper",
        target_column="class_label",
        target_id="paper_id",
        task=TaskType.CLASSIFICATION,
    ),
    "Countries": CTUDatasetDefault(
        target_table="target",
        target_column="2012",
        target_id="Country Code",
        task=TaskType.REGRESSION,
    ),
    "CraftBeer": CTUDatasetDefault(
        target_table="breweries",
        target_column="state",
        target_id="id",
        task=TaskType.CLASSIFICATION,
    ),
    "Credit": CTUDatasetDefault(
        target_table="member",
        target_column="region_no",
        target_id="member_no",
        timestamp_column="issue_dt",
        task=TaskType.CLASSIFICATION,
    ),
    "cs": CTUDatasetDefault(
        target_table="target_churn",
        target_column="target_churn",
        target_id="ACC_KEY",
        timestamp_column="date_horizon",
        task=TaskType.CLASSIFICATION,
    ),
    "Dallas": CTUDatasetDefault(
        target_table="incidents",
        target_column="subject_statuses",
        target_id="case_number",
        timestamp_column="date",
        task=TaskType.CLASSIFICATION,
    ),
    "DCG": CTUDatasetDefault(
        target_table="sentences",
        target_column="class",
        target_id="id",
        task=TaskType.CLASSIFICATION,
    ),
    "Dunur": CTUDatasetDefault(
        target_table="target",
        target_column="is_dunur",
        target_id="name1, name2",
        task=TaskType.CLASSIFICATION,
    ),
    "Elti": CTUDatasetDefault(
        target_table="target",
        target_column="is_elti",
        target_id="name1, name2",
        task=TaskType.CLASSIFICATION,
    ),
    "employee": CTUDatasetDefault(
        target_table="salaries",
        target_column="salary",
        target_id="emp_no",
        timestamp_column="from_date",
        task=TaskType.REGRESSION,
    ),
    "ErgastF1": CTUDatasetDefault(
        target_table="target",
        target_column="win",
        target_id="targetId",
        timestamp_column="raceId",
        task=TaskType.CLASSIFICATION,
    ),
    "Facebook": CTUDatasetDefault(
        target_table="feat",
        target_column="gender1",
        target_id="id",
        task=TaskType.CLASSIFICATION,
    ),
    "financial": CTUDatasetDefault(
        target_table="loan",
        target_column="status",
        target_id="account_id",
        timestamp_column="date",
        task=TaskType.CLASSIFICATION,
    ),
    "FNHK": CTUDatasetDefault(
        target_table="pripady",
        target_column="Delka_hospitalizace",
        target_id="Identifikace_pripadu",
        timestamp_column="Datum_prijeti",
        task=TaskType.REGRESSION,
    ),
    "ftp": CTUDatasetDefault(
        target_table="session",
        target_column="gender",
        target_id="session_id",
        task=TaskType.CLASSIFICATION,
    ),
    "geneea": CTUDatasetDefault(
        target_table="hl_hlasovani",
        target_column="vysledek",
        target_id="id_hlasovani",
        timestamp_column="datum",
        task=TaskType.CLASSIFICATION,
        force_collation="utf8mb3_unicode_ci",
    ),
    "genes": CTUDatasetDefault(
        target_table="Classification",
        target_column="Localization",
        target_id="GeneID",
        task=TaskType.CLASSIFICATION,
    ),
    "GOSales": CTUDatasetDefault(
        target_table="go_1k",
        target_column="Quantity",
        target_id="Retailer code, Product number",
        timestamp_column="Date",
        task=TaskType.REGRESSION,
    ),
    "Grants": CTUDatasetDefault(
        target_table="awards",
        target_column="award_amount",
        target_id="award_id",
        timestamp_column="award_effective_date",
        task=TaskType.REGRESSION,
    ),
    "Hepatitis_std": CTUDatasetDefault(
        target_table="dispat",
        target_column="Type",
        target_id="m_id",
        task=TaskType.CLASSIFICATION,
    ),
    "Hockey": CTUDatasetDefault(
        target_table="Master",
        target_column="shootCatch",
        target_id="playerId",
        task=TaskType.CLASSIFICATION,
    ),
    "imdb_ijs": CTUDatasetDefault(
        target_table="actors",
        target_column="gender",
        target_id="?",
        task=TaskType.CLASSIFICATION,
        db_distinct_counter="fetchall_unidecode_strip_ci",
    ),
    "KRK": CTUDatasetDefault(
        target_table="krk",
        target_column="class",
        target_id="id",
        task=TaskType.CLASSIFICATION,
    ),
    "lahman_2014": CTUDatasetDefault(
        target_table="salaries",
        target_column="salary",
        target_id="teamID, playerID, lgID",
        timestamp_column="yearID",
        task=TaskType.REGRESSION,
    ),
    "legalActs": CTUDatasetDefault(
        target_table="legalacts",
        target_column="ActKind",
        target_id="id",
        timestamp_column="update",
        task=TaskType.CLASSIFICATION,
        db_distinct_counter="fetchall_unidecode_strip_ci",
    ),
    "Mesh": CTUDatasetDefault(
        target_table="mesh",
        target_column="num",
        target_id="name",
        task=TaskType.CLASSIFICATION,
    ),
    "Mondial": CTUDatasetDefault(
        target_table="target",
        target_column="Target",
        target_id="Country",
        task=TaskType.CLASSIFICATION,
    ),
    "Mooney_Family": CTUDatasetDefault(
        target_table="uncle",
        target_column="?",
        target_id="name1, name2",
        task=TaskType.LINK_PREDICTION,
    ),
    "imdb_MovieLens": CTUDatasetDefault(
        target_table="users",
        target_column="u_gender",
        target_id="userid",
        task=TaskType.CLASSIFICATION,
    ),
    "medical": CTUDatasetDefault(
        target_table="Examination",
        target_column="Thrombosis",
        target_id="ID",
        timestamp_column="Examination Date",
        task=TaskType.CLASSIFICATION,
    ),
    "MuskSmall": CTUDatasetDefault(
        target_table="molecule",
        target_column="class",
        target_id="molecule_name",
        task=TaskType.CLASSIFICATION,
    ),
    "mutagenesis": CTUDatasetDefault(
        target_table="molecule",
        target_column="mutagenic",
        target_id="molecule_id",
        task=TaskType.CLASSIFICATION,
    ),
    "nations": CTUDatasetDefault(
        target_table="stat",
        target_column="femaleworkers",
        target_id="country_id",
        task=TaskType.CLASSIFICATION,
    ),
    "NBA": CTUDatasetDefault(
        target_table="Game",
        target_column="ResultOfTeam1",
        target_id="GameId",
        timestamp_column="Date",
        task=TaskType.CLASSIFICATION,
    ),
    "NCAA": CTUDatasetDefault(
        target_table="target",
        target_column="team_id1_wins",
        target_id="id",
        timestamp_column="season",
        task=TaskType.CLASSIFICATION,
    ),
    "northwind": CTUDatasetDefault(
        target_table="Orders",
        target_column="Freight",
        target_id="OrderID",
        timestamp_column="OrderId",
        task=TaskType.REGRESSION,
    ),
    "Pima": CTUDatasetDefault(
        target_table="pima",
        target_column="arg2",
        target_id="arg1",
        task=TaskType.CLASSIFICATION,
    ),
    "PremierLeague": CTUDatasetDefault(
        target_table="Matches",
        target_column="ResultOfTeamHome",
        target_id="MatchID",
        timestamp_column="MatchDate",
        task=TaskType.CLASSIFICATION,
    ),
    "PTC": CTUDatasetDefault(
        target_table="molecule",
        target_column="label",
        target_id="molecule_id",
        task=TaskType.REGRESSION,
    ),
    "PTE": CTUDatasetDefault(
        target_table="pte_active",
        target_column="is_active",
        target_id="drug_id",
        task=TaskType.CLASSIFICATION,
    ),
    "PubMed_Diabetes": CTUDatasetDefault(
        target_table="paper",
        target_column="class_label",
        target_id="paper_id",
        task=TaskType.CLASSIFICATION,
    ),
    "pubs": CTUDatasetDefault(
        target_table="titles",
        target_column="ytd_sales",
        target_id="title_id",
        timestamp_column="pubdate",
        task=TaskType.REGRESSION,
    ),
    "Pyrimidine": CTUDatasetDefault(
        target_table="molecule",
        target_column="activity",
        target_id="molecule_id",
        task=TaskType.REGRESSION,
    ),
    "restbase": CTUDatasetDefault(
        target_table="generalinfo",
        target_column="review",
        target_id="id_restaurant",
        task=TaskType.REGRESSION,
    ),
    "sakila": CTUDatasetDefault(
        target_table="payment",
        target_column="amount",
        target_id="payment_id",
        timestamp_column="payment_date",
        task=TaskType.REGRESSION,
    ),
    "SalesDB": CTUDatasetDefault(
        target_table="Sales",
        target_column="Quantity",
        target_id="SalesID",
        task=TaskType.REGRESSION,
    ),
    "Same_gen": CTUDatasetDefault(
        target_table="target",
        target_column="target",
        target_id="name1, name2",
        task=TaskType.CLASSIFICATION,
    ),
    "SAP": CTUDatasetDefault(
        target_table="Mailings1_2",
        target_column="RESPONSE",
        target_id="REFID",
        timestamp_column="REF_DATE",
        task=TaskType.CLASSIFICATION,
    ),
    "SAT": CTUDatasetDefault(
        target_table="fault",
        target_column="tf",
        target_id="?",
        timestamp_column="tm",
        task=TaskType.CLASSIFICATION,
    ),
    "Seznam": CTUDatasetDefault(
        target_table="probehnuto",
        target_column="kc_proklikano",
        target_id="client_id, sluzba",
        timestamp_column="month_year_datum_transakce",
        task=TaskType.REGRESSION,
    ),
    "SFScores": CTUDatasetDefault(
        target_table="inspections",
        target_column="score",
        target_id="business_id",
        timestamp_column="date",
        task=TaskType.REGRESSION,
    ),
    "Shakespeare": CTUDatasetDefault(
        target_table="paragraphs",
        target_column="character_id",
        target_id="id",
        task=TaskType.LINK_PREDICTION,
    ),
    "stats": CTUDatasetDefault(
        target_table="users",
        target_column="Reputation",
        target_id="Id",
        timestamp_column="LastAccessDate",
        task=TaskType.REGRESSION,
    ),
    "Student_loan": CTUDatasetDefault(
        target_table="no_payment_due",
        target_column="bool",
        target_id="name",
        task=TaskType.CLASSIFICATION,
    ),
    "Toxicology": CTUDatasetDefault(
        target_table="molecule",
        target_column="label",
        target_id="molecule_id",
        task=TaskType.CLASSIFICATION,
    ),
    "tpcc": CTUDatasetDefault(
        target_table="C_Customer",
        target_column="c_credit",
        target_id="c_id",
        timestamp_column="c_since",
        task=TaskType.CLASSIFICATION,
    ),
    "tpcd": CTUDatasetDefault(
        target_table="dss_customer",
        target_column="c_mktsegment",
        target_id="c_custkey",
        task=TaskType.CLASSIFICATION,
        schema_fixer=lambda schema: schema.dss_lineitem.foreign_keys.extend(
            [
                ForeignKeyDef(["l_partkey"], "dss_part", ["p_partkey"]),
                ForeignKeyDef(["l_suppkey"], "dss_supplier", ["s_suppkey"]),
            ]
        ),
    ),
    "tpcds": CTUDatasetDefault(
        target_table="customer",
        target_column="c_preferred_cust_flag",
        target_id="c_customer_sk",
        task=TaskType.CLASSIFICATION,
    ),
    "tpch": CTUDatasetDefault(
        target_table="customer",
        target_column="c_acctbal",
        target_id="c_custkey",
        task=TaskType.REGRESSION,
    ),
    "trains": CTUDatasetDefault(
        target_table="trains",
        target_column="direction",
        target_id="id",
        task=TaskType.CLASSIFICATION,
    ),
    "Triazine": CTUDatasetDefault(
        target_table="molecule",
        target_column="activity",
        target_id="molecule_id",
        task=TaskType.REGRESSION,
    ),
    "university": CTUDatasetDefault(
        target_table="student",
        target_column="intelligence",
        target_id="student_id",
        task=TaskType.CLASSIFICATION,
    ),
    "UTube": CTUDatasetDefault(
        target_table="utube_states",
        target_column="class",
        target_id="id",
        task=TaskType.CLASSIFICATION,
    ),
    "UW_std": CTUDatasetDefault(
        target_table="person",
        target_column="inPhase",
        target_id="p_id",
        task=TaskType.CLASSIFICATION,
    ),
    "VisualGenome": CTUDatasetDefault(
        target_table="IMG_OBJ",
        target_column="OBJ_CLASS_ID",
        target_id="IMG_ID, OBJ_SAMPLE_ID",
        task=TaskType.LINK_PREDICTION,
    ),
    "voc": CTUDatasetDefault(
        target_table="voyages",
        target_column="arrival_harbour",
        target_id="number, number_sup",
        timestamp_column="arrival_date",
        task=TaskType.CLASSIFICATION,
    ),
    "Walmart": CTUDatasetDefault(
        target_table="train",
        target_column="units",
        target_id="store_nbr, item_nbr",
        timestamp_column="date",
        task=TaskType.REGRESSION,
    ),
    "WebKP": CTUDatasetDefault(
        target_table="webpage",
        target_column="class_label",
        target_id="webpage_id",
        task=TaskType.CLASSIFICATION,
    ),
    "world": CTUDatasetDefault(
        target_table="Country",
        target_column="Continent",
        target_id="Code",
        task=TaskType.CLASSIFICATION,
    ),
}
