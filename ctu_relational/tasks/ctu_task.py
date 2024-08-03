from typing import Optional

from relbench.base import Dataset, EntityTask, RecommendationTask, TaskType

from ctu_relational.datasets import CTUDataset


class CTUDatasetEntityTask(EntityTask):
    entity_col = "__PK__"

    def __init__(self, dataset: CTUDataset, cache_dir: Optional[str] = None):
        super().__init__(dataset, cache_dir)


class CTUDatasetRecommendationTask(RecommendationTask):
    def __init__(self, dataset: CTUDataset, cache_dir: Optional[str] = None):
        super().__init__(dataset, cache_dir)


class AccidentsTask(CTUDatasetEntityTask):
    # entity_col= "id_nesreca"
    entity_table = "nesreca"
    time_col = "cas_nesreca"
    target_col = "klas_nesreca"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    # timedelta: pd.Timedelta
    # metrics: List[Callable[[NDArray, NDArray], float]]
    # num_eval_timestamps: int = 1


class AdventureWorks2014Task(CTUDatasetEntityTask):
    entity_table = "SalesOrderHeader"
    target_col = "TotalDue"
    # entity_col="CustomerID"
    time_col = "OrderDate"
    task_type = TaskType.REGRESSION


class AirlineTask(CTUDatasetEntityTask):
    entity_table = "On_Time_On_Time_Performance_2016_1"
    target_col = "ArrDel15"
    # entity_col="?"
    time_col = "FlightDate"
    task_type = TaskType.BINARY_CLASSIFICATION


class AtherosclerosisTask(CTUDatasetEntityTask):
    entity_table = "Entry"
    target_col = "STAV"
    # entity_col="ICO"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class AustralianFootballTask(CTUDatasetEntityTask):
    entity_table = "matches"
    target_col = "target"
    # entity_col="mid"
    time_col = "start_dt"
    task_type = TaskType.BINARY_CLASSIFICATION


class BasketballMenTask(CTUDatasetEntityTask):
    entity_table = "teams"
    target_col = "rank"
    # entity_col="tmID, year"
    time_col = "year"
    task_type = TaskType.REGRESSION


class BasketballWomenTask(CTUDatasetEntityTask):
    entity_table = "teams"
    target_col = "playoff"
    # entity_col="tmID, year"
    time_col = "year"
    task_type = TaskType.BINARY_CLASSIFICATION


class BiodegradabilityTask(CTUDatasetEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    # entity_col="molecule_id"
    task_type = TaskType.REGRESSION


# TODO: Fix access to the database
class BupaTask(CTUDatasetEntityTask):
    entity_table = "bupa"
    target_col = "arg2"
    # entity_col="arg1"
    task_type = TaskType.CLASSIFICATION


class CarcinogenesisTask(CTUDatasetEntityTask):
    entity_table = "canc"
    target_col = "class"
    # entity_col="drug_id"
    task_type = TaskType.BINARY_CLASSIFICATION


# TODO: Fix FKs in the database
class CCSTask(CTUDatasetEntityTask):
    entity_table = "transactions_1k"
    target_col = "Price"
    # entity_col="TransactionID"
    time_col = "Date"
    task_type = TaskType.REGRESSION


# TODO: Fix access to the database
class CDESchoolsTask(CTUDatasetEntityTask):
    entity_table = "satscores"
    target_col = "PctGE1500"
    # entity_col="cds"
    task_type = TaskType.REGRESSION


class ChessTask(CTUDatasetEntityTask):
    entity_table = "game"
    target_col = "game_result"
    # entity_col="game_id"
    time_col = "event_date"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


# TODO: Fix FKs in the database
class CiteSeerTask(CTUDatasetEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    # entity_col="paper_id"
    task_type = TaskType.CLASSIFICATION
    #     ForeignKeyDef(["cited_paper_id"], "paper", ["paper_id"])
    #     ForeignKeyDef(["citing_paper_id"], "paper", ["paper_id"])


class ClassicmodelsTask(CTUDatasetEntityTask):
    entity_table = "payments"
    target_col = "amount"
    # entity_col="checkNumber"
    time_col = "paymentDate"
    task_type = TaskType.REGRESSION


class ConsumerExpendituresTask(CTUDatasetEntityTask):
    entity_table = "EXPENDITURES"
    target_col = "GIFT"
    # entity_col="EXPENDITURE_ID"
    task_type = TaskType.REGRESSION


class CORATask(CTUDatasetEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    # entity_col="paper_id"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CountriesTask(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "2012"
    # entity_col="Country Code"
    task_type = TaskType.REGRESSION


class CraftBeerTask(CTUDatasetEntityTask):
    entity_table = "breweries"
    target_col = "state"
    # entity_col="id"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CreditTask(CTUDatasetEntityTask):
    entity_table = "member"
    target_col = "region_no"
    # entity_col="member_no"
    time_col = "issue_dt"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CSTask(CTUDatasetEntityTask):
    entity_table = "target_churn"
    target_col = "target_churn"
    # entity_col="ACC_KEY"
    time_col = "date_horizon"
    task_type = TaskType.BINARY_CLASSIFICATION


class DallasTask(CTUDatasetEntityTask):
    entity_table = "incidents"
    target_col = "subject_statuses"
    # entity_col="case_number"
    time_col = "date"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class DCGTask(CTUDatasetEntityTask):
    entity_table = "sentences"
    target_col = "class"
    # entity_col="id"
    task_type = TaskType.BINARY_CLASSIFICATION


class DunurTask(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "is_dunur"
    # entity_col="name1, name2"
    task_type = TaskType.BINARY_CLASSIFICATION


class EltiTask(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "is_elti"
    # entity_col="name1, name2"
    task_type = TaskType.BINARY_CLASSIFICATION


class employeeTask(CTUDatasetEntityTask):
    entity_table = "salaries"
    target_col = "salary"
    # entity_col="emp_no"
    time_col = "from_date"
    task_type = TaskType.REGRESSION


class ErgastF1Task(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "win"
    # entity_col="targetId"
    time_col = "raceId"
    task_type = TaskType.BINARY_CLASSIFICATION


class FacebookTask(CTUDatasetEntityTask):
    entity_table = "feat"
    target_col = "gender1"
    # entity_col="id"
    task_type = TaskType.BINARY_CLASSIFICATION


class FinancialTask(CTUDatasetEntityTask):
    entity_table = "loan"
    target_col = "status"
    # entity_col="account_id"
    time_col = "date"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class FNHKTask(CTUDatasetEntityTask):
    entity_table = "pripady"
    target_col = "Delka_hospitalizace"
    # entity_col="Identifikace_pripadu"
    time_col = "Datum_prijeti"
    task_type = TaskType.REGRESSION


class FTPTask(CTUDatasetEntityTask):
    entity_table = "session"
    target_col = "gender"
    # entity_col="session_id"
    task_type = TaskType.BINARY_CLASSIFICATION


class GeneeaTask(CTUDatasetEntityTask):
    entity_table = "hl_hlasovani"
    target_col = "vysledek"
    # entity_col="id_hlasovani"
    time_col = "datum"
    task_type = TaskType.BINARY_CLASSIFICATION
    # force_collation="utf8mb3_unicode_ci"


class GenesTask(CTUDatasetEntityTask):
    entity_table = "Classification"
    target_col = "Localization"
    # entity_col="GeneID"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class GOSalesTask(CTUDatasetEntityTask):
    entity_table = "go_1k"
    target_col = "Quantity"
    # entity_col="Retailer code, Product number"
    time_col = "Date"
    task_type = TaskType.REGRESSION


class GrantsTask(CTUDatasetEntityTask):
    entity_table = "awards"
    target_col = "award_amount"
    # entity_col="award_id"
    time_col = "award_effective_date"
    task_type = TaskType.REGRESSION


class HepatitisSTDTask(CTUDatasetEntityTask):
    entity_table = "dispat"
    target_col = "Type"
    # entity_col="m_id"
    task_type = TaskType.BINARY_CLASSIFICATION


class HockeyTask(CTUDatasetEntityTask):
    entity_table = "Master"
    target_col = "shootCatch"
    # entity_col="playerId"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class IMDBIJSTask(CTUDatasetEntityTask):
    entity_table = "actors"
    target_col = "gender"
    # entity_col="?"
    task_type = TaskType.BINARY_CLASSIFICATION
    # db_distinct_counter="fetchall_unidecode_strip_ci"


class KRKTask(CTUDatasetEntityTask):
    entity_table = "krk"
    target_col = "class"
    # entity_col="id"
    task_type = TaskType.BINARY_CLASSIFICATION


class Lahman2014Task(CTUDatasetEntityTask):
    entity_table = "salaries"
    target_col = "salary"
    # entity_col="teamID, playerID, lgID"
    time_col = "yearID"
    task_type = TaskType.REGRESSION


class LegalActsTask(CTUDatasetEntityTask):
    entity_table = "legalacts"
    target_col = "ActKind"
    # entity_col="id"
    time_col = "update"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    # db_distinct_counter="fetchall_unidecode_strip_ci"


class MeshTask(CTUDatasetEntityTask):
    entity_table = "mesh"
    target_col = "num"
    # entity_col="name"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class MondialTask(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "Target"
    # entity_col="Country"
    task_type = TaskType.BINARY_CLASSIFICATION


# TODO: what should be predicted?
class MooneyFamilyTask(CTUDatasetRecommendationTask):
    entity_table = "uncle"
    target_col = "?"
    # entity_col="name1, name2"
    task_type = TaskType.LINK_PREDICTION


class IMDBMovieLensTask(CTUDatasetEntityTask):
    entity_table = "users"
    target_col = "u_gender"
    # entity_col="userid"
    task_type = TaskType.BINARY_CLASSIFICATION


class MedicalTask(CTUDatasetEntityTask):
    entity_table = "Examination"
    target_col = "Thrombosis"
    # entity_col="ID"
    time_col = "Examination Date"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class MuskSmallTask(CTUDatasetEntityTask):
    entity_table = "molecule"
    target_col = "class"
    # entity_col="molecule_name"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class MutagenesisTask(CTUDatasetEntityTask):
    entity_table = "molecule"
    target_col = "mutagenic"
    # entity_col="molecule_id"
    task_type = TaskType.BINARY_CLASSIFICATION


class NationsTask(CTUDatasetEntityTask):
    entity_table = "stat"
    target_col = "femaleworkers"
    # entity_col="country_id"
    task_type = TaskType.BINARY_CLASSIFICATION


class NBATask(CTUDatasetEntityTask):
    entity_table = "Game"
    target_col = "ResultOfTeam1"
    # entity_col="GameId"
    time_col = "Date"
    task_type = TaskType.BINARY_CLASSIFICATION


class NCAATask(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "team_id1_wins"
    # entity_col="id"
    time_col = "season"
    task_type = TaskType.BINARY_CLASSIFICATION


class northwindTask(CTUDatasetEntityTask):
    entity_table = "Orders"
    target_col = "Freight"
    # entity_col="OrderID"
    time_col = "OrderId"
    task_type = TaskType.REGRESSION


class PimaTask(CTUDatasetEntityTask):
    entity_table = "pima"
    target_col = "arg2"
    # entity_col="arg1"
    task_type = TaskType.BINARY_CLASSIFICATION


class PremierLeagueTask(CTUDatasetEntityTask):
    entity_table = "Matches"
    target_col = "ResultOfTeamHome"
    # entity_col="MatchID"
    time_col = "MatchDate"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class PTETask(CTUDatasetEntityTask):
    entity_table = "pte_active"
    target_col = "is_active"
    # entity_col="drug_id"
    task_type = TaskType.BINARY_CLASSIFICATION


class PubMed_DiabetesTask(CTUDatasetEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    # entity_col="paper_id"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class pubsTask(CTUDatasetEntityTask):
    entity_table = "titles"
    target_col = "ytd_sales"
    # entity_col="title_id"
    time_col = "pubdate"
    task_type = TaskType.REGRESSION


class PyrimidineTask(CTUDatasetEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    # entity_col="molecule_id"
    task_type = TaskType.REGRESSION


class restbaseTask(CTUDatasetEntityTask):
    entity_table = "generalinfo"
    target_col = "review"
    # entity_col="id_restaurant"
    task_type = TaskType.REGRESSION


class sakilaTask(CTUDatasetEntityTask):
    entity_table = "payment"
    target_col = "amount"
    # entity_col="payment_id"
    time_col = "payment_date"
    task_type = TaskType.REGRESSION


class SalesDBTask(CTUDatasetEntityTask):
    entity_table = "Sales"
    target_col = "Quantity"
    # entity_col="SalesID"
    task_type = TaskType.REGRESSION


class SameGenTask(CTUDatasetEntityTask):
    entity_table = "target"
    target_col = "target"
    # entity_col="name1, name2"
    task_type = TaskType.BINARY_CLASSIFICATION


class SAPTask(CTUDatasetEntityTask):
    entity_table = "Mailings1_2"
    target_col = "RESPONSE"
    # entity_col="REFID"
    time_col = "REF_DATE"
    task_type = TaskType.BINARY_CLASSIFICATION


class SATTask(CTUDatasetEntityTask):
    entity_table = "fault"
    target_col = "tf"
    # entity_col="?"
    time_col = "tm"
    task_type = TaskType.BINARY_CLASSIFICATION


class SeznamTask(CTUDatasetEntityTask):
    entity_table = "probehnuto"
    target_col = "kc_proklikano"
    # entity_col="client_id, sluzba"
    time_col = "month_year_datum_transakce"
    task_type = TaskType.REGRESSION


class SFScoresTask(CTUDatasetEntityTask):
    entity_table = "inspections"
    target_col = "score"
    # entity_col="business_id"
    time_col = "date"
    task_type = TaskType.REGRESSION


# TODO: what should be predicted?
class ShakespeareTask(CTUDatasetRecommendationTask):
    entity_table = "paragraphs"
    target_col = "character_id"
    # entity_col="id"
    task_type = TaskType.LINK_PREDICTION


class StatsTask(CTUDatasetEntityTask):
    entity_table = "users"
    target_col = "Reputation"
    # entity_col="Id"
    time_col = "LastAccessDate"
    task_type = TaskType.REGRESSION


class StudentLoanTask(CTUDatasetEntityTask):
    entity_table = "no_payment_due"
    target_col = "bool"
    # entity_col="name"
    task_type = TaskType.BINARY_CLASSIFICATION


class ToxicologyTask(CTUDatasetEntityTask):
    entity_table = "molecule"
    target_col = "label"
    # entity_col="molecule_id"
    task_type = TaskType.CLASSIFICATION


class TPCCTask(CTUDatasetEntityTask):
    entity_table = "C_Customer"
    target_col = "c_credit"
    # entity_col="c_id"
    time_col = "c_since"
    task_type = TaskType.BINARY_CLASSIFICATION


# TODO: Fix FKs in the database
class TPCDTask(CTUDatasetEntityTask):
    entity_table = "dss_customer"
    target_col = "c_mktsegment"
    # entity_col="c_custkey"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    # ForeignKeyDef(["l_partkey"], "dss_part", ["p_partkey"])
    # ForeignKeyDef(["l_suppkey"], "dss_supplier", ["s_suppkey"])


class TPCDSTask(CTUDatasetEntityTask):
    entity_table = "customer"
    target_col = "c_preferred_cust_flag"
    # entity_col="c_customer_sk"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class TPCHTask(CTUDatasetEntityTask):
    entity_table = "customer"
    target_col = "c_acctbal"
    # entity_col="c_custkey"
    task_type = TaskType.REGRESSION


class TrainsTask(CTUDatasetEntityTask):
    entity_table = "trains"
    target_col = "direction"
    # entity_col="id"
    task_type = TaskType.BINARY_CLASSIFICATION


class TriazineTask(CTUDatasetEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    # entity_col="molecule_id"
    task_type = TaskType.REGRESSION


class UniversityTask(CTUDatasetEntityTask):
    entity_table = "student"
    target_col = "intelligence"
    # entity_col="student_id"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class UTubeTask(CTUDatasetEntityTask):
    entity_table = "utube_states"
    target_col = "class"
    # entity_col="id"
    task_type = TaskType.BINARY_CLASSIFICATION


class UWSTDTask(CTUDatasetEntityTask):
    entity_table = "person"
    target_col = "inPhase"
    # entity_col="p_id"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


# TODO: what should be predicted?
class VisualGenomeTask(CTUDatasetRecommendationTask):
    entity_table = "IMG_OBJ"
    target_col = "OBJ_CLASS_ID"
    # entity_col="IMG_ID, OBJ_SAMPLE_ID"
    task_type = TaskType.LINK_PREDICTION


# TODO: what should be predicted?
class VOCTask(CTUDatasetRecommendationTask):
    entity_table = "voyages"
    target_col = "arrival_harbour"
    # entity_col="number, number_sup"
    time_col = "arrival_date"
    task_type = TaskType.LINK_PREDICTION


class WalmartTask(CTUDatasetEntityTask):
    entity_table = "train"
    target_col = "units"
    # entity_col="store_nbr, item_nbr"
    time_col = "date"
    task_type = TaskType.REGRESSION


class WebKPTask(CTUDatasetEntityTask):
    entity_table = "webpage"
    target_col = "class_label"
    # entity_col="webpage_id"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class WorldTask(CTUDatasetEntityTask):
    entity_table = "Country"
    target_col = "Continent"
    # entity_col="Code"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
