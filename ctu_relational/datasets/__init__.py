from relbench.datasets import register_dataset

from .db_dataset import *
from .ctu_datasets import *

register_dataset("ctu-accidents", Accidents)
register_dataset("ctu-adventureworks", AdventureWorks)
register_dataset("ctu-airline", Airline)
register_dataset("ctu-credit", Credit)
register_dataset("ctu-ergastf1", ErgastF1)
register_dataset("ctu-expenditures", Expenditures)
register_dataset("ctu-employee", Employee)
register_dataset("ctu-financial", Financial)
register_dataset("ctu-fnhk", FNHK)
register_dataset("ctu-geneea", Geneea)
register_dataset("ctu-legalacts", LegalActs)
register_dataset("ctu-sap", SAP)
register_dataset("ctu-seznam", Seznam)
register_dataset("ctu-stats", Stats)
register_dataset("ctu-tpcc", TPCC)
register_dataset("ctu-tpcd", TPCD)
register_dataset("ctu-tpcds", TPCDS)
register_dataset("ctu-tpch", TPCH)
register_dataset("ctu-voc", VOC)
register_dataset("ctu-walmart", Walmart)
