from typing import Literal, Optional

from .db_dataset import DBDataset

__ALL__ = ["CTUDataset", "CTUDatabaseName"]

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
    'PremierLeague', 'PTC', 'PTE', 'PubMed_Diabetes', 'pubs', 'Pyrimidine', 'restbase',
    'sakila', 'SalesDB', 'Same_gen', 'SAP', 'SAT', 'Seznam', 'SFScores', 'Shakespeare',
    'stats', 'Student_loan', 'Toxicology', 'tpcc', 'tpcd', 'tpcds', 'tpch', 'trains',
    'Triazine', 'university', 'UTube', 'UW_std', 'VisualGenome', 'voc', 'Walmart','WebKP',
    'world'
]
# fmt: on


class CTUDataset(DBDataset):
    def __init__(self, database: CTUDatabaseName, cache_dir: Optional[str] = None):
        super().__init__(
            cache_dir=cache_dir,
            dialect="mariadb",
            driver="mysqlconnector",
            user="guest",
            password="ctu-relational",
            host="relational.fel.cvut.cz",
            port=3306,
            database=database,
        )
