# CTU Relational

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relational.fel.cvut.cz)
[![PyPI version](https://img.shields.io/pypi/v/ctu-relational?color=brightgreen)](https://pypi.org/project/ctu-relational/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

The CTU Prague Relational Learning Repository was originally published in [2015](https://arxiv.org/abs/1511.03086v1) with a goal to support machine learning research with multi-relational data. Today, the repository is hosted on https://relational.fel.cvut.cz and contains more than [80](https://relational.fel.cvut.cz/statistics) different datasets stored in SQL databases.

The [RelBench](https://github.com/snap-stanford/relbench) project is currently seeking a similar goal of establishing the Relational Deep Learning as a new subfield of deep learning. The goal of this library is to support the effort of RelBench team by providing the CTU Relational datasets in the standardized representation. As such, the library is an extension of the RelBench package.

## Installation

You can install CTU Relational package through pip:

```bash
pip install ctu-relational
```

## Contents

> :warning: The package is currenly in the development and contain only a subset of all available datasets. Rest will be added in the near future together with asociated tasks.

You can load datasets in same way as in the RelBench, e.g.:

```python
from relbench.datasets import get_dataset
import ctu_relational

dataset = get_dataset('ctu-seznam') # automatically cached through the relbench package
db = dataset.get_db()
```

or directly from CTU Relational:

```python
from ctu_relational import datasets as ctu_datasets

dataset = ctu_datasets.Seznam() # custom cache directory should be specified
db = dataset.get_db()
```

As opposed to the RelBench package, CTU Relational works directly with relational databases through the [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy?tab=readme-ov-file#sqlalchemy) package. `DBDataset` class provides a way of loading an SQL database in the RelBench format. You can load data from your SQL server with the following snippet.

```python
from ctu_relational.datasets import DBDataset

custom_dataset = DBDataset(
            dialect="mariadb", # other dialects should be supported but weren't tested
            driver="mysqlconnector",
            user=<user>,
            password=<password>,
            host=<host_url>,
            port=3306,
            database=<database_name>
        )

db = custom_dataset.get_db(upto_test_timestamp=False)
```

Although, directly loaded databases usually need some additional touches. Take a look at [`ctu_datasets.py`](https://github.com/jakubpeleska/ctu-relational-py/blob/d666c3694c10d3702a917db2fa162e2b259e6546/ctu_relational/datasets/ctu_datasets.py) for examples.
