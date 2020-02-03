.. _data_cleaning_user_guide:

Data cleaning
-------------
>>> import dabl
>>> import pandas as pd
>>> data = pd.read_csv(dabl.datasets.data_path("adult.csv.gz"))
>>> data_clean = dabl.clean(data)[::10]

The first step in any data analysis is data cleaning. dabl tries to detect the
types of your data and apply appropriate conversions.  It also tries to detect
potential data quality issues.
The field of data cleaning is impossibly broad, and dabl's approaches are by no
means sophisticated.  The goal of dabl is to get the data "clean enough" to
create useful visualizations and models, and to allow users to perform
custom cleaning operations themselves.
In particular if the detection of semantic types (continuous, categorical,
ordinal, text, etc) fails, the user can provide ``type_hints``:

>>> data_clean = dabl.clean(data, type_hints={"capital-gain": "continuous"})

Types of feature columns
^^^^^^^^^^^^^^^^^^^^^^^^

dabl uses certain heuristics to automatically segregate columns into ‘continuous’,
‘categorical’, ‘low_card_int’, ‘dirty_float’, ‘free_string’, ‘date’, and
‘useless’ categories. The ``detect_types`` method accomplishes this task.

While 'continuous' and 'categorical' are self explanatory, other feature types are:

-   low_card_int: a column with integer values is considered a low cardinality integer
    column if the number of distinct integers is less than ``max_int_cardinality``.
    The column is 'continuous' if number of distinct integers is greater than this
    threshold value. Apart from this, the column is 'categorical' if the number of
    distinct integers is less than or equal to ``5``.



