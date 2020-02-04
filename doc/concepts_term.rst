.. title:: Concepts and terminology

#############################
Concepts and terminology
#############################
dabl tries to reduce the turnaround time required for a quick baseline estimate
of a supervised learning problem. It does so by automating the task of iterating
through different techniques of data preprocessing, feature engineering,
parameter tuning and model building to generate efficacious baseline models.

Since the process involves automatic iterations, some of the techniques require
heuristics to generate the best possible outcome. dabl also employs the use of
specific terminologies related to datasets, plots or machine learning
techniques.

Column types
------------
One of the most important aspects of preprocessing is understanding the type of
data present in each column of your dataset. The columns could contain integers
that are binary, ordinal or nominal; float; character strings or even missing
and garbage values.

dabl segregates columns into ‘continuous’, ‘categorical’,
‘low_card_int’, ‘dirty_float’, ‘free_string’, ‘date’, and ‘useless’ categories:

-   **low_card_int**: a column with very few distinct integer values

    * Detection: a column with integer values is considered a low cardinality
      integer column if:

      * The number of distinct integers is less than ``max_int_cardinality`` in
        ``detect_types`` method. The column is 'continuous' if number of
        distinct integers is greater than this threshold value.

      * In case of ``plot``, if the rate of change of probability distribution
        curves for a set of distinct integer values in the columns is more than
        the rate of change of distribution curves of randomly shuffled values
        from the same set, we consider those values to be low cardinality
        integers. This is because in case of a higher rate of change, the
        probability distribution curve will be jagged signifying that the
        numbers are not continuous.

      Apart from this, the column is 'categorical' if the number of distinct
      integers is less than or equal to 5.

    * Treatment: all 'low_card_int' columns are treated as 'categorical' and
      'continuous' in preprocessing and model building.
