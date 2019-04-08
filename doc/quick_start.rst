###############################################
Quickstart to ML with dabl
###############################################

Let's dive right in!

Let's start with the classic. You have the titanic.csv file and want to predict
whether a passenger survived or not based on the information about the
passenger in that file.
We know, for tabular data like this, pandas is our friend.
Clearly we need to start with loading our data:

    >>> import pandas as pd
    >>> import dabl
    >>> titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))

Let's familiarize ourself with the data a bit; what's the shape, what are the
columns, what do they look like?

    >>> titanic.shape
    (1309, 14)

    >>> titanic.head()
       pclass  survived  ... body                        home.dest
    0       1         1  ...    ?                     St Louis, MO
    1       1         1  ...    ?  Montreal, PQ / Chesterville, ON
    2       1         0  ...    ?  Montreal, PQ / Chesterville, ON
    3       1         0  ...  135  Montreal, PQ / Chesterville, ON
    4       1         0  ...    ?  Montreal, PQ / Chesterville, ON
    <BLANKLINE>
    [5 rows x 14 columns]


So far so good! There's already a bunch of things going on in the data that we
can see here, but let's ask dabl what it thinks by cleaning up the data:

    >>> titanic_clean = dabl.clean(titanic, verbose=0)

This provides us with lots of information about what is happening in the
different columns. In this case, we might have been able to figure this out
quickly from the call to head,
but in larger datasets this might be a bit tricky.
For example we can see that there are several dirty columns with "?" in it.
This is probably a marker for a missing value and we could go back and fix our
parsing of the CSV, but let's try an continue with what dabl is doing
automatically for now.  In dabl, we can also get a best guess of the column
types in a convenient format:

    >>> types = dabl.detect_types(titanic_clean)
    >>> print(types)
                          continuous  dirty_float  ...  free_string  useless
    age_?                      False        False  ...        False    False
    age_dabl_continuous         True        False  ...        False    False
    boat                       False        False  ...        False    False
    body_?                     False        False  ...        False    False
    body_dabl_continuous        True        False  ...        False    False
    cabin                      False        False  ...         True    False
    embarked                   False        False  ...        False    False
    fare_?                     False        False  ...        False     True
    fare_dabl_continuous        True        False  ...        False    False
    home.dest                  False        False  ...         True    False
    name                       False        False  ...         True    False
    parch                      False        False  ...        False    False
    pclass                     False        False  ...        False    False
    sex                        False        False  ...        False    False
    sibsp                      False        False  ...        False    False
    survived                   False        False  ...        False    False
    ticket                     False        False  ...         True    False
    <BLANKLINE>
    [17 rows x 7 columns]


Having a very rough idea of the shape of our data, we can now start looking
at the actual content. The easiest way to do that is using visualization of
univariate and bivariate patterns. With plot_supervised,
we can create plot of the features deemed most important for our task.

    >>> dabl.plot_supervised(titanic, 'survived')
    Target looks like classification
    baseline score: 0.500

.. plot::

    >>> import pandas as pd
    >>> import dabl
    >>> titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))
    >>> dabl.plot_supervised(titanic, 'survived')
    Target looks like classification
    baseline score: 0.500
    >>> import matplotlib.pyplot as plt; plt.show()


Finally, we can find an initial model for our data. The SimpleClassifier does all
the work for us. It implements the familiar scikit-learn api of fit and
predict:

    >>> fc = dabl.SimpleClassifier(random_state=0)
    >>> X = titanic_clean.drop("survived", axis=1)
    >>> y = titanic_clean.survived
    >>> fc.fit(X, y)
    DummyClassifier(random_state=0, strategy='prior')
    accuracy: 0.6180    average_precision: 0.3820    recall_macro: 0.5000    roc_auc: 0.5000    
    new best (using recall_macro):
    accuracy             0.618028
    average_precision    0.381972
    recall_macro         0.500000
    roc_auc              0.500000
    Name: DummyClassifier(random_state=0, strategy='prior'), dtype: float64
    GaussianNB()
    accuracy: 0.8969    average_precision: 0.8705    recall_macro: 0.9021    roc_auc: 0.9190    
    new best (using recall_macro):
    accuracy             0.896903
    average_precision    0.870464
    recall_macro         0.902120
    roc_auc              0.919032
    Name: GaussianNB(), dtype: float64
    MultinomialNB()
    accuracy: 0.8885    average_precision: 0.9810    recall_macro: 0.8911    roc_auc: 0.9849    
    DecisionTreeClassifier(class_weight='balanced', max_depth=1, random_state=0)
    accuracy: 0.9755    average_precision: 0.9540    recall_macro: 0.9714    roc_auc: 0.9714    
    new best (using recall_macro):
    accuracy             0.975540
    average_precision    0.953971
    recall_macro         0.971441
    roc_auc              0.971441
    Name: DecisionTreeClassifier(class_weight='balanced', max_depth=1, random_state=0), dtype: float64
    DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=0)
    accuracy: 0.9572    average_precision: 0.9422    recall_macro: 0.9536    roc_auc: 0.9704    
    DecisionTreeClassifier(class_weight='balanced', min_impurity_decrease=0.01,
                random_state=0)
    accuracy: 0.9755    average_precision: 0.9540    recall_macro: 0.9714    roc_auc: 0.9714    
    LogisticRegression(C=0.1, class_weight='balanced', multi_class='auto',
              random_state=0, solver='lbfgs')
    accuracy: 0.9626    average_precision: 0.9861    recall_macro: 0.9609    roc_auc: 0.9888    
    Best model:
    DecisionTreeClassifier(class_weight='balanced', max_depth=1, random_state=0)
    Best Scores:
    accuracy             0.975540
    average_precision    0.953971
    recall_macro         0.971441
    roc_auc              0.971441
    Name: DecisionTreeClassifier(class_weight='balanced', max_depth=1, random_state=0), dtype: float64
    SimpleClassifier(random_state=0, refit=True, verbose=1)


