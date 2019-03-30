###############################################
Getting started with Machine Learning with dabl
###############################################

Let's dive right in!

Let's start with the classic. You have the titanic.csv file and want to predict
whether a passenger survived or not based on the information about the
passenger in that file.
We know, for tabular data like this, pandas is our friend.
Clearly we need to start with loading our data:

    >>> import pandas as pd
    >>> titanic = pd.read_csv("titanic.csv")

Let's familiarize ourself with the data a bit; what's the shape, what are the
columns, what do they look like?

    >>> titanic.shape
    >>> titanic.head()

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

    >>> types = dabl.detect_types_dataframe(titanic_clean)
    >>> print(types)

Having a very rough idea of the shape of our data, we can now start looking
at the actual content. The easiest way to do that is using visualization of
univariate and bivariate patterns. With plot_supervised,
we can create plot of the features deemed most important for our task.

    >>> plot_supervised(titanic, 'survived')

Finally, we can find a good model for our data. The SimpleClassifier does all
the work for us. It implements the familiar scikit-learn api of fit and
predict:

    >>> fc = SimpleClassifier()
    >>> X = titanic_clean.drop("survived", axis=1)
    >>> y = titanic_clean.survived
    >>> fc.fit(X, y)
