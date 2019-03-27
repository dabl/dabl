from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   FunctionTransformer)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from warnings import warn

_FLOAT_REGEX = "^[+-]?[0-9]*\.?[0-9]*$"


class DirtyFloatCleaner(BaseEstimator, TransformerMixin):
    # should this error if the inputs are not string?
    def fit(self, X, y=None):
        # FIXME ensure X is dataframe?
        # FIXME clean float columns will make this fail
        encoders = {}
        for col in X.columns:
            floats = X[col].str.match(_FLOAT_REGEX)
            # FIXME sparse
            encoders[col] = OneHotEncoder(sparse=False,
                                          handle_unknown='ignore').fit(
                X.loc[~floats, [col]])
        self.encoders_ = encoders
        self.columns_ = X.columns
        return self

    def transform(self, X):
        # FIXME check that columns are the same?
        result = []
        for col in self.columns_:
            nofloats = ~X[col].str.match(_FLOAT_REGEX)
            new_col = X[col].copy()
            new_col[nofloats] = np.NaN
            new_col = new_col.astype(np.float)
            enc = self.encoders_[col]
            cats = pd.DataFrame(0, index=X.index,
                                columns=enc.get_feature_names([str(col)]))
            if nofloats.any():
                cats.loc[nofloats, :] = enc.transform(X.loc[nofloats, [col]])
            # FIXME use types to distinguish outputs instead?
            cats["{}_dabl_continuous".format(col)] = new_col
            result.append(cats)
        return pd.concat(result, axis=1)


def guess_ordinal(values):
    # compare against http://proceedings.mlr.press/v70/valera17a/valera17a.pdf
    # there's some ways to guess month, day, week, year
    # but even if we have that, is that ordinal or categorical?
    # worst hack in the history of probability distributions, maybe ever
    # we compute second derivatives on the histogram. If they look smoother
    # than the shuffled histograms, we assume order is meaningful
    # why second derivatives? Why absolute norms? Why 1.5? good questions!
    counts = np.bincount(values)

    def norm(x):
        return np.abs(np.diff(np.diff(x))).sum()
    grad_norm = norm(counts)
    # shuffle 100 times
    grad_norm_shuffled = np.mean([
        norm(counts[np.random.permutation(len(counts))]) for i in range(100)])
    return grad_norm * 1.5 < grad_norm_shuffled


def _find_string_floats(X, dirty_float_threshold):
    is_float = X.apply(lambda x: x.str.match(_FLOAT_REGEX))
    clean_float_string = is_float.all()
    # remove 5 most common string values before checking if the rest is float
    # FIXME 5 hardcoded!!
    dirty_float = pd.Series(0, index=X.columns, dtype=bool)
    for col in X.columns:
        if clean_float_string[col]:
            # already know it's clean
            continue
        column = X[col]
        common_values = column.value_counts()[:5].index
        is_common = column.isin(common_values)
        if is_float[col][~is_common].mean() > dirty_float_threshold:
            dirty_float[col] = 1

    return clean_float_string, dirty_float


def detect_types(X, type_hints=None, max_int_cardinality='auto',
                 dirty_float_threshold=.9,
                 near_constant_threshold=.95, verbose=0):
    """Detect types of dataframe columns.

    Columns are labeled as one of the following types:
    'continuous', 'categorical', 'low_card_int', 'dirty_float',
    'free_string', 'date', 'useless'

    Pandas categorical variables, strings and integers of low cardinality and
    float values with two columns are labeled as categorical.
    Integers of high cardinality are labeled as continuous.
    Integers of intermediate cardinality are labeled as "low_card_int".
    Float variables that sometimes take string values are labeled "dirty_float"
    String variables with many unique values are labeled "free_text"
    (and currently not processed by dabl).
    Date types are labeled as "date" (and currently not processed by dabl).
    Anything that is constant, nearly constant, detected as an integer index,
    or doesn't match any of the above categories is labeled "useless".

    Parameters
    ----------
    X : dataframe
        input

    max_int_cardinality: int or 'auto', default='auto'
        Maximum number of distinct integers for an integer column
        to be considered categorical. 'auto' is ``max(42, n_samples/10)``.
        Integers are also always considered as continuous variables.

    dirty_float_threshold : float, default=.9
        The fraction of floats required in a dirty continuous
        column before it's considered "useless" or categorical
        (after removing top 5 string values)

    verbose : int
        How verbose to be

    Returns
    -------
    res : dataframe, shape (n_columns, 7)
        Boolean dataframe of detected types. Rows are columns in input X,
        columns are possible types (see above).
    """
    # FIXME integer indices are not dropped!
    # TODO detect encoding missing values as strings /weird values
    # TODO detect top coding
    # FIXME dirty int is detected as dirty float right now
    # TODO discard all constant and binary columns at the beginning?
    # TODO subsample large datsets? one level up?

    if not X.index.is_unique:
        raise ValueError("Non-unique index found. Reset index or call clean.")
    duplicated = X.columns.duplicated()
    if duplicated.any():
        raise ValueError("Duplicate Columns: {}".format(
            X.columns[duplicated]))
    X = _apply_type_hints(X, type_hints=type_hints)

    n_samples, n_features = X.shape
    if max_int_cardinality == "auto":
        max_int_cardinality = max(42, n_samples / 100)
    # FIXME only apply nunique to non-continuous?
    n_values = X.apply(lambda x: x.nunique())
    if verbose > 3:
        print(n_values)

    binary = n_values == 2
    if type_hints is not None:
        # force binary variables to be continuous
        # if type hints say so
        for k, v in type_hints.items():
            if v == 'continuous':
                binary[k] = False

    dtypes = X.dtypes
    kinds = dtypes.apply(lambda x: x.kind)
    # FIXME use pd.api.type.is_string_dtype etc maybe
    floats = kinds == "f"
    integers = (kinds == "i") | (kinds == "u")
    useless = pd.Series(0, index=X.columns, dtype=bool)

    # check if we have something that trivially is an index
    suspicious_index = (n_values == X.shape[0]) & integers
    if suspicious_index.any():
        warn_for = []
        for c in suspicious_index.index[suspicious_index]:
            if X[c][0] == 0:
                if (X[c] == np.arange(X.shape[0])).all():
                    # definitely an index
                    useless[c] = True
                else:
                    warn_for.append(c)
            elif X[c][0] == 1:
                if (X[c] == np.arange(1, X.shape[0] + 1)).all():
                    # definitely an index
                    useless[c] = True
                else:
                    warn_for.append(c)
        if len(warn_for):
            warn("Suspiciously looks like an index: {}, but unsure,"
                 " so keeping it for now".format(warn_for), UserWarning)
    categorical = dtypes == 'category'
    objects = (kinds == "O") & ~categorical  # FIXME string?
    dates = kinds == "M"
    other = - (floats | integers | objects | dates | categorical)
    # check if we can cast strings to float
    # we don't need to cast all, could so something smarter?
    if objects.any():
        clean_float_string, dirty_float = _find_string_floats(
            X.loc[:, objects], dirty_float_threshold)
    else:
        dirty_float = clean_float_string = pd.Series(0, index=X.columns,
                                                     dtype=bool)

    # using integers or string as categories only if low cardinality
    few_entries = n_values < max_int_cardinality
    # constant features are useless
    useless = (n_values < 2) | useless
    # also throw out near constant:
    most_common_count = X.apply(lambda x: x.value_counts().max())
    near_constant = most_common_count / X.count() > near_constant_threshold
    if near_constant.any():
        warn("Discarding near constant features: {}".format(
             near_constant.index[near_constant].tolist()))
    useless = useless | near_constant
    large_cardinality_int = integers & ~few_entries
    # hard coded very low cardinality integers are categorical
    cat_integers = integers & (n_values <= 5) & ~useless
    low_card_integers = (few_entries & integers
                         & ~binary & ~useless & ~cat_integers)
    non_float_objects = objects & ~dirty_float & ~clean_float_string
    cat_string = few_entries & non_float_objects & ~useless
    free_strings = ~few_entries & non_float_objects
    continuous = floats | large_cardinality_int | clean_float_string

    res = pd.DataFrame(
        {'continuous': continuous & ~binary & ~useless,
         'dirty_float': dirty_float,
         'low_card_int': low_card_integers,
         'categorical': ((cat_string | binary | categorical | cat_integers)
                         & ~useless),
         'date': dates,
         'free_string': free_strings, 'useless': useless,
         })
    # ensure we respected type hints
    if type_hints is not None:
        for k, v in type_hints.items():
            res.loc[k, v] = True
    res = res.fillna(False)
    res['useless'] = res['useless'] | (res.sum(axis=1) == 0)

    assert np.all(res.sum(axis=1) == 1)

    if verbose >= 1:
        print("Detected feature types:")
        desc = "{} float, {} int, {} object, {} date, {} other".format(
            floats.sum(), integers.sum(), objects.sum(), dates.sum(),
            other.sum())
        print(desc)
        print("Interpreted as:")
        print(res.sum())
    if verbose >= 2:
        if dirty_float.any():
            print("WARN Found dirty floats encoded as strings: {}".format(
                dirty_float.index[dirty_float].tolist()
            ))
        if res.useless.sum() > 0:
            print("WARN dropped columns (too many unique values): {}".format(
                res.index[res.useless].tolist()
            ))
    return res


def _apply_type_hints(X, type_hints):
    if type_hints is not None:
        # use type hints to convert columns
        # to possibly avoid some work.
        # means we need to copy X though.
        X = X.copy()
        for k, v in type_hints.items():
            if v == "continuous":
                X[k] = X[k].astype(np.float)
            elif v == "categorical":
                X[k] = X[k].astype('category')
            elif v == 'useless' and k in X.columns:
                X = X.drop(k, axis=1)
    return X


def select_cont(X):
    return X.columns.str.endswith("_dabl_continuous")


def _make_float(X):
    return X.astype(np.float, copy=False)


def clean(X, type_hints=None):
    """Public clean interface

    Parameters
    ----------
    type_hints : dict or None
        If dict, provide type information for columns.
        Keys are column names, values are types as provided by detect_types.
    """
    X = _apply_type_hints(X, type_hints=type_hints)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not X.index.is_unique:
        warn("Index not unique, resetting index!", UserWarning)
        X = X.reset_index()
    types = detect_types(X, type_hints=type_hints)
    for col in types.index[types.categorical]:
        X[col] = X[col].astype('category', copy=False)
    # get rid of dirty floats, add indicators
    X = _safe_clean(X, types=types)
    # deal with low cardinality ints
    for col in types.index[types.low_card_int]:
        # for now let's just check for smooth distributions
        pass
    # ensure that the indicator variables are also marked as categorical
    # we could certainly do this nicer, but at this point calling
    # detect_types shouldn't be expensive any more
    # though if we have actual string columns that are free strings... hum
    types = detect_types(X, type_hints=type_hints)
    for col in types.index[types.categorical]:
        X[col] = X[col].astype('category', copy=False)
    return X


def _safe_clean(X, onehot=False, types=None):
    """Cleaning / preprocessing outside of cross-validation

    FIXME this leads to duplicating integer columns! no good!

    This function is "safe" to use outside of cross-validation in that
    it only does preprocessing that doesn't use statistics that could
    leak information from the test set (like scaling or imputation).

    Using this method prior to analysis can speed up your pipelines.

    The main operation is checking for string-encoded floats.

    Parameters
    ----------
    X : dataframe
        Dirty data
    onehot : boolean, default=False
        Whether to do one-hot encoding of categorical data.

    Returns
    -------
    X_cleaner : dataframe
        Slightly cleaner dataframe.
    """
    if types is None:
        types = detect_types(X)
    res = []
    if types['dirty_float'].any():
        # don't use ColumnTransformer that can't return dataframe yet
        res.append(DirtyFloatCleaner().fit_transform(
            X.loc[:, types['dirty_float']]))
    if types['useless'].any() or types['dirty_float'].any():
        if onehot:
            # FIXME REMOVE THIS HERE
            res.append(X.loc[:, types['continuous']])
            cat_indices = types.index[types['categorical']]
            res.append(pd.get_dummies(X.loc[:, types['categorical']],
                                      columns=cat_indices))
        else:
            good_types = (types.continuous | types.categorical
                          | types.low_card_int | types.date
                          | types.free_string)
            res.append(X.loc[:, good_types])
        return pd.concat(res, axis=1)
    return X


class EasyPreprocessor(BaseEstimator, TransformerMixin):
    """A simple preprocessor

    Detects variable types, encodes everything as floats
    for use with sklearn.

    Applies one-hot encoding, missing value imputation and scaling.

    Attributes
    ----------
    ct_ : ColumnTransformer
        Main container for all transformations.

    columns_ : pandas columns
        Columns of training data

    dtypes_ : Series of dtypes
        Dtypes of training data columns.

    types_ : something
        Inferred input types.


    Parameters
    ----------
    scale : boolean, default=True
        Whether to scale continuous data.

    verbose : int, default=0
        Control output verbosity.

    """
    def __init__(self, scale=True, verbose=0, types=None):
        self.verbose = verbose
        self.scale = scale
        self.types = types

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns_ = X.columns
        self.dtypes_ = X.dtypes
        if self.types is None:
            # FIXME some sanity check?
            types = detect_types(X, verbose=self.verbose)
        else:
            types = self.types

        # go over variable blocks
        # check for missing values
        # scale etc
        pipe_categorical = OneHotEncoder(categories='auto',
                                         handle_unknown='ignore')

        steps_continuous = [FunctionTransformer(_make_float, validate=False)]
        if self.scale:
            steps_continuous.append(StandardScaler())
        # if X.loc[:, types['continuous']].isnull().values.any():
        # FIXME doesn't work if missing values only in dirty column
        steps_continuous.insert(0, SimpleImputer(strategy='median'))
        pipe_continuous = make_pipeline(*steps_continuous)
        # FIXME only have one imputer/standard scaler in all
        # (right now copied in dirty floats and floats)
        pipe_dirty_float = make_pipeline(
            DirtyFloatCleaner(),
            make_column_transformer(
                (pipe_continuous, select_cont), remainder="passthrough"))
        # construct column transformer
        transformer_cols = []
        if types['continuous'].any():
            transformer_cols.append(('continuous',
                                     pipe_continuous, types['continuous']))
        if types['categorical'].any():
            transformer_cols.append(('categorical',
                                     pipe_categorical, types['categorical']))
        if types['dirty_float'].any():
            transformer_cols.append(('dirty_float',
                                     pipe_dirty_float, types['dirty_float']))

        if not len(transformer_cols):
            raise ValueError("No feature columns found")
        self.ct_ = ColumnTransformer(transformer_cols, sparse_threshold=.1)

        self.ct_.fit(X)

        self.input_shape_ = X.shape
        self.types_ = types
        # Return the transformer
        return self

    def get_feature_names(self):
        return self.ct_.get_feature_names()

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['ct_'])
        return self.ct_.transform(X)
