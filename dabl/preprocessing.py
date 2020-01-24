from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import warnings
from warnings import warn

_FLOAT_REGEX = r"^[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))$"


class DirtyFloatCleaner(BaseEstimator, TransformerMixin):
    # should this error if the inputs are not string?
    def fit(self, X, y=None):
        # FIXME clean float columns will make this fail
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe. Convert or call `clean`.")
        encoders = {}
        for col in X.columns:
            floats = X[col].str.match(_FLOAT_REGEX)
            # FIXME sparse
            if (~floats).any():
                encoders[col] = OneHotEncoder(sparse=False,
                                              handle_unknown='ignore').fit(
                    X.loc[~floats, [col]])
            else:
                encoders[col] = None
        self.encoders_ = encoders
        self.columns_ = X.columns
        return self

    def transform(self, X):
        if (self.columns_ == X.columns).all() is False:
            raise ValueError("Given the same columns")
        result = []
        for col in self.columns_:
            nofloats = ~X[col].str.match(_FLOAT_REGEX)
            new_col = X[col].copy()
            new_col[nofloats] = np.NaN
            new_col = new_col.astype(np.float)
            enc = self.encoders_[col]
            if enc is None:
                if nofloats.any():
                    warnings.warn(
                        "Found non-floats {} in float column. It's "
                        "recommended"
                        " to call 'clean' on the whole dataset before "
                        "splitting into training and test set.".format(
                            X.loc[nofloats, col].unique()))
                new_col = new_col.rename("{}_dabl_continuous".format(col))
                result.append(new_col)
                continue
            cats = pd.DataFrame(0, index=X.index,
                                columns=enc.get_feature_names([str(col)]))
            if nofloats.any():
                cats.loc[nofloats, :] = enc.transform(X.loc[nofloats, [col]])
            cats["{}_dabl_continuous".format(col)] = new_col
            result.append(cats)
        return pd.concat(result, axis=1)

    def get_feature_names(self, input_features=None):
        feature_names = []
        for col in self.columns_:
            enc = self.encoders_[col]
            feature_names.extend(enc.get_feature_names([str(col)]))
            feature_names.append("{}_dabl_continuous".format(col))
        return feature_names


def guess_ordinal(values):
    # compare against http://proceedings.mlr.press/v70/valera17a/valera17a.pdf
    # there's some ways to guess month, day, week, year
    # but even if we have that, is that ordinal or categorical?
    # worst hack in the history of probability distributions, maybe ever
    # we compute second derivatives on the histogram. If they look smoother
    # than the shuffled histograms, we assume order is meaningful
    # why second derivatives? Why absolute norms? Why 1.5? good questions!
    if values.min() < 0:
        # we assume that negative numbers imply an ordering, not categories
        # probably needs testing
        return True
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
    non_string = is_float.isna()
    clean_float_string = is_float.all() & ~non_string.any()
    # remove 5 most common string values before checking if the rest is float
    # FIXME 5 hardcoded!!
    dirty_float = pd.Series(0, index=X.columns, dtype=bool)
    for col in X.columns:

        if clean_float_string[col]:
            # already know it's clean
            continue
        column = X[col]
        non_str = non_string[col]
        if non_str.any():
            # some non-strings in this column
            try:
                column_float = column[non_str].astype(np.float)
                # missing values are not counted as float
                # because they could be categorical as well
                is_float.loc[non_str, col] = ~column_float.isna()
            except ValueError:
                # FIXME of some are not float-able we assume
                # they all are not
                # it's unclear whether iteration manually is worth is
                is_float.loc[non_str, col] = False
        common_values = column.value_counts()[:5].index
        is_common = column.isin(common_values)
        if is_float.loc[~is_common, col].mean() > dirty_float_threshold:
            dirty_float[col] = 1

    return clean_float_string, dirty_float


def _float_col_is_int(series):
    # test on a small subset for speed
    # yes, a recursive call would be one line shorter.
    if series[:10].isna().any():
        return False
    if (series[:10] != series[:10].astype(int)).any():
        return False
    if series.isna().any():
        return False
    if (series != series.astype(int)).any():
        return False
    return True


def detect_types(X, type_hints=None, max_int_cardinality='auto',
                 dirty_float_threshold=.9,
                 near_constant_threshold=0.95, verbose=0):
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
        FIXME not true any more?

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
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X is not a dataframe. Convert or call `clean`.")
    if not X.index.is_unique:
        raise ValueError("Non-unique index found. Reset index or call clean.")
    duplicated = X.columns.duplicated()
    if duplicated.any():
        raise ValueError("Duplicate Columns: {}".format(
            X.columns[duplicated]))
    if type_hints is None:
        type_hints = dict()
    # apply type hints drops useless columns,
    # but in the end we want to check against original columns
    X_org = X
    X = _apply_type_hints(X, type_hints=type_hints)

    n_samples, n_features = X.shape
    if max_int_cardinality == "auto":
        max_int_cardinality = max(42, n_samples / 100)
    # FIXME only apply nunique to non-continuous?
    n_values = X.apply(lambda x: x.nunique())
    if verbose > 3:
        print(n_values)

    binary = n_values == 2
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
    # check if float column is actually all integers
    # we'll treat them as int for now.
    for col, isfloat in floats.items():
        if isfloat and (col not in type_hints
                        or type_hints[col] != "continuous"):
            if _float_col_is_int(X[col]):
                # it's int!
                integers[col] = True
                floats[col] = False

    useless = pd.Series(0, index=X.columns, dtype=bool)

    # check if we have something that trivially is an index
    suspicious_index = (n_values == X.shape[0]) & integers
    if suspicious_index.any():
        warn_for = []
        for c in suspicious_index.index[suspicious_index]:
            if X[c].iloc[0] == 0:
                if (X[c] == np.arange(X.shape[0])).all():
                    # definitely an index
                    useless[c] = True
                else:
                    warn_for.append(c)
            elif X[c].iloc[0] == 1:
                if (X[c] == np.arange(1, X.shape[0] + 1)).all():
                    # definitely an index
                    useless[c] = True
                else:
                    warn_for.append(c)
        if warn_for:
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
    near_constant = pd.Series(0, index=X.columns, dtype=bool)
    for col in X.columns:
        count = X[col].count()
        if n_values[col] / count > .9:
            # save some computation
            continue
        if X[col].value_counts().max() / count > near_constant_threshold:
            near_constant[col] = True
    if near_constant.any():
        warn("Discarding near-constant features: {}".format(
             near_constant.index[near_constant].tolist()))
    useless = useless | near_constant
    for k, v in type_hints.items():
        if v != "useless" and useless[k]:
            useless[k] = False
    large_cardinality_int = integers & ~few_entries
    # hard coded very low cardinality integers are categorical
    cat_integers = integers & (n_values <= 5) & ~useless
    low_card_integers = (few_entries & integers
                         & ~binary & ~useless & ~cat_integers)
    non_float_objects = objects & ~dirty_float & ~clean_float_string
    cat_string = few_entries & non_float_objects & ~useless
    free_strings = ~few_entries & non_float_objects
    continuous = floats | large_cardinality_int | clean_float_string
    categorical = cat_string | binary | categorical | cat_integers
    res = pd.DataFrame(
        {'continuous': continuous & ~binary & ~useless & ~categorical,
         'dirty_float': dirty_float,
         'low_card_int': low_card_integers,
         'categorical': categorical & ~useless,
         'date': dates,
         'free_string': free_strings, 'useless': useless,
         })
    # ensure we respected type hints
    for k, v in type_hints.items():
        res.loc[k, v] = True
    res = res.fillna(False)
    res['useless'] = res['useless'] | (res.sum(axis=1) == 0)
    # reorder res to have the same order as X.columns
    res = res.loc[X_org.columns]
    assert (X_org.columns == res.index).all()

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
            print("WARN dropped useless columns: {}".format(
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


def _select_cont(X):
    return X.columns.str.endswith("_dabl_continuous")


def _make_float(X):
    return X.astype(np.float, copy=False)


def clean(X, type_hints=None, return_types=False, verbose=0):
    """Public clean interface

    Parameters
    ----------
    type_hints : dict or None
        If dict, provide type information for columns.
        Keys are column names, values are types as provided by detect_types.
    return_types : bool, default=False
        Whether to return the inferred types
    verbose : int, default=0
        Verbosity control.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = _apply_type_hints(X, type_hints=type_hints)

    if not X.index.is_unique:
        warn("Index not unique, resetting index!", UserWarning)
        X = X.reset_index(drop=True)
    types = detect_types(X, type_hints=type_hints, verbose=verbose)
    # drop useless columns
    X = X.loc[:, ~types.useless].copy()
    types = types[~types.useless]
    for col in types.index[types.categorical]:
        X[col] = X[col].astype('category', copy=False)

    if types['dirty_float'].any():
        # don't use ColumnTransformer that can't return dataframe yet
        X_df = DirtyFloatCleaner().fit_transform(
            X.loc[:, types['dirty_float']])
        X = pd.concat([X.loc[:, ~types.dirty_float], X_df], axis=1)
        # we should know what these are but maybe running this again is fine?
        types_df = detect_types(X_df)
        types = pd.concat([types[~types.dirty_float], types_df])
    # deal with low cardinality ints
    # TODO ?
    # ensure that the indicator variables are also marked as categorical
    # we could certainly do this nicer, but at this point calling
    # detect_types shouldn't be expensive any more
    # though if we have actual string columns that are free strings... hum
    for col in types.index[types.categorical]:
        # ensure categories are strings, otherwise imputation might fail
        col_as_cat = X[col].astype('category', copy=False)
        if col_as_cat.cat.categories.astype("str").is_unique:
            # the world is good: converting to string keeps categories unique
            X[col] = col_as_cat.cat.rename_categories(
                lambda x: str(x))
        else:
            # we can't have nice things and need to convert to string
            # before making categories (again)
            warn("Duplicate categories of different types in column "
                 "{} considered equal {}".format(
                    col, col_as_cat.cat.categories))
            X[col] = X[col].astype(str).astype('category', copy=False)

    if return_types:
        return X, types
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

    force_imputation : bool, default=True
        Whether to create imputers even if not training data is missing.

    verbose : int, default=0
        Control output verbosity.

    """
    def __init__(self, scale=True, force_imputation=True, verbose=0,
                 types=None):
        self.verbose = verbose
        self.scale = scale
        self.types = types
        self.force_imputation = force_imputation

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

        types = types.copy()
        # low card int encoded as categorical and continuous for now:
        types.loc[types.low_card_int, 'continuous'] = True
        types.loc[types.low_card_int, 'categorical'] = True

        # go over variable blocks
        # check for missing values
        # scale etc
        steps_categorical = []
        if (self.force_imputation
                or X.loc[:, types.categorical].isna().any(axis=None)):
            steps_categorical.append(
                SimpleImputer(strategy='most_frequent', add_indicator=True))
        steps_categorical.append(
            OneHotEncoder(categories='auto', handle_unknown='ignore',
                          sparse=False))
        pipe_categorical = make_pipeline(*steps_categorical)

        steps_continuous = []
        if (self.force_imputation
                or X.loc[:, types.continuous].isna().any(axis=None)
                or types['dirty_float'].any()):
            # we could skip the imputer here, but if there's dirty
            # floats, they'll have NaN, and we reuse the cont pipeline
            steps_continuous.append(SimpleImputer(strategy='median'))
        if self.scale:
            steps_continuous.append(StandardScaler())
        # if X.loc[:, types['continuous']].isnull().values.any():
        # FIXME doesn't work if missing values only in dirty column
        pipe_continuous = make_pipeline(*steps_continuous)
        # FIXME only have one imputer/standard scaler in all
        # (right now copied in dirty floats and floats)

        pipe_dirty_float = make_pipeline(
            DirtyFloatCleaner(),
            make_column_transformer(
                (pipe_continuous, _select_cont), remainder="passthrough"))
        # construct column transformer
        transformer_cols = []
        if types['continuous'].any():
            transformer_cols.append(('continuous',
                                     pipe_continuous, types['continuous']))
        if types['categorical'].any():
            transformer_cols.append(('categorical',
                                     pipe_categorical, types['categorical']))
        if types['dirty_float'].any():
            # FIXME we're not really handling this here any more? (yes we are)
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
        # this can go soon hopefully
        feature_names = []
        for name, trans, cols in self.ct_.transformers_:
            if name == "continuous":
                # three should be no all-nan columns in the imputer
                if (trans.steps[0][0] == "simpleimputer"
                        and np.isnan(trans.steps[0][1].statistics_).any()):
                    raise ValueError("So unexpected! Looks like the imputer"
                                     " dropped some all-NaN columns."
                                     "Try calling 'clean' on your data first.")
                feature_names.extend(cols.index[cols])
            elif name == 'categorical':
                # this is the categorical pipe, extract one hot encoder
                ohe = trans.steps[-1][1]
                # FIXME that is really strange?!
                ohe_cols = self.columns_[self.columns_.map(cols)]
                feature_names.extend(ohe.get_feature_names(ohe_cols))
            elif name == "remainder":
                assert trans == "drop"
            elif name == "dirty_float":
                raise ValueError(
                    "Can't compute feature names when handling dirty floats. "
                    "Call 'clean' as a workaround")
            else:
                raise ValueError(
                    "Can't compute feature names for {}".format(name))
        return feature_names

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
        with warnings.catch_warnings():
            # fix when requiring sklearn 0.22
            # check_is_fitted will not have arguments any more
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            check_is_fitted(self, ['ct_'])
        return self.ct_.transform(X)
