from joblib import hash
import warnings
from warnings import warn

from dateutil.parser import ParserError

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

_FLOAT_REGEX = r"^[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))$"
_FLOAT_MATCHING_CACHE = {}
_MIXED_TYPE_WARNINGS = {}


def _float_matching(X_col, return_safe_col=False):
    is_floaty = X_col.str.match(_FLOAT_REGEX)
    # things that weren't strings
    not_strings = is_floaty.isna()
    if not_strings.any():
        rest = X_col[not_strings]
        all_castable = False
        try:
            # if we can convert them all to float we're done
            rest.astype(float)
            is_floaty[not_strings] = True
            all_castable = True
        except ValueError:
            pass
        if not all_castable:
            if X_col.name not in _MIXED_TYPE_WARNINGS:
                warn(f'Mixed types in column {X_col.name}')
                _MIXED_TYPE_WARNINGS[X_col.name] = True
            # make everything string
            rest = rest.astype(str)
            rest_is_floaty = _float_matching(rest)
            is_floaty[not_strings] = rest_is_floaty
            if return_safe_col:
                X_col = X_col.copy()
                X_col[not_strings] = rest

    if not is_floaty.dtype == bool:
        is_floaty = is_floaty.astype(bool)

    if return_safe_col:
        return is_floaty, X_col
    else:
        return is_floaty


def _float_matching_fetch(X, col, return_safe_col=False):
    """Retrieve _float_matching for X[col] from cache or function call.

    If not present in cache, stores function call results into cache.
    Uses dataframe object id and column name as cache key.
    """
    hash_key = f'{col}-{hash(X[col])}'

    if hash_key in _FLOAT_MATCHING_CACHE:
        floats, X_col = _FLOAT_MATCHING_CACHE[hash_key]
    else:
        floats, X_col = _float_matching(X[col], return_safe_col=True)
        _FLOAT_MATCHING_CACHE[hash_key] = floats, X_col

    if return_safe_col:
        return floats, X_col
    else:
        return floats


class DirtyFloatCleaner(BaseEstimator, TransformerMixin):
    # should this error if the inputs are not string?
    def fit(self, X, y=None):
        # FIXME clean float columns will make this fail
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe. Convert or call `clean`.")
        encoders = {}
        for col in X.columns:
            floats, X_col = _float_matching_fetch(X, col, return_safe_col=True)
            # FIXME sparse
            if (~floats).any():
                encoders[col] = OneHotEncoder(sparse=False,
                                              handle_unknown='ignore').fit(
                    pd.DataFrame(X_col[~floats]))
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
            floats, X_col = _float_matching_fetch(X, col, return_safe_col=True)
            nofloats = ~floats
            X_new_col = X_col.copy()
            X_new_col[nofloats] = np.NaN
            X_new_col = X_new_col.astype(float)
            enc = self.encoders_[col]
            if enc is None:
                if nofloats.any():
                    warnings.warn(
                        "Found non-floats {} in float column. It's "
                        "recommended"
                        " to call 'clean' on the whole dataset before "
                        "splitting into training and test set.".format(
                            X.loc[nofloats, col].unique()))
                X_new_col = X_new_col.rename("{}_dabl_continuous".format(col))
                result.append(X_new_col)
                continue
            cats = pd.DataFrame(0, index=X.index,
                                columns=enc.get_feature_names([str(col)]))
            if nofloats.any():
                cats.loc[nofloats, :] = enc.transform(pd.DataFrame(
                    X_col[nofloats]))
            cats["{}_dabl_continuous".format(col)] = X_new_col
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
    if values.max() > 100000:
        # really large numbers are probably identifiers.
        # also bincount will throw a memory error.
        return False
    counts = np.bincount(values)

    def norm(x):
        return np.abs(np.diff(np.diff(x))).sum()
    grad_norm = norm(counts)
    # shuffle 100 times
    grad_norm_shuffled = np.mean([
        norm(counts[np.random.permutation(len(counts))]) for i in range(100)])
    return grad_norm * 1.5 < grad_norm_shuffled


def _string_is_date(series):
    try:
        pd.to_datetime(series[:10])
    except (ParserError, pd.errors.OutOfBoundsDatetime, ValueError,
            TypeError, OverflowError):
        return False
    try:
        pd.to_datetime(series)
    except (ParserError, pd.errors.OutOfBoundsDatetime, ValueError,
            TypeError, OverflowError):
        return False
    return True


def _find_string_floats(X, dirty_float_threshold):
    if not isinstance(X, pd.DataFrame):
        # FIXME workaround to accept series
        X = pd.DataFrame(X)
    is_float = X.apply(_float_matching)
    clean_float_string = is_float.all()
    # remove 5 most common string values before checking if the rest is float
    # FIXME 5 hardcoded!!
    dirty_float = pd.Series(0, index=X.columns, dtype=bool)
    for col in X.columns:
        if clean_float_string[col]:
            # already know it's clean
            continue
        X_col = X[col]
        common_distinct_values = X_col.value_counts()[:5].index
        is_common = X_col.isin(common_distinct_values) | X_col.isna()
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


_FLOAT_TYPES = ['floating', 'mixed-interger-float', 'decimal']
_INTEGER_TYPES = ['integer']
_DATE_TYPES = ['datetime64', 'datetime', 'date',
               'timedelta64', 'timedelta', 'time', 'period']
# FIXME we should be able to do better for mixed-integer
_OBJECT_TYPES = ['string', 'bytes', 'mixed', 'mixed-integer']
_CATEGORICAL_TYPES = ['categorical', 'boolean']


def _type_detection_int(series, max_int_cardinality='auto'):
    n_distinct_values = series.nunique()
    if n_distinct_values == len(series):
        # could be an index
        if series.iloc[0] == 0:
            if (series == np.arange(len(series))).all():
                # definitely an index
                return 'useless'
        elif series.iloc[0] == 1:
            if (series == np.arange(1, len(series) + 1)).all():
                # definitely an index
                return 'useless'
    if n_distinct_values > max_int_cardinality:
        return 'continuous'
    elif n_distinct_values <= 5:
        # weird hack / edge case
        return 'categorical'
    else:
        return 'low_card_int'


def _type_detection_float(series, max_int_cardinality='auto'):
    if _float_col_is_int(series):
        return _type_detection_int(
            series, max_int_cardinality=max_int_cardinality)
    return 'continuous'


def _type_detection_object(series, *, dirty_float_threshold,
                           max_int_cardinality='auto'):
    clean_float_string, dirty_float = _find_string_floats(
            series, dirty_float_threshold)
    if dirty_float.any():
        return 'dirty_float'
    elif clean_float_string.any():
        return _type_detection_float(
            series.astype(float), max_int_cardinality=max_int_cardinality)
    if _string_is_date(series):
        return 'date'
    if series.nunique() <= max_int_cardinality:
        return 'categorical'
    return "free_string"


def detect_type_series(series, *, dirty_float_threshold=0.9,
                       max_int_cardinality='auto',
                       near_constant_threshold=0.95, target_col=None):
    n_distinct_values = series.nunique()
    if series.isna().mean() > 0.99:
        return 'useless'
    # infer near-constant-values
    # fast-pass if n_distinct_values is high
    count = series.count()

    if n_distinct_values == 1:
        return 'useless'

    if (n_distinct_values < (1 - near_constant_threshold) * count
            and series.name != target_col):
        if series.value_counts().max() > near_constant_threshold * count:
            return 'useless'
    if n_distinct_values == 2:
        return 'categorical'

    inferred_type = pd.api.types.infer_dtype(series)
    if inferred_type in _DATE_TYPES:
        return 'date'
    elif inferred_type in _CATEGORICAL_TYPES:
        return 'categorical'
    elif inferred_type in _FLOAT_TYPES:
        return _type_detection_float(
            series, max_int_cardinality=max_int_cardinality)
    elif inferred_type in _INTEGER_TYPES:
        return _type_detection_int(
            series, max_int_cardinality=max_int_cardinality)
    elif inferred_type in _OBJECT_TYPES:
        return _type_detection_object(
            series, max_int_cardinality=max_int_cardinality,
            dirty_float_threshold=dirty_float_threshold
        )
    else:
        raise ValueError("WEEEEEIIIRRD")


def detect_types(X, type_hints=None, max_int_cardinality='auto',
                 dirty_float_threshold=.9,
                 near_constant_threshold=0.95, target_col=None,
                 verbose=0):
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
        to be considered categorical. 'auto' is ``max(42, n_samples/100)``.
        Integers are also always considered as continuous variables.
        FIXME not true any more?

    dirty_float_threshold : float, default=.9
        The fraction of floats required in a dirty continuous
        column before it's considered "useless" or categorical
        (after removing top 5 string values)

    target_col : string, int or None
        Specifies the target column in the data, if any.
        Target columns are never dropped.

    verbose : int
        How verbose to be

    Returns
    -------
    res : dataframe, shape (n_columns, 7)
        Boolean dataframe of detected types. Rows are columns in input X,
        columns are possible types (see above).
    """
    # TODO detect top coding
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

    n_samples, _ = X.shape
    if max_int_cardinality == "auto":
        max_int_cardinality = max(42, n_samples / 100)
        if n_samples <= 42:
            # this is pretty hacky
            max_int_cardinality = n_samples // 2

    types_series = X.apply(lambda col: detect_type_series(
        col, max_int_cardinality=max_int_cardinality,
        near_constant_threshold=near_constant_threshold,
        target_col=target_col, dirty_float_threshold=dirty_float_threshold))

    for t in type_hints:
        if t in X.columns:
            types_series[t] = type_hints[t]

    known_types = ['continuous', 'dirty_float', 'low_card_int', 'categorical',
                   'date', 'free_string', 'useless']
    if X.empty:
        return pd.DataFrame(columns=known_types, dtype=bool)
    res = pd.DataFrame({t: types_series == t for t in known_types})
    assert (X.columns == res.index).all()

    assert np.all(res.sum(axis=1) == 1)

    assert (types_series == res.idxmax(axis=1)).all()

    if verbose >= 1:
        print("Detected feature types:")
        print(res.sum())
    return res


def _apply_type_hints(X, type_hints):
    if type_hints is not None:
        # use type hints to convert columns
        # to possibly avoid some work.
        # means we need to copy X though.
        X = X.copy()
        for k, v in type_hints.items():
            if v == "continuous":
                X[k] = X[k].astype(float)
            elif v == "categorical":
                X[k] = X[k].astype('category')
            elif v == 'useless' and k in X.columns:
                X = X.drop(k, axis=1)
    return X


def _select_cont(X):
    return X.columns.str.endswith("_dabl_continuous")


def clean(X, type_hints=None, return_types=False,
          target_col=None, verbose=0):
    """Public clean interface

    Parameters
    ----------
    type_hints : dict or None
        If dict, provide type information for columns.
        Keys are column names, values are types as provided by detect_types.
    return_types : bool, default=False
        Whether to return the inferred types
    target_col : string, int or None
        If not None specifies a target column in the data.
        Target columns are never dropped.
    verbose : int, default=0
        Verbosity control.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = _apply_type_hints(X, type_hints=type_hints)

    if not X.index.is_unique:
        warn("Index not unique, resetting index!", UserWarning)
        X = X.reset_index(drop=True)
    types_p = types = detect_types(X, type_hints=type_hints, verbose=verbose,
                                   target_col=target_col)
    # drop useless columns
    X = X.loc[:, ~types.useless].copy()
    types = types.loc[~types.useless, :]
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

        # discard dirty float targets that cant be converted to float
        if target_col is not None and types_p['dirty_float'][target_col]:
            warn("Discarding dirty_float targets that cannot be converted "
                 "to float.", UserWarning)
            X = X.dropna(subset=["{}_dabl_continuous".format(target_col)])
            X = X.rename(columns={"{}_dabl_continuous".format(
                target_col): "{}".format(target_col)})
            types = types.rename(index={"{}_dabl_continuous".format(
                target_col): "{}".format(target_col)})

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
    """A simple preprocessor.

    Detects variable types, encodes everything as floats
    for use with sklearn.

    Applies one-hot encoding, missing value imputation and scaling.

    Attributes
    ----------
    ct_ : ColumnTransformer
        Main container for all transformations.

    columns_ : pandas columns
        Columns of training data.

    dtypes_ : Series of dtypes
        Dtypes of training data columns.

    types_ : something
        Inferred input types.


    Parameters
    ----------
    scale : boolean, default=True
        Whether to scale continuous data.

    force_imputation : bool, default=True
        Whether to create imputers even if no training data is missing.

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
                imputer = trans.steps[0][1]

                ohe_cols = cols[cols].index
                added_cols = ohe_cols[imputer.indicator_.features_].map(
                    lambda x: '{}_imputed'.format(x))
                ohe_cols = ohe_cols.to_list()
                ohe_cols.extend(added_cols)

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
            in `X`.
        """
        # Check is fit had been called
        with warnings.catch_warnings():
            # fix when requiring sklearn 0.22
            # check_is_fitted will not have arguments any more
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            check_is_fitted(self, ['ct_'])
        return self.ct_.transform(X)
