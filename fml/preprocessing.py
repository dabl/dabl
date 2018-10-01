from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   FunctionTransformer)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

_FLOAT_REGEX = "^[+-]?[0-9]*\.?[0-9]*$"


class DirtyFloatCleaner(TransformerMixin):
    # should this error if the inputs are not string?
    def fit(self, X, y=None):
        # FIXME ensure X is dataframe?
        # FIXME clean float columns will make this fail
        encoders = {}
        for col in X.columns:
            floats = X[col].str.match(_FLOAT_REGEX)
            # FIXME sparse
            encoders[col] = OneHotEncoder(sparse=False).fit(
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
                                columns=enc.get_feature_names())
            if nofloats.any():
                cats.loc[nofloats, :] = enc.transform(X.loc[nofloats, [col]])
            # FIXME use types to distinguish outputs instead?
            cats["{}_fml_continuous".format(col)] = new_col
            result.append(cats)
        return pd.concat(result, axis=1)


def detect_types_dataframe(X, max_int_cardinality='auto',
                           dirty_float_threshold=.5, verbose=0):
    """
    Parameters
    ----------
    X : dataframe
        input

    dirty_float_threshold : float, default=.5
        The fraction of floats required in a dirty continuous
        column before it's considered "useless".

    max_int_cardinality: int or 'auto', default='auto'
        Maximum number of distinct integers for an integer column
        to be considered categorical. 'auto' is ``max(42, n_samples/10)``.
        Integers are also always considered as continuous variables.

    verbose : int
        How verbose to be

    Returns
    -------
    res : dataframe
        boolean dataframe of detected types.

    recognized types:
    continuous
    categorical
    dirty float string
    free string TODO
    dirty category string TODO
    date
    useless
    """
    # TODO detect top coding
    # Todo: detect index / unique integers
    # todo: detect near constant features
    # TODO subsample large datsets? one level up?
    # TODO detect encoding missing values as strings /weird values
    n_samples, n_features = X.shape
    if max_int_cardinality == "auto":
        max_int_cardinality = max(42, n_samples / 10)
    n_values = X.apply(lambda x: x.nunique())
    dtypes = X.dtypes
    kinds = dtypes.apply(lambda x: x.kind)
    # FIXME use pd.api.type.is_string_dtype etc maybe
    floats = kinds == "f"
    integers = kinds == "i"
    objects = kinds == "O"  # FIXME string?
    dates = kinds == "M"
    other = - (floats | integers | objects | dates)
    # check if we can cast strings to float
    # we don't need to cast all, could
    if objects.any():
        float_frequencies = X.loc[:, objects].apply(
            lambda x: x.str.match(_FLOAT_REGEX).mean())
    else:
        float_frequencies = pd.Series(0, index=X.columns)
    clean_float_string = float_frequencies == 1.0
    dirty_float = ((float_frequencies > dirty_float_threshold)
                   & (~clean_float_string))

    # using categories as integers is not that bad usually
    cont_integers = integers.copy()
    # using integers as categories only if low cardinality
    few_entries = n_values < max_int_cardinality
    cat_integers = few_entries & integers
    cat_string = few_entries & objects

    res = pd.DataFrame(
        {'continuous': floats | cont_integers | clean_float_string,
         'categorical': cat_integers | cat_string, 'date': dates,
         'dirty_float': dirty_float})
    res = res.fillna(False)
    res['useless'] = res.sum(axis=1) == 0

    if verbose >= 1:
        print("Detected feature types:")
        desc = "{} float, {} int, {} object, {} date, {} other".format(
            floats.sum(), integers.sum(), objects.sum(), dates.sum(),
            other.sum())
        print(desc)
        print("Interpreted as:")
        interp = ("{} continuous, {} categorical, {} date, "
                  "{} dirty float, {} dropped").format(
            res.continuous.sum(), res.categorical.sum(), res.date.sum(),
            dirty_float.sum(), res.useless.sum()
        )
        print(interp)
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


def select_cont(X):
    return X.columns.str.endswith("_fml_continuous")


class FriendlyPreprocessor(BaseEstimator, TransformerMixin):
    """ An simple preprocessor

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
    def __init__(self, scale=True, verbose=0):
        self.verbose = verbose
        self.scale = scale

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
        types = detect_types_dataframe(X, verbose=self.verbose)
        
        # go over variable blocks
        # check for missing values
        # scale etc
        pipe_categorical = OneHotEncoder()

        steps_continuous = [FunctionTransformer(lambda x: x.astype(np.float),
                                                validate=False)]
        if self.scale:
            steps_continuous.append(StandardScaler())
        # if X.loc[:, types['continuous']].isnull().values.any():
        # FIXME doesn't work if missing values only in dirty column
        steps_continuous.insert(0, SimpleImputer(strategy='median'))
        pipe_continuous = make_pipeline(*steps_continuous)
        # FIXME only have one imputer/standard scaler in all
        pipe_dirty_float = make_pipeline(
            DirtyFloatCleaner(),
            make_column_transformer(
                (select_cont, pipe_continuous), remainder="passthrough"))
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
