from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   FunctionTransformer)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.utils.validation import check_array, check_is_fitted
import pandas as pd
import numpy as np

_FLOAT_REGEX = "^[+-]?[0-9]*\.?[0-9]*$"


class DirtyFloatCleaner(TransformerMixin):
    def fit(self, X, y=None):
        # FIXME ensure X is dataframe?
        # FIXME clean float columns will make this fail
        encoders = {}
        for col in X.columns:
            floats = X[col].str.match(_FLOAT_REGEX)
            # FIXME sparse
            encoders[col] = OneHotEncoder(sparse=False).fit(X.loc[~floats, [col]])
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
            cats = pd.DataFrame(0, index=np.arange(len(X)),
                                columns=enc.get_feature_names())
            cats[nofloats] = enc.transform(X.loc[nofloats, [col]])
            # FIXME use types to distinguish outputs instead?
            cats["{}_fml_continuous".format(col)] = new_col
            result.append(cats)
        return pd.concat(result, axis=1)


def detect_types_dataframe(X, verbose=0):
    """
    recognized types:
    continuous
    categorical
    dirty float string
    free string TODO
    dirty category string TODO
    date
    """
    # Todo: detect index / unique integers
    # todo: detect near constant features
    # TODO subsample large datsets? one level up?
    # TODO detect encoding missing values as strings /weird values
    n_samples, n_features = X.shape
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
    float_frequencies = X.loc[:, objects].apply(
        lambda x: x.str.match(_FLOAT_REGEX).mean())
    clean_float_string = float_frequencies == 1.0
    dirty_float = (float_frequencies > .9) & (~clean_float_string)

    # using categories as integers is not that bad usually
    cont_integers = integers.copy()
    # using integers as categories only if low cardinality
    # FIXME hardcoded
    few_entries = n_values < max(42, n_samples / 10)
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


def detect_types_ndarray(X):
    raise NotImplementedError


def select_cont(X):
    X.columns.str.endswith("_fml_continuous")


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
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            self.dtypes_ = X.dtypes
            types = detect_types_dataframe(X, verbose=self.verbose)
        else:
            X = check_array(X)
            types = detect_types_ndarray(X)

        # go over variable blocks
        # check for missing values
        # scale etc
        pipe_categorical = OneHotEncoder()

        steps_continuous = [FunctionTransformer(lambda x: x.astype(np.float),
                                                validate=False)]
        if self.scale:
            steps_continuous.append(StandardScaler())
        if X.loc[:, types['continuous']].isnull().values.any():
            steps_continuous.insert(0, SimpleImputer(strategy='median'))
        pipe_continuous = make_pipeline(*steps_continuous)
        # FIXME only have one imputer/standard scaler in all
        pipe_dirty_float = make_pipeline(
            DirtyFloatCleaner(),
            make_column_transformer(
                (select_cont, pipe_continuous), remainder="passthrough"))
        # construct column transformer
        column_types = []
        if types['continuous'].any():
            column_types.append((types['continuous'], pipe_continuous))
        if types['categorical'].any():
            column_types.append((types['categorical'], pipe_categorical))
        if types['dirty_float'].any():
            column_types.append((types['dirty_float'], pipe_dirty_float))

        if not len(column_types):
            raise ValueError("No feature columns found")
        self.ct_ = make_column_transformer(*column_types)
        self.ct_.sparse_threshold = .1  # FIXME HACK

        self.ct_.fit(X)

        self.input_shape_ = X.shape
        self.types_ = types
        # Return the transformer
        return self

    def get_feature_names(self):
        return self.ct_.get_feature_names(self.columns_)

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
