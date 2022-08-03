import matplotlib
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from itertools import cycle
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from copy import deepcopy

from ..preprocessing import detect_types
from .utils import _prune_categories


def _get_top_features(data, target_col, show_top=5):
    features = data.drop(columns=target_col)
    ordinal_encoded = features.apply(lambda x: x.astype("category").cat.codes)
    target = data[target_col]
    f = mutual_info_classif(
        ordinal_encoded, target,
        discrete_features=np.ones(data.shape[1], dtype=bool))
    top_k = np.argsort(f)[-show_top:][::-1]
    return data.columns[top_k]


def _make_alluvial_curve(verts, verts_out, weight, width, color, alpha):
    """A single flow for a combination of values for all variables.


    Parameters
    ----------
    verts : list of tuples
        Vertex coordinates
    weight : float
        Height (strength) of flow.
    width : float
        Width of pillar for each variable.
    color : color
        Color of flow.
    alpha: float
        Transparency of flow.
    """
    path_points = []
    codes = []
    # vert is lower left hand corner of flow for each segment
    for i, vert in enumerate(verts):
        if i == 0:
            codes.append(Path.MOVETO)
            path_points.append(vert)
        else:
            last_point = path_points[-1]
            # first bezier handle
            codes.append(Path.CURVE4)
            path_points.append(((last_point[0] + vert[0]) / 2, last_point[1]))
            # second bezier handle
            codes.append(Path.CURVE4)
            path_points.append(((last_point[0] + vert[0]) / 2, vert[1]))
            # first point
            codes.append(Path.LINETO)
            # codes.append(Path.LINETO)
            path_points.append(vert)
        if i in [0, len(verts) - 1]:
            # we don't have a separate out-vert for the first or last one
            out_vert = vert
        else:
            out_vert = verts_out[i - 1]
        # second point in within block
        # first bezier handle
        out_vert = (out_vert[0] + width, out_vert[1])
        codes.append(Path.CURVE4)
        path_points.append(((out_vert[0] + vert[0]) / 2, vert[1]))
        # second bezier handle
        codes.append(Path.CURVE4)
        path_points.append(((out_vert[0] + vert[0]) / 2, out_vert[1]))

        codes.append(Path.LINETO)
        path_points.append((out_vert[0], out_vert[1]))

    # go back, shift up by weight
    path_points.extend([(x, y + weight) for x, y in path_points[::-1]])
    codes.extend(codes[::-1])
    codes[-1] = Path.LINETO
    # close poly
    path_points.append(path_points[-1])
    codes.append(Path.CLOSEPOLY)
    path = Path(path_points, codes)
    return patches.PathPatch(path, alpha=alpha, lw=0, color=color)


def alluvian_diagram(source, value_cols, by_col, weight_col='weight',
                     vertical_margin=.1, horizontal_margin=.2, width=.05,
                     alpha=.4, ax=None):
    """Sankey layouting given a table of co-occurences.

    Actually an alluvian diagram.

    For a high-level API based on a dataframe of data, look at plot_sankey.

    Parameters
    ----------
    source : ndarray
        Count of co-occurences for variables.

    value_cols : list of column names
        List of columns to include as variables.

    by_col : string
        Column name for grouping. Usually the class name.
        This would be "hue" in seaborn.

    weight_col : string, default='weight'
        Name of column to use as weight / flow strength.

    vertical_margin : float, default=.1
        Margin between values within a column.

    horizontal_margin : float, default=.2
        Margin between columns corresponding to different variables.

    width : float, default=.05
        Width of column corresponding to a variable.

    alpha : float, default=.4
        Transparency for flows.

    ax : matplotlib Axes, default=None
        Axes to plot into.
    """
    source = source.copy()
    source[weight_col] = source[weight_col] / source[weight_col].sum()
    if ax is None:
        _, ax = plt.subplots(dpi=100)
    ax.set_ylim(-0.1, 1 + vertical_margin + 0.2)
    ax.set_xlim(-0.1, width * len(value_cols)
                + horizontal_margin * (len(value_cols) - 1) + 0.1)
    left = 0

    coords = defaultdict(dict)
    for col in value_cols:
        top = 0
        # compute blocks per category for each feature/value column
        # weights is height of each individual category
        # coords stores left and top for each value
        weights = source.groupby(col)[weight_col].sum()
        n_values = len(weights)
        this_margin = vertical_margin / (n_values - 1)
        ax.text(left, 0, col, rotation=90, horizontalalignment="right",
                verticalalignment="bottom")
        for val, this_weight in weights.items():
            if this_weight == 0:
                continue
            coords[col][val] = (left, top)
            patch = patches.Rectangle((left, top), width=width,
                                      height=this_weight, ec='k', fc='none')
            ax.text(left, top, val)
            top += this_weight + this_margin
            ax.add_patch(patch)
        left += width + horizontal_margin

    colors = {val: col for val, col in zip(
        source[by_col].unique(),
        cycle(matplotlib.rcParams['axes.prop_cycle'].by_key()['color']))}

    coords_backwards = deepcopy(coords)
    # coords has "top" for each category
    # for each row in data, put in current position,
    # and increase top for that block.
    verts = defaultdict(list)
    # for first column, sort just by that column.
    sort_col = value_cols[0]
    for col in value_cols:
        coord_col_name = f"coord_{col}"
        source[coord_col_name] = 0
        for i, row in source.sort_values([by_col, sort_col]).iterrows():
            # start of block for value of row in thie columns
            coord = coords[col][row[col]]
            coords[col][row[col]] = (coord[0], coord[1] + row[weight_col])
            verts[i].append(coord)
            source.loc[i, coord_col_name] = coord[1]
        sort_col = coord_col_name

    # now go backwards and change the "out" of each of the flows
    # in the previous column
    verts_backwards = defaultdict(list)
    sort_col = f"coord_{value_cols[-1]}"
    # start with second to last col, leave out first
    for col in value_cols[-2:0:-1]:
        for i, row in source.sort_values([by_col, sort_col]).iterrows():
            coord = coords_backwards[col][row[col]]
            coords_backwards[col][row[col]] = (
                coord[0], coord[1] + row[weight_col])
            verts_backwards[i].append(coord)
        sort_col = f"coord_{col}"

    for i, row in source.iterrows():
        patch = _make_alluvial_curve(
            verts[i], verts_backwards[i][::-1], row[weight_col], width,
            colors[row[by_col]], alpha=alpha)
        ax.add_patch(patch)

    ax.set_axis_off()


def plot_sankey(data, target_col, value_cols=None, prune_percentile=.9,
                show_top=5, max_categories=5, dpi=150, figure=None):
    """Plot sankey diagram based on categorical variable co-occurences.

    Actually an alluvian diagram.

    Parameters
    ----------
    data : DataFrame
        Input data.

    target_col : string
        Target column, used to color flows.

    value_cols : list of string
        Column names to include in diagram.
        If none, all categorical columns are considered.

    prune_percentile : float, default=.9
        Prune small flows to retain prune_percentile of total weight.
        This will make the plot simpler and faster,
        but might distort the data. In particular, it might hide minority
        populations and only show large trends.

    show_top : integer, default=5
        Number of categorical features to show if value_cols is not given.
        Selected based on mutual information.

    max_categories : integer, default=5
        Maximum number of categories per variable/column.
        Only the max_categories largest values are considere, the rest is
        grouped into an "other" category.

    dpi : integer, default=150
        Resolution for figure. Ignored if figure is passed.

    figure : matplotlib figure, default=None
        If passed, dpi is ignored.

    """
    if value_cols is None:
        cats = detect_types(data).categorical
        data = data.loc[:, cats].copy()
        for col in data.columns:
            if col == target_col:
                continue
            data[col] = _prune_categories(
                data[col], max_categories=max_categories)

        value_cols = _get_top_features(data, target_col, show_top=show_top)
    else:
        data = data.copy()
        for col in value_cols:
            data[col] = _prune_categories(
                data[col], max_categories=max_categories)

    cols = ([target_col] + list(value_cols)
            if target_col is not None else value_cols)
    data = data[cols]

    sizes = data.groupby(data.columns.tolist()).size()
    sizes.name = 'weight'
    sankey_data = sizes.reset_index()
    if prune_percentile > 0:
        total = sankey_data.weight.sum()
        vals = sankey_data.weight.sort_values(ascending=False)
        smallest_allowed = vals.iloc[np.searchsorted(
            vals.cumsum(), prune_percentile * total)]
        sankey_data = sankey_data[sankey_data.weight >= smallest_allowed]
    if figure is None:
        figure = plt.figure(dpi=dpi, figsize=(len(value_cols) * 2, 6))
    # data_sorted = sankey_data.sort_values(target_col)
    value_cols = [x for x in data.columns if x != target_col]
    alluvian_diagram(sankey_data, value_cols=value_cols, by_col=target_col,
                     vertical_margin=.1, horizontal_margin=.2, width=.05,
                     alpha=.4, ax=figure.gca())
