"""
port of sklearn.tree.plot_tere until 0.21 gets released
"""

import warnings
from io import StringIO

from numbers import Integral

import numpy as np

from sklearn.utils.validation import check_is_fitted

from sklearn.tree import _criterion
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier

from ._reingold_tilford import buchheim, Tree


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list

def plot_tree(decision_tree, max_depth=None, feature_names=None,
              class_names=None, label='all', filled=False,
              impurity=True, node_ids=False,
              proportion=False, rotate=False, rounded=False,
              precision=3, ax=None, fontsize=None):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.
    This function requires matplotlib, and works best with matplotlib >= 1.5.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be exported to GraphViz.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    class_names : list of strings, bool or None, optional (default=None)
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, optional (default='all')
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, optional (default=False)
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    impurity : bool, optional (default=True)
        When set to ``True``, show the impurity at each node.

    node_ids : bool, optional (default=False)
        When set to ``True``, show the ID number on each node.

    proportion : bool, optional (default=False)
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, optional (default=False)
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional (default=False)
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, optional (default=3)
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, optional (default=None)
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, optional (default=None)
        Size of text font. If None, determined automatically to fit figure.

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.plot_tree(clf)  # doctest: +SKIP
    [Text(251.5,345.217,'X[3] <= 0.8...

    """
    exporter = _MPLTreeExporter(
        max_depth=max_depth, feature_names=feature_names,
        class_names=class_names, label=label, filled=filled,
        impurity=impurity, node_ids=node_ids,
        proportion=proportion, rotate=rotate, rounded=rounded,
        precision=precision, fontsize=fontsize)
    return exporter.export(decision_tree, ax=ax)


class _BaseTreeExporter(object):
    def __init__(self, max_depth=None, feature_names=None,
                 class_names=None, label='all', filled=False,
                 impurity=True, node_ids=False,
                 proportion=False, rotate=False, rounded=False,
                 precision=3, fontsize=None):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rotate = rotate
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors['bounds'] is None:
            # Classification tree
            color = list(self.colors['rgb'][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = ((sorted_values[0] - sorted_values[1])
                         / (1 - sorted_values[1]))
        else:
            # Regression tree or multi-output
            color = list(self.colors['rgb'][0])
            alpha = ((value - self.colors['bounds'][0]) /
                     (self.colors['bounds'][1] - self.colors['bounds'][0]))
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return '#%2x%2x%2x' % tuple(color)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # Initialize colors and bounds if required
            self.colors['rgb'] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors['bounds'] = (np.min(-tree.impurity),
                                         np.max(-tree.impurity))
            elif (tree.n_classes[0] == 1 and
                  len(np.unique(tree.value)) != 1):
                # Find max and min values in leaf nodes for regression
                self.colors['bounds'] = (np.min(tree.value),
                                         np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = (tree.value[node_id][0, :] /
                        tree.weighted_n_node_samples[node_id])
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == 'root' and node_id == 0) or self.label == 'all'

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (characters[1],
                                       tree.feature[node_id],
                                       characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           characters[3],
                                           round(tree.threshold[node_id],
                                                 self.precision),
                                           characters[4])

        # Write impurity
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += (str(round(tree.impurity[node_id], self.precision))
                            + characters[4])

        # Write node sample count
        if labels:
            node_string += 'samples = '
        if self.proportion:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            node_string += (str(round(percent, 1)) + '%' +
                            characters[4])
        else:
            node_string += (str(tree.n_node_samples[node_id]) +
                            characters[4])

        # Write node class distribution / regression value
        if self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (self.class_names is not None and
                tree.n_classes[0] != 1 and
                tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (characters[1],
                                          np.argmax(value),
                                          characters[2])
            node_string += class_name

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[:-len(characters[4])]

        return node_string + characters[5]


class _MPLTreeExporter(_BaseTreeExporter):
    def __init__(self, max_depth=None, feature_names=None,
                 class_names=None, label='all', filled=False,
                 impurity=True, node_ids=False,
                 proportion=False, rotate=False, rounded=False,
                 precision=3, fontsize=None):

        super().__init__(
            max_depth=max_depth, feature_names=feature_names,
            class_names=class_names, label=label, filled=filled,
            impurity=impurity, node_ids=node_ids, proportion=proportion,
            rotate=rotate, rounded=rounded, precision=precision)
        self.fontsize = fontsize

        # validate
        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError("'precision' should be greater or equal to 0."
                                 " Got {} instead.".format(precision))
        else:
            raise ValueError("'precision' should be an integer. Got {}"
                             " instead.".format(type(precision)))

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {'leaves': []}
        # The colors to render each node with
        self.colors = {'bounds': None}

        self.characters = ['#', '[', ']', '<=', '\n', '', '']

        self.bbox_args = dict(fc='w')
        if self.rounded:
            self.bbox_args['boxstyle'] = "round"
        else:
            # matplotlib <1.5 requires explicit boxstyle
            self.bbox_args['boxstyle'] = "square"

        self.arrow_args = dict(arrowstyle="<-")

    def _make_tree(self, node_id, et, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(et, node_id, criterion='entropy')
        if (et.children_left[node_id] != _tree.TREE_LEAF
                and (self.max_depth is None or depth <= self.max_depth)):
            children = [self._make_tree(et.children_left[node_id], et,
                                        depth=depth + 1),
                        self._make_tree(et.children_right[node_id], et,
                                        depth=depth + 1)]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_)
        draw_tree = buchheim(my_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        self.recurse(draw_tree, decision_tree.tree_, ax,
                     scale_x, scale_y, ax_height)

        anns = [ann for ann in ax.get_children()
                if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            try:
                extents = [ann.get_bbox_patch().get_window_extent()
                           for ann in anns]
                max_width = max([extent.width for extent in extents])
                max_height = max([extent.height for extent in extents])
                # width should be around scale_x in axis coordinates
                size = anns[0].get_fontsize() * min(scale_x / max_width,
                                                    scale_y / max_height)
                for ann in anns:
                    ann.set_fontsize(size)
            except AttributeError:
                # matplotlib < 1.5
                warnings.warn("Automatic scaling of tree plots requires "
                              "matplotlib 1.5 or higher. Please specify "
                              "fontsize.")

        return anns

    def recurse(self, node, tree, ax, scale_x, scale_y, height, depth=0):
        # need to copy bbox args because matplotib <1.5 modifies them
        kwargs = dict(bbox=self.bbox_args.copy(), ha='center', va='center',
                      zorder=100 - 10 * depth, xycoords='axes pixels')

        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + .5) * scale_x, height - (node.y + .5) * scale_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs['bbox']['fc'] = self.get_fill_color(tree,
                                                           node.tree.node_id)
            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + .5) * scale_x,
                             height - (node.parent.y + .5) * scale_y)
                kwargs["arrowprops"] = self.arrow_args
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, scale_x, scale_y, height,
                             depth=depth + 1)

        else:
            xy_parent = ((node.parent.x + .5) * scale_x,
                         height - (node.parent.y + .5) * scale_y)
            kwargs["arrowprops"] = self.arrow_args
            kwargs['bbox']['fc'] = 'grey'
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)
