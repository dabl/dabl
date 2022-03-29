"""
Comparing categorical variable visualizations
=============================================
This example showcases the four types of visualization supported for
categorical variables for classification,
which are 'count', 'proportion', 'mosaic' and 'sankey'.

The 'count' plot is easiest to understand and closest to the data, as it
simply provides a bar-plot of class counts per category.
However, it makes it hard to make comparisons between different categories.
For example, for workclass, it is hard to see the differences in proportions
among the categories.

The 'proportion' plot on the other hand *only* shows the proportion, so we can
see that the proportions in state-government, government, and self-employed
are nearly the same. However, 'proportion' does not show how many samples are
in each category. How much each category is actually present in the data can be
very important, though.

The 'mosaic' plot shows both the class proportions within each category
(on the x axis) as well as the proportion of the category in the data
(on the y axis). The 'mosaic' plot can be a bit busy; in particular if
there are many classes and many catgories, it becomes harder to interpret.

The 'sankey' plot is even busier, as it combines the features of the 'count'
plot with an alluvial flow diagram of interactions.
By default, only the 5 most common features are included in the sankey diagram,
which can be adjusted by calling the plot_sankey function directly.

"""
from dabl.plot import plot_classification_categorical
from dabl.datasets import load_adult
import matplotlib.pyplot as plt

# load the adult census dataset
# returns a plain dataframe
data = load_adult()
# visualize the interactions of the 5 most important categorical variables
for kind in ['count', 'proportion', 'mosaic', 'sankey']:
    plot_classification_categorical(data, 'income', kind=kind)
    figure = plt.gcf()
    figure.suptitle("")
plt.show()
