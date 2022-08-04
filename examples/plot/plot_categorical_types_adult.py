"""
Comparing categorical variable visualizations
=============================================
This example showcases the four types of visualization supported for
categorical variables for classification,
which are 'count', 'proportion', 'mosaic' and 'sankey'.
"""

from dabl.plot import plot_classification_categorical
from dabl.datasets import load_adult

data = load_adult()

# %%
#
# The 'count' plot is easiest to understand and closest to the data, as it
# simply provides a bar-plot of class counts per category.
# However, it makes it hard to make comparisons between different categories.
# For example, for workclass, it is hard to see the differences in proportions
# among the categories.
plot_classification_categorical(data, target_col='income', kind="count")

# %%
#
# The 'proportion' plot on the other hand *only* shows the proportion, so we
# can see that the proportions in state-government, government, and
# self-employed are nearly the same. However, 'proportion' does not show how
# many samples are in each category. How much each category is actually
# present in the data can be very important, though.
plot_classification_categorical(data, target_col='income', kind="proportion")

# %%
#
# The 'mosaic' plot shows both the class proportions within each category
# (on the x axis) as well as the proportion of the category in the data
# (on the y axis). The 'mosaic' plot can be a bit busy; in particular if
# there are many classes and many catgories, it becomes harder to interpret.
plot_classification_categorical(data, target_col='income', kind="mosaic")
# %%
#
# The 'sankey' plot is even busier, as it combines the features of the 'count'
# plot with an alluvial flow diagram of interactions.
# By default, only the 5 most common features are included in the sankey
# diagram, which can be adjusted by calling the plot_sankey function directly.

plot_classification_categorical(data, target_col='income', kind="sankey")
