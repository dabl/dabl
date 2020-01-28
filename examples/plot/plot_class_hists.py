"""
Class Histogram Example
==========================================
"""
# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from dabl.datasets import load_adult
from dabl.plot import class_hists

data = load_adult()
class_hists(data, "age", "gender", legend=True)
plt.show()
