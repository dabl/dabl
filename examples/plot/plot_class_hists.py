"""
Class Histogram Example
==========================================
"""
import matplotlib.pyplot as plt
from dabl.datasets import load_adult
from dabl.plot import class_hists

data = load_adult()

# Plots the histogram of age per gender
class_hists(data, "age", "gender", legend=True)
plt.show()
