from __future__ import print_function
from setuptools import setup, find_packages


setup(name='dabl',
      version='0.1.1',
      description='Data Analysis Baseline Library',
      author='Andreas Mueller',
      packages=find_packages(),
      install_requires=["numpy", "scipy", "scikit-learn", "pandas",
                        "matplotlib", "seaborn"],
      author_email='t3kcit+githubspam@gmail.com',
      )
