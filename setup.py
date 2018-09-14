from __future__ import print_function
import sys
from setuptools import setup, find_packages


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='fml',
      version='0.0.1',
      description='A Friendlier Machine Learning Library',
      author='Andreas Mueller',
      packages=find_packages(),
      install_requires=[],
      author_email='t3kcit+githubspam@gmail.com',
      )
