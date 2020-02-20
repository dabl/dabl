import io
from setuptools import setup, find_packages


with io.open('README.md', encoding='utf_8') as fp:
    readme = fp.read()

setup(name='dabl',
      version='0.1.8',
      description='Data Analysis Baseline Library',
      author='Andreas Mueller',
      url='https://github.com/amueller/dabl',
      long_description=readme,
      long_description_content_type='text/markdown; charset=UTF-8',
      packages=find_packages(),
      install_requires=["numpy", "scipy", "scikit-learn", "pandas",
                        "matplotlib", "seaborn"],
      author_email='t3kcit+githubspam@gmail.com',
      package_data={'': ['datasets/titanic.csv',
                         'datasets/ames_housing.pkl.bz2',
                         'datasets/adult.csv.gz'
                         ]},
      )
