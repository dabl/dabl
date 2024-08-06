import io
from setuptools import setup, find_packages


with io.open('README.md', encoding='utf_8') as fp:
    readme = fp.read()

setup(name='dabl',
      version='0.2.7-dev',
      description='Data Analysis Baseline Library',
      author='Andreas Mueller',
      url='https://github.com/amueller/dabl',
      long_description=readme,
      long_description_content_type='text/markdown; charset=UTF-8',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "scikit-learn>=1.1",

          # Workaround (currently) undeclared numpy 2.x incompatibility with older
          # scikit-learn versions.
          # See Also: https://github.com/scikit-learn/scikit-learn/issues/29630
          "numpy<2.0; python_version<'3.9'",
          "scikit-learn<1.4; python_version<'3.9'",
          "scikit-learn>=1.3; python_version>='3.9'",

          "pandas",
          "matplotlib < 3.8;python_version<'3.9'",
          "matplotlib >= 3.8;python_version>='3.9'",
          "seaborn",
      ],
      python_requires=">=3.8",
      author_email='t3kcit+githubspam@gmail.com',
      package_data={'': ['datasets/titanic.csv',
                         'datasets/ames_housing.pkl.bz2',
                         'datasets/adult.csv.gz'
                         ]},
      )
