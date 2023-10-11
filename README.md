# `dabl`: the Database Analysis Baseline Library

- "Mr Sanchez, are you a data scientist?"
- "I dabl, Mr president."


dabl creates a faster way for you to start your supervised machine learning project by offering convenient data preprocessing, creating strong baseline models and data visualization among the many tools it offers.

Learn more about dabl and its capabilities on our [website](https://dabl.github.io/).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
    - [Install](#install)
    - [Try it out](#try-it-out)
- [Current scope and Upcoming features](#current-scope-and-upcoming-features)
- [Related Packages](#related-packages)
    - [Lux](#lux)
    - [Pandas Profiling](#pandas-Profiling)
- [How to Contribute](#how-to-contribute)
- [License](#license)

## Overview

This project focuses on the iterative tasks of building a machine learning model like cleaning and inspecting data, creating strong baseline models, and basic data visualization. This makes it easier for beginners and professional by reducing common task allowing them to focus on problem solving and data analysis.  

Tasks dabl can currently help you with:
- Data Cleaning
- Exploratory Data Analysis
- Initial Model Building
- Enhanced Model Building
- Searching optimal parameters with successive halving
    - Choosing the budget (number of samples)
    - Exhausting the budget
    - Enhanced Model Building

Below are a few examples of different data visualizations dabl is capable of creating with your data. 

![Diamond Data Set Image](https://dabl.github.io/0.1.9/_images/sphx_glr_plot_diamonds_002.png)

![Diamond Data Set Image](https://dabl.github.io/0.1.9/_images/sphx_glr_plot_ames_003.png)

![Diamond Data Set Image](https://dabl.github.io/0.1.9/_images/sphx_glr_plot_wine_004.png)

To view more examples checkout the [example gallery](https://dabl.github.io/0.1.9/auto_examples/index.html) to see more examples and how to implement them.

## Installation

### Install

Installation is very simple just follow these steps:

1. **Download**

    * Download a ZIP file of the project ([Download](https://github.com/dabl/dabl/archive/refs/heads/main.zip))
    
2. **Extract**

    * Extract the ZIP file in any location

3. **Compile**

    * In the extracted folder run the following command to set up dabl in your *terminal* (mac/linux) or *command prompt* (windows)

    ```
    pip install dabl
    ```

4. **Using dabl**

    * You can start using dabl by simply importing it to your python project.

    ```
    import dabl
    ```

    * Here is a list of all the functions provided by the [dabl API](https://dabl.github.io/0.1.9/api.html)

### Try it out

Or you could try out dabl by using [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dabl/dabl/main)

## Current scope and upcoming features

This library is very much still under development. Current code focuses mostly on exploratory visualization and preprocessing.
There are also drop-in replacements for GridSearchCV and RandomizedSearchCV using successive halfing.
There are preliminary portfolios in the style of
[POSH
auto-sklearn](https://ml.informatik.uni-freiburg.de/papers/18-AUTOML-AutoChallenge.pdf)
to find strong models quickly.  In essence that boils down to a quick search
over different gradient boosting models and other tree ensembles and
potentially kernel methods.

Check out the [the website](https://dabl.github.io/dev/) and [example gallery](https://dabl.github.io/0.1.9/auto_examples/index.html) to get an idea of the visualizations that are available.

Stay Tuned!

## Related packages

### Lux

[Lux](https://github.com/lux-org/lux) is an awesome project for easy interactive visualization of pandas dataframes within notebooks.

### Pandas Profiling

[Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) can
provide a thorough summary of the data in only a single line of code. Using the
```ProfileReport()``` method, you are able to access a HTML report of your data
that can help you find correlations and identify missing data.

`dabl` focuses less on statistical measures of individual columns, and more on
providing a quick overview via visualizations, as well as convienient
preprocessing and model search for machine learning.

## How to Contribute

If you choose to contribute to dabl please:

- Follow a ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.

- Before making any considerable contribution open a issue, to discuss the proposed changes or bugs.

## License

This project is licensed under [BSD-3-Clause](https://github.com/dabl/dabl/blob/main/LICENSE) .