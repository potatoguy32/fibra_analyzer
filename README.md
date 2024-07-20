# FIBRA Portfolio Optimization

This project provides a methodology for optimizing an investment portfolio of FIBRAs (Fideicomisos de Inversión en Bienes Raíces) using Python. The methodology includes collecting historical data, using predictive models for price forecasting, scrapping web pages to get future divideds, optimizing the portfolio based on predicted data, and automating the rebalancing of the portfolio.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project organization](#project-organization)

## Project Overview

FIBRAs are financial instruments in Mexico similar to Real Estate Investment Trusts (REITs) in the United States. This project aims to automate the optimization and rebalancing of a portfolio of FIBRAs to maximize returns while managing risk.

### Objectives

1. Collect historical price and dividend data for FIBRAs.
2. Analyze the data to understand asset characteristics.
3. Use time series models to predict future prices.
4. Extract future dividends from financial reports.
5. Optimize the portfolio based on predictions.

## Features

- **Data Collection**: Automatically fetch historical prices and dividend data using the Yahoo Finance API.
- **Data Analysis**: Visualize and analyze historical data to gain insights into asset performance.
- **Predictive Modeling**: Use time series models to predict future prices.
- **Dividends**: Use a web scrapper to download the latest financial reports and extract dividends information.
- **Portfolio Optimization**: Optimize the portfolio using Modern Portfolio Theory and alternative risk-return measures.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/fibra-portfolio-optimization.git
   cd fibra-portfolio-optimization

### Environment variables
This project requires a .env file in the home (outer) directoy with the following environment variables:
- DATABASE_PATH: This project works with sqlite3, this path will point to a local database (.db) file. 
- MODEL_STORE: Folder where all trained models will be stored and accessed for predictions.
- REPORTS_STORE_PATH: Folder where all the financial reports will be stored for text extraction.
- UTILITY_FILES_PATH: Files used to complement this tool workflow such as the MXN-USD historical exchange rate extracted from "BANXICO".
- DATA_FOLDER_PATH: It is not included in the repor, but this tools asumes a folder in the root directory named "data" structured as the project organization section points.

## Usage
You can find a sample of the workflow of this project in the jupyter notebook "workflow.ipynb" in the notebooks folder.

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data (not included in git repo)
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    │── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    │
    └── .env            <- Enviornment variables to be used in the entire project


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
