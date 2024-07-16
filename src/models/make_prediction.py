# out of the box python modules
import sqlite3
import os
import pathlib

from dotenv import load_dotenv

# some
from src.models.make_model import MyModelStruct

# Data wrangling stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from darts.dataprocessing.transformers import Scaler

# Load environment variables
load_dotenv()
DATABASE_PATH = os.environ['DATABASE_PATH']
MODEL_STORE_PATH = pathlib.Path(os.environ['MODEL_STORE'])


def get_prediction(dataset, window):
    sample = dataset.copy()
    ticker_name = sample.ticker.unique()[0].split('.')[0]
    struct = MyModelStruct(ticker=ticker_name, model_type='price', dataset=sample, scaler=Scaler())
    struct.set_model()
    struct.set_train_test_data(train_size=0.85)
    struct.train_model()
    struct.run_backtest()
    struct.run_prediction(window=window, series=struct.val_scaled)
    struct.append_prediction()

    prediction_df = struct.get_aggregated_dataset()
    return prediction_df