import math
import os
import pathlib
import pickle

# Local stuff
from src.data import make_dataset

# Data wrangling and optimizatin stuff
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Neural network stuff
from darts import TimeSeries
from darts.utils.callbacks import TFMProgressBar
from darts.models import TCNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from torch.nn import MSELoss

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_STORE_PATH = pathlib.Path(os.environ['MODEL_STORE'])


def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }


class MyModelStruct:
    def __init__(self, ticker, model_type='price', scaler=None, dataset=None):
        self.ticker = str(ticker)
        self.model_type = str(model_type)
        self.model_name = '{0}_{1}_model'.format(ticker, model_type)
        self.model_path = MODEL_STORE_PATH.joinpath('price_models/' + self.model_name + '.pt')
        self.scaler_path = MODEL_STORE_PATH.joinpath('price_scalers/' + self.model_name + '.pkl')
        if isinstance(dataset, pd.DataFrame):
            self.set_dataset(dataset)

        if self.model_path.exists():
            print("Found an existing model for this. Use load_model method to use it or continue with training to replace it")

        if scaler:
            self.set_scaler(scaler)

        return
    
    def set_train_test_data(self, train_size=0.85):
        if not self.scaler:
            print("You have not set a scaler for the model. Call set_scaler method before training the data")
            return

        training_data_len = math.ceil(len(self._dataset) * train_size)
        scaler = self.scaler
        self._train_size = train_size
        self._training_data_len = training_data_len
        self.train_data_date = self._dataset.iloc[training_data_len, 1]
        train, val = self.ts_dataset.split_after(train_size)
        train_scaled = scaler.fit_transform(train)
        val_scaled = scaler.transform(val)
        ts_scaled = scaler.transform(self.ts_dataset)
        
        # Store values in class instance
        self.ts_dataset_scaled = ts_scaled
        self.train = train
        self.train_scaled = train_scaled
        self.val = val
        self.val_scaled = val_scaled
        self.month_series = datetime_attribute_timeseries(self.ts_dataset, attribute="month", one_hot=True)
        return
    
    def train_model(self, **kwargs):       
        self.model.fit(
            series=self.train_scaled,
            past_covariates=self.month_series,
            val_series=self.val_scaled,
            val_past_covariates=self.month_series,
            verbose=False
        )

        self.model = TCNModel.load_from_checkpoint(model_name=self.model_name, best=True)
        return
    
    def run_backtest(self, **kwargs):
        self.backtest = self.model.historical_forecasts(
            series=self.ts_dataset_scaled,
            past_covariates=self.month_series,
            start=self.val_scaled.start_time(),
            forecast_horizon=10,
            retrain=False,
            verbose=True,
            # num_samples=100
        )
        
        self.reversed_backtest = self.scaler.inverse_transform(self.backtest)
        return
    
    def run_prediction(self, window=14, **kwargs):
        self.prediction = self.model.predict(window, verbose=False, **kwargs)
        self.reversed_prediction = self.scaler.inverse_transform(self.prediction)
        return
    
    def append_prediction(self):
        self._dataset = (pd
                         .concat([self._dataset,
                                  self.reversed_prediction.pd_dataframe().reset_index()]
                                , ignore_index=True)
                        .ffill()
                        .drop_duplicates(subset=['date'], keep='last'))
        self.set_dataset(self._dataset)
        
    def get_aggregated_dataset(self):
        validation_df = self.val.pd_dataframe().reset_index()
        validation_df['source'] = 'actual'
        prediction_df = self.reversed_prediction.pd_dataframe().reset_index()
        prediction_df['source'] = 'prediction'
        final_df = pd.concat([validation_df, prediction_df], ignore_index=True)
        return final_df

    def set_scaler(self, scaler):
        self.scaler = scaler
        return

    def set_dataset(self, dataset):
        self._dataset = pd.DataFrame(dataset)
        ts = TimeSeries.from_dataframe(df=self._dataset, time_col='date', value_cols='avg_price', freq='B')
        ts = fill_missing_values(ts, fill='auto')
        self.ts_dataset = ts

    def set_model(self, model=None):
        if model is None:
            # self.model_name = 'TCN'
            model = TCNModel(
                input_chunk_length=22,
                output_chunk_length=20,
                n_epochs=100,
                loss_fn=MSELoss(),
                dropout=0.1,
                dilation_base=4,
                weight_norm=True,
                kernel_size=8,
                num_filters=6,
                optimizer_kwargs={"lr": 0.06},
                random_state=0,
                model_name=self.model_name,
                save_checkpoints=True,
                force_reset=True,
                # **generate_torch_kwargs()
            )
            
        self.model = model
    
    def load_model(self, path=None):
        if path is None and not self.model_path.exists():
            print('Model not found in default path.')
            return

        if path is not None and not pathlib.Path(path).exists():
            print("Model path not found.")
            return

        if path is not None:
            self.model_path = pathlib.Path(path)
        
        self.model = TCNModel.load(str(self.model_path))
        return
    
    def load_scaler(self, path=None):
        if path is None and not self.model_path.exists():
            print('Model not found in default path.')
            return

        if path is not None and not pathlib.Path(path).exists():
            print("Model path not found.")
            return

        if path is not None:
            self.model_path = pathlib.Path(path)

        with open(self.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        self.set_scaler(scaler)
        return
    
    def save_model(self, path=None):
        if self.model is None:
            print("Model not available")
            return

        if path is not None:
            self.model_path = pathlib.Path(path)
        
        self.model.save(str(self.model_path))
        return
