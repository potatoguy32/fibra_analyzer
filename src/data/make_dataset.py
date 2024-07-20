# Python base modules
import datetime
import sqlite3
import os
import pathlib

# Python external (installed) modules
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import numpy as np

# Projct (local) modules
from src.data import CRUD
from src.data import text_handlers
from src.models.make_prediction import get_prediction

from pydantic import ValidationError


# Load environment variables
load_dotenv()
DATABASE_PATH = os.environ['DATABASE_PATH']
REPORTS_STORE_PATH = os.environ['REPORTS_STORE_PATH']

def update_market_data():
    with sqlite3.connect(DATABASE_PATH) as conn:
        tickers_data = CRUD.get_tickers_last_dates(conn)
        
    last_dates_df = (pd
                 .DataFrame(tickers_data, columns=['ticker', 'last_date'])
                 .groupby(['last_date'])
                 .agg(list)
                 .reset_index()
                 )
    
    df_list = []
    for date, tickers in last_dates_df.itertuples(index=False, name=None):
        try:
            start_date = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            market_data = get_market_data_from_yf(tickers, start_date, end_date)
            current_df_list = preprocess_market_data_from_yf(market_data)
            df_list.extend(current_df_list)
        except Exception as e:
            print(e)
            continue
    
    appended_df = pd.concat(df_list, ignore_index=True)
    appended_df = appended_df.get(['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',])
    print(f"Loading {appended_df.shape[0]} entries into database...")
    with sqlite3.connect(DATABASE_PATH) as conn:
        CRUD.load_market_data(conn, appended_df.itertuples(index=False, name=None))
        
    print(f"Succesfully loaded new data in database, returning dataset for validations...")
    return appended_df

def add_new_tickers(tickers, start_date, end_date):
    new_market_data = get_market_data_from_yf(tickers, start_date, end_date)
    new_processed_df = pd.concat(preprocess_market_data_from_yf(new_market_data), ignore_index=True)
    
    print(f"Loading {new_processed_df.shape[0]} entries into database...")
    with sqlite3.connect(DATABASE_PATH) as conn:
        CRUD.load_market_data(conn, new_processed_df.itertuples(index=False, name=None))
    
    print(f"Succesfully loaded new data in database, returning dataset for validations...")
    return new_processed_df

def delete_tickers(tickers):
    if not isinstance(tickers, list):
        tickers = [tickers, ]

    with sqlite3.connect(DATABASE_PATH) as conn:
        CRUD.delete_tickers(connection=conn, tickers=tickers)

    print("Succesfully deleted tickers from DB...")
    return

def get_market_data_from_yf(tickers, start_date, end_date):
    tickers = list(tickers) if isinstance(tickers, list) else [tickers, ]
    tickers_instance = yf.Tickers(tickers) if len(tickers) > 1 else yf.Ticker(tickers[0])
    market_data = tickers_instance.history(start=start_date, end=end_date, group_by='ticker') if len(tickers) > 1 else tickers_instance.history(start=start_date, end=end_date)
    if len(tickers) == 1:
        multi_index_cols = pd.MultiIndex.from_tuples([(tickers[0], col) for col in market_data.columns])
        market_data.columns = multi_index_cols
    return market_data

def preprocess_market_data_from_yf(market_data):
    df_list = []
    for ticker in set([col[0] for col in market_data.columns]):
        current_df = market_data.get(ticker).reset_index()
        current_df.insert(0, 'ticker', ticker)
        current_df.dropna(
            subset=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'],
            how='all',
            inplace=True
            )
        current_df['Date'] = current_df['Date'].dt.date.astype(str)
        df_list.append(current_df)
        
    return df_list

def extract_dividend_data(handler, file_path):
    dividend_dict = handler(file_path).get_dividend_data()
    return dividend_dict

def get_aggregared_dividend_data(asset_map: dict) -> list[pd.DataFrame]:
    asset_map = dict(asset_map)
    dividend_data = []
    for folder, handler in asset_map.items():
        print("=================== ", handler.__name__, "=================== ")
        for file in os.listdir(pathlib.Path(REPORTS_STORE_PATH).joinpath(folder + '/')):
            try:
                curr_file_path = pathlib.Path(REPORTS_STORE_PATH).joinpath(folder).joinpath(file)
                current_data = extract_dividend_data(handler, curr_file_path)
                dividend_data.append(current_data)
                print(current_data)
            except:
                pass

    return dividend_data

def update_dividend_data(asset_map: dict) -> None:
    extracted_data = get_aggregared_dividend_data(asset_map)
    filtered_entries = []
    for dividend_data_dict in extracted_data:
        try:
            entry = text_handlers.DividendModel(**dividend_data_dict).dump_tuple()
        except ValidationError as e:
            print('Skipping line. Details:')
            print(e.errors())
        
        filtered_entries.append(entry)

    with sqlite3.connect(DATABASE_PATH) as conn:
        CRUD.load_dividends(conn, filtered_entries)

    return

def get_market_dataset(ticker, start_date, end_date):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cur = conn.cursor()
        
        market_data_query = "SELECT * FROM market_data order by ticker, date;"
        market_data_res = cur.execute(market_data_query)
        market_data_entries = market_data_res.fetchall()
    
    full_market_df = pd.DataFrame(market_data_entries, columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'])
    full_market_df['date'] = pd.to_datetime(full_market_df['date'])
    full_market_df['avg_price'] = full_market_df[['open', 'high', 'low', 'close']].mean(axis=1)
    ticker_df = full_market_df.query("ticker == @ticker and date >= @start_date & date <= @end_date")
    sample = ticker_df.filter(['date', 'ticker', 'avg_price', 'dividends'])
    return sample

def get_dividends_dataset(ticker_name):
    future_dividends_query = "SELECT * FROM dividends;"
    with sqlite3.connect(DATABASE_PATH) as conn:
        cur = conn.cursor()
        future_dividends_res = cur.execute(future_dividends_query)
        future_dividends_entries = future_dividends_res.fetchall()

    future_dividends_df = pd.DataFrame(future_dividends_entries, columns=['ticker', 'announcement_date', 'dividend_date', 'dividend_amount'])
    future_dividends_df['announcement_date'] = pd.to_datetime(future_dividends_df['announcement_date'])
    future_dividends_df['dividend_date'] = pd.to_datetime(future_dividends_df['dividend_date'])
    ticker_dividends_df = future_dividends_df.query("ticker == @ticker_name")
    return ticker_dividends_df

def get_ticker_dataset(ticker_name: str, start_date: str, end_date: str, prediction_window: int = 10):
    # Create and filter market data dataframe
    sample = get_market_dataset(ticker_name, start_date, end_date)

    # Create and filter dividends data dataframe
    ticker_dividends_df = get_dividends_dataset(ticker_name)

    # Make prediction with ML model
    prediction_df = get_prediction(sample, prediction_window)

    # Join data
    consolidated_price_df = sample.merge(prediction_df, how='outer')
    consolidated_price_df['ticker'] = consolidated_price_df['ticker'].ffill()
    consolidated_price_df['source'] = consolidated_price_df['source'].bfill()
    consolidated_price_df['dividends'] = consolidated_price_df['dividends'].fillna(0)
    latest_actual_date = consolidated_price_df.query("source == 'actual'").date.max().date()
    future_dividends_df = (ticker_dividends_df
                           .filter(['ticker', 'dividend_date', 'dividend_amount'])
                           .rename(columns={'dividend_date': 'date'})
                           .query("date > @latest_actual_date"))

    consolidated_df = consolidated_price_df.merge(future_dividends_df, how='left')
    consolidated_df['dividends'] = np.where(consolidated_df.dividend_amount.notnull(), consolidated_df.dividend_amount, consolidated_df.dividends)
    consolidated_df.drop(columns=['dividend_amount'], inplace=True)
    consolidated_df = consolidated_df.sort_values(by=['date', 'source'], ascending=[True, False]).drop_duplicates(subset=['date'], keep='last')

    return consolidated_df
