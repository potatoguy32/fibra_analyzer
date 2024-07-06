# Python base modules
import datetime
import sqlite3
import os
import pathlib

# Python external (installed) modules
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Projct (local) modules
from src.data import CRUD
from src.data import text_handlers

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

def get_aggregared_dividend_data():
    asset_map = {
        'FMTY14': text_handlers.TextHandlerFMTY14,
        'FIBRAPL14': text_handlers.TextHandlerFIBRAPL14,
        'FNOVA17': text_handlers.TextHandlerFNOVA17,
        'FIBRAMQ12': text_handlers.TextHandlerFIBRAMQ12,
        'FSHOP13': text_handlers.TextHandlerFSHOP13,
        }
    dividend_data = []
    for folder, handler in asset_map.items():
        for file in os.listdir(pathlib.Path(REPORTS_STORE_PATH).joinpath(folder + '/')):
            curr_file_path = pathlib.Path(REPORTS_STORE_PATH).joinpath(folder).joinpath(file)
            current_data = extract_dividend_data(handler, curr_file_path)
            dividend_data.append(current_data)

    return dividend_data

def update_dividend_data():
    extracted_data = get_aggregared_dividend_data()
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
        