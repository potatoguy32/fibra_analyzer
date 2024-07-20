# Local stuff
from src.data import make_dataset
import datetime
import math
from functools import reduce
import pathlib
import os

# Data wrangling and optimizatin stuff
import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Load environment variables
from dotenv import load_dotenv
DATA_FOLDER_PATH = pathlib.Path(os.environ['DATA_FOLDER_PATH'])
load_dotenv()


def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.sum(weights * returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_volatility


class PortfolioOptimizer:
    def __init__(self, tickers: list) -> None:
        self.tickers = list(tickers)
        self.weights = np.ones(len(tickers)) / len(tickers)
        self.datasets = None
        self.df_prices = None
        pass

    def make_tickers_dataset(self, start_date: str, end_date: str, prediction_window: int = 20):
        df_prices_holder = []
        self.datasets = []
        for fibra in self.tickers:
            current_df = make_dataset.get_ticker_dataset(fibra, start_date, end_date, prediction_window)
            self.datasets.append(current_df)
            df_prices_holder.append(current_df.query("source == 'actual'").avg_price)

        self.df_prices = pd.concat(df_prices_holder, axis=1, ignore_index=True)
        return
    
    def set_evaluation_dates(self):
        self.evaluation_dates = [df.date.max().date() for df in self.datasets]
        return
    
    def get_evaluation_dates(self):
        return list(self.evaluation_dates)

    def get_df_prices(self):
        return self.df_prices.copy()

    def get_datasets_list(self):
        return [df.copy() for df in self.datasets]

    def compute_optimal_weights(self):
        expected_returns = []
        expected_risks = []
        for df in self.datasets:
            asseet_std = df.query("source == 'actual'").avg_price.diff().std()
            current_price = df.query("source == 'actual'").iloc[-1].avg_price
            predicted_price = df.query("source == 'prediction'").iloc[-1].avg_price
            period_dividends = df.query("source == 'prediction'").dividends.sum()
            future_price_diff = predicted_price - current_price
            expected_pnl = future_price_diff + period_dividends

            expected_returns.append(expected_pnl)
            expected_risks.append(asseet_std)
        
        cov_matrix = self.df_prices.pct_change().cov()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))

        optimized = minimize(lambda x: -sharpe_ratio(x, expected_returns, cov_matrix), 
                            self.weights, method='SLSQP', bounds=bounds, constraints=constraints)

        self.weights =  optimized.x
        return
    
    def get_weights(self):
        return [round(item, 4) for item in self.weights]
    
    def run_backtest(self, backtest_start_date, backtest_end_date, window, initial_investment):
        trailing_days = 450
        balance = initial_investment

        pnl_log = {}
        portfolio_share_log = {}
        portfolio_composition_log = {}
        dividends_log = {}
        
        end_date = backtest_start_date.date()
        start_date = end_date - datetime.timedelta(days=trailing_days)

        # Get initial values
        self.make_tickers_dataset(start_date, end_date, window)
        self.compute_optimal_weights()
        
        buy_date = min([df.query("source == 'actual'").date.max().date() for df in self.datasets])
        buy_prices = [df.get(df.date >= str(buy_date)).avg_price.values[0] for df in self.datasets]
        forecast_end_date = min([df.date.max().date() for df in self.datasets])

        last_weights = self.get_weights()
        share_number = [math.floor(balance * weight / price) for weight, price in zip(last_weights, buy_prices)]
        reminder = sum([(balance * weight) - (price * math.floor(balance * weight / price))  for weight, price in zip(last_weights, buy_prices)])
        
        iter_count = 0
        while True:
            self.make_tickers_dataset(start_date, forecast_end_date, window)
            self.compute_optimal_weights()

            evaluation_date = max([df.query("source == 'actual'").date.max().date() for df in self.datasets])
            sell_prices = [df.get(df.date == str(evaluation_date)).avg_price.values[-1] for df in self.datasets]

            sells = [math.floor(np.nan_to_num(min(shares * (new/last - 1), 0), nan=0)) for last, new, shares in zip(last_weights, self.get_weights(), share_number)]
            dividends = [df.get(df.date.between(str(buy_date), str(evaluation_date), inclusive='both')).dividends.sum() * share_quantity for df, share_quantity in zip(self.datasets, share_number)]
            balance = -sum([number * sell_price for number, sell_price in zip(sells, sell_prices)]) + reminder + sum(dividends)

            buy_date = min([df.query("source == 'actual'").date.max().date() for df in self.datasets])
            buy_prices = [df.get(df.date >= str(buy_date)).avg_price.values[0] for df in self.datasets]
            forecast_end_date = min([df.date.max().date() for df in self.datasets])

            new_shares = [math.floor(balance * weight / price) for weight, price in zip(self.get_weights(), sell_prices)]
            new_share_number = [last_number + sell_number + new_number for new_number, last_number, sell_number in zip(new_shares, share_number, sells)]
            reminder = sum([(balance * weight) - (price * math.floor(balance * weight / price))  for weight, price in zip(self.get_weights(), buy_prices)])

            current_pnl = sum([share * price for share, price in zip(new_share_number, sell_prices)]) + reminder - initial_investment
            
            pnl_log[str(evaluation_date)] = current_pnl
            portfolio_share_log[str(evaluation_date)] = self.get_weights()
            portfolio_composition_log[str(evaluation_date)] = new_share_number
            dividends_log[str(evaluation_date)] = sum(dividends)


            end_date = evaluation_date
            if (end_date >= backtest_end_date.date()) or (iter_count == 1000):
                break

        datafrmaes = [
        pd.DataFrame(portfolio_share_log.items(), columns=['date', 'portfolio_share']),
        pd.DataFrame(portfolio_composition_log.items(), columns=['date', 'portfolio_composition']),
        pd.DataFrame(dividends_log.items(), columns=['date', 'paid_dividends']),
        pd.DataFrame(pnl_log.items(), columns=['date', 'pnl']),
        ]

        summary_dataframe = reduce(lambda left, right: pd.merge(left, right), datafrmaes)
        summary_dataframe.insert(0, 'tickers', [self.tickers for _ in range(len(summary_dataframe))])
        summary_dataframe['tickers'] = summary_dataframe.tickers.apply(lambda x: [val.split('.')[0] for val in x])
        self.backtest_summary_df = summary_dataframe
        return
    
    # def run_backtest(self, backtest_start_date, backtest_end_date, window, innitial_investment):
    #     trailing_days = 450

    #     pnl_log = {}
    #     portfolio_share_log = {}
    #     portfolio_composition_log = {}
    #     dividends_log = {}

    #     end_date = backtest_start_date.date()
    #     start_date = end_date - datetime.timedelta(days=trailing_days)

    #     self.make_tickers_dataset(str(start_date), str(end_date), window)
        
    #     # Get initial values and number of shares
    #     # buy_date = study_start_date.date()
    #     buy_date = min([df.query("source == 'actual'").date.max().date() for df in self.datasets])
    #     evaluation_date = min([df.date.max().date() for df in self.datasets])
    #     buy_prices = [df.get(df.date >= str(buy_date)).avg_price.values[0] for df in self.datasets]
    #     sell_prices = [df.get(df.date == str(evaluation_date)).avg_price.values[-1] for df in self.datasets]

    #     last_weights = self.get_weights()
    #     share_number = [math.floor(innitial_investment * weight / price) for weight, price in zip(last_weights, buy_prices)]
    #     reminder = sum([(innitial_investment * weight) - price * math.floor(innitial_investment * weight / price)  for weight, price in zip(last_weights, buy_prices)])

    #     # Optmize weights with forecasted prices
    #     self.compute_optimal_weights()

    #     # Rebalance portfolio and compute values
    #     sells = [math.floor(np.nan_to_num(min(shares * (new/last - 1), 0), nan=0)) for last, new, shares in zip(last_weights, self.get_weights(), share_number)]
    #     dividends = [df.query("source == 'prediction'").dividends.sum() * share_quantity for df, share_quantity in zip(self.datasets, share_number)]
    #     new_balance = -sum([number * sell_price for number, sell_price in zip(sells, sell_prices)]) + reminder + sum(dividends)
    #     new_shares = [math.floor(new_balance * weight / price) for weight, price in zip(self.get_weights(), sell_prices)]
    #     new_share_number = [last_number + sell_number + new_number for new_number, last_number, sell_number in zip(new_shares, share_number, sells)]

    #     current_pnl = sum([share * price for share, price in zip(new_share_number, sell_prices)]) + reminder - innitial_investment

    #     pnl_log[str(evaluation_date)] = current_pnl
    #     portfolio_share_log[str(evaluation_date)] = self.get_weights()
    #     portfolio_composition_log[str(evaluation_date)] = new_share_number
    #     dividends_log[str(evaluation_date)] = dividends
        
    #     iter_count = 0
    #     while True:
    #         global last_rebalance
    #         last_rebalance = end_date

    #         end_date = evaluation_date
    #         self.make_tickers_dataset(str(start_date), str(end_date), window)

    #         # Get initial values
    #         sell_prices = [df.get(df.date == str(evaluation_date)).avg_price.values[-1] for df in self.datasets]
    #         evaluation_date = min([df.date.max().date() for df in self.datasets])

    #         # Get initial/current number of shares
    #         last_weights = self.get_weights()
    #         share_number = list(new_share_number)

    #         # Optmize weights with forecasted prices
    #         self.compute_optimal_weights()

    #         # Rebalance portfolio and compute values
    #         sells = [math.floor(np.nan_to_num(min(shares * (new/last - 1), 0), nan=0)) for last, new, shares in zip(last_weights, self.get_weights(), share_number)]
    #         dividends = [df.query("date > @last_rebalance").dividends.sum() * share_quantity for df, share_quantity in zip(self.datasets, share_number)]
    #         new_balance = -sum([number * sell_price for number, sell_price in zip(sells, sell_prices)]) + sum(dividends) + reminder
    #         new_shares = [math.floor(new_balance * weight / price) for weight, price in zip(self.get_weights(), sell_prices)]
    #         new_share_number = [last_number + sell_number + new_number for new_number, last_number, sell_number in zip(new_shares, share_number, sells)]
    #         reminder = sum([(new_balance * weight) - (price * math.floor(new_balance * weight / price))  for weight, price in zip(self.get_weights(), sell_prices)])

    #         current_pnl = sum([share * price for share, price in zip(new_share_number, sell_prices)]) + reminder - innitial_investment
            
    #         pnl_log[str(evaluation_date)] = current_pnl
    #         portfolio_share_log[str(evaluation_date)] = self.get_weights()
    #         portfolio_composition_log[str(evaluation_date)] = new_share_number
    #         dividends_log[str(evaluation_date)] = dividends

    #         iter_count += 1
    #         if (end_date >= backtest_end_date.date()) or (iter_count >= 1000):
    #             break

    #     datafrmaes = [
    #         pd.DataFrame(portfolio_share_log.items(), columns=['date', 'portfolio_share']),
    #         pd.DataFrame(portfolio_composition_log.items(), columns=['date', 'portfolio_composition']),
    #         pd.DataFrame(dividends_log.items(), columns=['date', 'paid_dividends']),
    #         pd.DataFrame(pnl_log.items(), columns=['date', 'pnl']),
    #         ]

    #     summary_dataframe = reduce(lambda left, right: pd.merge(left, right), datafrmaes)
    #     summary_dataframe.insert(0, 'tickers', [self.tickers for _ in range(len(summary_dataframe))])
    #     summary_dataframe['tickers'] = summary_dataframe.tickers.apply(lambda x: [val.split('.')[0] for val in x])
    #     self.backtest_summary_df = summary_dataframe
    #     summary_dataframe.to_csv(DATA_FOLDER_PATH.joinpath('processed/backtest_summary.csv'), index=False)
    #     return