def create_db(connection):
    # Create (connect) database
    cur = connection.cursor()

    # Create users and passwords tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        ticker TEXT NOT NULL,
        "date" DATE NOT NULL,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume FLOAT,
        dividends FLOAT,
        stock_splits FLOAT,
        PRIMARY KEY(ticker, date) ON CONFLICT REPLACE
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dividends (
        ticker TEXT NOT NULL,
        announcement_date DATE NOT NULL,
        dividend_date DATE NOT NULL,
        dividend_amount FLOAT,
        PRIMARY KEY(ticker, announcement_date, dividend_date) ON CONFLICT REPLACE
    );
    """)
    connection.commit()
    return

def load_market_data(connection, data):
    create_db(connection=connection)
    cur = connection.cursor()
    cur.executemany("""INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
    connection.commit()
    return

def get_tickers_last_dates(connection):
    cur = connection.cursor()
    res = cur.execute("SELECT ticker, MAX(date) as latest_date from market_data group by ticker;")
    return res.fetchall()

def delete_tickers(connection, tickers):
    if len(tickers) == 1 and isinstance(tickers, list):
        tickers = tickers.pop()

    if isinstance(tickers, str):
        query_string = """DELETE FROM market_data
            WHERE ticker = '{}';""".format(tickers)
    elif isinstance(tickers, list):
        query_string = """DELETE FROM market_data
            WHERE ticker in {};""".format(tuple(tickers))
    
    cur = connection.cursor()
    cur.execute(query_string)
    connection.commit()
    return

def load_dividends(connection, data):
    create_db(connection=connection)
    cur = connection.cursor()
    cur.executemany("""INSERT INTO dividends VALUES (?, ?, ?, ?)""", data)
    return