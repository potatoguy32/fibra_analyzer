import sqlite3
import os
import pathlib
import re

import requests
from bs4 import BeautifulSoup

import dateparser

from dotenv import load_dotenv
load_dotenv()

DATABASE_PATH = pathlib.Path(os.environ['DATABASE_PATH'])
REPORTS_STORE_PATH = pathlib.Path(os.environ['REPORTS_STORE_PATH'])


class BaseScrapper:
    def __init__(self) -> None:
        self.get_latest_dividend_info()
        self.create_dir()
        return
    
    def update_financial_reports(self) -> None:
        self.scrap_financial_reports()
        self.save_financial_reports()
        print(f"Sucessfully saved {len(self.distributions_info)} pdf reports.")
        return

    def save_financial_reports(self) -> None:
        for link in self.distributions_info.values():
            file_name = link.split('/')[-1]
            file_path = self.reports_path.joinpath(file_name)

            response = requests.get(link)

            with open(file_path, "wb") as f:
                f.write(response.content)

        return
    
    def create_dir(self):
        if not self.reports_path.exists():
            self.reports_path.mkdir(parents=True)
        return

    def get_latest_dividend_info(self) -> None:
        query = "SELECT ticker, max(announcement_date) FROM dividends WHERE ticker = '{}' GROUP BY ticker;".format(self.ticker)
        with sqlite3.connect(DATABASE_PATH) as conn:
            cur = conn.cursor()
            self.last_dividend_info = cur.execute(query).fetchone()


class ScrapperFMTY14(BaseScrapper):
    def __init__(self):
        self.ticker = 'FMTY14.MX'
        self.reports_path = REPORTS_STORE_PATH.joinpath('FMTY14')
        self.URL = 'https://www.fibramty.com/distribuciones'
        super().__init__()
        return

    def scrap_financial_reports(self) -> None:
        page = requests.get(self.URL)
        soup = BeautifulSoup(page.content, "html.parser")
        distributions_table = soup.find("div", class_="table-responsive")

        links = distributions_table.find_all("a")
        dates = distributions_table.find_all("td", class_="text-right")

        if len(links) > 5:
            links = links[:5]
            dates = dates[:5]

        distributions_info = {date.text: link['href'] for date, link in zip(dates, links)}
        
        if isinstance(self.last_dividend_info, tuple):
            dict_keys = list(distributions_info.keys())
            [distributions_info.pop(key) for key in dict_keys if key <= self.last_dividend_info[1]]

        self.distributions_info = distributions_info
        return


class ScrapperFIBRAPL14(BaseScrapper):
    def __init__(self) -> None:
        self.ticker = 'FIBRAPL14.MX'
        self.reports_path = REPORTS_STORE_PATH.joinpath('FIBRAPL14')
        self.URL = 'https://www.fibraprologis.com/es-MX/inversionistas/informacion-fiscal'
        super().__init__()
        return

    def scrap_financial_reports(self) -> None:
        page = requests.get(self.URL)
        soup = BeautifulSoup(page.content, "html.parser")
        distributions_table = soup.find("div", class_="table-wrapper")
        table_items = distributions_table.find_next('tbody').find_all('tr')

        distributions_info = {
            date.text: link.a['href'] for date, title, link in 
            [entry.find_all('td') for entry in table_items]
            if title.text == 'Qualified Notice – Sec. 1446(a) and (f)'
                              }

        self.distributions_info = distributions_info
        return


class ScrapperFNOVA17(BaseScrapper):
    def __init__(self) -> None:
        self.ticker = 'FNOVA17.MX'
        self.reports_path = REPORTS_STORE_PATH.joinpath('FNOVA17')
        self.URL = 'https://www.fibra-nova.com/bursatil/distribuciones'
        super().__init__()
        return

    def scrap_financial_reports(self) -> None:
        page = requests.get(self.URL)
        soup = BeautifulSoup(page.content, "html.parser")

        table_elements = soup.find_all('tr')[1:6]
        
        self.distributions_info = {date.text: link.a['href'] for link, date in [element.find_all('td') for element in table_elements]}
        return


class ScrapperDANHOS13(BaseScrapper):
    def __init__(self) -> None:
        self.ticker = 'DANHOS13.MX'
        self.reports_path = REPORTS_STORE_PATH.joinpath('DANHOS13')
        self.URL = 'https://www.fibradanhos.com.mx/inversionistas/recursosanalistas'
        super().__init__()
        return

    def scrap_financial_reports(self) -> None:
        regex = re.compile('((\d{1,2})\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre))', re.I)
        page = requests.get(self.URL)
        soup = BeautifulSoup(page.content, "html.parser")

        table_elements = soup.find('div', class_='uk-accordion-content').find_all('td')[:10]
        
        distributions_info = {regex.search(element.text).group(0) + ' ' + element.parent.parent.parent.parent.find('a', class_='uk-accordion-title').text: element.a['href']
                              for element in table_elements if 'distribución' in element.text.lower()}
        
        distributions_info = {dateparser.parse(date, languages=['es']).strftime('%Y-%m-%d'): link for date, link in distributions_info.items()}
        self.distributions_info = distributions_info
        return