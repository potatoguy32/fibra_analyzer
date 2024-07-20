import os
import re
import datetime
import warnings

from dotenv import load_dotenv
import pathlib

from pydantic import BaseModel

import dateparser

from pypdf import PdfReader
from tabula import read_pdf

import pandas as pd

load_dotenv()
warnings.simplefilter(action='ignore', category=FutureWarning)

UTILITY_FILES_PATH = os.environ['UTILITY_FILES_PATH']


class DividendModel(BaseModel):
    ticker: str
    announcement_date: datetime.date
    dividend_date: datetime.date
    dividend_amount: float
    
    def dump_tuple(self):
        model_dict = self.model_dump()
        model_dict.update({
            'announcement_date': str(model_dict['announcement_date']),
            'dividend_date': str(model_dict['dividend_date']),
        })
        
        return tuple(model_dict.values())


class TextHandler:
    def __init__(self, path) -> None:
        self.matching_dict = {
            'ticker': None,
            'announcement_date': None,
            'dividend_date': None,
            'dividend_amount': None,
            # 'dividend_type': None,
            }
        
        self.file_path = str(path)
        self.reader = PdfReader(path)
        return

class TextHandlerFMTY14(TextHandler):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.matching_dict.update({'ticker': 'FMTY14.MX'})
        self.distribution_regex = re.compile('(distribución)(.|\n])*(CBFI)', re.I)
        self.payment_date_regex = re.compile('fecha de pago' , re.I)
        self.announcement_date_regex = re.compile('\d{2}/\d{2}/\d{4}' , re.I)
        self.payment_type_regex = re.compile('proveniente de (reembolso de capital|resultado fiscal)' , re.I)
        return
    
    def get_dividend_data(self):
        for line in re.split('\n', self.reader.pages[-1].extract_text()): # r'\b. \n\b'
            matching_distribution = self.distribution_regex.search(line)
            matching_date = self.payment_date_regex.search(line)
            matching_announcement = self.announcement_date_regex.search(line)
            matching_type = self.payment_type_regex.search(line)
            if matching_distribution:
                self.matching_dict['dividend_amount'] = line.split('$ ')[-1]
            elif matching_date:
                self.matching_dict['dividend_date'] = dateparser.parse(line.split(': ')[-1], languages=['es']).strftime('%Y-%m-%d')
            # elif matching_type:
            #     self.matching_dict['dividend_type'] = 'capital' if 'capital' in matching_type.group(0).lower() else 'fiscal'
            elif matching_announcement:
                self.matching_dict['announcement_date'] = datetime.datetime.strptime(matching_announcement.group(0), '%d/%m/%Y').strftime('%Y-%m-%d')
            else:
                continue
        return dict(self.matching_dict)


class TextHandlerFIBRAPL14(TextHandler):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.matching_dict.update({'ticker': 'FIBRAPL14.MX'})
        self.dates_regex = re.compile('((January|February|March|April|May|June|July|August|September|October|November|December)\s+([1-9]|[12][0-9]|3[01])\s*,\s+(\d{4}))', re.I)
        self.dividends_table = read_pdf(self.file_path, pages='all')[0]
        self.exchange_rate_path = pathlib.Path(UTILITY_FILES_PATH).joinpath('exchange_rate.csv')
        self.exchange_rates_df = pd.read_csv(self.exchange_rate_path, header=None, names=['date', 'rate'])
        self.exchange_rates_df['date'] = pd.to_datetime(self.exchange_rates_df['date'], format='%d/%m/%Y')
        return
    
    def get_dividend_data(self):
        for line in self.reader.pages[0].extract_text().split('\n \n'):
            matching_regex = self.dates_regex.findall(line)
            if len(matching_regex) > 1:
                if len(self.reader.pages[0].extract_text().split('\n \n')) == 1:
                    matching_regex.pop(0)

                announcement_date = matching_regex[0][0].replace(' ,', ',').replace('  ', ' ')
                dividend_date = matching_regex[1][0].replace(' ,', ',').replace('  ', ' ')
                self.matching_dict['announcement_date'] = datetime.datetime.strptime(announcement_date, '%B %d, %Y').strftime('%Y-%m-%d')
                self.matching_dict['dividend_date'] = datetime.datetime.strptime(dividend_date, '%B %d, %Y').strftime('%Y-%m-%d')
        
        self.dividend_amount = self.dividends_table.iloc[:, 1].apply(lambda x: x.split('$')[-1]).astype(float).sum()
        dividend_date_df = str(self.matching_dict['dividend_date'])
        self.dividend_exchange_rate = self.exchange_rates_df.query("date == @dividend_date_df").rate.values[0]
        self.matching_dict['dividend_amount'] = str(round(self.dividend_amount * self.dividend_exchange_rate, 4))
        return dict(self.matching_dict)
    
class TextHandlerFNOVA17(TextHandler):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.matching_dict.update({'ticker': 'FNOVA17.MX'})
        self.payment_date_regex = re.compile('fecha de pago' , re.I)
        self.announcement_date_regex = re.compile('\d{2}/\d{2}/\d{4}' , re.I)
        self.payment_type_regex = re.compile('Concepto del Aviso de Distribución: (Reembolso de Capital|Resultado Fiscal)' , re.I)
        return
    
    def get_dividend_data(self):
        tables = read_pdf(self.file_path)
        self.matching_dict['dividend_amount'] = tables[2].loc[0, 'IMPORTE']
        for line in re.split('\n', self.reader.pages[-1].extract_text()): # r'\b. \n\b'
            matching_date = self.payment_date_regex.search(line)
            matching_announcement = self.announcement_date_regex.search(line)
            matching_type = self.payment_type_regex.search(line)
            if matching_date:
                self.matching_dict['dividend_date'] = dateparser.parse(line.split(': ')[-1], languages=['es']).strftime('%Y-%m-%d')
            # elif matching_type:
            #     self.matching_dict['dividend_type'] = 'capital' if 'capital' in matching_type.group(0).lower() else 'fiscal'
            elif matching_announcement:
                self.matching_dict['announcement_date'] = datetime.datetime.strptime(matching_announcement.group(0), '%d/%m/%Y').strftime('%Y-%m-%d')
            else:
                continue
        return dict(self.matching_dict)
    
class TextHandlerFIBRAMQ12(TextHandler):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.matching_dict.update({'ticker': 'FIBRAMQ12.MX'})
        self.distribution_regex = re.compile('((\d{1,2})(.|\n)*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(.|\n)*(FIBRAMQ declaró una distribución)(.|\n)*(ps\.)(.|\n)*(0\.\d+)(.|\n)*(\d{1,2})(.|\n)*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre))', re.I)
        self.date_regex = re.compile('((\d{1,2})( de )(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)( de )(\d{4}))', re.I)
        self.dividend_regex = re.compile('Ps\. \d\.\d+')
        self.payment_type_regex = re.compile('proveniente de (reembolso de capital|resultado fiscal)' , re.I)
        return
    
    def get_dividend_data(self):
        for page in self.reader.pages:
            current_page = page.extract_text()
            for paragraph in current_page.split('\n \n'):
                if self.distribution_regex.findall(paragraph):
                    matching_paragraph = str(paragraph)
                    break

        continuous_paragraph = self.distribution_regex.findall(matching_paragraph)[0][0].replace('\n', '').replace('  ', ' ')
        matching_regex = self.date_regex.findall(continuous_paragraph)
        announcement_date, payment_date = str(matching_regex[0][0]), str(matching_regex[2][0])
        self.matching_dict['announcement_date'] = dateparser.parse(announcement_date, languages=['es']).strftime('%Y-%m-%d')
        self.matching_dict['dividend_date'] = dateparser.parse(payment_date, languages=['es']).strftime('%Y-%m-%d')
        self.matching_dict['dividend_amount'] = self.dividend_regex.findall(continuous_paragraph).pop().split()[-1]
        return dict(self.matching_dict)

class TextHandlerFSHOP13(TextHandler):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.matching_dict.update({'ticker': 'FSHOP13.MX'})
        self.dividend_regex =re.compile('\d\.\d+ pesos por CBFI', re.I)
        self.dates_regex = re.compile('((\d{1,2})( de )(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre))', re.I)
        self.payment_type_regex = re.compile('(reembolso de capital|resultado fiscal)' , re.I)
        return
    
    def get_dividend_data(self):
        page_text = self.reader.pages[0].extract_text().replace('\n', ' ').replace('  ', ' ')
        holder = []
        [holder.append(entry) for entry in self.dates_regex.findall(page_text) if entry not in holder]
        dividend_amount = self.dividend_regex.findall(page_text).pop(0).split()[0]
        announcement_date = holder.pop(0)[0]
        dividend_date = holder.pop(0)[0]
        self.matching_dict.update({
            'announcement_date': dateparser.parse(announcement_date, languages=['es']).strftime('%Y-%m-%d'),
            'dividend_date': dateparser.parse(dividend_date, languages=['es']).strftime('%Y-%m-%d'),
            'dividend_amount': dividend_amount
        })

        return dict(self.matching_dict)


class TextHandlerDANHOS13(TextHandler):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.matching_dict.update({'ticker': 'DANHOS13.MX'})
        self.dividend_regex = re.compile('\$(\d\.\d+ pesos)', re.I)
        self.payment_date_regex = re.compile('(el (\d{1,2})( de )(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)( de )(\d{4}))', re.I)
        self.payment_type_regex = re.compile('(reembolso de capital|resultado fiscal)' , re.I)
        return
    
    def get_dividend_data(self):
        page_text = self.reader.pages[0].extract_text().replace('\n', ' ').replace('  ', ' ')
        dividend_amount = self.dividend_regex.findall(page_text)[0].split(' ')[0]
        dividend_date = dateparser.parse(self.payment_date_regex.findall(page_text)[0][0].replace('el', '').strip(), languages=['es']).strftime('%Y-%m-%d')
        self.matching_dict.update({
            'announcement_date': None,
            'dividend_date': dividend_date,
            'dividend_amount': dividend_amount
        })

        return dict(self.matching_dict)