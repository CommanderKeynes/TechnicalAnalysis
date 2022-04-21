

import csv
import logging
from ftplib import FTP
from io import BytesIO, TextIOWrapper
from typing import List
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import declarative_base


Base = declarative_base()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class NasdaqListed(Base):

    __tablename__ = 'NasdaqListed'

    symbol = sqlalchemy.Column(sqlalchemy.String)
    security_name = sqlalchemy.Column(sqlalchemy.String)
    market_category = sqlalchemy.Column(sqlalchemy.String)
    test_issue = sqlalchemy.Column(sqlalchemy.String)
    financial_status = sqlalchemy.Column(sqlalchemy.String)
    round_lot_size = sqlalchemy.Column(sqlalchemy.String)
    etf = sqlalchemy.Column(sqlalchemy.String)
    next_shares = sqlalchemy.Column(sqlalchemy.String)


def get_nasdaq_listed_from_ftp() -> pd.DataFrame:
    logger.info('Querying for tickers.')
    raw = BytesIO()

    # FTP locations taken from https://quant.stackexchange.com/a/1862.
    with FTP('ftp.nasdaqtrader.com') as conn:
        conn.login()
        conn.retrbinary('RETR SymbolDirectory/nasdaqlisted.txt', raw.write)

    raw.seek(0)
    reader = csv.DictReader(TextIOWrapper(raw), delimiter='|')

    # After http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs:
    # - data['Test Issue'] == 'N' indicates the asset is not a test security
    # - data['Financial Status'] == 'N' indicates the asset is normal, and
    #   continues to meet the Nasdaq's requirements for listing.
    tickers = [row for row in reader
               if row['Test Issue'] == 'N'
               if row['Financial Status'] == 'N']

    tickers_df = pd.DataFrame(tickers)

    return tickers_df


def main():

    tickers_df = get_nasdaq_listed_from_ftp()
    tickers_df.to_csv('nasdaq_securities.csv', index=False, mode='a')


if __name__ == '__main__':
    main()
