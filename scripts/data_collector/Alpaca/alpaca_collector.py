# alpaca_collector.py

import abc
import os
import sys
import copy
import time
import datetime
import multiprocessing
from pathlib import Path
from typing import Iterable, List

import fire
import pandas as pd
import numpy as np
from loguru import logger

# --- Alpaca-py Imports ---
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# --- Qlib Imports ---
# Make sure qlib is installed: pip install qlib
try:
    import qlib
    from qlib.data import D
    from qlib.tests.data import GetData
    from qlib.utils import exists_qlib_data
except ImportError:
    print("Please install qlib: 'pip install qlib'")
    sys.exit(1)

# --- Local Imports from Original Script ---
# These are supporting components from the qlib data collector scripts.
# For simplicity, they are included directly here.
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    generate_minutes_calendar_from_daily,
    calc_adjusted_price, # Used for 1-min data normalization
)


class AlpacaCollector(BaseCollector):
    """
    Collector for fetching US stock data from Alpaca.

    NOTE: Alpaca's data is already adjusted for splits and dividends by default.
    """

    # How many times to retry on API failure.
    retry = 5

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        # --- Alpaca-specific arguments ---
        api_key: str = None,
        api_secret: str = None,
        paper: bool = True,
    ):
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        # Handle Alpaca API keys
        if api_key is None:
            api_key = os.getenv("APCA_API_KEY_ID")
        if api_secret is None:
            api_secret = os.getenv("APCA_API_SECRET_KEY")
        if api_key is None or api_secret is None:
            raise ValueError("Alpaca API key and secret are required. Pass them as arguments or set environment variables APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

        # Initialize Alpaca clients
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.init_datetime()

    def init_datetime(self):
        # Alpaca requires timezone-aware datetime objects
        self.start_datetime = pd.Timestamp(self.start_datetime, tz=self._timezone)
        self.end_datetime = pd.Timestamp(self.end_datetime, tz=self._timezone)

    @property
    def _timezone(self):
        return "America/New_York"

    def get_instrument_list(self) -> List[str]:
        """
        Get all active, tradable US equity symbols from Alpaca.
        Includes major index ETFs as substitutes for indices like ^GSPC.
        """
        logger.info("Getting active US equity symbols from Alpaca...")
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = self.trading_client.get_all_assets(search_params)
        symbols = [asset.symbol for asset in assets if asset.tradable]
        
        # Add major ETFs that track indices
        index_etfs = ["SPY", "QQQ", "DIA"] 
        for etf in index_etfs:
            if etf not in symbols:
                symbols.append(etf)
        
        logger.info(f"Obtained {len(symbols)} tradable US stock symbols.")
        return symbols
    
    def normalize_symbol(self, symbol: str) -> str:
        """Alpaca uses standard uppercase symbols."""
        return symbol.upper()

    def get_data_from_remote(self, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch data for a single symbol from Alpaca.
        """
        if interval.lower() in ["1d", "day"]:
            timeframe = TimeFrame.Day
        elif interval.lower() in ["1min", "minute"]:
            timeframe = TimeFrame.Minute
        else:
            raise ValueError(f"Interval '{interval}' is not supported by this collector.")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment='all' # Default, ensures data is adjusted for splits and dividends
        )
        
        try:
            bars_df = self.stock_client.get_stock_bars(request_params).df
            if not bars_df.empty:
                bars_df = bars_df.reset_index()
                # Rename columns to match Qlib's expected format
                bars_df = bars_df.rename(columns={"timestamp": "date", "trade_count": "count"})
                bars_df["date"] = pd.to_datetime(bars_df["date"])
                return bars_df
        except APIError as e:
            logger.warning(f"Failed to fetch data for {symbol} from Alpaca: {e}")
        
        return pd.DataFrame()

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Wrapper for fetching data with retry logic.
        """
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple():
            self.sleep()
            resp = self.get_data_from_remote(symbol, interval, start_datetime, end_datetime)
            if resp is None or resp.empty:
                raise ValueError(f"No data fetched for {symbol}.")
            return resp

        try:
            return _get_simple()
        except ValueError as e:
            logger.warning(f"Giving up on {symbol}: {e}")
            return pd.DataFrame()

    def download_index_data(self):
        """
        Index data (e.g., SPY, QQQ) is collected as part of the main `get_instrument_list`
        and `collector_data` process. This method is not needed.
        """
        pass


class AlpacaCollectorUS1d(AlpacaCollector):
    pass


class AlpacaCollectorUS1min(AlpacaCollector):
    pass


class AlpacaNormalize(BaseNormalize):
    """
    Base class for normalizing data from Alpaca.
    """
    COLUMNS = ["open", "close", "high", "low", "volume"]

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_alpaca(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        """
        Shared normalization logic.
        """
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None) # Remove timezone
        df = df[~df.index.duplicated(keep="first")]

        if calendar_list is not None:
            df = df.reindex(pd.to_datetime(calendar_list))
        
        df.sort_index(inplace=True)
        df["change"] = AlpacaNormalize.calc_change(df, last_close)
        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalize_alpaca(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        return df

class AlpacaNormalize1d(AlpacaNormalize):

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("US_ALL")

    def _get_first_close(self, df: pd.DataFrame) -> float:
        df = df.loc[df["close"].first_valid_index() :]
        return df["close"].iloc[0] if not df.empty else 1.0

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function is kept for compatibility with Qlib's data format, which often
        standardizes prices relative to the first day's value.
        """
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)
        for _col in df.columns:
            if _col in [self._symbol_field_name, "change", "factor"]:
                continue
            # Volume is treated differently in the original script.
            # Here we assume the provided volume is the actual traded volume.
            if _col != "volume": 
                df[_col] = df[_col] / _close
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().normalize(df)
        df = self._manual_adj_data(df)
        # Alpaca data is already adjusted, so the adjustment factor is 1.0.
        # This column is added for compatibility with some Qlib features.
        df["factor"] = 1.0
        return df


class AlpacaNormalize1dExtend(AlpacaNormalize1d):

    def __init__(self, old_qlib_data_dir: [str, Path], **kwargs):
        super().__init__(**kwargs)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Perform standard 1d normalization on the new data chunk.
        df = super(AlpacaNormalize1d, self).normalize(df) # Call parent's normalize
        df = self._manual_adj_data(df) # Call parent's manual adjustment
        
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        
        if str(symbol_name).upper() not in old_symbol_list:
            logger.warning(f"Symbol {symbol_name} not in old Qlib data, cannot extend. Returning new data only.")
            return df.reset_index()
            
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        
        # Filter new data to start after the last date of old data
        df = df.loc[df.index > latest_date]
        if df.empty:
            return pd.DataFrame() # No new data to append

        # Step 2: Rescale the new data to be continuous with the old data.
        # This is crucial because `_manual_adj_data` scales each chunk independently.
        first_new_day = df.index[0]
        last_old_day_data = old_df.loc[latest_date]
        
        # Get the unscaled first-day close from the new data (before manual adjustment)
        # This requires re-reading the raw source file.
        raw_df = pd.read_csv(self.source_dir.joinpath(f"{symbol_name}.csv"), parse_dates=[self._date_field_name])
        raw_df = raw_df.set_index(self._date_field_name).sort_index()
        first_new_close_unscaled = raw_df.loc[first_new_day]["close"]
        last_old_close_scaled = last_old_day_data["close"]

        # Calculate the scaling ratio needed to connect the series
        # Ratio = (last day of old series) / (first day of new series)
        scaling_ratio = last_old_close_scaled / first_new_close_unscaled

        # Apply this scaling ratio to all OHLC columns in the new data chunk
        for col in ["open", "high", "low", "close"]:
            df[col] = raw_df.loc[df.index, col] * scaling_ratio
        
        # Volume and factor don't need this scaling
        df["volume"] = raw_df.loc[df.index, "volume"]
        df["factor"] = 1.0
        
        # Recalculate change based on the new continuous series
        last_close = last_old_day_data["close"]
        df["change"] = self.calc_change(df, last_close=last_close)

        return df.reset_index()


class AlpacaNormalize1min(AlpacaNormalize):

    def __init__(self, qlib_data_1d_dir: [str, Path], **kwargs):
        super().__init__(**kwargs)
        if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
            raise ValueError("`qlib_data_1d_dir` is required for 1-min normalization.")
        qlib.init(provider_uri=qlib_data_1d_dir)
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        self.calendar_list_1d = list(D.calendar(freq="day"))

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return generate_minutes_calendar_from_daily(self.calendar_list_1d, freq="1min")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For 1-min data, the main task is re-indexing to a full trading-day calendar.
        Price adjustment is not needed as Alpaca data is already adjusted.
        """
        df = super().normalize(df)
        # Add factor=1.0 for compatibility
        df["factor"] = 1.0
        # The `calc_adjusted_price` utility can still be useful for aligning
        # with the 1d calendar and calculating `paused` status.
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="1min",
            consistent_1d=True, # Align 1min calendar with 1d
            calc_paused=False, # Paused calculation can be complex, disabled for US
            _1d_data_all=self.all_1d_data,
        )
        return df


class Run(BaseRun):
    """
    Task runner for collecting and processing US stock data from Alpaca.
    """
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"AlpacaCollectorUS{self.interval}"

    @property
    def normalize_class_name(self):
        return f"AlpacaNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
        api_key: str = None,
        api_secret: str = None,
        paper: bool = True
    ):
        """
        Download data from Alpaca.

        Parameters
        ----------
        api_key: str
            Alpaca API Key. Can also be set by env var APCA_API_KEY_ID.
        api_secret: str
            Alpaca API Secret. Can also be set by env var APCA_API_SECRET_KEY.
        paper: bool
            Whether to use Alpaca's paper trading endpoint for account info.
        (other params are described in the original file)
        """
        super().download_data(max_collector_count, delay, start, end, check_data_length, limit_nums,
                                api_key=api_key, api_secret=api_secret, paper=paper)
    
    def normalize_data(self, *args, **kwargs):
        """Please see the original script docstring for this method."""
        super().normalize_data(*args, **kwargs)

    def normalize_data_1d_extend(self, *args, **kwargs):
        """Please see the original script docstring for this method."""
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            **kwargs
        )
        yc.normalize()
    
    def download_today_data(self, *args, **kwargs):
        """Please see the original script docstring for this method."""
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            *args, **kwargs
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 0,
        exists_skip: bool = False,
        api_key: str = None,
        api_secret: str = None,
    ):
        """
        Automated workflow to update a Qlib 1D data directory with the latest data from Alpaca.
        """
        if self.interval.lower() != "1d":
            raise ValueError("This workflow currently only supports '1d' interval.")

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            logger.warning(f"Qlib data not found in {qlib_data_1d_dir}. Will create a new one.")
            # This will create calendars and an empty instruments directory
            GetData().qlib_data(target_dir=qlib_data_1d_dir, interval=self.interval, region="us", exists_skip=exists_skip)

        # Determine start date from existing calendar
        calendar_path = Path(qlib_data_1d_dir).joinpath("calendars/day.txt")
        if calendar_path.exists():
            calendar_df = pd.read_csv(calendar_path, header=None)
            # Start from the day after the last known date
            start_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # If no calendar, start from a default date
            start_date = "2015-01-01" 
            logger.warning(f"No calendar found. Starting data collection from {start_date}.")

        # Download new data
        self.download_data(
            delay=delay, 
            start=start_date, 
            end=end_date, 
            check_data_length=check_data_length, 
            api_key=api_key, 
            api_secret=api_secret
        )
        
        # Set a higher worker count for CPU-bound normalization
        self.max_workers = max(multiprocessing.cpu_count() - 1, 1)

        # Normalize data, extending the existing dataset
        self.normalize_data_1d_extend(old_qlib_data_dir=qlib_data_1d_dir)

        # Dump the new, normalized CSVs into Qlib's binary format
        logger.info("Dumping normalized data into Qlib binary format...")
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()
        logger.info("Data update complete.")


if __name__ == "__main__":
    fire.Fire(Run)