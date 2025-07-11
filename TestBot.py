import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import logging
import nest_asyncio
import ccxt.async_support as ccxt
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import lightgbm as lgb
import json
import matplotlib.pyplot as plt
import sys
from dotenv import load_dotenv
import telegram
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.metrics import log_loss
from threading import Lock

# Настройка логирования
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot_prehost.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding='utf-8')

# Применение nest_asyncio
nest_asyncio.apply()

# Конфигурация
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, 'token.env'))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
ADMIN_ID = int(os.getenv('ADMIN_ID', 0))
ALLOWED_USERS_PATH = os.path.join(BASE_DIR, 'allowed_users.json')
SETTINGS_PATH = os.path.join(BASE_DIR, 'settings.json')
SCREENSHOT_PATH = os.path.join(BASE_DIR, 'screenshot.png')
MODEL_PATH = os.path.join(BASE_DIR, 'model_improved1.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'features1.pkl')
DATA_CACHE_PATH = os.path.join(BASE_DIR, 'data_cache')
CAPITAL = 100
RISK_PER_TRADE = 0.01
SENTIMENT_CACHE = {}
SENTIMENT_CACHE_TTL = 3600  # 1 час
TIMEFRAME = '15m'
MAX_RETRIES = 3
RETRY_DELAY = 5
CACHE_TTL = 3600  # 1 час
DEFAULT_AUTO_INTERVAL = 300  # 5 минут

# Конфигурационные параметры для торговых стратегий
ACTIVE_FEATURES = [
    'price_change_1h', 'price_change_2h', 'price_change_6h', 'volume_score',
    'volume_change', 'atr_normalized', 'rsi', 'macd', 'vwap_signal', 'obv',
    'adx', 'bb_upper', 'bb_lower', 'support_level', 'resistance_level',
    'sentiment', 'smart_money_score'
]
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_1_PCT = 0.03
TAKE_PROFIT_2_PCT = 0.05
PRICE_THRESHOLD = 0.5
VOLUME_THRESHOLD = 5.0
RSI_THRESHOLD = 40.0

# Список криптовалют
CRYPTO_PAIRS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "SOL/USDT", "DOT/USDT", "DOGE/USDT", "POL/USDT", "TRX/USDT"
]

# SQLite
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    symbol = Column(String)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    rr_ratio = Column(Float)
    position_size = Column(Float)
    probability = Column(Float)
    institutional_score = Column(Float)
    sentiment_score = Column(Float)
    trader_level = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    result = Column(String, nullable=True)

class TradeMetrics(Base):
    __tablename__ = 'trade_metrics'
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer)
    symbol = Column(String)
    entry_price = Column(Float)
    price_after_1h = Column(Float, nullable=True)
    price_after_2h = Column(Float, nullable=True)
    volume_change = Column(Float, nullable=True)
    institutional_score = Column(Float, nullable=True)
    vwap_signal = Column(Float, nullable=True)
    sentiment = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    obv = Column(Float, nullable=True)
    smart_money_score = Column(Float, nullable=True)
    probability = Column(Float, nullable=True)
    success = Column(String, nullable=True)

engine = create_engine('sqlite:///trades.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class TradingModel:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.scalers = {}
        self.active_features_dict = {}
        self.LOW_RECALL_SYMBOLS = ["BTC/USDT", "BNB/USDT"]
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.active_features_dict = model_data.get('active_features', {})
                
                if os.path.exists(FEATURES_PATH):
                    features_data = joblib.load(FEATURES_PATH)
                    if isinstance(features_data, dict):
                        self.active_features_dict = features_data
                    elif isinstance(features_data, list):
                        self.active_features_dict = {'combined': features_data}
                
                logger.info(f"Models and scalers loaded successfully. {len(self.models)} models available")
            else:
                logger.warning("Model file not found")
                self.models = {}
                self.scalers = {}
                self.active_features_dict = {'combined': ACTIVE_FEATURES}
        except Exception as e:
            logger.error(f"Failed to load model data: {e}")
            self.models = {}
            self.scalers = {}
            self.active_features_dict = {'combined': ACTIVE_FEATURES}

    def get_model_for_symbol(self, symbol):
        base_symbol = symbol.replace('/USDT', 'USDT')
        if 'combined' in self.models:
            return self.models['combined'], self.scalers['combined'], self.active_features_dict['combined']
        elif base_symbol in self.models:
            return self.models[base_symbol], self.scalers[base_symbol], self.active_features_dict[base_symbol]
        else:
            return None, None, None

    async def get_historical_data(self, symbol, timeframe='15m', limit=1000):
        cache_file = os.path.join(DATA_CACHE_PATH, f"{symbol.replace('/', '_')}_{timeframe}_historical.pkl")
        if os.path.exists(cache_file):
            try:
                cache_mtime = os.path.getmtime(cache_file)
                if (datetime.utcnow().timestamp() - cache_mtime) < CACHE_TTL:
                    df = pd.read_pickle(cache_file)
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if not df.empty and all(col in df.columns for col in required_columns):
                        logger.info(f"Using cached data for {symbol}: {len(df)} records")
                        return df
                    else:
                        logger.warning(f"Invalid cache for {symbol}, deleting")
                        os.remove(cache_file)
            except Exception as e:
                logger.error(f"Error reading cache for {symbol}: {e}")
                if os.path.exists(cache_file):
                    os.remove(cache_file)

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                markets = await self.exchange.load_markets()
                if symbol not in markets:
                    logger.warning(f"Pair {symbol} not found")
                    return pd.DataFrame()
                
                since = int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)
                all_ohlcv = []
                while len(all_ohlcv) < limit:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(limit, 1000))
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    await asyncio.sleep(0.1)
                
                if all_ohlcv:
                    df = pd.DataFrame(all_ohlcv[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['price'] = df['close'].astype(float)
                    df['taker_buy_base'] = df['volume'].astype(float) * 0.5
                    df['symbol'] = symbol
                    
                    if df['price'].isna().any() or (df['price'] <= 0).any():
                        logger.warning(f"Invalid data for {symbol}, skipping")
                        return pd.DataFrame()
                    
                    os.makedirs(DATA_CACHE_PATH, exist_ok=True)
                    df.to_pickle(cache_file)
                    logger.info(f"Fetched {len(df)} records for {symbol}")
                    return df
                break
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt}/{MAX_RETRIES} failed for {symbol}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    await notify_admin(f"Failed to fetch data for {symbol}: {e}")
            finally:
                await self.exchange.close()
        logger.warning(f"Failed to get data for {symbol}")
        return pd.DataFrame()

    def calculate_indicators(self, df):
        if df.empty or len(df) < 14:
            logger.warning("Not enough data for indicators")
            return df
            
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return df

            df = df.dropna(subset=required_columns)
            df = df[(df['close'] > 0) & (df['volume'] >= 0)]
            
            if len(df) < 14:
                logger.warning("Not enough data after cleaning")
                return df

            if 'price' not in df.columns:
                df['price'] = df['close']

            # Волатильность
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()

            # Моментум
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

            # MACD
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            # ADX
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            tr = tr.replace(0, 0.0001)
            plus_dm = high_diff.where(high_diff > low_diff, 0)
            minus_dm = low_diff.where(low_diff > high_diff, 0)
            plus_di = 100 * plus_dm.rolling(window=14).mean() / tr
            minus_di = 100 * minus_dm.rolling(window=14).mean() / tr
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df['adx'] = dx.rolling(window=14).mean()

            # OBV
            df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

            # VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_signal'] = np.where(df['close'] > df['vwap'], 1.0, -1.0)

            # Другие признаки
            df['price_change_1h'] = df['close'].pct_change(4) * 100
            df['price_change_2h'] = df['close'].pct_change(8) * 100
            df['price_change_6h'] = df['close'].pct_change(24) * 100
            df['volume_score'] = df['volume'] / df['volume'].rolling(window=6).mean() * 100
            df['volume_change'] = df['volume'].pct_change() * 100
            df['atr_normalized'] = (df['high'] - df['low']) / df['close'].replace(0, 0.0001) * 100

            # Уровни поддержки/сопротивления
            df['support_level'] = df['low'].rolling(window=20).min() / df['close'].replace(0, 0.0001)
            df['resistance_level'] = df['high'].rolling(window=20).max() / df['close'].replace(0, 0.0001)

            # Синтетические признаки
            df['sentiment'] = pd.Series(50.0 + (df['rsi'] - 50) * 0.5 + df['macd'] * 10, index=df.index).clip(0, 100)
            df['smart_money_score'] = (df['sentiment'] * 0.4 + (df['rsi'] / 100) * 30 + (df['adx'] / 100) * 30) / 0.7
            df['smart_money_score'] = df['smart_money_score'].clip(0, 100).fillna(50)
            
            df = df.ffill().fillna(0)
            logger.info(f"Indicators calculated for {df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def prepare_features(self, df):
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        features = [
            'price_change_1h', 'price_change_2h', 'price_change_6h', 'volume_score',
            'volume_change', 'atr_normalized', 'rsi', 'macd', 'vwap_signal', 'obv',
            'adx', 'bb_upper', 'bb_lower', 'support_level', 'resistance_level',
            'sentiment', 'smart_money_score'
        ]
        
        df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df[features].iloc[-2:], df.iloc[-2:]

    async def predict_probability(self, symbol, direction='LONG', stop_loss=None, position_size=0):
        try:
            df = await self.get_historical_data(symbol)
            if df.empty or len(df) < 48:
                logger.warning(f"Not enough data for {symbol}")
                return 0.0
            
            df = self.calculate_indicators(df)
            X, _ = self.prepare_features(df)
            
            model, scaler, active_features = self.get_model_for_symbol(symbol)
            if model is None or scaler is None or not active_features:
                logger.error(f"No model/scaler for {symbol}")
                return 0.0

            threshold = 0.3160 if symbol in self.LOW_RECALL_SYMBOLS else 0.5
            
            latest_data = X[active_features].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_scaled = scaler.transform(latest_data)
            y_pred_proba = model.predict_proba(X_scaled)[0][1] * 100
            
            if direction == 'SHORT':
                y_pred_proba = 100 - y_pred_proba
            
            # Дополнительные фильтры
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_adx = df['adx'].iloc[-1]
            
            if direction == 'LONG':
                if current_rsi > 70: y_pred_proba *= 0.7
                if current_macd < 0: y_pred_proba *= 0.8
            else:  # SHORT
                if current_rsi < 30: y_pred_proba *= 0.7
                if current_macd > 0: y_pred_proba *= 0.8
            
            if current_adx < 25: y_pred_proba *= 0.9
            
            final_proba = max(1.0, min(99.0, y_pred_proba))
            logger.info(f"Predicted probability for {symbol} {direction}: {final_proba:.1f}%")
            return final_proba
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return 0.0

    async def analyze_symbol(self, symbol, coin_id, price_change_1h, taker_buy_base, volume, balance):
        try:
            df = await self.get_historical_data(symbol)
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}")
                return None
                
            df = self.calculate_indicators(df)
            current_price = df['close'].iloc[-1]
            atr = df['atr_normalized'].iloc[-1] * current_price
            
            # Расчет уровней
            stop_loss_long = current_price - max(2 * atr, current_price * STOP_LOSS_PCT)
            take_profit_1_long = current_price + 6 * atr
            take_profit_2_long = current_price + 10 * atr
            stop_loss_short = current_price + max(2 * atr, current_price * STOP_LOSS_PCT)
            take_profit_1_short = current_price - 6 * atr
            take_profit_2_short = current_price - 10 * atr
            
            # Прогноз вероятности
            long_prob = await self.predict_probability(
                symbol, 'LONG', stop_loss_long, 
                abs(calculate_position_size(current_price, stop_loss_long, balance)[0]))
            short_prob = await self.predict_probability(
                symbol, 'SHORT', stop_loss_short,
                abs(calculate_position_size(current_price, stop_loss_short, balance)[0]))
            
            institutional_score = taker_buy_base / volume * 100 if volume > 0 else 50.0
            vwap_signal = (current_price - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] * 100
            sentiment = get_news_sentiment(coin_id)
            
            return {
                'symbol': symbol,
                'coin_id': coin_id,
                'price': current_price,
                'price_change': price_change_1h,
                'volume_change': ((df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] * 100) if len(df) >= 2 else 0.0,
                'institutional_score': institutional_score,
                'vwap_signal': vwap_signal,
                'sentiment': sentiment,
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'adx': df['adx'].iloc[-1],
                'obv': df['obv'].iloc[-1],
                'long_prob': long_prob,
                'short_prob': short_prob,
                'stop_loss_long': stop_loss_long,
                'take_profit_1_long': take_profit_1_long,
                'take_profit_2_long': take_profit_2_long,
                'stop_loss_short': stop_loss_short,
                'take_profit_1_short': take_profit_1_short,
                'take_profit_2_short': take_profit_2_short,
                'df': df
            }
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

# Инициализация модели
trading_model = TradingModel()

# Управление файлами
def ensure_files_exist():
    if not os.path.exists(DATA_CACHE_PATH):
        os.makedirs(DATA_CACHE_PATH)
        logger.info(f"Created directory: {DATA_CACHE_PATH}")
    if not os.path.exists(ALLOWED_USERS_PATH):
        with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump([ADMIN_ID], f)
        logger.info(f"Created file: {ALLOWED_USERS_PATH}")
    if not os.path.exists(SETTINGS_PATH):
        default_settings = {str(ADMIN_ID): {"auto_trade": False, "interval": 300, "risk_level": "low"}}
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, indent=4)
        logger.info(f"Created file: {SETTINGS_PATH}")

def load_settings():
    ensure_files_exist()
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_settings(settings):
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved")
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

def get_user_settings(user_id: int) -> dict:
    try:
        settings_file = os.path.join(BASE_DIR, 'settings.json')
        default = {
            'price_threshold': 0.3,
            'volume_threshold': 5,
            'rsi_threshold': 40,
            'use_rsi': True,
            'auto_interval': DEFAULT_AUTO_INTERVAL,
            'balance': 1000,
            'min_probability': 60.0
        }
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                all_settings = json.load(f)
            user_settings = all_settings.get(str(user_id), {})
            merged = {**default, **user_settings}
            logger.info(f"Loaded settings for user_id={user_id}, balance={merged.get('balance', 'not set')}")
            return merged
        return default
    except Exception as e:
        logger.error(f"Error getting settings for user_id={user_id}: {e}")
        asyncio.create_task(notify_admin(f"Error in get_user_settings for user_id={user_id}: {e}"))
        return default

def save_user_settings(user_id: int, settings: dict):
    try:
        settings_file = os.path.join(BASE_DIR, 'settings.json')
        all_settings = {}
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    all_settings = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error reading JSON for user_id={user_id}: {e}")
        all_settings[str(user_id)] = settings
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(all_settings, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved settings for user_id={user_id}, balance={settings.get('balance', 'not set')}")
    except Exception as e:
        logger.error(f"Error saving settings for user_id={user_id}: {e}")
        asyncio.create_task(notify_admin(f"Error in save_user_settings for user_id={user_id}: {e}"))

def load_allowed_users():
    ensure_files_exist()
    with open(ALLOWED_USERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_allowed_users(users):
    try:
        with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f)
        logger.info("Allowed users saved")
    except Exception as e:
        logger.error(f"Error saving allowed users: {e}")

def is_authorized(user_id):
    return user_id in load_allowed_users()

async def notify_admin(message):
    try:
        bot = Application.builder().token(TELEGRAM_TOKEN).build()
        await bot.bot.send_message(chat_id=ADMIN_ID, text=f"🚨 **Bot Error**: {message}")
    except Exception as e:
        logger.error(f"Error sending notification: {e}")

def get_news_sentiment(coin_id):
    cache_key = f"{coin_id}_{datetime.utcnow().strftime('%Y%m%d%H')}"
    if cache_key in SENTIMENT_CACHE:
        sentiment, timestamp = SENTIMENT_CACHE[cache_key]
        if (datetime.utcnow() - timestamp).total_seconds() < SENTIMENT_CACHE_TTL:
            logger.info(f"Using cached sentiment for {coin_id}: {sentiment:.2f}")
            return sentiment
    try:
        sources = [
            "https://www.coindesk.com/feed",
            "https://cryptoslate.com/feed/",
            "https://cointelegraph.com/rss"
        ]
        analyzer = SentimentIntensityAnalyzer()
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        sentiment_scores = []
        for source in sources:
            feed = feedparser.parse(source)
            relevant_articles = [
                entry for entry in feed.entries
                if coin_id.lower() in (entry.title.lower() + " " + entry.summary.lower())
                and 'published_parsed' in entry and entry.published_parsed
                and datetime(*entry.published_parsed[:6]) > cutoff_time
            ]
            scores = [analyzer.polarity_scores(entry.title + " " + entry.summary)['compound']
                      for entry in relevant_articles]
            sentiment_scores.extend(scores)
        sentiment = np.mean(sentiment_scores) * 100 if sentiment_scores else 0
        SENTIMENT_CACHE[cache_key] = (sentiment, datetime.utcnow())
        logger.info(f"Calculated sentiment for {coin_id}: {sentiment:.2f} ({len(sentiment_scores)} articles)")
        return sentiment
    except Exception as e:
        logger.error(f"Error getting sentiment for {coin_id}: {e}")
        return 0

async def get_current_price(symbol):
    try:
        binance_symbol = symbol.replace('/', '')
        ticker = await trading_model.exchange.fetch_ticker(binance_symbol)
        current_price = float(ticker['last'])
        logger.info(f"Current price for {symbol}: ${current_price:.6f}")
        return current_price
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        await notify_admin(f"Error getting price for {symbol}: {e}")
        return 0.0

def calculate_position_size(entry_price, stop_loss, balance):
    if balance <= 0:
        logger.error(f"Invalid balance: {balance}")
        return 0, 0
    risk_amount = balance * RISK_PER_TRADE
    price_diff = max(abs(entry_price - stop_loss), entry_price * 0.001)
    position_size = risk_amount / price_diff if price_diff > 0 else 0.000018
    position_size_percent = (position_size * entry_price / balance) * 100
    logger.info(f"Position size: Balance={balance}, Risk={risk_amount}, Size={position_size}")
    return position_size, position_size_percent

def create_price_chart(df, symbol, price_change):
    try:
        if df.empty:
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.title(f"{symbol} Price Chart (15m) - No Data")
            plt.text(0.5, 0.5, f"Price change: {price_change:.2f}%", horizontalalignment='center')
            plt.grid()
            plt.savefig(SCREENSHOT_PATH)
            plt.close()
            return SCREENSHOT_PATH
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(df['timestamp'], df['price'], label=f"{symbol} Price", color='blue')
        ax1.set_title(f"{symbol} Price Chart (15m)")
        ax1.set_ylabel("Price (USD)")
        ax1.legend()
        ax1.grid()
        ax2.bar(df['timestamp'], df['volume'], label='Volume', color='green')
        ax2.set_ylabel("Volume")
        ax2.legend()
        ax2.grid()
        plt.tight_layout()
        plt.savefig(SCREENSHOT_PATH)
        plt.close()
        logger.info(f"Created chart for {symbol}")
        return SCREENSHOT_PATH
    except Exception as e:
        logger.error(f"Error creating chart for {symbol}: {e}")
        asyncio.create_task(notify_admin(f"Error creating chart for {symbol}: {e}"))
        return None

def get_top_cryptos():
    try:
        result = []
        for symbol in CRYPTO_PAIRS:
            coin_id = symbol.replace('/USDT', '')
            try:
                ticker = asyncio.run(trading_model.exchange.fetch_ticker(symbol))
                result.append((
                    symbol,
                    coin_id,
                    float(ticker.get('percentage', 0)),
                    float(ticker.get('baseVolume', 0) * 0.5),
                    float(ticker.get('baseVolume', 0))
                ))
            except Exception as e:
                logger.error(f"Error getting ticker for {symbol}: {e}")
                result.append((symbol, coin_id, 0, 0, 0))
        session = Session()
        try:
            disabled_pairs = []
            for symbol, _, _, _, _ in result:
                metrics = session.query(TradeMetrics).join(Trade, Trade.id == TradeMetrics.trade_id).filter(
                    Trade.symbol == symbol,
                    Trade.timestamp >= datetime.utcnow() - timedelta(days=30)
                ).order_by(Trade.timestamp.desc()).limit(20).all()
                if len(metrics) >= 20:
                    wins = len([m for m in metrics if m.success in ['TP1', 'TP2']])
                    if wins / len(metrics) < 0.6:
                        disabled_pairs.append(symbol)
            result = [item for item in result if item[0] not in disabled_pairs]
            logger.info(f"Disabled pairs with win rate <60%: {disabled_pairs}")
        finally:
            session.close()
        return result
    except Exception as e:
        logger.error(f"Error getting top cryptos: {e}")
        return [(symbol, symbol.replace('/USDT', ''), 0, 0, 0) for symbol in CRYPTO_PAIRS]

async def check_trade_result(symbol, entry_price, stop_loss, tp1, tp2, trade_id):
    session = Session()
    try:
        await asyncio.sleep(3600)
        price_1h = await get_current_price(symbol)
        await asyncio.sleep(3600)
        price_2h = await get_current_price(symbol)
        success = None
        if price_1h <= stop_loss or price_2h <= stop_loss:
            success = 'SL'
        elif price_2h >= tp2:
            success = 'TP2'
        elif price_1h >= tp1 or price_2h >= tp1:
            success = 'TP1'
        trade_metrics = TradeMetrics(
            trade_id=trade_id,
            symbol=symbol,
            entry_price=entry_price,
            price_after_1h=price_1h,
            price_after_2h=price_2h,
            success=success
        )
        trade = session.query(Trade).filter_by(id=trade_id).first()
        if trade and success:
            trade.result = success
        session.add(trade_metrics)
        session.commit()
        logger.info(f"Trade #{trade_id} ({symbol}): result={success}")
    except Exception as e:
        logger.error(f"Error checking trade result for {symbol}: {e}")
        await notify_admin(f"Error checking trade result for {symbol}: {e}")
    finally:
        session.close()

trade_lock = Lock()

async def check_active_trades(context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    try:
        user_id = context.job.data
        logger.info(f"Checking active trades for user_id={user_id}")
        with trade_lock:
            active_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                (Trade.result.is_(None) | (Trade.result == 'TP1'))
            ).all()
            if not active_trades:
                logger.info(f"No active trades for user_id={user_id}")
                return
            for trade in active_trades:
                trade_exists = session.query(Trade).filter_by(id=trade.id).first()
                if not trade_exists:
                    logger.warning(f"Trade #{trade.id} ({trade.symbol}) not found")
                    continue
                current_price = await get_current_price(trade.symbol)
                if current_price is None or current_price <= 0:
                    logger.warning(f"Invalid price for {trade.symbol}: {current_price}")
                    continue
                price_precision = 6 if trade.entry_price < 1 else 2
                user_settings = get_user_settings(user_id)
                balance = user_settings.get('balance', 0)
                update_needed = False
                new_result = None
                pnl = 0
                if trade.position_size > 0:  # LONG
                    if current_price <= trade.stop_loss:
                        new_result = 'SL'
                        final_price = trade.stop_loss
                        pnl = (final_price - trade.entry_price) * trade.position_size
                    elif trade.result is None and current_price >= trade.take_profit_1:
                        new_result = 'TP1'
                        final_price = trade.take_profit_1
                        pnl = (final_price - trade.entry_price) * trade.position_size
                    elif trade.result == 'TP1' and current_price >= trade.take_profit_2:
                        new_result = 'TP2'
                        final_price = trade.take_profit_2
                        pnl = (final_price - trade.entry_price) * trade.position_size
                else:  # SHORT
                    if current_price >= trade.stop_loss:
                        new_result = 'SL'
                        final_price = trade.stop_loss
                        pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                    elif trade.result is None and current_price <= trade.take_profit_1:
                        new_result = 'TP1'
                        final_price = trade.take_profit_1
                        pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                    elif trade.result == 'TP1' and current_price <= trade.take_profit_2:
                        new_result = 'TP2'
                        final_price = trade.take_profit_2
                        pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                
                if new_result:
                    try:
                        trade.result = new_result
                        trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade.id).first()
                        if trade_metrics:
                            trade_metrics.success = new_result
                        session.commit()
                        
                        # Обновление баланса
                        balance += pnl
                        user_settings['balance'] = balance
                        save_user_settings(user_id, user_settings)
                        
                        message = (
                            f"📊 **Trade #{trade.id}** ({trade.symbol}): "
                            f"{'✅ TP1' if new_result == 'TP1' else '✅ TP2' if new_result == 'TP2' else '❌ SL'} reached\n"
                            f"🎯 Entry: ${trade.entry_price:.{price_precision}f} | Exit: ${final_price:.{price_precision}f}\n"
                            f"💸 PNL: {pnl:.2f} USDT\n"
                            f"💰 New balance: ${balance:.2f}"
                        )
                        await context.bot.send_message(
                            chat_id=trade.user_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"Trade #{trade.id} ({trade.symbol}) updated: result={new_result}, PNL={pnl:.2f}, balance={balance:.2f}")
                    except Exception as e:
                        logger.error(f"Error updating trade #{trade.id} ({trade.symbol}): {e}")
                        await notify_admin(f"Error updating trade #{trade.id} for user_id={user_id}: {e}")
                        session.rollback()
    except Exception as e:
        logger.error(f"Error in check_active_trades for user_id={user_id}: {e}")
        await notify_admin(f"Error in check_active_trades for user_id={user_id}: {e}")
    finally:
        session.close()

async def retrain_model_daily(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Starting daily model retraining")
    session = None
    try:
        session = Session()
        trades = session.query(Trade, TradeMetrics).join(
            TradeMetrics,
            Trade.id == TradeMetrics.trade_id
        ).filter(
            Trade.result.isnot(None)
        ).all()
        
        if len(trades) < 5:
            logger.warning(f"Not enough data for retraining ({len(trades)} trades)")
            return
        
        X = []
        y = []
        pnls = []
        for trade, metrics in trades:
            features = [
                metrics.volume_change or 0,
                metrics.institutional_score or 0,
                metrics.vwap_signal or 0,
                metrics.sentiment or 0,
                metrics.rsi or 0,
                metrics.macd or 0,
                metrics.adx or 0,
                metrics.obv or 0,
                metrics.smart_money_score or 0,
                trade.entry_price / (trade.stop_loss + 1e-10),
                trade.rr_ratio or 0,
                metrics.probability or 0,
                (trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100 if trade.position_size > 0 else (trade.entry_price - trade.take_profit_1) / trade.entry_price * 100,
                (trade.take_profit_2 - trade.entry_price) / trade.entry_price * 100 if trade.position_size > 0 else (trade.entry_price - trade.take_profit_2) / trade.entry_price * 100,
                trade.probability or 0,
                metrics.volume_change / (metrics.institutional_score + 1e-10) if metrics.institutional_score else 0,
                metrics.rsi / 100 if metrics.rsi else 0
            ]
            X.append(features)
            y.append(1 if trade.result in ['TP1', 'TP2'] else 0)
            final_price = trade.stop_loss if trade.result == 'SL' else trade.take_profit_1 if trade.result == 'TP1' else trade.take_profit_2
            if trade.position_size > 0:
                pnl = (final_price - trade.entry_price) * trade.position_size
            else:
                pnl = (trade.entry_price - final_price) * abs(trade.position_size)
            pnls.append(pnl)
        
        X = pd.DataFrame(X, columns=trading_model.active_features_dict.get('combined', ACTIVE_FEATURES))
        y = np.array(y)
        
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_counts}")
        
        if len(class_counts) < 2 or min(counts) < 2:
            logger.warning(f"Unbalanced classes: {class_counts}")
            return
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = trading_model.scalers.get('combined')
        if not scaler:
            scaler = StandardScaler()
            trading_model.scalers['combined'] = scaler
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = trading_model.models.get('combined')
        if not model:
            model = lgb.LGBMClassifier(random_state=42)
            trading_model.models['combined'] = model
        
        loss_before = model.score(X_test_scaled, y_test)
        logger.info(f"Accuracy before retraining: {loss_before:.4f}")
        
        model.fit(X_train_scaled, y_train, sample_weight=[2.0 if y == 0 else 0.5 if y == 1 else 1.0 for y in y_train])
        
        loss_after = model.score(X_test_scaled, y_test)
        logger.info(f"Accuracy after retraining: {loss_after:.4f}, samples: {len(X)}")
        
        trading_model.models['combined'] = model
        trading_model.scalers['combined'] = scaler
        trading_model.save_model()
        
    except Exception as e:
        logger.error(f"Error in retrain_model_daily: {str(e)}")
        await notify_admin(f"Error in retrain_model_daily: {str(e)}")
    finally:
        if session is not None:
            session.close()

async def auto_search_trades(context: ContextTypes.DEFAULT_TYPE):
    settings = load_settings()
    for user_id_str in settings:
        user_id = int(user_id_str)
        if not is_authorized(user_id):
            continue
        user_settings = get_user_settings(user_id)
        min_probability = user_settings.get('min_probability', 60.0)
        balance = user_settings.get('balance', 1000)
        cryptos = get_top_cryptos()
        session = Session()
        try:
            opportunities = []
            for symbol, coin_id, price_change_1h, taker_buy_base, volume in cryptos:
                analysis = await trading_model.analyze_symbol(symbol, coin_id, price_change_1h, taker_buy_base, volume, balance)
                if not analysis:
                    continue
                
                # Проверка условий входа
                current_rsi = analysis['rsi']
                current_macd = analysis['macd']
                current_adx = analysis['adx']
                
                if analysis['long_prob'] >= min_probability and current_rsi >= 30 and current_rsi <= 70 and current_macd > 0 and current_adx > 20:
                    opportunities.append({
                        **analysis,
                        'direction': 'LONG',
                        'probability': analysis['long_prob'],
                        'stop_loss': analysis['stop_loss_long'],
                        'take_profit_1': analysis['take_profit_1_long'],
                        'take_profit_2': analysis['take_profit_2_long']
                    })
                
                if analysis['short_prob'] >= min_probability and current_rsi >= 30 and current_rsi <= 70 and current_macd < 0 and current_adx > 20:
                    opportunities.append({
                        **analysis,
                        'direction': 'SHORT',
                        'probability': analysis['short_prob'],
                        'stop_loss': analysis['stop_loss_short'],
                        'take_profit_1': analysis['take_profit_1_short'],
                        'take_profit_2': analysis['take_profit_2_short']
                    })
            
            if not opportunities:
                continue
                
            opportunities.sort(key=lambda x: abs(x['probability'] - 50), reverse=True)
            
            for opp in opportunities[:3]:  # Ограничим 3 лучшими возможностями
                symbol = opp['symbol']
                direction = opp['direction']
                current_price = opp['price']
                df = opp['df']
                price_change = opp['price_change']
                volume_change = opp['volume_change']
                institutional_score = opp['institutional_score']
                vwap_signal = opp['vwap_signal']
                sentiment = opp['sentiment']
                rsi = opp['rsi']
                macd = opp['macd']
                adx = opp['adx']
                obv = opp['obv']
                probability = opp['probability']
                stop_loss = opp['stop_loss']
                take_profit_1 = opp['take_profit_1']
                take_profit_2 = opp['take_profit_2']
                
                existing_trades = session.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.symbol == symbol,
                    Trade.result.is_(None) | (Trade.result == 'TP1')
                ).all()
                
                is_duplicate = False
                for trade in existing_trades:
                    entry_diff = abs(trade.entry_price - current_price) / current_price
                    sl_diff = abs(trade.stop_loss - stop_loss) / current_price
                    tp1_diff = abs(trade.take_profit_1 - take_profit_1) / current_price
                    if entry_diff < 0.005 and sl_diff < 0.01 and tp1_diff < 0.01:
                        is_duplicate = True
                        logger.info(f"Skipping {symbol}, duplicate of trade #{trade.id}")
                        break
                
                if is_duplicate:
                    continue
                
                rr_ratio = (take_profit_1 - current_price) / (current_price - stop_loss) if direction == 'LONG' else (current_price - take_profit_1) / (stop_loss - current_price)
                position_size, position_size_percent = calculate_position_size(current_price, stop_loss, balance)
                position_size = position_size if direction == 'LONG' else -position_size
                potential_profit_tp1 = (take_profit_1 - current_price) * position_size if direction == 'LONG' else (current_price - take_profit_1) * abs(position_size)
                potential_profit_tp2 = (take_profit_2 - current_price) * position_size if direction == 'LONG' else (current_price - take_profit_2) * abs(position_size)
                
                trade = Trade(
                    user_id=user_id,
                    symbol=symbol,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    rr_ratio=rr_ratio,
                    position_size=position_size,
                    probability=probability,
                    institutional_score=institutional_score,
                    sentiment_score=sentiment,
                    trader_level="Auto"
                )
                session.add(trade)
                session.flush()
                
                trade_metrics = TradeMetrics(
                    trade_id=trade.id,
                    symbol=symbol,
                    entry_price=current_price,
                    volume_change=volume_change,
                    institutional_score=institutional_score,
                    vwap_signal=vwap_signal,
                    sentiment=sentiment,
                    rsi=rsi,
                    macd=macd,
                    adx=adx,
                    obv=obv,
                    smart_money_score=min(100, institutional_score + (volume_change / 5)),
                    probability=probability
                )
                session.add(trade_metrics)
                session.commit()
                
                price_precision = 6 if current_price < 1 else 2
                vwap_text = '🟢 Bullish' if vwap_signal > 0 else '🔴 Bearish'
                macd_text = '🟢 Bullish' if macd > 0 else '🔴 Bearish'
                tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
                
                message = (
                    f"🔔 **New Trade: {symbol} {direction}**\n"
                    f"💰 **Balance**: ${balance:.2f}\n"
                    f"🎯 Entry: ${current_price:.{price_precision}f}\n"
                    f"⛔ Stop Loss: ${stop_loss:.{price_precision}f}\n"
                    f"💰 TP1: ${take_profit_1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                    f"💰 TP2: ${take_profit_2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                    f"📊 RR: {rr_ratio:.1f}:1\n"
                    f"📏 Size: {position_size_percent:.2f}% ({abs(position_size):.6f} {symbol.split('/')[0]})\n"
                    f"🎲 Probability: {probability:.1f}%\n"
                    f"🏛️ Institutional: {institutional_score:.1f}%\n"
                    f"📈 VWAP: {vwap_text}\n"
                    f"📮 Sentiment: {sentiment:.1f}%\n"
                    f"📊 RSI: {rsi:.1f} | MACD: {macd_text} | ADX: {adx:.1f}\n"
                    f"💡 Logic: Change {price_change:.2f}%, Volume +{volume_change:.1f}%\n"
                    f"📈 Chart: {tradingview_url}\n"
                    f"💾 Trade saved. Mark result:"
                )
                
                keyboard = [
                    [InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade.id}"),
                     InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade.id}"),
                     InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}"),
                     InlineKeyboardButton("🚫 Cancel", callback_data=f"CANCEL_{trade.id}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                chart_path = create_price_chart(df, symbol, price_change)
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as photo:
                        await context.bot.send_photo(
                            chat_id=user_id,
                            photo=photo,
                            caption=message,
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                    os.remove(chart_path)
                else:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=message + "\n⚠️ Failed to create chart.",
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                
                logger.info(f"Created trade #{trade.id} for {symbol} ({direction})")
                asyncio.create_task(check_trade_result(symbol, current_price, stop_loss, take_profit_1, take_profit_2, trade.id))
                break  # Ограничиваем одну сделку за раз
                
        finally:
            session.close()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except telegram.error.BadRequest as e:
        logger.warning(f"Stale or invalid query: {e}")
        return
        
    user_id = query.from_user.id
    if not is_authorized(user_id):
        await query.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    session = Session()
    try:
        action, trade_id = query.data.split('_')
        trade_id = int(trade_id)
        trade = session.query(Trade).filter_by(id=trade_id, user_id=user_id).first()
        if not trade:
            await query.message.edit_text("🚫 **Trade not found or doesn't belong to you.**", parse_mode='Markdown')
            return
            
        user_settings = get_user_settings(user_id)
        balance = user_settings.get('balance', 0)
        price_precision = 6 if trade.entry_price < 1 else 2
        
        if action in ['TP1', 'TP2', 'SL']:
            final_price = trade.take_profit_1 if action == 'TP1' else trade.take_profit_2 if action == 'TP2' else trade.stop_loss
            if trade.position_size > 0:
                pnl = (final_price - trade.entry_price) * trade.position_size
            else:
                pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                
            trade.result = action
            trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade.id).first()
            if trade_metrics:
                trade_metrics.success = action
                
            balance += pnl
            user_settings['balance'] = balance
            save_user_settings(user_id, user_settings)
            session.commit()
            
            message = (
                f"📊 **Trade #{trade.id}** ({trade.symbol}): "
                f"{'✅ TP1' if action == 'TP1' else '✅ TP2' if action == 'TP2' else '❌ SL'} reached\n"
                f"🎯 Entry: ${trade.entry_price:.{price_precision}f} | Exit: ${final_price:.{price_precision}f}\n"
                f"💸 PNL: {pnl:.2f} USDT\n"
                f"💰 New balance: ${balance:.2f}"
            )
            await query.message.edit_text(message, parse_mode='Markdown')
            logger.info(f"Trade #{trade.id} ({trade.symbol}) updated: {action}, PNL={pnl:.2f}")
            
        elif action == 'CANCEL':
            session.delete(trade)
            session.query(TradeMetrics).filter_by(trade_id=trade.id).delete()
            session.commit()
            await query.message.edit_text(f"🗑️ **Trade #{trade.id}** ({trade.symbol}) canceled.", parse_mode='Markdown')
            logger.info(f"Trade #{trade.id} ({trade.symbol}) canceled")
            
    except Exception as e:
        logger.error(f"Error in button handler for user_id={user_id}: {e}")
        await query.message.reply_text(f"🚨 **Error**: {e}", parse_mode='Markdown')
        await notify_admin(f"Error in button handler for user_id={user_id}: {e}")
    finally:
        session.close()

async def set_min_probability(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    args = context.args
    settings = load_settings()
    user_settings = get_user_settings(user_id)
    user_id_str = str(user_id)
    
    try:
        if not args:
            min_probability = user_settings.get('min_probability', 60.0)
            message = (
                f"⚙️ **Current min probability**: {min_probability}%\n"
                f"Usage: `/setminprobability <percent>`\n"
                f"Example: `/setminprobability 60`"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
            return
            
        min_probability = float(args[0])
        if min_probability < 0 or min_probability > 100:
            await update.message.reply_text("🚫 **Probability must be between 0 and 100%.**", parse_mode='Markdown')
            return
            
        user_settings['min_probability'] = min_probability
        settings[user_id_str] = user_settings
        save_settings(settings)
        await update.message.reply_text(f"✅ **Min probability set**: {min_probability}%", parse_mode='Markdown')
        logger.info(f"User {user_id} set min probability to {min_probability}%")
        
    except Exception as e:
        logger.error(f"Error in set_min_probability: {e}")
        await update.message.reply_text(f"🚨 **Error**: {e}\nFormat: `/setminprobability <percent>`", parse_mode='Markdown')
        await notify_admin(f"Error in /setminprobability: {e}")

async def set_criteria(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    args = context.args
    user_settings = get_user_settings(user_id)
    
    try:
        if len(args) < 3:
            message = (
                f"⚙️ **Current criteria**:\n"
                f"📊 Change: {user_settings['price_threshold']}% | Volume: {user_settings['volume_threshold']}% | RSI: {user_settings['rsi_threshold']} ({'on' if user_settings['use_rsi'] else 'off'})\n"
                f"⏱ Interval: {user_settings['auto_interval']//60} min\n"
                f"Usage: `/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes]`\n"
                f"Example: `/setcriteria 5 0.3 40 rsi_off 5`"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
            return
            
        volume_threshold = float(args[0])
        price_threshold = float(args[1])
        rsi_threshold = float(args[2])
        use_rsi = 'rsi_off' not in args
        interval_minutes = int(args[args.index('interval_minutes') + 1]) if 'interval_minutes' in args else user_settings['auto_interval'] // 60
        
        user_settings.update({
            'volume_threshold': volume_threshold,
            'price_threshold': price_threshold,
            'rsi_threshold': rsi_threshold,
            'use_rsi': use_rsi,
            'auto_interval': interval_minutes * 60
        })
        save_user_settings(user_id, user_settings)
        
        await update.message.reply_text(
            f"✅ **Criteria updated**:\n"
            f"📊 Change: {price_threshold}% | Volume: {volume_threshold}% | RSI: {rsi_threshold} ({'on' if use_rsi else 'off'})\n"
            f"⏱ Interval: {interval_minutes} min",
            parse_mode='Markdown'
        )
        logger.info(f"User {user_id} updated criteria: {user_settings}")
        
    except Exception as e:
        logger.error(f"Error in set_criteria: {e}")
        await update.message.reply_text(f"🚨 **Error**: {e}\nFormat: `/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes]`", parse_mode='Markdown')
        await notify_admin(f"Error in /setcriteria: {e}")

async def set_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    args = context.args
    settings = load_settings()
    user_id_str = str(user_id)
    
    try:
        if not args:
            balance = get_user_settings(user_id).get('balance', None)
            message = f"💰 **Current balance**: {balance if balance is not None else 'Not set'}\n" \
                      f"Usage: `/setbalance <amount>`\n" \
                      f"Example: `/setbalance 1000`"
            await update.message.reply_text(message, parse_mode='Markdown')
            return
            
        new_balance = float(args[0])
        if new_balance <= 0:
            await update.message.reply_text("🚫 **Balance must be greater than 0.**", parse_mode='Markdown')
            return
            
        user_settings = get_user_settings(user_id)
        user_settings['balance'] = new_balance
        settings[user_id_str] = user_settings
        save_settings(settings)
        await update.message.reply_text(f"✅ **Balance set**: ${new_balance:.2f}", parse_mode='Markdown')
        logger.info(f"User {user_id} set balance to ${new_balance:.2f}")
        
    except Exception as e:
        logger.error(f"Error in set_balance: {e}")
        await update.message.reply_text(f"🚨 **Error**: {e}\nFormat: `/setbalance <amount>`", parse_mode='Markdown')
        await notify_admin(f"Error in /setbalance: {e}")

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("🚫 **Only admin can add users!**", parse_mode='Markdown')
        return
        
    try:
        new_user_id = int(context.args[0])
        users = load_allowed_users()
        if new_user_id not in users:
            users.append(new_user_id)
            save_allowed_users(users)
            await update.message.reply_text(f"✅ User `{new_user_id}` added.", parse_mode='Markdown')
            logger.info(f"Added user {new_user_id}")
        else:
            await update.message.reply_text("ℹ️ User already in list.", parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in add_user: {e}")
        await update.message.reply_text(f"🚨 **Error**: {e}", parse_mode='Markdown')
        await notify_admin(f"Error in /add_user: {e}")

async def idea(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Idea command from user {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    session = Session()
    try:
        user_settings = get_user_settings(user_id)
        balance = user_settings.get('balance', None)
        min_probability = user_settings.get('min_probability', 60.0)
        
        if balance is None:
            await update.message.reply_text(
                "🚫 **Balance not set.**\nUse `/setbalance <amount>` to set balance.",
                parse_mode='Markdown'
            )
            return
            
        cryptos = get_top_cryptos()
        opportunities = []
        
        for symbol, coin_id, price_change_1h, taker_buy_base, volume in cryptos:
            analysis = await trading_model.analyze_symbol(symbol, coin_id, price_change_1h, taker_buy_base, volume, balance)
            if not analysis:
                continue
                
            current_rsi = analysis['rsi']
            current_macd = analysis['macd']
            current_adx = analysis['adx']
            
            # LONG conditions
            if (analysis['long_prob'] >= min_probability and 
                current_rsi >= 30 and current_rsi <= 70 and 
                current_macd > 0 and current_adx > 20):
                opportunities.append({
                    **analysis,
                    'direction': 'LONG',
                    'probability': analysis['long_prob'],
                    'stop_loss': analysis['stop_loss_long'],
                    'take_profit_1': analysis['take_profit_1_long'],
                    'take_profit_2': analysis['take_profit_2_long']
                })
                
            # SHORT conditions
            if (analysis['short_prob'] >= min_probability and 
                current_rsi >= 30 and current_rsi <= 70 and 
                current_macd < 0 and current_adx > 20):
                opportunities.append({
                    **analysis,
                    'direction': 'SHORT',
                    'probability': analysis['short_prob'],
                    'stop_loss': analysis['stop_loss_short'],
                    'take_profit_1': analysis['take_profit_1_short'],
                    'take_profit_2': analysis['take_profit_2_short']
                })
        
        if not opportunities:
            await update.message.reply_text(
                f"🔍 **No trading opportunities found.**\nCurrent min probability: {min_probability}%",
                parse_mode='Markdown'
            )
            return
            
        # Sort by probability deviation from 50%
        opportunities.sort(key=lambda x: abs(x['probability'] - 50), reverse=True)
        
        # Process best opportunity
        best_opp = opportunities[0]
        symbol = best_opp['symbol']
        direction = best_opp['direction']
        current_price = best_opp['price']
        
        # Check for duplicate trades
        existing_trades = session.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.symbol == symbol,
            Trade.result.is_(None) | (Trade.result == 'TP1')
        ).all()
        
        is_duplicate = any(
            abs(trade.entry_price - current_price) / current_price < 0.005
            for trade in existing_trades
        )
        
        if is_duplicate:
            await update.message.reply_text("🔔 **Similar trade already active.**", parse_mode='Markdown')
            return
            
        # Calculate position size and RR ratio
        rr_ratio = ((best_opp['take_profit_1'] - current_price) / 
                   (current_price - best_opp['stop_loss'])) if direction == 'LONG' else (
                   (current_price - best_opp['take_profit_1']) / 
                   (best_opp['stop_loss'] - current_price))
                   
        position_size, position_size_percent = calculate_position_size(
            current_price, best_opp['stop_loss'], balance)
        position_size = position_size if direction == 'LONG' else -position_size
        
        # Calculate potential profits
        potential_profit_tp1 = (best_opp['take_profit_1'] - current_price) * position_size if direction == 'LONG' else (current_price - best_opp['take_profit_1']) * abs(position_size)
        potential_profit_tp2 = (best_opp['take_profit_2'] - current_price) * position_size if direction == 'LONG' else (current_price - best_opp['take_profit_2']) * abs(position_size)
        
        # Create trade record
        trade = Trade(
            user_id=user_id,
            symbol=symbol,
            entry_price=current_price,
            stop_loss=best_opp['stop_loss'],
            take_profit_1=best_opp['take_profit_1'],
            take_profit_2=best_opp['take_profit_2'],
            rr_ratio=rr_ratio,
            position_size=position_size,
            probability=best_opp['probability'],
            institutional_score=best_opp['institutional_score'],
            sentiment_score=best_opp['sentiment'],
            trader_level="Manual"
        )
        
        session.add(trade)
        session.flush()
        
        # Create trade metrics
        trade_metrics = TradeMetrics(
            trade_id=trade.id,
            symbol=symbol,
            entry_price=current_price,
            volume_change=best_opp['volume_change'],
            institutional_score=best_opp['institutional_score'],
            vwap_signal=best_opp['vwap_signal'],
            sentiment=best_opp['sentiment'],
            rsi=best_opp['rsi'],
            macd=best_opp['macd'],
            adx=best_opp['adx'],
            obv=best_opp['obv'],
            smart_money_score=min(100, best_opp['institutional_score'] + (best_opp['volume_change'] / 5)),
            probability=best_opp['probability']
        )
        
        session.add(trade_metrics)
        session.commit()
        
        # Prepare message for user
        price_precision = 6 if current_price < 1 else 2
        vwap_text = '🟢 Bullish' if best_opp['vwap_signal'] > 0 else '🔴 Bearish'
        macd_text = '🟢 Bullish' if best_opp['macd'] > 0 else '🔴 Bearish'
        tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
        
        message = (
            f"🔔 **New Trade: {symbol} {direction}**\n"
            f"💰 **Balance**: ${balance:.2f}\n"
            f"🎯 Entry: ${current_price:.{price_precision}f}\n"
            f"⛔ Stop Loss: ${best_opp['stop_loss']:.{price_precision}f}\n"
            f"💰 TP1: ${best_opp['take_profit_1']:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
            f"💰 TP2: ${best_opp['take_profit_2']:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
            f"📊 RR: {rr_ratio:.1f}:1\n"
            f"📏 Size: {position_size_percent:.2f}% ({abs(position_size):.6f} {symbol.split('/')[0]})\n"
            f"🎲 Probability: {best_opp['probability']:.1f}%\n"
            f"🏛️ Institutional: {best_opp['institutional_score']:.1f}%\n"
            f"📈 VWAP: {vwap_text}\n"
            f"📮 Sentiment: {best_opp['sentiment']:.1f}%\n"
            f"📊 RSI: {best_opp['rsi']:.1f} | MACD: {macd_text} | ADX: {best_opp['adx']:.1f}\n"
            f"💡 Logic: Change {best_opp['price_change']:.2f}%, Volume +{best_opp['volume_change']:.1f}%\n"
            f"📈 Chart: {tradingview_url}\n"
            f"💾 Trade saved. Mark result:"
        )
        
        # Create inline keyboard
        keyboard = [
            [InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade.id}"),
             InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade.id}"),
             InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}"),
             InlineKeyboardButton("🚫 Cancel", callback_data=f"CANCEL_{trade.id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send message with chart
        chart_path = create_price_chart(best_opp['df'], symbol, best_opp['price_change'])
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                await context.bot.send_photo(
                    chat_id=user_id,
                    photo=photo,
                    caption=message,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            os.remove(chart_path)
        else:
            await context.bot.send_message(
                chat_id=user_id,
                text=message + "\n⚠️ Failed to create chart.",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        logger.info(f"Created trade #{trade.id} for {symbol} ({direction})")
        asyncio.create_task(check_trade_result(symbol, current_price, best_opp['stop_loss'], best_opp['take_profit_1'], best_opp['take_profit_2'], trade.id))
            
    except Exception as e:
        logger.error(f"Error in idea command for user_id={user_id}: {str(e)}")
        await update.message.reply_text(f"🚨 **Error**: {str(e)}", parse_mode='Markdown')
        await notify_admin(f"Error in idea command for user_id={user_id}: {str(e)}")
    finally:
        session.close()

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Test command from user {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    session = None
    try:
        user_settings = get_user_settings(user_id)
        balance = user_settings.get('balance', None)
        if balance is None:
            await update.message.reply_text(
                "🚫 **Balance not set.**\nUse `/setbalance <amount>` to set balance.",
                parse_mode='Markdown'
            )
            return
            
        session = Session()
        symbols = [('BTC/USDT', 'BTC'), ('ETH/USDT', 'ETH')]
        directions = ['LONG', 'SHORT']
        
        for (symbol, coin_id), direction in zip(symbols, directions):
            current_price = await get_current_price(symbol)
            if current_price <= 0.0:
                logger.warning(f"Failed to get price for {symbol}, skipping")
                await update.message.reply_text(f"⚠️ Failed to get price for {symbol}.", parse_mode='Markdown')
                continue
                
            df = await trading_model.get_historical_data(symbol)
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}, skipping")
                await update.message.reply_text(f"⚠️ Failed to get data for {symbol}.", parse_mode='Markdown')
                continue
                
            df = trading_model.calculate_indicators(df)
            atr = df['atr_normalized'].iloc[-1] * current_price
            
            if direction == 'LONG':
                stop_loss = current_price - max(2 * atr, current_price * 0.005 if current_price >= 1 else current_price * 0.01)
                take_profit_1 = current_price + 6 * atr
                take_profit_2 = current_price + 10 * atr
            else:
                stop_loss = current_price + max(2 * atr, current_price * 0.005 if current_price >= 1 else current_price * 0.01)
                take_profit_1 = current_price - 6 * atr
                take_profit_2 = current_price - 10 * atr
                
            probability = await trading_model.predict_probability(
                symbol, direction, stop_loss, 
                abs(calculate_position_size(current_price, stop_loss, balance)[0]))
            display_probability = probability if direction == 'LONG' else 100.0 - probability
            
            # Create test trade
            existing_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.symbol == symbol,
                Trade.result.is_(None) | (Trade.result == 'TP1')
            ).all()
            
            is_duplicate = False
            for trade in existing_trades:
                entry_diff = abs(trade.entry_price - current_price) / current_price
                sl_diff = abs(trade.stop_loss - stop_loss) / current_price
                tp1_diff = abs(trade.take_profit_1 - take_profit_1) / current_price
                if entry_diff < 0.005 and sl_diff < 0.01 and tp1_diff < 0.01:
                    is_duplicate = True
                    logger.info(f"Skipping test trade for {symbol}, duplicate of trade #{trade.id}")
                    break
                    
            if is_duplicate:
                await update.message.reply_text(
                    f"🔔 **Test trade ({symbol}) already active.** Skipping...",
                    parse_mode='Markdown'
                )
                continue
                
            rr_ratio = (take_profit_1 - current_price) / (current_price - stop_loss) if direction == 'LONG' else (current_price - take_profit_1) / (stop_loss - current_price)
            position_size, position_size_percent = calculate_position_size(current_price, stop_loss, balance)
            position_size = position_size if direction == 'LONG' else -position_size
            potential_profit_tp1 = (take_profit_1 - current_price) * position_size if direction == 'LONG' else (current_price - take_profit_1) * abs(position_size)
            potential_profit_tp2 = (take_profit_2 - current_price) * position_size if direction == 'LONG' else (current_price - take_profit_2) * abs(position_size)
            
            trade = Trade(
                user_id=user_id,
                symbol=symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                rr_ratio=rr_ratio,
                position_size=position_size,
                probability=probability,
                institutional_score=80.0,
                sentiment_score=75.0 if direction == 'LONG' else 25.0,
                trader_level="Test"
            )
            session.add(trade)
            session.flush()
            
            trade_metrics = TradeMetrics(
                trade_id=trade.id,
                symbol=symbol,
                entry_price=current_price,
                volume_change=40.0,
                institutional_score=80.0,
                vwap_signal=1.0 if direction == 'LONG' else -1.0,
                sentiment=75.0 if direction == 'LONG' else 25.0,
                rsi=65.0 if direction == 'LONG' else 35.0,
                macd=1.0 if direction == 'LONG' else -1.0,
                adx=30.0,
                obv=1000000.0 if direction == 'LONG' else -1000000.0,
                smart_money_score=90.0,
                probability=probability
            )
            session.add(trade_metrics)
            session.commit()
            
            price_precision = 6 if current_price < 1 else 2
            vwap_text = '🟢 Bullish' if trade_metrics.vwap_signal > 0 else '🔴 Bearish'
            macd_text = '🟢 Bullish' if trade_metrics.macd > 0 else '🔴 Bearish'
            tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
            
            message = (
                f"🔔 **Test Trade: {symbol} {direction}**\n"
                f"💰 **Balance**: ${balance:.2f}\n"
                f"🎯 Entry: ${current_price:.{price_precision}f}\n"
                f"⛔ Stop Loss: ${stop_loss:.{price_precision}f}\n"
                f"💰 TP1: ${take_profit_1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                f"💰 TP2: ${take_profit_2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                f"📊 RR: {rr_ratio:.1f}:1\n"
                f"📏 Size: {position_size_percent:.2f}% ({abs(position_size):.6f} {coin_id})\n"
                f"🎲 Probability: {display_probability:.1f}%\n"
                f"🏛️ Institutional: {trade_metrics.institutional_score:.1f}%\n"
                f"📈 VWAP: {vwap_text}\n"
                f"📮 Sentiment: {trade_metrics.sentiment:.1f}%\n"
                f"📊 RSI: {trade_metrics.rsi:.1f} | MACD: {macd_text} | ADX: {trade_metrics.adx:.1f}\n"
                f"💡 Logic: Change 2.50%, Volume +40.0%\n"
                f"📈 Chart: {tradingview_url}\n"
                f"💾 Test trade saved. Mark result:"
            )
            
            keyboard = [
                [InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade.id}"),
                 InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade.id}"),
                 InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}"),
                 InlineKeyboardButton("🚫 Cancel", callback_data=f"CANCEL_{trade.id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            chart_path = create_price_chart(df, symbol, 2.5)
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as photo:
                    await context.bot.send_photo(
                        chat_id=user_id,
                        photo=photo,
                        caption=message,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                os.remove(chart_path)
            else:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=message + "\n⚠️ Failed to create chart.",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            
            logger.info(f"Created test trade #{trade.id} for {symbol} ({direction})")
            asyncio.create_task(check_trade_result(symbol, current_price, stop_loss, take_profit_1, take_profit_2, trade.id))
            
    except Exception as e:
        logger.error(f"Error in test command for user_id={user_id}: {str(e)}")
        await update.message.reply_text(f"🚨 **Error**: {str(e)}", parse_mode='Markdown')
        await notify_admin(f"Error in test command for user_id={user_id}: {str(e)}")
    finally:
        if session is not None:
            session.close()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Start command from user {user_id}")
    
    if not is_authorized(user_id):
        await update.message.reply_text(
            "🚫 **Access denied.** Contact admin for access.",
            parse_mode='Markdown'
        )
        return
        
    user_settings = get_user_settings(user_id)
    balance = user_settings.get('balance', None)
    min_probability = user_settings.get('min_probability', 60.0)
    auto_interval = user_settings.get('auto_interval', DEFAULT_AUTO_INTERVAL)

    message = (
        f"👋 **Welcome to Trading Bot!**\n\n"
        f"💰 **Your balance**: {f'${balance:.2f}' if balance is not None else 'Not set'}\n"
        f"🎲 **Min probability**: {min_probability}%\n"
        f"⏱ **Auto-search interval**: {auto_interval//60} min\n\n"
        f"📖 **Available commands**:\n"
        f"/idea - Find trading opportunity\n"
        f"/test - Create test trade\n"
        f"/setbalance <amount> - Set balance\n"
        f"/setminprobability <percent> - Set min probability\n"
        f"/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes] - Set criteria\n"
        f"/stats - Show trading statistics\n"
        f"/active - Show active trades\n"
        f"/history - Show trade history\n"
        f"/clear_trades - Clear your trades\n"
    )
    
    if user_id == ADMIN_ID:
        message += f"/add_user <user_id> - Add user (admin only)\n"

    await update.message.reply_text(message, parse_mode='Markdown')

    # Setup jobs for user
    job_name = f"auto_search_{user_id}"
    current_jobs = context.job_queue.get_jobs_by_name(job_name)
    for job in current_jobs:
        job.schedule_removal()
        logger.info(f"Removed old auto-search job {job_name} for user_id={user_id}")

    context.job_queue.run_repeating(
        auto_search_trades,
        interval=auto_interval,
        first=auto_interval,
        name=job_name,
        data=user_id
    )
    logger.info(f"Started auto-search job for user_id={user_id} with interval {auto_interval} sec")

    # Setup trade checking job
    job_check_trades = f"check_trades_{user_id}"
    current_check_jobs = context.job_queue.get_jobs_by_name(job_check_trades)
    for job in current_check_jobs:
        job.schedule_removal()
        logger.info(f"Removed old trade check job {job_check_trades} for user_id={user_id}")

    context.job_queue.run_repeating(
        check_active_trades,
        interval=60,  # Check every minute
        first=10,
        name=job_check_trades,
        data=user_id
    )
    logger.info(f"Started trade check job for user_id={user_id}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")
    await notify_admin(f"Bot error: {context.error}")
    if update and update.effective_user:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text="🚨 **An error occurred.** Please try again or contact admin.",
            parse_mode='Markdown'
        )

async def clear_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Clear trades command from user {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    session = None
    try:
        session = Session()
        deleted_trades = session.query(Trade).filter_by(user_id=user_id).delete()
        deleted_metrics = session.query(TradeMetrics).filter(
            TradeMetrics.trade_id.in_(
                session.query(Trade.id).filter_by(user_id=user_id)
            )
        ).delete()
        session.commit()
        await update.message.reply_text(
            f"🗑️ **Deleted {deleted_trades} trades and {deleted_metrics} metrics for your account.**",
            parse_mode='Markdown'
        )
        logger.info(f"Deleted {deleted_trades} trades and {deleted_metrics} metrics for user_id={user_id}")
    except Exception as e:
        logger.error(f"Error clearing trades for user_id={user_id}: {e}")
        await update.message.reply_text(f"🚨 **Error clearing trades**: {e}", parse_mode='Markdown')
        await notify_admin(f"Error in clear_trades for user_id={user_id}: {e}")
    finally:
        if session is not None:
            session.close()

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 You are not authorized to use this bot.")
        return
        
    session = Session()
    try:
        trades = session.query(Trade).filter(Trade.user_id == user_id, Trade.result.isnot(None)).all()
        if not trades:
            await update.message.reply_text("No completed trades yet.")
            return
            
        total_trades = len(trades)
        successful_trades = sum(1 for trade in trades if trade.result in ['TP1', 'TP2'])
        success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = 0
        
        for trade in trades:
            trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade.id).first()
            if trade_metrics:
                final_price = trade.stop_loss if trade_metrics.success == 'SL' else trade.take_profit_1 if trade_metrics.success == 'TP1' else trade.take_profit_2
                if trade.position_size > 0:  # LONG
                    pnl = (final_price - trade.entry_price) * trade.position_size
                else:  # SHORT
                    pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                total_pnl += pnl
                logger.warning(f"Trade #{trade.id}, PNL={pnl:.2f}, final_price={final_price:.2f}, entry_price={trade.entry_price:.2f}, position_size={trade.position_size}")
                
        user_settings = get_user_settings(user_id)
        balance = user_settings.get('balance', 0)
        
        text = (
            f"📊 **Statistics**:\n"
            f"Total trades: {total_trades}\n"
            f"Successful trades: {successful_trades} ({success_rate:.2f}%)\n"
            f"Total PNL: {total_pnl:.2f} USDT\n"
            f"Current balance: {balance:.2f} USDT"
        )
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Error in stats for user_id={user_id}: {e}")
        await update.message.reply_text(f"🚫 Error: {str(e)}")
        await notify_admin(f"Error in stats for user_id={user_id}: {e}")
    finally:
        session.close()

async def active(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    session = Session()
    try:
        active_trades = session.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.result.is_(None) | (Trade.result == 'TP1')
        ).order_by(Trade.timestamp.desc()).limit(5).all()
        
        if not active_trades:
            await update.message.reply_text("📊 **Active trades**: No active trades.", parse_mode='Markdown')
            return
            
        message = "📊 **Active trades**:\n"
        for trade in active_trades:
            price_precision = 6 if trade.entry_price < 1 else 2
            current_price = await get_current_price(trade.symbol)
            status = '🟡 Pending' if trade.result is None else '✅ TP1 reached'
            message += (
                f"#{trade.id}: *{trade.symbol} LONG*\n"
                f"🎯 Entry: ${trade.entry_price:.{price_precision}f} | Current: ${current_price:.{price_precision}f}\n"
                f"⛔ SL: ${trade.stop_loss:.{price_precision}f} | 💰 TP1: ${trade.take_profit_1:.{price_precision}f} | 💰 TP2: ${trade.take_profit_2:.{price_precision}f}\n"
                f"📊 Status: {status}\n"
                f"⏰ Time: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
            )
            
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_active")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in active: {e}")
        await update.message.reply_text(f"🚨 **Error**: {e}", parse_mode='Markdown')
        await notify_admin(f"Error in /active: {e}")
    finally:
        session.close()

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Access denied.**", parse_mode='Markdown')
        return
        
    session = Session()
    try:
        trades = session.query(Trade).filter_by(user_id=user_id).order_by(Trade.timestamp.desc()).limit(5).all()
        if not trades:
            await update.message.reply_text("📜 **Trade history**: No trades yet.", parse_mode='Markdown')
            return
            
        message = "📜 **Trade history**:\n"
        for trade in trades:
            price_precision = 6 if trade.entry_price < 1 else 2
            status = '🟡 Active' if trade.result is None or trade.result == 'TP1' else ('✅ TP2' if trade.result == 'TP2' else '❌ SL')
            message += (
                f"#{trade.id}: *{trade.symbol} LONG*\n"
                f"🎯 Entry: ${trade.entry_price:.{price_precision}f}\n"
                f"⛔ SL: ${trade.stop_loss:.{price_precision}f} | 💰 TP1: ${trade.take_profit_1:.{price_precision}f} | 💰 TP2: ${trade.take_profit_2:.{price_precision}f}\n"
                f"📊 Status: {status}\n"
                f"⏰ Time: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
            )
            
        keyboard = [
            [InlineKeyboardButton("🟡 Active", callback_data="filter_active")],
            [InlineKeyboardButton("✅ Completed", callback_data="filter_completed")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in history: {e}")
        await update.message.reply_text(f"🚨 **Error**: {e}", parse_mode='Markdown')
        await notify_admin(f"Error in /history: {e}")
    finally:
        session.close()

def main():
    try:
        # Initialize application
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Register command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("idea", idea))
        application.add_handler(CommandHandler("test", test))
        application.add_handler(CommandHandler("setbalance", set_balance))
        application.add_handler(CommandHandler("setminprobability", set_min_probability))
        application.add_handler(CommandHandler("setcriteria", set_criteria))
        application.add_handler(CommandHandler("add_user", add_user))
        application.add_handler(CommandHandler("clear_trades", clear_trades))
        application.add_handler(CommandHandler("active", active))
        application.add_handler(CommandHandler("history", history))
        application.add_handler(CommandHandler("stats", stats))
        application.add_handler(CallbackQueryHandler(button))

        # Register error handler
        application.add_error_handler(error_handler)

        # Setup daily model retraining
        application.job_queue.run_daily(
            retrain_model_daily,
            time(hour=0, minute=0),
            days=(0, 1, 2, 3, 4, 5, 6),
            name="retrain_model_daily"
        )
        logger.info("Scheduled daily model retraining")

        # Ensure required files exist
        ensure_files_exist()

        # Start bot
        logger.info("Starting bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Critical error starting bot: {e}")
        asyncio.run(notify_admin(f"Critical error starting bot: {e}"))

if __name__ == '__main__':
    main()
