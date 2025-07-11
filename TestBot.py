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
    'adx', 'bb_upper', 'bb_lower', 'support_level', 'resistance_level'
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

def get_model_for_symbol(self, symbol):
        """Выбираем модель для символа"""
        base_symbol = symbol.replace('/USDT', 'USDT')
        if 'combined' in self.models:
            return self.models['combined'], self.scalers['combined'], self.active_features_dict['combined']
        elif base_symbol in self.models:
            return self.models[base_symbol], self.scalers[base_symbol], self.active_features_dict[base_symbol]
        else:
            return None, None, None
# Класс TradingModel
class TradingModel:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.scalers = {}
        self.active_features = []
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.active_features = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else ACTIVE_FEATURES
                logger.info("Модель и скейлеры успешно загружены")
            else:
                logger.warning("Файл модели не найден")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            asyncio.create_task(notify_admin(f"Ошибка загрузки модели: {e}"))

    def save_model(self):
        try:
            joblib.dump({'models': self.models, 'scalers': self.scalers}, MODEL_PATH)
            joblib.dump(self.active_features, FEATURES_PATH)
            logger.info("Модель и скейлеры сохранены")
        except Exception as e:
            logger.error(f"save_model: Не удалось сохранить модель: {e}")
            asyncio.create_task(notify_admin(f"Не удалось сохранить модель: {e}"))

    async def get_historical_data(self, symbol, timeframe='15m', limit=1000):
        cache_file = os.path.join(DATA_CACHE_PATH, f"{symbol.replace('/', '_')}_{timeframe}_historical.pkl")
        if os.path.exists(cache_file):
            try:
                cache_mtime = os.path.getmtime(cache_file)
                if (datetime.utcnow().timestamp() - cache_mtime) < CACHE_TTL:
                    df = pd.read_pickle(cache_file)
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'price', 'taker_buy_base', 'symbol']
                    if not df.empty and all(col in df.columns for col in required_columns):
                        logger.info(f"get_historical_data: Кэш для {symbol}: {len(df)} записей")
                        return df
                    else:
                        logger.warning(f"get_historical_data: Кэш для {symbol} повреждён, удаляем")
                        os.remove(cache_file)
            except Exception as e:
                logger.error(f"get_historical_data: Ошибка чтения кэша для {symbol}: {e}")
                await notify_admin(f"Ошибка чтения кэша для {symbol}: {e}")
                if os.path.exists(cache_file):
                    os.remove(cache_file)

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                markets = await self.exchange.load_markets()
                if symbol not in markets:
                    logger.warning(f"get_historical_data: Пара {symbol} не найдена")
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
                    # Проверка корректности данных
                    if df['price'].isna().any() or (df['price'] <= 0).any():
                        logger.warning(f"get_historical_data: Некорректные данные для {symbol}, пропуски или нулевые цены")
                        return pd.DataFrame()
                    os.makedirs(DATA_CACHE_PATH, exist_ok=True)
                    df.to_pickle(cache_file)
                    logger.info(f"get_historical_data: Получено {len(df)} записей для {symbol}")
                    return df
                break
            except Exception as e:
                attempt += 1
                logger.error(f"get_historical_data: Попытка {attempt}/{MAX_RETRIES} не удалась для {symbol}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    await notify_admin(f"Не удалось получить данные для {symbol}: {e}")
            finally:
                await self.exchange.close()
        logger.warning(f"get_historical_data: Не удалось получить данные для {symbol}")
        return pd.DataFrame()

    
    async def predict_probability(self, symbol, direction='LONG', stop_loss=None, position_size=0):
        """
        Прогнозирует вероятность успешной сделки для заданного символа и направления
        
        Параметры:
            symbol (str): Торговая пара (например, 'BTC/USDT')
            direction (str): Направление сделки ('LONG' или 'SHORT')
            stop_loss (float): Цена стоп-лосса (опционально)
            position_size (float): Размер позиции (опционально)
        
        Возвращает:
            float: Вероятность успеха в процентах (0-100)
        """
        try:
            # Получаем исторические данные
            df = await self.get_historical_data(symbol)
            if df.empty:
                logger.warning(f"predict_probability: Пустой DataFrame для {symbol}")
                return 0.0
            
            # Рассчитываем индикаторы
            df = self.calculate_indicators(df)
            
            # Получаем модель и скейлер для символа
            model, scaler, active_features = self.get_model_for_symbol(symbol)
            if not model or not scaler:
                logger.error(f"predict_probability: Модель или скейлер не найдены для {symbol}")
                return 0.0

            # Проверяем доступность признаков
            available_features = [f for f in active_features if f in df.columns]
            if not available_features:
                logger.error(f"predict_probability: Нет доступных признаков для {symbol}")
                return 0.0

            # Подготавливаем данные для предсказания
            features = df[available_features].iloc[-1:].values
            if features.size == 0:
                logger.error(f"predict_probability: Нет данных для предсказания {symbol}")
                return 0.0

            # Масштабируем признаки и делаем предсказание
            features_scaled = scaler.transform(features)
            proba = model.predict_proba(features_scaled)[0][1] * 100
            
            # Корректируем вероятность для SHORT сделок
            if direction == 'SHORT':
                proba = 100 - proba
            
            # Ограничиваем вероятность диапазоном 0-100%
            final_proba = max(0.0, min(100.0, proba))
            
            logger.info(
                f"predict_probability: {symbol} {direction} | "
                f"Probability: {final_proba:.1f}% | "
                f"Price: {df['price'].iloc[-1]:.4f} | "
                f"RSI: {df['rsi'].iloc[-1]:.1f} | "
                f"MACD: {df['macd'].iloc[-1]:.4f}"
            )
            
            return final_proba
            
        except ccxt.NetworkError as e:
            logger.error(f"predict_probability: Ошибка сети для {symbol}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"predict_probability: Ошибка биржи для {symbol}: {e}")
        except Exception as e:
            logger.error(f"predict_probability: Неожиданная ошибка для {symbol}: {e}", exc_info=True)
        
        return 0.0


    
    def calculate_indicators(self, df):
        if df.empty or len(df) < 14:
            logger.warning("calculate_indicators: Недостаточно данных")
            return df
            
        try:
            # Проверка и подготовка данных
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"calculate_indicators: Отсутствуют столбцы: {missing_columns}")
                return df

            # Очистка данных
            df = df.dropna(subset=required_columns)
            df = df[(df['close'] > 0) & (df['volume'] >= 0)]
            
            if len(df) < 14:
                logger.warning("calculate_indicators: После очистки недостаточно данных")
                return df

            # Добавляем price если его нет
            if 'price' not in df.columns:
                df['price'] = df['close']

            # Волатильность (Bollinger Bands)
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()

            # Моментум (RSI)
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

            # Расчет MACD
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            # Тренд (ADX)
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

            # Объем (OBV)
            df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

            # VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_signal'] = np.where(df['close'] > df['vwap'], 1.0, -1.0)

            # Прочие признаки
            df['price_change_1h'] = df['close'].pct_change(4) * 100
            df['price_change_2h'] = df['close'].pct_change(8) * 100
            df['price_change_6h'] = df['close'].pct_change(24) * 100
            df['volume_score'] = df['volume'] / df['volume'].rolling(window=6).mean() * 100
            df['volume_change'] = df['volume'].pct_change() * 100
            df['atr_normalized'] = (df['high'] - df['low']) / df['close'].replace(0, 0.0001) * 100

            # Уровни поддержки и сопротивления
            df['support_level'] = df['low'].rolling(window=20).min() / df['close'].replace(0, 0.0001)
            df['resistance_level'] = df['high'].rolling(window=20).max() / df['close'].replace(0, 0.0001)

            # Заполнение пропусков (исправленная версия без устаревшего метода)
            df = df.ffill().fillna(0)
            
            logger.info(f"calculate_indicators: Сформированы признаки для {df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'}")
            return df
            
        except Exception as e:
            logger.error(f"calculate_indicators: Ошибка: {e}")
            return df
    

    

    

    async def analyze_symbol(self, symbol, coin_id, price_change_1h, taker_buy_base, volume, balance):
        try:
            df = await self.get_historical_data(symbol)
            if df.empty:
                logger.warning(f"analyze_symbol: Пустой DataFrame для {symbol}")
                return None
            df = self.calculate_indicators(df)
            current_price = df['price'].iloc[-1]
            atr = df['atr_normalized'].iloc[-1] * current_price
            stop_loss_long = current_price - max(2 * atr, current_price * STOP_LOSS_PCT)
            take_profit_1_long = current_price + 6 * atr
            take_profit_2_long = current_price + 10 * atr
            stop_loss_short = current_price + max(2 * atr, current_price * STOP_LOSS_PCT)
            take_profit_1_short = current_price - 6 * atr
            take_profit_2_short = current_price - 10 * atr
            long_prob = await self.predict_probability(symbol, 'LONG', stop_loss_long, abs(calculate_position_size(current_price, stop_loss_long, balance)[0]))
            short_prob = await self.predict_probability(symbol, 'SHORT', stop_loss_short, abs(calculate_position_size(current_price, stop_loss_short, balance)[0]))
            institutional_score = taker_buy_base / volume * 100 if volume > 0 else 50.0
            vwap_signal = (current_price - df['price'].rolling(window=20).mean().iloc[-1]) / df['price'].rolling(window=20).mean().iloc[-1] * 100
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
            logger.error(f"analyze_symbol: Ошибка для {symbol}: {e}")
            return None

# Инициализация модели
trading_model = TradingModel()

# Управление файлами
def ensure_files_exist():
    if not os.path.exists(DATA_CACHE_PATH):
        os.makedirs(DATA_CACHE_PATH)
        logger.info(f"Создана директория: {DATA_CACHE_PATH}")
    if not os.path.exists(ALLOWED_USERS_PATH):
        with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump([ADMIN_ID], f)
        logger.info(f"Создан файл: {ALLOWED_USERS_PATH}")
    if not os.path.exists(SETTINGS_PATH):
        default_settings = {str(ADMIN_ID): {"auto_trade": False, "interval": 300, "risk_level": "low"}}
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, indent=4)
        logger.info(f"Создан файл: {SETTINGS_PATH}")

def load_settings():
    ensure_files_exist()
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_settings(settings):
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        logger.info("save_settings: Настройки сохранены")
    except Exception as e:
        logger.error(f"save_settings: Ошибка сохранения: {e}")

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
            logger.info(f"get_user_settings: Загружены настройки для user_id={user_id}, balance={merged.get('balance', 'не указан')}")
            return merged
        return default
    except Exception as e:
        logger.error(f"get_user_settings: Ошибка для user_id={user_id}: {e}")
        asyncio.create_task(notify_admin(f"Ошибка в get_user_settings для user_id={user_id}: {e}"))
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
                logger.error(f"save_user_settings: Ошибка чтения JSON для user_id={user_id}: {e}")
        all_settings[str(user_id)] = settings
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(all_settings, f, indent=4, ensure_ascii=False)
        logger.info(f"save_user_settings: Настройки сохранены для user_id={user_id}, balance={settings.get('balance', 'не указан')}")
    except Exception as e:
        logger.error(f"save_user_settings: Ошибка для user_id={user_id}: {e}")
        asyncio.create_task(notify_admin(f"Ошибка в save_user_settings для user_id={user_id}: {e}"))

def load_allowed_users():
    ensure_files_exist()
    with open(ALLOWED_USERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_allowed_users(users):
    try:
        with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f)
        logger.info("save_allowed_users: Список пользователей сохранён")
    except Exception as e:
        logger.error(f"save_allowed_users: Ошибка сохранения: {e}")

def is_authorized(user_id):
    return user_id in load_allowed_users()

async def notify_admin(message):
    try:
        bot = Application.builder().token(TELEGRAM_TOKEN).build()
        await bot.bot.send_message(chat_id=ADMIN_ID, text=f"🚨 **Ошибка бота**: {message}")
    except Exception as e:
        logger.error(f"notify_admin: Не удалось отправить уведомление: {e}")

def get_news_sentiment(coin_id):
    cache_key = f"{coin_id}_{datetime.utcnow().strftime('%Y%m%d%H')}"
    if cache_key in SENTIMENT_CACHE:
        sentiment, timestamp = SENTIMENT_CACHE[cache_key]
        if (datetime.utcnow() - timestamp).total_seconds() < SENTIMENT_CACHE_TTL:
            logger.info(f"get_news_sentiment: Кэш для {coin_id}: {sentiment:.2f}")
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
        logger.info(f"get_news_sentiment: Сентимент для {coin_id}: {sentiment:.2f} ({len(sentiment_scores)} статей)")
        return sentiment
    except Exception as e:
        logger.error(f"get_news_sentiment: Ошибка для {coin_id}: {e}")
        return 0

async def get_current_price(symbol):
    try:
        binance_symbol = symbol.replace('/', '')
        ticker = await trading_model.exchange.fetch_ticker(binance_symbol)
        current_price = float(ticker['last'])
        logger.info(f"get_current_price: Цена для {symbol}: ${current_price:.6f}")
        return current_price
    except Exception as e:
        logger.error(f"get_current_price: Ошибка для {symbol}: {e}")
        await notify_admin(f"Ошибка получения цены для {symbol}: {e}")
        return 0.0

def calculate_position_size(entry_price, stop_loss, balance):
    if balance <= 0:
        logger.error(f"calculate_position_size: Некорректный баланс: {balance}")
        return 0, 0
    risk_amount = balance * RISK_PER_TRADE
    price_diff = max(abs(entry_price - stop_loss), entry_price * 0.001)
    position_size = risk_amount / price_diff if price_diff > 0 else 0.000018
    position_size_percent = (position_size * entry_price / balance) * 100
    logger.info(f"calculate_position_size: Баланс={balance}, Риск={risk_amount}, Размер позиции={position_size}")
    return position_size, position_size_percent

def create_price_chart(df, symbol, price_change):
    try:
        if df.empty:
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.title(f"{symbol} График цены (15 мин) - Нет данных")
            plt.text(0.5, 0.5, f"Изменение цены: {price_change:.2f}%", horizontalalignment='center')
            plt.grid()
            plt.savefig(SCREENSHOT_PATH)
            plt.close()
            return SCREENSHOT_PATH
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(df['timestamp'], df['price'], label=f"{symbol} Цена", color='blue')
        ax1.set_title(f"{symbol} График цены (15 мин)")
        ax1.set_ylabel("Цена (USD)")
        ax1.legend()
        ax1.grid()
        ax2.bar(df['timestamp'], df['volume'], label='Объём', color='green')
        ax2.set_ylabel("Объём")
        ax2.legend()
        ax2.grid()
        plt.tight_layout()
        plt.savefig(SCREENSHOT_PATH)
        plt.close()
        logger.info(f"create_price_chart: График создан для {symbol}")
        return SCREENSHOT_PATH
    except Exception as e:
        logger.error(f"create_price_chart: Ошибка для {symbol}: {e}")
        asyncio.create_task(notify_admin(f"Ошибка создания графика для {symbol}: {e}"))
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
                logger.error(f"get_top_cryptos: Ошибка тикера для {symbol}: {e}")
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
            logger.info(f"get_top_cryptos: Отключены пары с винрейтом <60%: {disabled_pairs}")
        finally:
            session.close()
        return result
    except Exception as e:
        logger.error(f"get_top_cryptos: Ошибка: {e}")
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
        logger.info(f"check_trade_result: Сделка #{trade_id} ({symbol}): success={success}")
    except Exception as e:
        logger.error(f"check_trade_result: Ошибка для {symbol}: {e}")
        await notify_admin(f"Ошибка проверки результата для {symbol}: {e}")
    finally:
        session.close()

trade_lock = Lock()

async def check_active_trades(context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    try:
        user_id = context.job.data
        logger.info(f"check_active_trades: Проверка активных сделок для user_id={user_id}")
        with trade_lock:
            active_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                (Trade.result.is_(None) | (Trade.result == 'TP1'))
            ).all()
            if not active_trades:
                logger.info(f"check_active_trades: Нет активных сделок для user_id={user_id}")
                return
            for trade in active_trades:
                trade_exists = session.query(Trade).filter_by(id=trade.id).first()
                if not trade_exists:
                    logger.warning(f"check_active_trades: Сделка #{trade.id} ({trade.symbol}) не найдена")
                    continue
                current_price = await get_current_price(trade.symbol)
                if current_price is None or current_price <= 0:
                    logger.warning(f"check_active_trades: Некорректная цена для {trade.symbol}: {current_price}")
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
                        balance += pnl
                        user_settings['balance'] = balance
                        save_user_settings(user_id, user_settings)
                        message = (
                            f"📊 **Сделка #{trade.id}** ({trade.symbol}): "
                            f"{'✅ TP1' if new_result == 'TP1' else '✅ TP2' if new_result == 'TP2' else '❌ SL'} достигнут\n"
                            f"🎯 Вход: ${trade.entry_price:.{price_precision}f} | Выход: ${final_price:.{price_precision}f}\n"
                            f"💸 PNL: {pnl:.2f} USDT\n"
                            f"💰 Новый баланс: ${balance:.2f}"
                        )
                        await context.bot.send_message(
                            chat_id=trade.user_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"check_active_trades: Сделка #{trade.id} ({trade.symbol}) обновлена: result={new_result}, PNL={pnl:.2f}, balance={balance:.2f}")
                    except Exception as e:
                        logger.error(f"check_active_trades: Ошибка при обновлении сделки #{trade.id} ({trade.symbol}): {e}")
                        await notify_admin(f"Ошибка при обновлении сделки #{trade.id} для user_id={user_id}: {e}")
                        session.rollback()
    except Exception as e:
        logger.error(f"check_active_trades: Общая ошибка для user_id={user_id}: {e}")
        await notify_admin(f"Ошибка в check_active_trades для user_id={user_id}: {e}")
    finally:
        session.close()

async def retrain_model_daily(self, context: ContextTypes.DEFAULT_TYPE):
    logger.info("retrain_model_daily: Начало дообучения модели")
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
            logger.warning(f"retrain_model_daily: Недостаточно данных для дообучения ({len(trades)} сделок)")
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
                trade.entry_price / (trade.stop_loss + 1e-10),  # Пример дополнительного признака
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
        
        X = pd.DataFrame(X, columns=self.active_features)
        y = np.array(y)
        
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        logger.info(f"retrain_model_daily: Распределение классов: {class_counts}")
        
        if len(class_counts) < 2 or min(counts) < 2:
            logger.warning(f"retrain_model_daily: Несбалансированные классы: {class_counts}")
            return
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = self.scalers.get('combined')
        if not scaler:
            scaler = StandardScaler()
            self.scalers['combined'] = scaler
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = self.models.get('combined')
        if not model:
            model = lgb.LGBMClassifier(random_state=42)
            self.models['combined'] = model
        
        loss_before = model.score(X_test_scaled, y_test)
        logger.info(f"retrain_model_daily: Точность до дообучения: {loss_before:.4f}")
        
        model.fit(X_train_scaled, y_train, sample_weight=[2.0 if y == 0 else 0.5 if y == 1 else 1.0 for y in y_train])
        
        loss_after = model.score(X_test_scaled, y_test)
        logger.info(f"retrain_model_daily: Точность после дообучения: {loss_after:.4f}, сэмплов: {len(X)}")
        
        self.models['combined'] = model
        self.scalers['combined'] = scaler
        self.save_model()
        
    except Exception as e:
        logger.error(f"retrain_model_daily: Ошибка при дообучении: {str(e)}")
        await notify_admin(f"Ошибка в retrain_model_daily: {str(e)}")
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
                if analysis['long_prob'] >= min_probability:
                    opportunities.append({
                        **analysis,
                        'direction': 'LONG',
                        'probability': analysis['long_prob'],
                        'stop_loss': analysis['stop_loss_long'],
                        'take_profit_1': analysis['take_profit_1_long'],
                        'take_profit_2': analysis['take_profit_2_long']
                    })
                if analysis['short_prob'] >= min_probability:
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
            for opp in opportunities:
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
                if (direction == 'LONG' and probability < min_probability) or (direction == 'SHORT' and (100 - probability) < min_probability):
                    continue
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
                        logger.info(f"auto_search_trades: Пропущена сделка для {symbol}, уже есть активная сделка #{trade.id}")
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
                    trader_level="Новичок"
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
                vwap_text = '🟢 Бычий' if vwap_signal > 0 else '🔴 Медвежий'
                macd_text = '🟢 Бычий' if macd > 0 else '🔴 Медвежий'
                tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
                message = (
                    f"🔔 **Новая сделка: {symbol} {direction}**\n"
                    f"💰 **Баланс**: ${balance:.2f}\n"
                    f"🎯 Вход: ${current_price:.{price_precision}f}\n"
                    f"⛔ Стоп-лосс: ${stop_loss:.{price_precision}f}\n"
                    f"💰 TP1: ${take_profit_1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                    f"💰 TP2: ${take_profit_2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                    f"📊 RR: {rr_ratio:.1f}:1\n"
                    f"📏 Размер: {position_size_percent:.2f}% ({abs(position_size):.6f} {symbol.split('/')[0]})\n"
                    f"🎲 Вероятность: {probability:.1f}%\n"
                    f"🏛️ Институц.: {institutional_score:.1f}%\n"
                    f"📈 VWAP: {vwap_text}\n"
                    f"📮 Сентимент: {sentiment:.1f}%\n"
                    f"📊 RSI: {rsi:.1f} | MACD: {macd_text} | ADX: {adx:.1f}\n"
                    f"💡 Логика: Рост {price_change:.2f}%, Объём +{volume_change:.1f}%\n"
                    f"📈 График: {tradingview_url}\n"
                    f"💾 Сделка сохранена. Отметьте результат:"
                )
                keyboard = [
                    [InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade.id}"),
                     InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade.id}"),
                     InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}"),
                     InlineKeyboardButton("🚫 Отмена", callback_data=f"CANCEL_{trade.id}")]
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
                        text=message + "\n⚠️ Не удалось создать график.",
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                logger.info(f"auto_search_trades: Сделка #{trade.id} создана для {symbol} ({direction})")
                break
        finally:
            session.close()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except telegram.error.BadRequest as e:
        logger.warning(f"button: Устаревший или недействительный запрос: {e}")
        return
    user_id = query.from_user.id
    if not is_authorized(user_id):
        await query.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    session = Session()
    try:
        action, trade_id = query.data.split('_')
        trade_id = int(trade_id)
        trade = session.query(Trade).filter_by(id=trade_id, user_id=user_id).first()
        if not trade:
            await query.message.edit_text("🚫 **Сделка не найдена или не принадлежит вам.**", parse_mode='Markdown')
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
                f"📊 **Сделка #{trade.id}** ({trade.symbol}): "
                f"{'✅ TP1' if action == 'TP1' else '✅ TP2' if action == 'TP2' else '❌ SL'} достигнут\n"
                f"🎯 Вход: ${trade.entry_price:.{price_precision}f} | Выход: ${final_price:.{price_precision}f}\n"
                f"💸 PNL: {pnl:.2f} USDT\n"
                f"💰 Новый баланс: ${balance:.2f}"
            )
            await query.message.edit_text(message, parse_mode='Markdown')
            logger.info(f"button: Сделка #{trade.id} ({trade.symbol}) обновлена: {action}, PNL={pnl:.2f}")
        elif action == 'CANCEL':
            session.delete(trade)
            session.query(TradeMetrics).filter_by(trade_id=trade.id).delete()
            session.commit()
            await query.message.edit_text(f"🗑️ **Сделка #{trade.id}** ({trade.symbol}) отменена.", parse_mode='Markdown')
            logger.info(f"button: Сделка #{trade.id} ({trade.symbol}) отменена")
    except Exception as e:
        logger.error(f"button: Ошибка для user_id={user_id}: {e}")
        await query.message.reply_text(f"🚨 **Ошибка**: {e}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в button для user_id={user_id}: {e}")
    finally:
        session.close()

async def set_min_probability(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    args = context.args
    settings = load_settings()
    user_settings = get_user_settings(user_id)
    user_id_str = str(user_id)
    try:
        if not args:
            min_probability = user_settings.get('min_probability', 60.0)
            message = (
                f"⚙️ **Текущая минимальная вероятность**: {min_probability}%\n"
                f"Используйте: `/setminprobability <процент>`\n"
                f"Пример: `/setminprobability 60`"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
            return
        min_probability = float(args[0])
        if min_probability < 0 or min_probability > 100:
            await update.message.reply_text("🚫 **Вероятность должна быть от 0 до 100%.**", parse_mode='Markdown')
            return
        user_settings['min_probability'] = min_probability
        settings[user_id_str] = user_settings
        save_settings(settings)
        await update.message.reply_text(f"✅ **Минимальная вероятность установлена**: {min_probability}%", parse_mode='Markdown')
        logger.info(f"set_min_probability: Пользователь {user_id} установил минимальную вероятность: {min_probability}%")
    except Exception as e:
        logger.error(f"set_min_probability: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка**: {e}\nФормат: `/setminprobability <процент>`", parse_mode='Markdown')
        await notify_admin(f"Ошибка в /setminprobability: {e}")

async def set_criteria(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    args = context.args
    user_settings = get_user_settings(user_id)
    try:
        if len(args) < 3:
            message = (
                f"⚙️ **Текущие критерии**:\n"
                f"📊 Рост: {user_settings['price_threshold']}% | Объём: {user_settings['volume_threshold']}% | RSI: {user_settings['rsi_threshold']} ({'вкл' if user_settings['use_rsi'] else 'выкл'})\n"
                f"⏱ Интервал: {user_settings['auto_interval']//60} мин\n"
                f"Используйте: `/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes]`\n"
                f"Пример: `/setcriteria 5 0.3 40 rsi_off 5`"
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
            f"✅ **Критерии обновлены**:\n"
            f"📊 Рост: {price_threshold}% | Объём: {volume_threshold}% | RSI: {rsi_threshold} ({'вкл' if use_rsi else 'выкл'})\n"
            f"⏱ Интервал: {interval_minutes} мин",
            parse_mode='Markdown'
        )
        logger.info(f"set_criteria: Пользователь {user_id} обновил критерии: {user_settings}")
    except Exception as e:
        logger.error(f"set_criteria: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка**: {e}\nФормат: `/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes]`", parse_mode='Markdown')
        await notify_admin(f"Ошибка в /setcriteria: {e}")

async def set_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    args = context.args
    settings = load_settings()
    user_id_str = str(user_id)
    try:
        if not args:
            balance = get_user_settings(user_id).get('balance', None)
            message = f"💰 **Текущий баланс**: {balance if balance is not None else 'Не установлен'}\n" \
                      f"Используйте: `/setbalance <сумма>`\n" \
                      f"Пример: `/setbalance 1000`"
            await update.message.reply_text(message, parse_mode='Markdown')
            return
        new_balance = float(args[0])
        if new_balance <= 0:
            await update.message.reply_text("🚫 **Баланс должен быть больше 0.**", parse_mode='Markdown')
            return
        user_settings = get_user_settings(user_id)
        user_settings['balance'] = new_balance
        settings[user_id_str] = user_settings
        save_settings(settings)
        await update.message.reply_text(f"✅ **Баланс установлен**: ${new_balance:.2f}", parse_mode='Markdown')
        logger.info(f"set_balance: Пользователь {user_id} установил баланс: ${new_balance:.2f}")
    except Exception as e:
        logger.error(f"set_balance: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка**: {e}\nФормат: `/setbalance <сумма>`", parse_mode='Markdown')
        await notify_admin(f"Ошибка в /setbalance: {e}")

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("🚫 **Только админ может добавлять пользователей!**", parse_mode='Markdown')
        return
    try:
        new_user_id = int(context.args[0])
        users = load_allowed_users()
        if new_user_id not in users:
            users.append(new_user_id)
            save_allowed_users(users)
            await update.message.reply_text(f"✅ Пользователь `{new_user_id}` добавлен.", parse_mode='Markdown')
            logger.info(f"add_user: Добавлен пользователь {new_user_id}")
        else:
            await update.message.reply_text("ℹ️ Пользователь уже в списке.", parse_mode='Markdown')
    except Exception as e:
        logger.error(f"add_user: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка**: {e}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в /add_user: {e}")

async def idea(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"idea: Команда от пользователя {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    session = Session()
    try:
        # Загрузка настроек пользователя
        user_settings = get_user_settings(user_id)
        balance = user_settings.get('balance', None)
        min_probability = user_settings.get('min_probability', 60.0)
        auto_interval = user_settings.get('auto_interval', DEFAULT_AUTO_INTERVAL)

        # Проверка наличия баланса
        if balance is None:
            await update.message.reply_text(
                "🚫 **Баланс не установлен.**\nИспользуйте `/setbalance <сумма>` для установки баланса.",
                parse_mode='Markdown'
            )
            return

        # Сброс текущих задач автопоиска
        job_name = f"auto_search_{user_id}"
        current_jobs = context.job_queue.get_jobs_by_name(job_name)
        for job in current_jobs:
            job.schedule_removal()
            logger.info(f"idea: Задача автопоиска {job_name} сброшена для user_id={user_id}")

        # Получение списка криптовалют
        top_cryptos = get_top_cryptos()
        if not top_cryptos:
            await update.message.reply_text("🔍 **Нет доступных данных по криптовалютам.**", parse_mode='Markdown')
            return

        opportunities = []
        # Анализ каждой криптовалютной пары
        for symbol, coin_id, price_change_1h, taker_buy_base, volume in top_cryptos:
            if not symbol.replace('/', '').isalnum():
                logger.warning(f"idea: Некорректный символ {symbol}, пропускаем")
                continue
            analysis = await trading_model.analyze_symbol(symbol, coin_id, price_change_1h, taker_buy_base, volume, balance)
            if not analysis:
                logger.warning(f"idea: Анализ для {symbol} не удался, пропускаем")
                continue
            # Добавление LONG и SHORT возможностей
            if analysis['long_prob'] >= min_probability:
                opportunities.append({
                    **analysis,
                    'direction': 'LONG',
                    'probability': analysis['long_prob'],
                    'stop_loss': analysis['stop_loss_long'],
                    'take_profit_1': analysis['take_profit_1_long'],
                    'take_profit_2': analysis['take_profit_2_long']
                })
            if analysis['short_prob'] >= min_probability:
                opportunities.append({
                    **analysis,
                    'direction': 'SHORT',
                    'probability': analysis['short_prob'],
                    'stop_loss': analysis['stop_loss_short'],
                    'take_profit_1': analysis['take_profit_1_short'],
                    'take_profit_2': analysis['take_profit_2_short']
                })

        # Если нет подходящих возможностей
        if not opportunities:
            await update.message.reply_text(
                f"🔍 **Нет подходящих возможностей для торговли.**\nТекущая минимальная вероятность: {min_probability}%",
                parse_mode='Markdown'
            )
            return

        # Сортировка возможностей по вероятности (чем дальше от 50%, тем лучше)
        opportunities.sort(key=lambda x: abs(x['probability'] - 50), reverse=True)

        # Обработка лучшей возможности
        for opp in opportunities:
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
            display_probability = probability if direction == 'LONG' else 100.0 - probability

            # Проверка минимальной вероятности
            if (direction == 'LONG' and probability < min_probability) or (direction == 'SHORT' and (100 - probability) < min_probability):
                logger.info(f"idea: Пропущена возможность для {symbol} ({direction}), вероятность {display_probability:.1f}% ниже {min_probability}%")
                continue

            # Проверка на дубликаты активных сделок
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
                    logger.info(f"idea: Пропущена сделка для {symbol}, уже есть активная сделка #{trade.id}")
                    break
            if is_duplicate:
                await update.message.reply_text(
                    f"🔔 **Возможность ({symbol}) уже активна.** Пробуем другую...",
                    parse_mode='Markdown'
                )
                await asyncio.sleep(0.5)
                continue

            # Расчет параметров сделки
            rr_ratio = (take_profit_1 - current_price) / (current_price - stop_loss) if direction == 'LONG' else (current_price - take_profit_1) / (stop_loss - current_price)
            position_size, position_size_percent = calculate_position_size(current_price, stop_loss, balance)
            position_size = position_size if direction == 'LONG' else -position_size
            potential_profit_tp1 = (take_profit_1 - current_price) * position_size if direction == 'LONG' else (current_price - take_profit_1) * abs(position_size)
            potential_profit_tp2 = (take_profit_2 - current_price) * position_size if direction == 'LONG' else (current_price - take_profit_2) * abs(position_size)

            # Сохранение сделки в базу данных
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
                trader_level="Новичок"
            )
            session.add(trade)
            session.flush()

            # Сохранение метрик сделки
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

            # Формирование сообщения
            price_precision = 6 if current_price < 1 else 2
            vwap_text = '🟢 Бычий' if vwap_signal > 0 else '🔴 Медвежий'
            macd_text = '🟢 Бычий' if macd > 0 else '🔴 Медвежий'
            tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
            message = (
                f"🔔 **Новая сделка: {symbol} {direction}**\n"
                f"💰 **Баланс**: ${balance:.2f}\n"
                f"🎯 Вход: ${current_price:.{price_precision}f}\n"
                f"⛔ Стоп-лосс: ${stop_loss:.{price_precision}f}\n"
                f"💰 TP1: ${take_profit_1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                f"💰 TP2: ${take_profit_2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                f"📊 RR: {rr_ratio:.1f}:1\n"
                f"📏 Размер: {position_size_percent:.2f}% ({abs(position_size):.6f} {symbol.split('/')[0]})\n"
                f"🎲 Вероятность: {display_probability:.1f}%\n"
                f"🏛️ Институц.: {institutional_score:.1f}%\n"
                f"📈 VWAP: {vwap_text}\n"
                f"📮 Сентимент: {sentiment:.1f}%\n"
                f"📊 RSI: {rsi:.1f} | MACD: {macd_text} | ADX: {adx:.1f}\n"
                f"💡 Логика: Рост {price_change:.2f}%, Объём +{volume_change:.1f}%\n"
                f"📈 График: {tradingview_url}\n"
                f"💾 Сделка сохранена. Отметьте результат:"
            )

            # Создание кнопок управления
            keyboard = [
                [InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade.id}"),
                 InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade.id}"),
                 InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}"),
                 InlineKeyboardButton("🚫 Отмена", callback_data=f"CANCEL_{trade.id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Отправка сообщения с графиком
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
                    text=message + "\n⚠️ Не удалось создать график.",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )

            logger.info(f"idea: Сделка #{trade.id} создана для {symbol} ({direction})")

            # Запуск задачи автопоиска
            context.job_queue.run_repeating(
                auto_search_trades,
                interval=auto_interval,
                first=auto_interval,
                name=f"auto_search_{user_id}",
                data=user_id
            )
            logger.info(f"idea: Задача автопоиска перезапущена для user_id={user_id} с интервалом {auto_interval} сек")

            # Запуск проверки результата сделки
            asyncio.create_task(check_trade_result(symbol, current_price, stop_loss, take_profit_1, take_profit_2, trade.id))
            return  # Останавливаемся после первой подходящей сделки
    except Exception as e:
        logger.error(f"idea: Ошибка для user_id={user_id}: {str(e)}")
        await update.message.reply_text(f"🚨 **Ошибка**: {str(e)}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в idea для user_id={user_id}: {str(e)}")
    finally:
        session.close()

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"test: Команда от пользователя {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    session = None
    try:
        user_settings = get_user_settings(user_id)
        balance = user_settings.get('balance', None)
        if balance is None:
            await update.message.reply_text(
                "🚫 **Баланс не установлен.**\nИспользуйте `/setbalance <сумма>` для установки баланса.",
                parse_mode='Markdown'
            )
            return
        session = Session()
        symbols = [('BTC/USDT', 'BTC'), ('ETH/USDT', 'ETH')]
        directions = ['LONG', 'SHORT']
        for (symbol, coin_id), direction in zip(symbols, directions):
            current_price = await get_current_price(symbol)
            if current_price <= 0.0:
                logger.warning(f"test: Не удалось получить цену для {symbol}, пропускаем")
                await update.message.reply_text(f"⚠️ Не удалось получить цену для {symbol}.", parse_mode='Markdown')
                continue
            df = await trading_model.get_historical_data(symbol)
            if df.empty:
                logger.warning(f"test: Пустой DataFrame для {symbol}, пропускаем")
                await update.message.reply_text(f"⚠️ Не удалось получить данные для {symbol}.", parse_mode='Markdown')
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
            probability = await trading_model.predict_probability(symbol, direction, stop_loss, abs(calculate_position_size(current_price, stop_loss, balance)[0]))
            display_probability = probability if direction == 'LONG' else 100.0 - probability
            price_change = 2.5 if direction == 'LONG' else -2.5
            volume_change = 40.0
            institutional_score = 80.0
            vwap_signal = 1.0 if direction == 'LONG' else -1.0
            sentiment = 75.0 if direction == 'LONG' else 25.0
            rsi = 65.0 if direction == 'LONG' else 35.0
            macd = 1.0 if direction == 'LONG' else -1.0
            adx = 30.0
            obv = 1000000.0 if direction == 'LONG' else -1000000.0
            smart_money_score = 90.0
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
                    logger.info(f"test: Пропущена тестовая сделка для {symbol}, уже есть активная сделка #{trade.id}")
                    break
            if is_duplicate:
                await update.message.reply_text(
                    f"🔔 **Тестовая сделка ({symbol}) уже активна.** Пропускаем...",
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
                institutional_score=institutional_score,
                sentiment_score=sentiment,
                trader_level="Новичок"
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
                smart_money_score=smart_money_score,
                probability=probability
            )
            session.add(trade_metrics)
            session.commit()
            price_precision = 6 if current_price < 1 else 2
            vwap_text = '🟢 Бычий' if vwap_signal > 0 else '🔴 Медвежий'
            macd_text = '🟢 Бычий' if macd > 0 else '🔴 Медвежий'
            tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
            message = (
                f"🔔 **Тестовая сделка: {symbol} {direction}**\n"
                f"💰 **Баланс**: ${balance:.2f}\n"
                f"🎯 Вход: ${current_price:.{price_precision}f}\n"
                f"⛔ Стоп-лосс: ${stop_loss:.{price_precision}f}\n"
                f"💰 TP1: ${take_profit_1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                f"💰 TP2: ${take_profit_2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                f"📊 RR: {rr_ratio:.1f}:1\n"
                f"📏 Размер: {position_size_percent:.2f}% ({abs(position_size):.6f} {coin_id})\n"
                f"🎲 Вероятность: {display_probability:.1f}%\n"
                f"🏛️ Институц.: {institutional_score:.1f}%\n"
                f"📈 VWAP: {vwap_text}\n"
                f"📮 Сентимент: {sentiment:.1f}%\n"
                f"📊 RSI: {rsi:.1f} | MACD: {macd_text} | ADX: {adx:.1f}\n"
                f"💡 Логика: Рост {price_change:.2f}%, Объём +{volume_change:.1f}%\n"
                f"📈 График: {tradingview_url}\n"
                f"💾 Тестовая сделка сохранена. Отметьте результат:"
            )
            keyboard = [
                [InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade.id}"),
                 InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade.id}"),
                 InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}"),
                 InlineKeyboardButton("🚫 Отмена", callback_data=f"CANCEL_{trade.id}")]
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
                    text=message + "\n⚠️ Не удалось создать график.",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            logger.info(f"test: Тестовая сделка #{trade.id} создана для {symbol} ({direction})")
            asyncio.create_task(check_trade_result(symbol, current_price, stop_loss, take_profit_1, take_profit_2, trade.id))
    except Exception as e:
        logger.error(f"test: Ошибка для user_id={user_id}: {str(e)}")
        await update.message.reply_text(f"🚨 **Ошибка**: {str(e)}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в test для user_id={user_id}: {str(e)}")
    finally:
        if session is not None:
            session.close()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"start: Команда от пользователя {user_id}")
    
    # Проверка авторизации
    if not is_authorized(user_id):
        await update.message.reply_text(
            "🚫 **Доступ запрещён.** Обратитесь к администратору для получения доступа.",
            parse_mode='Markdown'
        )
        return

    # Загрузка настроек пользователя
    user_settings = get_user_settings(user_id)
    balance = user_settings.get('balance', None)
    min_probability = user_settings.get('min_probability', 60.0)
    auto_interval = user_settings.get('auto_interval', DEFAULT_AUTO_INTERVAL)

    # Формирование приветственного сообщения
    message = (
        f"👋 **Добро пожаловать в трейдинг-бота!**\n\n"
        f"💰 **Ваш баланс**: {f'${balance:.2f}' if balance is not None else 'Не установлен'}\n"
        f"🎲 **Минимальная вероятность**: {min_probability}%\n"
        f"⏱ **Интервал автопоиска**: {auto_interval//60} мин\n\n"
        f"📖 **Доступные команды**:\n"
        f"/idea - Найти торговую возможность\n"
        f"/test - Создать тестовую сделку\n"
        f"/setbalance <сумма> - Установить баланс\n"
        f"/setminprobability <процент> - Установить минимальную вероятность\n"
        f"/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes] - Настроить критерии\n"
    )
    
    if user_id == ADMIN_ID:
        message += f"/add_user <user_id> - Добавить пользователя (только для админа)\n"

    await update.message.reply_text(message, parse_mode='Markdown')

    # Запуск задач для пользователя
    job_name = f"auto_search_{user_id}"
    current_jobs = context.job_queue.get_jobs_by_name(job_name)
    for job in current_jobs:
        job.schedule_removal()
        logger.info(f"start: Задача автопоиска {job_name} сброшена для user_id={user_id}")

    context.job_queue.run_repeating(
        auto_search_trades,
        interval=auto_interval,
        first=auto_interval,
        name=job_name,
        data=user_id
    )
    logger.info(f"start: Задача автопоиска запущена для user_id={user_id} с интервалом {auto_interval} сек")

    # Запуск проверки активных сделок
    job_check_trades = f"check_trades_{user_id}"
    current_check_jobs = context.job_queue.get_jobs_by_name(job_check_trades)
    for job in current_check_jobs:
        job.schedule_removal()
        logger.info(f"start: Задача проверки сделок {job_check_trades} сброшена для user_id={user_id}")

    context.job_queue.run_repeating(
        check_active_trades,
        interval=60,  # Проверка каждую минуту
        first=10,
        name=job_check_trades,
        data=user_id
    )
    logger.info(f"start: Задача проверки активных сделок запущена для user_id={user_id}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"error_handler: Ошибка: {context.error}")
    await notify_admin(f"Ошибка в боте: {context.error}")
    if update and update.effective_user:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text="🚨 **Произошла ошибка.** Пожалуйста, попробуйте снова или свяжитесь с администратором.",
            parse_mode='Markdown'
        )
async def clear_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        logger.info(f"clear_trades: Команда от пользователя {user_id}")
        if not is_authorized(user_id):
            await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
            return
        session = None
        try:
            session = Session()
            # Удаление только пользовательских записей из Trade
            deleted_trades = session.query(Trade).filter_by(user_id=user_id).delete()
            # Удаление связанных записей из TradeMetrics
            deleted_metrics = session.query(TradeMetrics).filter(
                TradeMetrics.trade_id.in_(
                    session.query(Trade.id).filter_by(user_id=user_id)
                )
            ).delete()
            session.commit()
            await update.message.reply_text(
                f"🗑️ **Удалено {deleted_trades} сделок и {deleted_metrics} метрик для вашего аккаунта.**",
                parse_mode='Markdown'
            )
            logger.info(f"clear_trades: Удалено {deleted_trades} сделок и {deleted_metrics} метрик для user_id={user_id}")
        except Exception as e:
            logger.error(f"clear_trades: Ошибка для user_id={user_id}: {e}")
            await update.message.reply_text(f"🚨 **Ошибка при очистке сделок**: {e}", parse_mode='Markdown')
            await notify_admin(f"Ошибка в clear_trades для user_id={user_id}: {e}")
        finally:
            if session is not None:
                session.close()

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not is_authorized(user_id):
            await update.message.reply_text("🚫 Вы не авторизованы для использования бота.")
            return
        session = Session()
        try:
            trades = session.query(Trade).filter(Trade.user_id == user_id, Trade.result.isnot(None)).all()
            if not trades:
                await update.message.reply_text("Нет завершённых сделок.")
                return
            total_trades = len(trades)
            successful_trades = sum(1 for trade in trades if trade.result in ['TP1', 'TP2'])
            success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = 0
            for trade in trades:
                trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade.id).first()
                if trade_metrics:
                    final_price = trade.stop_loss if trade_metrics.success == 'SL' else trade.take_profit_1 if trade_metrics.success == 'TP1' else trade.take_profit_2
                    # Учитываем направление сделки (покупка/продажа)
                    if trade.position_size > 0:  # Покупка (long)
                        pnl = (final_price - trade.entry_price) * trade.position_size
                    else:  # Продажа (short)
                        pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                    total_pnl += pnl
                    logger.warning(f"stats: Сделка #{trade.id}, PNL={pnl:.2f}, final_price={final_price:.2f}, entry_price={trade.entry_price:.2f}, position_size={trade.position_size}")
            user_settings = get_user_settings(user_id)
            balance = user_settings.get('balance', 0)
            text = (
                f"📊 Статистика:\n"
                f"Всего сделок: {total_trades}\n"
                f"Успешных сделок: {successful_trades} ({success_rate:.2f}%)\n"
                f"Общий PNL: {total_pnl:.2f} USDT\n"
                f"Текущий баланс: {balance:.2f} USDT"
            )
            await update.message.reply_text(text)
        except Exception as e:
            logger.error(f"stats: Ошибка для user_id={user_id}: {e}")
            await update.message.reply_text(f"🚫 Ошибка: {str(e)}")
            await notify_admin(f"Ошибка в stats для user_id={user_id}: {e}")
        finally:
            session.close()

async def active(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not is_authorized(user_id):
            await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
            return
        session = Session()
        try:
            active_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.result.is_(None) | (Trade.result == 'TP1')
            ).order_by(Trade.timestamp.desc()).limit(5).all()
            if not active_trades:
                await update.message.reply_text("📊 **Активные сделки**: Нет активных сделок.", parse_mode='Markdown')
                return
            message = "📊 **Активные сделки**:\n"
            for trade in active_trades:
                price_precision = 6 if trade.entry_price < 1 else 2
                current_price = await get_current_price(trade.symbol)
                status = '🟡 Ожидает' if trade.result is None else '✅ TP1 достигнут'
                message += (
                    f"#{trade.id}: *{trade.symbol} LONG*\n"
                    f"🎯 Вход: ${trade.entry_price:.{price_precision}f} | Текущая: ${current_price:.{price_precision}f}\n"
                    f"⛔ SL: ${trade.stop_loss:.{price_precision}f} | 💰 TP1: ${trade.take_profit_1:.{price_precision}f} | 💰 TP2: ${trade.take_profit_2:.{price_precision}f}\n"
                    f"📊 Статус: {status}\n"
                    f"⏰ Время: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            keyboard = [[InlineKeyboardButton("🔄 Обновить", callback_data="refresh_active")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"active: Ошибка: {e}")
            await update.message.reply_text(f"🚨 **Ошибка**: {e}", parse_mode='Markdown')
            await notify_admin(f"Ошибка в /active: {e}")
        finally:
            session.close()

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not is_authorized(user_id):
            await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
            return
        session = Session()
        try:
            trades = session.query(Trade).filter_by(user_id=user_id).order_by(Trade.timestamp.desc()).limit(5).all()
            if not trades:
                await update.message.reply_text("📜 **История сделок**: Нет сделок.", parse_mode='Markdown')
                return
            message = "📜 **История сделок**:\n"
            for trade in trades:
                price_precision = 6 if trade.entry_price < 1 else 2
                status = '🟡 Активна' if trade.result is None or trade.result == 'TP1' else ('✅ TP2' if trade.result == 'TP2' else '❌ SL')
                message += (
                    f"#{trade.id}: *{trade.symbol} LONG*\n"
                    f"🎯 Вход: ${trade.entry_price:.{price_precision}f}\n"
                    f"⛔ SL: ${trade.stop_loss:.{price_precision}f} | 💰 TP1: ${trade.take_profit_1:.{price_precision}f} | 💰 TP2: ${trade.take_profit_2:.{price_precision}f}\n"
                    f"📊 Статус: {status}\n"
                    f"⏰ Время: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            keyboard = [
                [InlineKeyboardButton("🟡 Активные", callback_data="filter_active")],
                [InlineKeyboardButton("✅ Завершённые", callback_data="filter_completed")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"history: Ошибка: {e}")
            await update.message.reply_text(f"🚨 **Ошибка**: {e}", parse_mode='Markdown')
            await notify_admin(f"Ошибка в /history: {e}")
        finally:
            session.close()

def main():
    try:
        # Инициализация приложения
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Регистрация обработчиков команд
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

        # Регистрация обработчика ошибок
        application.add_error_handler(error_handler)

        # Запуск ежедневного дообучения модели
        application.job_queue.run_daily(
            retrain_model_daily,
            time(hour=0, minute=0),
            days=(0, 1, 2, 3, 4, 5, 6),
            name="retrain_model_daily"
        )
        logger.info("main: Задача ежедневного дообучения модели запущена")

        # Обеспечение существования необходимых файлов
        ensure_files_exist()

        # Запуск бота
        logger.info("main: Бот запускается...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"main: Критическая ошибка при запуске бота: {e}")
        asyncio.run(notify_admin(f"Критическая ошибка при запуске бота: {e}"))

if __name__ == '__main__':
    main()



