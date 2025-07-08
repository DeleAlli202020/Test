import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import logging
import nest_asyncio
import ccxt.async_support as ccxt
import joblib
from sklearn.preprocessing import RobustScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import lightgbm as lgb
import json
import matplotlib.pyplot as plt
import sys
from dotenv import load_dotenv # type: ignore
import telegram

# Настройка логирования
logging.basicConfig(
    level=logging.WARNING,  # Изменяем уровень на WARNING
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot_prehost.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Установка кодировки UTF-8 для вывода в консоль
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
MODEL_PATH = os.path.join(BASE_DIR, 'model_improved.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.pkl')
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

STOP_LOSS_PCT = 0.01  # 1% стоп-лосс
TAKE_PROFIT_1_PCT = 0.03  # 3% для первого тейк-профита
TAKE_PROFIT_2_PCT = 0.05  # 5% для второго тейк-профита
PRICE_THRESHOLD = 0.5  # Порог изменения цены в процентах
VOLUME_THRESHOLD = 5.0  # Порог изменения объёма в процентах
RSI_THRESHOLD = 40.0  # Порог RSI

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
    result = Column(String, nullable=True)  # None, 'SL', 'TP1', 'TP2'

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

# Инициализация биржи
exchange = ccxt.binance({'enableRateLimit': True})

def ensure_files_exist():
    """Проверяет и создаёт необходимые файлы и директории."""
    # Создание папки data_cache
    if not os.path.exists(DATA_CACHE_PATH):
        os.makedirs(DATA_CACHE_PATH)
        logger.info(f"Создана директория: {DATA_CACHE_PATH}")

    # Создание allowed_users.json
    if not os.path.exists(ALLOWED_USERS_PATH):
        with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump([123456789], f)  # Добавляем ADMIN_ID по умолчанию
        logger.info(f"Создан файл: {ALLOWED_USERS_PATH}")

    # Создание settings.json
    if not os.path.exists(SETTINGS_PATH):
        default_settings = {
            "123456789": {
                "auto_trade": False,
                "interval": 300,
                "risk_level": "low"
            }
        }
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, indent=4)
        logger.info(f"Создан файл: {SETTINGS_PATH}")

# Управление настройками
def load_settings():
    """Загружает настройки пользователей."""
    ensure_files_exist()
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_settings(settings):
    try:
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info("save_settings: Настройки сохранены")
    except Exception as e:
        logger.error(f"save_settings: Ошибка сохранения: {e}")

def get_user_settings(user_id: int) -> dict:
    """Получает настройки пользователя из JSON-файла."""
    try:
        settings_file = os.path.join(BASE_DIR, 'settings.json')
        default = {
            'price_threshold': 0.3,
            'volume_threshold': 5,
            'rsi_threshold': 40,
            'use_rsi': True,
            'auto_interval': DEFAULT_AUTO_INTERVAL,
            'balance': 1000  # Начальный баланс по умолчанию
        }
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                all_settings = json.load(f)
            user_settings = all_settings.get(str(user_id), {})
            merged = {**default, **user_settings}
            logger.info(f"get_user_settings: Загружены настройки для user_id={user_id}, balance={merged.get('balance', 'не указан')}")
            return merged
        logger.info(f"get_user_settings: Файл не существует, возвращены настройки по умолчанию для user_id={user_id}")
        return default
    except Exception as e:
        logger.error(f"get_user_settings: Ошибка для user_id={user_id}: {e}")
        asyncio.create_task(notify_admin(f"Ошибка в get_user_settings для user_id={user_id}: {e}"))
        return default

def save_user_settings(user_id: int, settings: dict):
    """Сохраняет настройки пользователя в JSON-файл."""
    try:
        settings_file = os.path.join(BASE_DIR, 'settings.json')
        all_settings = {}
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    all_settings = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"save_user_settings: Ошибка чтения JSON для user_id={user_id}: {e}")
                all_settings = {}
        all_settings[str(user_id)] = settings
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(all_settings, f, indent=4, ensure_ascii=False)
        logger.info(f"save_user_settings: Настройки сохранены для user_id={user_id}, balance={settings.get('balance', 'не указан')}")
    except Exception as e:
        logger.error(f"save_user_settings: Ошибка для user_id={user_id}: {e}")
        asyncio.create_task(notify_admin(f"Ошибка в save_user_settings для user_id={user_id}: {e}"))

# Загрузка модели
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            models = data.get('models', {})
            scalers = data.get('scalers', {})
            active_features = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else []
            model = models.get('combined')
            scaler = scalers.get('combined')
            if model and scaler and active_features:
                logger.info("load_model: Модель и скейлер загружены")
                return model, scaler, active_features
            logger.warning("load_model: Неполные данные модели")
        except Exception as e:
            logger.error(f"load_model: Не удалось загрузить модель: {e}")
            asyncio.run(notify_admin(f"Не удалось загрузить модель: {e}"))
    logger.warning("load_model: Модель не найдена")
    return None, None, []

# Сохранение модели
def save_model(model, scaler, active_features):
    try:
        joblib.dump({'models': {'combined': model}, 'scalers': {'combined': scaler}}, MODEL_PATH)
        joblib.dump(active_features, FEATURES_PATH)
        logger.info("save_model: Модель и скейлер сохранены")
    except Exception as e:
        logger.error(f"save_model: Не удалось сохранить модель: {e}")
        asyncio.run(notify_admin(f"Не удалось сохранить модель: {e}"))

# Управление пользователями
def load_allowed_users():
    """Загружает список разрешённых пользователей."""
    ensure_files_exist()
    with open(ALLOWED_USERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_allowed_users(users):
    try:
        with open(ALLOWED_USERS_PATH, 'w') as f:
            json.dump(users, f)
        logger.info("save_allowed_users: Список пользователей сохранён")
    except Exception as e:
        logger.error(f"save_allowed_users: Ошибка сохранения: {e}")

def is_authorized(user_id):
    return user_id in load_allowed_users()

# Уведомления админу
async def notify_admin(message):
    try:
        bot = Application.builder().token(TELEGRAM_TOKEN).build()
        await bot.bot.send_message(chat_id=ADMIN_ID, text=f"🚨 **Ошибка бота**: {message}")
    except Exception as e:
        logger.error(f"notify_admin: Не удалось отправить уведомление: {e}")

# Получение исторических данных
async def get_historical_data(symbol, timeframe='15m', limit=1000):
    cache_file = os.path.join(DATA_CACHE_PATH, f"{symbol.replace('/', '_')}_{timeframe}_historical.pkl")
    if os.path.exists(cache_file):
        try:
            cache_mtime = os.path.getmtime(cache_file)
            if (datetime.utcnow().timestamp() - cache_mtime) < CACHE_TTL:
                df = pd.read_pickle(cache_file)
                logger.info(f"get_historical_data: Кэш для {symbol}: {len(df)} записей")
                return df
        except Exception as e:
            logger.error(f"get_historical_data: Ошибка чтения кэша для {symbol}: {e}")
            await notify_admin(f"Ошибка чтения кэша для {symbol}: {e}")

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            markets = await exchange.load_markets()
            if symbol not in markets:
                logger.warning(f"get_historical_data: Пара {symbol} не найдена")
                return pd.DataFrame()
            since = int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)
            all_ohlcv = []
            while len(all_ohlcv) < limit:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(limit, 1000))
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
            await exchange.close()
    logger.warning(f"get_historical_data: Не удалось получить данные для {symbol}")
    return pd.DataFrame()

# Технические индикаторы
def calculate_rsi(df, periods=14):
    if df.empty or len(df) < periods:
        logger.warning("calculate_rsi: Недостаточно данных")
        return pd.Series(0, index=df.index)
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss.where(loss != 0, 0.0001)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(df, fast=12, slow=26):
    if df.empty or len(df) < slow:
        logger.warning("calculate_macd: Недостаточно данных")
        return pd.Series(0, index=df.index)
    exp1 = df['price'].ewm(span=fast, adjust=False).mean()
    exp2 = df['price'].ewm(span=slow, adjust=False).mean()
    macd = (exp1 - exp2) / df['price'].iloc[-1] * 100 if not df.empty else 0
    return macd.fillna(0)

def calculate_adx(df, periods=14):
    if df.empty or len(df) < periods:
        logger.warning("calculate_adx: Недостаточно данных")
        return pd.Series(0, index=df.index)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['price'].astype(float)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=periods, adjust=False).mean()
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.ewm(span=periods, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=periods, adjust=False).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(span=periods, adjust=False).mean()
    return adx.fillna(0)

def calculate_obv(df):
    if df.empty or len(df) < 2:
        logger.warning("calculate_obv: Недостаточно данных")
        return pd.Series(0, index=df.index)
    price_diff = df['price'].diff()
    direction = np.sign(price_diff)
    obv = (direction * df['volume']).cumsum()
    return obv.fillna(0)

def calculate_vwap(df):
    if df.empty:
        logger.warning("calculate_vwap: Недостаточно данных")
        return pd.Series(0, index=df.index)
    typical_price = (df['high'].astype(float) + df['low'].astype(float) + df['price'].astype(float)) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap.fillna(0)

def calculate_vwap_signal(df):
    if df.empty:
        logger.warning("calculate_vwap_signal: Недостаточно данных")
        return pd.Series(0, index=df.index)
    vwap = calculate_vwap(df)
    current_price = df['price']
    return ((current_price - vwap) / vwap * 100).fillna(0)

def calculate_bb_width(df, periods=20):
    if df.empty or len(df) < periods:
        logger.warning("calculate_bb_width: Недостаточно данных")
        return pd.Series(0, index=df.index)
    sma = df['price'].rolling(window=periods).mean()
    std = df['price'].rolling(window=periods).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_width = (upper - lower) / sma
    return bb_width.fillna(0)

def calculate_atr_normalized(df, periods=14):
    if df.empty or len(df) < periods:
        logger.warning("calculate_atr_normalized: Недостаточно данных")
        return pd.Series(0, index=df.index)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['price'].astype(float)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=periods, adjust=False).mean()
    price = df['price']
    return (atr / price).fillna(0)

def calculate_support_resistance(df, window=20):
    if df.empty or len(df) < window:
        logger.warning("calculate_support_resistance: Недостаточно данных")
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    support = df['low'].rolling(window=window).min()
    resistance = df['high'].rolling(window=window).max()
    return support.fillna(df['price'].min()), resistance.fillna(df['price'].max())

# Анализ новостей
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

# Текущая цена
async def get_current_price(symbol):
    try:
        # Нормализуем символ для Binance API (например, ETH/USDT -> ETHUSDT)
        binance_symbol = symbol.replace('/', '')
        ticker = await exchange.fetch_ticker(binance_symbol)
        current_price = float(ticker['last'])
        logger.info(f"get_current_price: Цена для {symbol}: ${current_price:.6f}")
        return current_price
    except Exception as e:
        logger.error(f"get_current_price: Ошибка для {symbol}: {e}")
        await notify_admin(f"Ошибка получения цены для {symbol}: {e}")
        return 0.0
    
from sklearn.metrics import log_loss

def pnl_loss(y_true, y_pred_proba, trade_results, pnls=None):
    """
    Кастомная функция потерь для классификации, учитывающая результат сделки и PnL.
    
    Parameters:
    - y_true: истинные метки (1 для роста цены, 0 для падения).
    - y_pred_proba: предсказанные вероятности (из model.predict_proba).
    - trade_results: список результатов сделок ('SL', 'TP1', 'TP2', None).
    - pnls: список значений PnL (опционально, для весов пропорциональных прибыли/убытку).
    
    Returns:
    - loss: взвешенная логарифмическая потеря.
    """
    weights = []
    for i, result in enumerate(trade_results):
        if result == 'SL':
            weight = 2.0 * abs(pnls[i]) if pnls and pnls[i] is not None else 2.0
            weights.append(weight)  # Усиленный штраф за убыточные сделки
        elif result in ['TP1', 'TP2']:
            weight = 0.5 * abs(pnls[i]) if pnls and pnls[i] is not None else 0.5
            weights.append(weight)  # Меньший штраф для прибыльных сделок
        else:
            weights.append(1.0)  # Нейтральный вес для неизвестных результатов
    return log_loss(y_true, y_pred_proba, sample_weight=weights)

# Подготовка данных для дообучения
def prepare_training_data(df):
    try:
        if df.empty or len(df) < 48:
            logger.warning("prepare_training_data: Пустой или недостаточный датафрейм")
            return pd.DataFrame(columns=ACTIVE_FEATURES), np.array([])
        # Проверка наличия необходимых столбцов
        required_columns = ['price', 'volume', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"prepare_training_data: Отсутствуют столбцы: {missing_columns}")
            return pd.DataFrame(columns=ACTIVE_FEATURES), np.array([])
        
        X = pd.DataFrame(index=df.index)
        X['price_change_1h'] = df['price'].pct_change(4) * 100
        X['price_change_2h'] = df['price'].pct_change(8) * 100
        X['price_change_6h'] = df['price'].pct_change(24) * 100
        X['volume_score'] = df['volume'] / df['volume'].rolling(window=6).mean() * 100
        X['volume_change'] = df['volume'].pct_change() * 100
        X['atr_normalized'] = calculate_atr_normalized(df)
        X['rsi'] = calculate_rsi(df)
        X['macd'] = calculate_macd(df)
        X['vwap_signal'] = calculate_vwap_signal(df)
        X['obv'] = calculate_obv(df)
        X['adx'] = calculate_adx(df)
        bb_upper, bb_lower, bb_width = calculate_bollinger_bands(df)
        # Убедимся, что столбец 'price' существует
        if 'price' not in df.columns:
            logger.error("prepare_training_data: Столбец 'price' отсутствует в DataFrame")
            return pd.DataFrame(columns=ACTIVE_FEATURES), np.array([])
        X['bb_upper'] = bb_upper / df['price']
        X['bb_lower'] = bb_lower / df['price']
        support, resistance = calculate_support_resistance(df)
        X['support_level'] = support / df['price']
        X['resistance_level'] = resistance / df['price']
        # Проверка на константные признаки
        for column in X.columns:
            if X[column].nunique() <= 1:
                logger.warning(f"prepare_training_data: Признак {column} константный (уникальных значений: {X[column].nunique()})")
                X[column] = X[column] + np.random.normal(0, 1e-6, X[column].shape)  # Добавляем небольшой шум
        
        # Проверка наличия всех признаков из ACTIVE_FEATURES
        missing_features = [f for f in ACTIVE_FEATURES if f not in X.columns]
        if missing_features:
            logger.warning(f"prepare_training_data: Отсутствуют признаки: {missing_features}")
            for feature in missing_features:
                X[feature] = 0.0
        
        # Обработка пропусков
        X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        # Создание целевой переменной
        price = df['price']
        future_prices = price.shift(-2)
        labels = ((future_prices - price) / price * 100 >= 0.5).astype(int)
        
        # Удаляем последние две строки, чтобы синхронизировать с labels
        X = X.iloc[:-2][ACTIVE_FEATURES].copy()  # Создаём копию для безопасности
        labels = labels.iloc[:-2].copy()
        
        logger.info(f"prepare_training_data: Сформирован DataFrame с {len(X)} строками, признаки: {X.columns.tolist()}")
        return X, labels
    except Exception as e:
        logger.error(f"prepare_training_data: Ошибка: {e}")
        return pd.DataFrame(columns=ACTIVE_FEATURES), np.array([])

def calculate_bollinger_bands(df, window=20, window_dev=2):
    if df.empty or len(df) < window:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    sma = df['price'].rolling(window=window).mean()
    std = df['price'].rolling(window=window).std()
    upper = sma + window_dev * std
    lower = sma - window_dev * std
    width = (upper - lower) / sma
    return upper.fillna(0), lower.fillna(0), width.fillna(0)

# Дообучение модели
async def retrain_model_daily(context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    try:
        model, scaler, active_features = load_model()
        if not model or not scaler or not active_features:
            logger.error("retrain_model_daily: Модель или скейлер не загружены")
            await notify_admin("Не удалось загрузить модель для дообучения")
            return
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        trades = session.query(Trade).join(TradeMetrics, Trade.id == TradeMetrics.trade_id).filter(
            Trade.timestamp >= cutoff_time,
            TradeMetrics.success.isnot(None)
        ).all()
        if len(trades) < 5:
            logger.warning(f"retrain_model_daily: Недостаточно сделок ({len(trades)} < 5) для дообучения")
            await notify_admin(f"Недостаточно сделок ({len(trades)}) для дообучения за последние 7 дней")
            return
        X_new = []
        y_new = []
        trade_results = []
        pnls = []
        for trade in trades:
            df = await get_historical_data(trade.symbol)
            if df.empty:
                logger.warning(f"retrain_model_daily: Нет данных для {trade.symbol}")
                continue
            X, _ = prepare_training_data(df)
            if X.empty:
                logger.warning(f"retrain_model_daily: Пустой DataFrame для {trade.symbol}")
                continue
            trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade.id).first()
            if not trade_metrics:
                logger.warning(f"retrain_model_daily: Метрика для сделки #{trade.id} не найдена")
                continue
            label = 1 if trade_metrics.success in ['TP1', 'TP2'] else 0
            if trade_metrics.success == 'SL':
                final_price = trade.stop_loss
            elif trade_metrics.success == 'TP1':
                final_price = trade.take_profit_1
            elif trade_metrics.success == 'TP2':
                final_price = trade.take_profit_2
            else:
                final_price = trade.entry_price
            pnl = (final_price - trade.entry_price) * trade.position_size
            X_new.append(X.iloc[-1][active_features])
            y_new.append(label)
            trade_results.append(trade_metrics.success)
            pnls.append(pnl)
        if len(X_new) < 5:
            logger.warning(f"retrain_model_daily: Недостаточно данных для дообучения ({len(X_new)} < 5)")
            await notify_admin(f"Недостаточно данных для дообучения ({len(X_new)})")
            return
        X_new = pd.DataFrame(X_new, columns=active_features)
        X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=active_features)
        y_new = np.array(y_new)
        unique_labels = np.unique(y_new)
        if len(unique_labels) < 2:
            logger.warning(f"retrain_model_daily: Только один класс в данных: {unique_labels}. Добавляем синтетические данные.")
            num_synthetic = min(3, len(X_new))
            for _ in range(num_synthetic):
                idx = np.random.randint(0, len(X_new))
                synthetic_X = X_new.iloc[idx].copy()
                for col in synthetic_X.index:
                    feature_range = X_new[col].std() or 1.0
                    synthetic_X[col] += np.random.normal(0, feature_range * 0.01, 1)[0]
                X_new = pd.concat([X_new, pd.DataFrame([synthetic_X], columns=active_features)], ignore_index=True)
                y_new = np.append(y_new, 1)
                trade_results.append('TP1')
                pnls.append(pnls[idx] * -1 if pnls[idx] < 0 else pnls[idx])
            X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=active_features)
            logger.warning(f"retrain_model_daily: Добавлено {num_synthetic} синтетических сэмплов, новые метки: {np.unique(y_new)}")
        for column in X_new.columns:
            if X_new[column].nunique() <= 1:
                logger.warning(f"retrain_model_daily: Признак {column} константный, добавляем шум")
                X_new[column] = X_new[column] + np.random.normal(0, 1e-6, X_new[column].shape)
        try:
            y_pred_proba = model.predict_proba(X_new_scaled)
            loss = pnl_loss(y_new, y_pred_proba, trade_results, pnls)
            logger.warning(f"retrain_model_daily: Значение потерь до дообучения: {loss:.4f}, метки: {list(y_new)}")
            # Устанавливаем cv=2 для StackingClassifier, если это необходимо
            if hasattr(model, 'cv') and len(np.unique(y_new)) >= 2:
                model.set_params(cv=min(2, len(np.unique(y_new))))
            model.fit(X_new_scaled, y_new)
            save_model(model, scaler, active_features)
            logger.warning(f"retrain_model_daily: Модель дообучена на {len(X_new)} сэмплов, тестовая вероятность: {model.predict_proba(X_new_scaled.iloc[-1:])[0][1] * 100:.2f}%")
        except Exception as e:
            logger.error(f"retrain_model_daily: Ошибка при дообучении: {e}")
            await notify_admin(f"Ошибка при дообучении: {e}")
            return
    except Exception as e:
        logger.error(f"retrain_model_daily: Критическая ошибка: {e}")
        await notify_admin(f"Критическая ошибка в retrain_model_daily: {e}")
    finally:
        session.close()



# Прогноз вероятности
def predict_probability(model, scaler, active_features, df, coin, stop_loss, position_size):
    try:
        X, _ = prepare_training_data(df)
        if X.empty:
            logger.warning(f"predict_probability: Пустой DataFrame для {coin}")
            return 0.0
        X_last = X.iloc[-1][active_features]
        X_last_df = pd.DataFrame([X_last], columns=active_features)  # Сохраняем как DataFrame
        X_last_scaled = pd.DataFrame(scaler.transform(X_last_df), columns=active_features)
        probability = model.predict_proba(X_last_scaled)[0][1] * 100
        base_probability = probability
        rr_ratio = 0.0
        if stop_loss > 0 and position_size > 0:
            rr_ratio = (df['price'].iloc[-1] - stop_loss) / position_size
            rr_ratio = min(max(rr_ratio, -1.0), 3.0)
        probability = min(probability * (1 + rr_ratio * 0.1), 95.0)
        logger.warning(f"predict_probability: {coin}: base_probability={base_probability:.2f}%, скорректированная={probability:.2f}%")
        return probability
    except Exception as e:
        logger.error(f"predict_probability: Ошибка для {coin}: {e}")
        return 0.0


# Список криптовалют
def get_top_cryptos():
    try:
        result = []
        for symbol in CRYPTO_PAIRS:
            coin_id = symbol.replace('/USDT', '')
            try:
                ticker = asyncio.run(exchange.fetch_ticker(symbol))
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
                    Trade.timestamp >= datetime.utcnow() - timedelta(days=30)  # Проверяем сделки за последние 30 дней
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
    
# Smart Money анализ
def smart_money_analysis(df, taker_buy_base, volume, coin_id):
    if df.empty:
        logger.warning(f"smart_money_analysis: Пустой датафрейм для {coin_id}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    avg_volume = df['volume'].rolling(window=6).mean().iloc[-1] if 'volume' in df.columns else 100
    recent_volume = df['volume'].iloc[-1] if 'volume' in df.columns else volume
    volume_change = min(recent_volume / avg_volume * 100, 100)
    taker_buy = df['taker_buy_base'].iloc[-1] if 'taker_buy_base' in df.columns else taker_buy_base
    institutional_score = (taker_buy / recent_volume * 100) if recent_volume > 0 else 0
    vwap_signal = calculate_vwap_signal(df).iloc[-1]
    sentiment = get_news_sentiment(coin_id)
    rsi = calculate_rsi(df).iloc[-1]
    macd = calculate_macd(df).iloc[-1]
    adx = calculate_adx(df).iloc[-1]
    obv = calculate_obv(df).iloc[-1]
    bb_width = calculate_bb_width(df).iloc[-1]
    score = (
        0.2 * min(df['price'].pct_change().iloc[-1] * 100, 100) +
        0.2 * volume_change +
        0.2 * institutional_score +
        0.15 * rsi +
        0.1 * macd * 100 +
        0.1 * vwap_signal +
        0.1 * adx +
        0.05 * bb_width * 100
    ) / 2
    logger.info(
        f"smart_money_analysis: {coin_id}: price_change={df['price'].pct_change().iloc[-1] * 100:.2f}%, "
        f"volume_change={volume_change:.2f}%, institutional={institutional_score:.2f}%, "
        f"sentiment={sentiment:.2f}%, rsi={rsi:.2f}, macd={macd:.2f}, adx={adx:.2f}, "
        f"obv={obv:.2f}, bb_width={bb_width:.2f}, score={score:.2f}"
    )
    return volume_change, institutional_score, vwap_signal, sentiment, rsi, macd, adx, obv, bb_width, score

# Размер позиции
def calculate_position_size(entry_price, stop_loss, balance):
    if balance <= 0:
        logger.error(f"calculate_position_size: Некорректный баланс: {balance}")
        return 0, 0
    risk_per_trade = 0.01  # 1% риска на сделку
    risk_amount = balance * risk_per_trade
    logger.info(f"calculate_position_size: Баланс={balance}, Риск={risk_amount}")
    price_diff = max(abs(entry_price - stop_loss), entry_price * 0.001)  # Минимум 0.1%
    position_size = risk_amount / price_diff if price_diff > 0 else 0.000018
    position_size_percent = (position_size * entry_price / balance) * 100
    return position_size, position_size_percent

# График цен
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
        asyncio.run(notify_admin(f"Ошибка создания графика для {symbol}: {e}"))
        return None

# Анализ торговой возможности
async def analyze_trade_opportunity(model, scaler, active_features, df, price_change_1h, current_price, symbol, taker_buy_base, volume, coin_id):
    try:
        if df.empty:
            logger.info(f"analyze_trade_opportunity: Пустой DataFrame для {symbol}")
            return False, 0, 0, 50.0, 0, 0, 50.0, 0, 0, 0, 0, 0
        # Проверка наличия необходимых столбцов
        required_columns = ['price', 'volume', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"analyze_trade_opportunity: Отсутствуют столбцы для {symbol}: {missing_columns}")
            return False, 0, 0, 50.0, 0, 0, 50.0, 0, 0, 0, 0, 0
        price_change = ((df['price'].iloc[-1] / df['price'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
        volume_change = ((df['volume'].iloc[-1] / df['volume'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
        institutional_score = (taker_buy_base / volume * 100) if volume > 0 else 50.0
        vwap_signal = (df['price'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] * 100 if 'vwap' in df and len(df) >= 2 else 0
        sentiment = get_news_sentiment(coin_id)
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50.0
        macd = df['macd'].iloc[-1] if 'macd' in df else 0
        adx = df['adx'].iloc[-1] if 'adx' in df else 0
        obv = df['obv'].iloc[-1] if 'obv' in df else 0
        smart_money_score = (
            (price_change / PRICE_THRESHOLD if PRICE_THRESHOLD != 0 else 0) * 0.2 +
            (volume_change / VOLUME_THRESHOLD if VOLUME_THRESHOLD != 0 else 0) * 0.3 +
            (institutional_score / 100) * 0.2 +
            (sentiment / 100) * 0.2 +
            (rsi / 100) * 0.1
        ) * 100
        probability = predict_probability(model, scaler, active_features, df, coin_id, sentiment, institutional_score)
        is_opportunity = (
            volume_change > VOLUME_THRESHOLD * 1.5 and
            rsi > RSI_THRESHOLD - 10 and
            probability > 5
        )
        logger.info(
            f"analyze_trade_opportunity: {symbol}: price_change={price_change:.2f}%, "
            f"volume_change={volume_change:.2f}%, rsi={rsi:.2f}, probability={probability:.2f}%, "
            f"is_opportunity={is_opportunity}"
        )
        return (
            is_opportunity, price_change, volume_change, institutional_score, vwap_signal,
            sentiment, rsi, macd, adx, obv, smart_money_score, probability
        )
    except Exception as e:
        logger.error(f"analyze_trade_opportunity: Ошибка для {symbol}: {e}")
        return False, 0, 0, 50.0, 0, 0, 50.0, 0, 0, 0, 0, 0

# Проверка результата сделки
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

# Проверка активных сделок
from threading import Lock

trade_lock = Lock()

async def check_active_trades(context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    try:
        user_id = context.job.data
        logger.info(f"check_active_trades: Проверка активных сделок для user_id={user_id}")
        with trade_lock:  # Блокировка для предотвращения конкурентных обновлений
            active_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                (Trade.result.is_(None) | (Trade.result == 'TP1'))
            ).all()
            if not active_trades:
                logger.info(f"check_active_trades: Нет активных сделок для user_id={user_id}")
                return
            for trade in active_trades:
                # Проверка существования сделки
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
                        # Отправка уведомления
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

# Автоматический поиск сделок
async def auto_search_trades(context: ContextTypes.DEFAULT_TYPE):
    settings = load_settings()
    for user_id_str in settings:
        user_id = int(user_id_str)
        if not is_authorized(user_id):
            continue
        user_settings = get_user_settings(user_id)
        model, scaler, active_features = load_model()
        if not model or not scaler or not active_features:
            logger.error("auto_search_trades: Модель не загружена для user_id={user_id}")
            await context.bot.send_message(user_id, "🚨 **Ошибка**: Модель не загружена.", parse_mode='Markdown')
            continue
        cryptos = get_top_cryptos()
        session = Session()
        try:
            opportunities = []
            for symbol, coin_id, price_change_1h, taker_buy_base, volume in cryptos:
                df = await get_historical_data(symbol)
                if not df.empty:
                    df['symbol'] = symbol
                current_price = df['price'].iloc[-1] if not df.empty else await get_current_price(symbol)
                if current_price <= 0.0:
                    logger.warning(f"auto_search_trades: Не удалось получить цену для {symbol}, пропускаем")
                    continue
                is_opportunity, price_change, volume_change, institutional_score, vwap_signal, sentiment, rsi, macd, adx, obv, smart_money_score, probability = await analyze_trade_opportunity(
                    model, scaler, active_features, df, price_change_1h, current_price, symbol, taker_buy_base, volume, coin_id
                )
                if is_opportunity:
                    # Проверка на дублирующиеся активные сделки
                    existing_trades = session.query(Trade).filter(
                        Trade.user_id == user_id,
                        Trade.symbol == symbol,
                        Trade.result.is_(None) | (Trade.result == 'TP1')
                    ).all()
                    is_duplicate = False
                    atr = calculate_atr_normalized(df).iloc[-1] * current_price
                    stop_loss = current_price - max(2 * atr, current_price * 0.005 if current_price >= 1 else current_price * 0.01)
                    tp1 = current_price + 6 * atr
                    for trade in existing_trades:
                        entry_diff = abs(trade.entry_price - current_price) / current_price
                        sl_diff = abs(trade.stop_loss - stop_loss) / current_price
                        tp1_diff = abs(trade.take_profit_1 - tp1) / current_price
                        if entry_diff < 0.005 and sl_diff < 0.01 and tp1_diff < 0.01:
                            is_duplicate = True
                            logger.warning(f"auto_search_trades: Пропущена сделка для {symbol}, уже есть активная сделка #{trade.id}")
                            break
                    if is_duplicate:
                        continue
                    opportunities.append({
                        'symbol': symbol,
                        'coin_id': coin_id,
                        'price_change': price_change,
                        'volume_change': volume_change,
                        'institutional_score': institutional_score,
                        'vwap_signal': vwap_signal,
                        'sentiment': sentiment,
                        'rsi': rsi,
                        'macd': macd,
                        'adx': adx,
                        'obv': obv,
                        'smart_money_score': smart_money_score,
                        'probability': probability,
                        'df': df,
                        'current_price': current_price
                    })
            if not opportunities:
                logger.warning(f"auto_search_trades: Сделки не найдены для user_id={user_id}")
                continue
            best_opportunity = max(opportunities, key=lambda x: x['smart_money_score'])
            if best_opportunity['probability'] == 0:
                logger.warning(f"auto_search_trades: Пропущена сделка для {best_opportunity['symbol']} из-за нулевой вероятности")
                continue
            symbol = best_opportunity['symbol']
            coin_id = best_opportunity['coin_id']
            price_change = best_opportunity['price_change']
            volume_change = best_opportunity['volume_change']
            institutional_score = best_opportunity['institutional_score']
            vwap_signal = best_opportunity['vwap_signal']
            sentiment = best_opportunity['sentiment']
            rsi = best_opportunity['rsi']
            macd = best_opportunity['macd']
            adx = best_opportunity['adx']
            obv = best_opportunity['obv']
            smart_money_score = best_opportunity['smart_money_score']
            probability = best_opportunity['probability']
            df = best_opportunity['df']
            current_price = best_opportunity['current_price']
            atr = calculate_atr_normalized(df).iloc[-1] * current_price
            entry_price = current_price
            min_stop_loss = entry_price * 0.01 if entry_price < 1 else entry_price * 0.005
            stop_loss = entry_price - max(2 * atr, min_stop_loss)
            tp1 = entry_price + 6 * atr
            tp2 = entry_price + 9 * atr
            rr_ratio = (tp1 - entry_price) / (entry_price - stop_loss) if (entry_price - stop_loss) > 0 else 3.0
            user_settings = get_user_settings(user_id)
            balance = user_settings.get('balance', None)
            if balance is None:
                logger.warning(f"auto_search_trades: Баланс не установлен для user_id={user_id}")
                await context.bot.send_message(
                    chat_id=user_id,
                    text="🚫 **Баланс не установлен.**\nИспользуйте `/setbalance <сумма>` для установки баланса.",
                    parse_mode='Markdown'
                )
                return
            position_size, position_size_percent = calculate_position_size(entry_price, stop_loss, balance)
            potential_profit_tp1 = (tp1 - entry_price) * position_size
            potential_profit_tp2 = (tp2 - entry_price) * position_size
            trader_level = "Новичок"
            tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
            price_precision = 6 if entry_price < 1 else 2
            message = (
                f"📈 **Новая сделка: {symbol} LONG** (авто)\n"
                f"💰 **Баланс**: ${balance:.2f}\n"
                f"🎯 Вход: ${entry_price:.{price_precision}f}\n"
                f"⛔ Стоп-лосс: ${stop_loss:.{price_precision}f}\n"
                f"💰 TP1: ${tp1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                f"💰 TP2: ${tp2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                f"📊 RR: {rr_ratio:.1f}:1\n"
                f"📏 Размер: {position_size_percent:.2f}% ({position_size:.6f} {coin_id})\n"
                f"🎲 Вероятность: {probability:.1f}%\n"
                f"🏛️ Институц.: {institutional_score:.1f}%\n"
                f"📈 VWAP: {'🟢 Бычий' if vwap_signal > 0 else '🔴 Медвежий'}\n"
                f"📮 Сентимент: {sentiment:.1f}%\n"
                f"📊 RSI: {rsi:.1f} | MACD: {'🟢' if macd > 0 else '🔴'} | ADX: {adx:.1f}\n"
                f"💡 Логика: Рост {price_change:.2f}%, Объём +{volume_change:.1f}%\n"
                f"📈 График: {tradingview_url}\n"
                f"💾 Сделка сохранена. Отметьте результат:"
            )
            trade = Trade(
                user_id=user_id,
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                rr_ratio=rr_ratio,
                position_size=position_size,
                probability=probability,
                institutional_score=institutional_score,
                sentiment_score=sentiment,
                trader_level=trader_level
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
            trade_metrics = TradeMetrics(
                trade_id=trade_id,
                symbol=symbol,
                entry_price=entry_price,
                price_after_1h=None,
                price_after_2h=None,
                volume_change=volume_change,
                institutional_score=institutional_score,
                vwap_signal=vwap_signal,
                sentiment=sentiment,
                rsi=rsi,
                macd=macd,
                adx=adx,
                obv=obv,
                smart_money_score=smart_money_score,
                probability=probability,
                success=None
            )
            session.add(trade_metrics)
            session.commit()
            asyncio.create_task(check_trade_result(symbol, entry_price, stop_loss, tp1, tp2, trade_id))
            keyboard = [
                [
                    InlineKeyboardButton("✅ TP1", callback_data=f"TP1_{trade_id}"),
                    InlineKeyboardButton("✅ TP2", callback_data=f"TP2_{trade_id}"),
                    InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            chart_path = create_price_chart(df, symbol, price_change)
            try:
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as photo:
                        await context.bot.send_photo(chat_id=user_id, photo=photo, caption=message, reply_markup=reply_markup, parse_mode='Markdown')
                    os.remove(chart_path)
                else:
                    await context.bot.send_message(chat_id=user_id, text=message + "\n⚠️ Не удалось создать график.", reply_markup=reply_markup, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"auto_search_trades: Ошибка отправки для {symbol}: {str(e)}")
                await context.bot.send_message(chat_id=user_id, text=message + f"\n⚠️ Ошибка: {str(e)}", reply_markup=reply_markup, parse_mode='Markdown')
                await notify_admin(f"Ошибка отправки авто-сделки для {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"auto_search_trades: Ошибка для user_id={user_id}: {str(e)}")
            await context.bot.send_message(chat_id=user_id, text=f"🚨 **Ошибка**: {str(e)}", parse_mode='Markdown')
            await notify_admin(f"Ошибка в auto_search_trades: {str(e)}")
        finally:
            session.close()

# Обработчик кнопок
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except telegram.error.BadRequest as e:
        logger.warning(f"button: Устаревший или недействительный запрос: {e}")
        return
    user_id = query.from_user.id
    if not is_authorized(user_id):
        await query.message.reply_text("🚫 Вы не авторизованы для использования бота.")
        return
    data = query.data
    if data in ('filter_active', 'filter_completed', 'refresh_active'):
        return  # Пропускаем, так как эти запросы обрабатываются в history_filter
    if data.startswith(("TP1_", "TP2_", "SL_")):
        try:
            result, trade_id = data.split("_")
            trade_id = int(trade_id)
        except ValueError as e:
            logger.error(f"button: Ошибка разбора query.data={data}: {e}")
            await query.message.reply_text("🚫 Неверный формат запроса.")
            await notify_admin(f"Ошибка в button для user_id={user_id}, query.data={data}: {e}")
            return
        session = Session()
        try:
            trade = session.query(Trade).filter_by(id=trade_id, user_id=user_id).first()
            if not trade:
                await query.message.reply_text("🚫 Сделка не найдена или не принадлежит вам.")
                return
            if trade.result:
                await query.message.reply_text(f"🚫 Сделка #{trade_id} уже отмечена как {trade.result}.")
                return
            trade.result = result
            trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade_id).first()
            if not trade_metrics:
                await query.message.reply_text(f"🚫 Метрика для сделки #{trade_id} не найдена.")
                return
            trade_metrics.success = result
            user_settings = get_user_settings(user_id)
            balance = user_settings.get('balance', 0)
            final_price = trade.stop_loss if result == 'SL' else trade.take_profit_1 if result == 'TP1' else trade.take_profit_2
            pnl = (final_price - trade.entry_price) * trade.position_size
            balance += pnl
            user_settings['balance'] = balance
            save_user_settings(user_id, user_settings)
            session.commit()
            await query.message.reply_text(
                f"✅ Сделка #{trade_id} отмечена как {result}. PNL: {pnl:.2f} USDT. Новый баланс: {balance:.2f} USDT."
            )
            closed_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.result.isnot(None)
            ).count()
            if closed_trades >= 5:
                asyncio.create_task(retrain_model_daily(context))
            else:
                logger.warning(f"button: Дообучение не запущено для user_id={user_id}, закрытых сделок: {closed_trades}")
        except Exception as e:
            logger.error(f"button: Ошибка при обработке сделки #{trade_id}: {e}")
            await query.message.reply_text(f"🚫 Ошибка при обработке: {str(e)}")
            await notify_admin(f"Ошибка в button для user_id={user_id}, trade_id={trade_id}: {e}")
        finally:
            session.close()
    else:
        await query.message.reply_text("🚫 Неверный выбор.")
        await notify_admin(f"Неверный выбор в button для user_id={user_id}, query.data={data}")

# Команда /setcriteria
async def set_criteria(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            message = (
                f"⚙️ **Текущие настройки**:\n"
                f"📈 Цена: {user_settings['price_threshold']}% | Объём: {user_settings['volume_threshold']}% | RSI: {user_settings['rsi_threshold']} ({'вкл' if user_settings['use_rsi'] else 'выкл'})\n"
                f"⏱ Авто-поиск: каждые {user_settings['auto_interval']//60} мин\n"
                f"Используйте: `/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes]`\n"
                f"Пример: `/setcriteria 5 0.3 40 rsi_off 10`"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
            return
        volume_threshold = float(args[0])
        price_threshold = float(args[1])
        rsi_threshold = float(args[2]) if len(args) >= 3 else user_settings['rsi_threshold']
        use_rsi = 'rsi_off' not in args
        auto_interval = int(float(args[args.index(args[-1])]) * 60) if len(args) >= 4 and args[-1].replace('.', '').isdigit() else user_settings['auto_interval']
        user_settings.update({
            'price_threshold': price_threshold,
            'volume_threshold': volume_threshold,
            'rsi_threshold': rsi_threshold,
            'use_rsi': use_rsi,
            'auto_interval': auto_interval
        })
        settings[user_id_str] = user_settings
        save_settings(settings)
        message = (
            f"✅ **Настройки обновлены**:\n"
            f"📈 Цена: {price_threshold}% | Объём: {volume_threshold}% | RSI: {rsi_threshold} ({'вкл' if use_rsi else 'выкл'})\n"
            f"⏱ Авто-поиск: каждые {auto_interval//60} мин"
        )
        await update.message.reply_text(message, parse_mode='Markdown')
        logger.info(f"set_criteria: Пользователь {user_id} обновил настройки: {user_settings}")
        # Перезапуск задачи авто-поиска
        job = context.job_queue.get_jobs_by_name(f"auto_search_{user_id}")
        if job:
            job[0].schedule_removal()
        context.job_queue.run_repeating(
            auto_search_trades,
            interval=user_settings['auto_interval'],
            name=f"auto_search_{user_id}",
            data=user_id
        )
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

# Команда /add_user
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

# Команда /idea
async def idea(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"idea: Команда от пользователя {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    session = None
    try:
        # Загрузка настроек пользователя
        user_settings = get_user_settings(user_id)
        auto_interval = user_settings['auto_interval']
        
        # Сброс текущей задачи автопоиска
        job_name = f"auto_search_{user_id}"
        current_jobs = context.job_queue.get_jobs_by_name(job_name)
        for job in current_jobs:
            job.schedule_removal()
            logger.info(f"idea: Задача автопоиска {job_name} сброшена для user_id={user_id}")

        result = load_model()
        if len(result) == 2:
            model, scaler = result
            active_features = ACTIVE_FEATURES  # Используем глобальный список, если модель старая
        else:
            model, scaler, active_features = result
        top_cryptos = get_top_cryptos()
        session = Session()
        opportunities = []
        for symbol, coin_id, price_change_1h, taker_buy_base, volume in top_cryptos:
            if not symbol.replace('/', '').isalnum():
                logger.warning(f"idea: Некорректный символ {symbol}, пропускаем")
                continue
            df = await get_historical_data(symbol)
            if df.empty:
                logger.warning(f"idea: Пустой DataFrame для {symbol}, пропускаем")
                continue
            current_price = await get_current_price(symbol)
            if current_price <= 0.0:
                logger.warning(f"idea: Не удалось получить цену для {symbol}, пропускаем")
                continue
            is_opportunity, price_change, volume_change, institutional_score, vwap_signal, sentiment, rsi, macd, adx, obv, smart_money_score, probability = await analyze_trade_opportunity(
                model, scaler, active_features, df, price_change_1h, current_price, symbol, taker_buy_base, volume, coin_id
            )
            if is_opportunity:
                opportunities.append({
                    'symbol': symbol,
                    'coin_id': coin_id,
                    'price_change': price_change,
                    'volume_change': volume_change,
                    'institutional_score': institutional_score,
                    'vwap_signal': vwap_signal,
                    'sentiment': sentiment,
                    'rsi': rsi,
                    'macd': macd,
                    'adx': adx,
                    'obv': obv,
                    'smart_money_score': smart_money_score,
                    'probability': probability,
                    'current_price': current_price,
                    'df': df
                })
        opportunities.sort(key=lambda x: x['probability'], reverse=True)
        for opp in opportunities:
            symbol = opp['symbol']
            current_price = opp['current_price']
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
            smart_money_score = opp['smart_money_score']
            probability = opp['probability']
            existing_trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.symbol == symbol,
                Trade.result.is_(None)
            ).all()
            is_duplicate = False
            for trade in existing_trades:
                entry_diff = abs(trade.entry_price - current_price) / current_price
                sl_diff = abs(trade.stop_loss - current_price * (1 - STOP_LOSS_PCT)) / current_price
                tp1_diff = abs(trade.take_profit_1 - current_price * (1 + TAKE_PROFIT_1_PCT)) / current_price
                if entry_diff < 0.005 and sl_diff < 0.01 and tp1_diff < 0.01:
                    is_duplicate = True
                    logger.info(f"idea: Пропущена сделка для {symbol}, уже есть активная сделка #{trade.id}")
                    break
            if is_duplicate:
                await update.message.reply_text(
                    f"🔔 **Лучшая возможность ({symbol}) уже активна.** Пробуем другую...",
                    parse_mode='Markdown'
                )
                await asyncio.sleep(0.5)  # Задержка для избежания rate limits
                continue
            if probability == 0:
                logger.warning(f"idea: Пропущена сделка для {symbol} из-за нулевой вероятности")
                continue
            stop_loss = current_price * (1 - STOP_LOSS_PCT)
            take_profit_1 = current_price * (1 + TAKE_PROFIT_1_PCT)
            take_profit_2 = current_price * (1 + TAKE_PROFIT_2_PCT)
            if stop_loss <= 0 or take_profit_1 <= current_price or take_profit_2 <= take_profit_1:
                logger.warning(f"idea: Некорректные параметры для {symbol}: SL={stop_loss}, TP1={take_profit_1}, TP2={take_profit_2}")
                continue
            rr_ratio = (take_profit_1 - current_price) / (current_price - stop_loss)
            user_settings = get_user_settings(user_id)
            balance = user_settings.get('balance', None)
            if balance is None:
                await update.message.reply_text(
                    "🚫 **Баланс не установлен.**\n"
                    "Используйте `/setbalance <сумма>` для установки баланса.",
                    parse_mode='Markdown'
                )
                return
            position_size, position_size_percent = calculate_position_size(current_price, stop_loss, balance)
            potential_profit_tp1 = (take_profit_1 - current_price) * position_size
            potential_profit_tp2 = (take_profit_2 - current_price) * position_size
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
                price_after_1h=None,
                price_after_2h=None,
                volume_change=volume_change,
                institutional_score=institutional_score,
                vwap_signal=vwap_signal,
                sentiment=sentiment,
                rsi=rsi,
                macd=macd,
                adx=adx,
                obv=obv,
                smart_money_score=smart_money_score,
                probability=probability,
                success=None
            )
            session.add(trade_metrics)
            session.commit()
            price_precision = 6 if current_price < 1 else 2
            vwap_text = '🟢 Бычий' if vwap_signal > 0 else '🔴 Медвежий'
            macd_text = '🟢 Бычий' if macd > 0 else '🔴 Медвежий'
            chart_path = create_price_chart(df, symbol, price_change)
            tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.replace('/', '')}&interval=15"
            message = (
                f"🔔 **Новая сделка: {symbol} LONG**\n"
                f"💰 **Баланс**: ${balance:.2f}\n"
                f"🎯 Вход: ${current_price:.{price_precision}f}\n"
                f"⛔ Стоп-лосс: ${stop_loss:.{price_precision}f}\n"
                f"💰 TP1: ${take_profit_1:.{price_precision}f} (+${potential_profit_tp1:.2f})\n"
                f"💰 TP2: ${take_profit_2:.{price_precision}f} (+${potential_profit_tp2:.2f})\n"
                f"📊 RR: {rr_ratio:.1f}:1\n"
                f"📏 Размер: {position_size_percent:.2f}% ({position_size:.6f} {symbol.split('/')[0]})\n"
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
                 InlineKeyboardButton("❌ SL", callback_data=f"SL_{trade.id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
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
            logger.info(f"idea: Сделка #{trade.id} создана для {symbol}")
            
            # Перезапуск задачи автопоиска с задержкой
            context.job_queue.run_repeating(
                auto_search_trades,
                interval=auto_interval,
                first=auto_interval,  # Задержка перед первым запуском
                name=f"auto_search_{user_id}",
                data=user_id
            )
            logger.info(f"idea: Задача автопоиска перезапущена для user_id={user_id} с интервалом {auto_interval} сек")
            return
        await update.message.reply_text("🔍 **Нет подходящих возможностей для торговли.**", parse_mode='Markdown')
    except Exception as e:
        logger.error(f"idea: Ошибка: {str(e)}")
        await update.message.reply_text(f"🚨 **Ошибка**: {str(e)}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в idea: {str(e)}")
    finally:
        if session is not None:
            session.close()
# Команда /active
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

# Команда /history
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

# Обработчик фильтров истории
async def history_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except telegram.error.BadRequest as e:
        logger.warning(f"history_filter: Устаревший или недействительный запрос: {e}")
        return
    user_id = query.from_user.id
    if not is_authorized(user_id):
        await query.message.reply_text("🚫 Вы не авторизованы для использования бота.", parse_mode='Markdown')
        return
    session = Session()
    try:
        if query.data == 'filter_active':
            trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                (Trade.result.is_(None) | (Trade.result == 'TP1'))
            ).order_by(Trade.timestamp.desc()).limit(10).all()
            if not trades:
                text = "📊 **Активные сделки**: Нет активных сделок."
                reply_markup = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🟡 Активные", callback_data='filter_active'),
                     InlineKeyboardButton("✅ Завершённые", callback_data='filter_completed'),
                     InlineKeyboardButton("🔄 Обновить", callback_data='refresh_active')]
                ])
                await query.message.edit_text(text, reply_markup=reply_markup, parse_mode='Markdown')
                return
            text = "📊 **Активные сделки**:\n"
            for trade in trades:
                price_precision = 6 if trade.entry_price < 1 else 2
                status = '🟡 Ожидает' if trade.result is None else '✅ TP1 достигнут'
                current_price = await get_current_price(trade.symbol)
                text += (
                    f"#{trade.id}: *{trade.symbol} LONG*\n"
                    f"🎯 Вход: ${trade.entry_price:.{price_precision}f} | Текущая: ${current_price:.{price_precision}f}\n"
                    f"⛔ SL: ${trade.stop_loss:.{price_precision}f} | 💰 TP1: ${trade.take_profit_1:.{price_precision}f} | 💰 TP2: ${trade.take_profit_2:.{price_precision}f}\n"
                    f"📊 Статус: {status}\n"
                    f"⏰ Время: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("🟡 Активные", callback_data='filter_active'),
                 InlineKeyboardButton("✅ Завершённые", callback_data='filter_completed'),
                 InlineKeyboardButton("🔄 Обновить", callback_data='refresh_active')]
            ])
            if len(text) > 4000:  # Ограничение Telegram на длину сообщения
                text = text[:3900] + "...\n(Список усечён из-за ограничений Telegram)"
            await query.message.edit_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        elif query.data == 'filter_completed':
            trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.result.isnot(None)
            ).order_by(Trade.timestamp.desc()).limit(10).all()
            if not trades:
                text = "📜 **Завершённые сделки**: Нет завершённых сделок."
                reply_markup = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🟡 Активные", callback_data='filter_active'),
                     InlineKeyboardButton("✅ Завершённые", callback_data='filter_completed'),
                     InlineKeyboardButton("🔄 Обновить", callback_data='refresh_active')]
                ])
                await query.message.edit_text(text, reply_markup=reply_markup, parse_mode='Markdown')
                return
            text = "📜 **Последние 10 завершённых сделок**:\n"
            for trade in trades:
                trade_metrics = session.query(TradeMetrics).filter_by(trade_id=trade.id).first()
                result = trade_metrics.success if trade_metrics else trade.result
                final_price = trade.stop_loss if result == 'SL' else trade.take_profit_1 if result == 'TP1' else trade.take_profit_2
                if trade.position_size > 0:  # Покупка (long)
                    pnl = (final_price - trade.entry_price) * trade.position_size
                else:  # Продажа (short)
                    pnl = (trade.entry_price - final_price) * abs(trade.position_size)
                price_precision = 6 if trade.entry_price < 1 else 2
                text += (
                    f"#{trade.id}: *{trade.symbol} {'LONG' if trade.position_size > 0 else 'SHORT'}*\n"
                    f"🎯 Вход: ${trade.entry_price:.{price_precision}f} | Выход: ${final_price:.{price_precision}f}\n"
                    f"📊 Результат: {'✅ TP1' if result == 'TP1' else '✅ TP2' if result == 'TP2' else '❌ SL'}\n"
                    f"💸 PNL: {pnl:.2f} USDT\n"
                    f"⏰ Время: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("🟡 Активные", callback_data='filter_active'),
                 InlineKeyboardButton("✅ Завершённые", callback_data='filter_completed'),
                 InlineKeyboardButton("🔄 Обновить", callback_data='refresh_active')]
            ])
            if len(text) > 4000:
                text = text[:3900] + "...\n(Список усечён из-за ограничений Telegram)"
            await query.message.edit_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        elif query.data == 'refresh_active':
            trades = session.query(Trade).filter(
                Trade.user_id == user_id,
                (Trade.result.is_(None) | (Trade.result == 'TP1'))
            ).order_by(Trade.timestamp.desc()).limit(10).all()
            if not trades:
                text = "📊 **Активные сделки**: Нет активных сделок."
                reply_markup = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🟡 Активные", callback_data='filter_active'),
                     InlineKeyboardButton("✅ Завершённые", callback_data='filter_completed'),
                     InlineKeyboardButton("🔄 Обновить", callback_data='refresh_active')]
                ])
                await query.message.edit_text(text, reply_markup=reply_markup, parse_mode='Markdown')
                return
            text = "📊 **Активные сделки**:\n"
            for trade in trades:
                price_precision = 6 if trade.entry_price < 1 else 2
                status = '🟡 Ожидает' if trade.result is None else '✅ TP1 достигнут'
                current_price = await get_current_price(trade.symbol)
                text += (
                    f"#{trade.id}: *{trade.symbol} LONG*\n"
                    f"🎯 Вход: ${trade.entry_price:.{price_precision}f} | Текущая: ${current_price:.{price_precision}f}\n"
                    f"⛔ SL: ${trade.stop_loss:.{price_precision}f} | 💰 TP1: ${trade.take_profit_1:.{price_precision}f} | 💰 TP2: ${trade.take_profit_2:.{price_precision}f}\n"
                    f"📊 Статус: {status}\n"
                    f"⏰ Время: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("🟡 Активные", callback_data='filter_active'),
                 InlineKeyboardButton("✅ Завершённые", callback_data='filter_completed'),
                 InlineKeyboardButton("🔄 Обновить", callback_data='refresh_active')]
            ])
            if len(text) > 4000:
                text = text[:3900] + "...\n(Список усечён из-за ограничений Telegram)"
            await query.message.edit_text(text, reply_markup=reply_markup, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"history_filter: Ошибка для user_id={user_id}: {e}")
        await query.message.reply_text(f"🚫 **Ошибка**: {str(e)}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в history_filter для user_id={user_id}: {e}")
    finally:
        session.close()
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 Вы не авторизованы для использования бота.")
        return
    help_text = (
        "📚 *Руководство по использованию торгового бота*\n\n"
        "Бот помогает находить торговые возможности и управлять сделками. Доступные команды:\n"
        "- `/start` - Запускает бота и отображает приветственное сообщение.\n"
        "- `/idea` - Ищет торговые сигналы для всех доступных монет и предлагает сделки с вероятностью успеха.\n"
        "- `/setcriteria` - Устанавливает критерии для автопоиска (интервал, минимальная вероятность).\n"
        "- `/active` - Показывает активные сделки с кнопками для отметки результатов (TP1, TP2, SL).\n"
        "- `/history` - Отображает историю сделок (активные или последние 10 завершённых).\n"
        "- `/stats` - Показывает статистику: общее количество сделок, процент успешных, общий PNL.\n"
        "- `/metrics` - Показывает метрики модели (значение потерь, вероятность на тестовых данных).\n"
        "- `/add_user` - (Только для админа) Добавляет нового пользователя.\n"
        "- `/stop` - Останавливает автопоиск сделок.\n"
        "- `/clear_trades` - Очищает все сделки пользователя.\n"
        "- `/setbalance <сумма>` - Устанавливает начальный баланс.\n"
        "- `/help` - Показывает это руководство.\n\n"
        "🔔 *Автопоиск*: Бот автоматически ищет сделки с заданным интервалом (настраивается через `/setcriteria`).\n"
        "📊 *Дообучение*: Модель дообучается ежедневно или после 5 закрытых сделок.\n"
        "⚠️ *Ошибки*: Все ошибки отправляются администратору для анализа.\n"
        "📈 *PNL*: Рассчитывается как `(выходная_цена - входная_цена) * размер_позиции`.\n"
        "Свяжитесь с админом для поддержки!"
    )
    try:
        await update.message.reply_text(help_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"help_command: Ошибка для user_id={user_id}: {e}")
        await update.message.reply_text("🚫 Ошибка при отображении помощи.")
        await notify_admin(f"Ошибка в help_command для user_id={user_id}: {e}")

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    user_settings = get_user_settings(user_id)
    balance = user_settings.get('balance', None)
    await update.message.reply_text(
        f"👋 **Добро пожаловать в крипто-бота!**\n"
        f"💰 **Баланс**: {balance if balance is not None else 'Не установлен'}\n"
        f"📈 **Пары**: {', '.join(CRYPTO_PAIRS)}\n"
        f"⚙️ **Текущие критерии**:\n"
        f"  📊 Рост > {user_settings['price_threshold']}% | Объём > {user_settings['volume_threshold']}% | RSI > {user_settings['rsi_threshold']} ({'вкл' if user_settings['use_rsi'] else 'выкл'})\n"
        f"  ⏱ Авто-поиск: каждые {user_settings['auto_interval']//60} мин\n"
        f"📚 **Команды**:\n"
        f"  `/setbalance <сумма>` - Установить баланс\n"
        f"  `/idea [volume price rsi | rsi_off]` - Торговая идея\n"
        f"  `/setcriteria <volume> <price> <rsi> [rsi_off] [interval_minutes]` - Настроить критерии\n"
        f"  `/active` - Активные сделки\n"
        f"  `/history` - История сделок\n"
        f"  `/stats` - Статистика\n"
        f"  `/metrics` - Метрики\n"
        f"  `/add_user <user_id>` - Добавить пользователя (админ)\n"
        f"  `/stop` - Остановить бота (админ)\n"
        f"  `/clear_trades` - Очистить базу сделок",
        parse_mode='Markdown'
    )


# Команда /stop
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("🚫 **У вас нет прав.**", parse_mode='Markdown')
        return
    logger.info("stop: Остановка бота")
    await update.message.reply_text("🛑 **Бот останавливается...**", parse_mode='Markdown')
    try:
        await context.application.stop()
        await context.application.shutdown()
        os._exit(0)
    except Exception as e:
        logger.error(f"stop: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка**: {e}", parse_mode='Markdown')
        await notify_admin(f"Ошибка при остановке: {e}")

# Команда /stats
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

# Команда /metrics
async def metrics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    session = Session()
    try:
        metrics = session.query(TradeMetrics).join(Trade, Trade.id == TradeMetrics.trade_id).filter(Trade.user_id == user_id).all()
        total_trades = len(metrics)
        wins = len([m for m in metrics if m.success in ['TP1', 'TP2']])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_rr = sum([t.rr_ratio for t in session.query(Trade).filter_by(user_id=user_id).all()]) / total_trades if total_trades > 0 else 0
        message = (
            f"📊 **Метрики**\n"
            f"🔢 Всего сделок: {total_trades}\n"
            f"🏆 Процент выигрышей: {win_rate:.2f}%\n"
            f"📈 Средний RR: {avg_rr:.1f}:1\n"
            f"📋 Последние (до 5):\n"
        )
        for i, m in enumerate(metrics[:5], 1):
            trade = session.query(Trade).filter_by(id=m.trade_id).first()
            price_precision = 6 if m.entry_price < 1 else 2
            price_change = ((m.price_after_2h - m.entry_price) / m.entry_price * 100) if m.price_after_2h else 0
            status = '🟡 Активна' if m.success is None or m.success == 'TP1' else ('✅ TP2' if m.success == 'TP2' else '❌ SL')
            message += (
                f"#{i}: *{m.symbol} LONG*\n"
                f"🎯 Вход: ${m.entry_price:.{price_precision}f}\n"
                f"📈 Изменение: {price_change:.2f}%\n"
                f"📊 Статус: {status}\n"
                f"⏰ Время: {trade.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
            )
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"metrics: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка**: {e}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в /metrics: {e}")
    finally:
        session.close()

async def clear_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"clear_trades: Команда от пользователя {user_id}")
    if not is_authorized(user_id):
        await update.message.reply_text("🚫 **Доступ запрещён.**", parse_mode='Markdown')
        return
    session = None
    try:
        session = Session()
        # Удаление всех записей из TradeMetrics
        session.query(TradeMetrics).delete()
        # Удаление всех записей из Trade
        session.query(Trade).delete()
        session.commit()
        await update.message.reply_text("🗑️ **База данных сделок очищена.**", parse_mode='Markdown')
        logger.info("clear_trades: База данных успешно очищена")
    except Exception as e:
        logger.error(f"clear_trades: Ошибка: {e}")
        await update.message.reply_text(f"🚨 **Ошибка при очистке базы данных**: {e}", parse_mode='Markdown')
        await notify_admin(f"Ошибка в clear_trades: {e}")
    finally:
        if session is not None:
            session.close()

# Основная функция
async def main():
    global application
    
    # Инициализация логгера
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Проверка на уже запущенный бот
    if 'application' in globals() and hasattr(application, 'running') and application.running:
        logger.warning("Бот уже запущен, пропускаем повторный запуск")
        return

    try:
        # Создаем экземпляр Application с явной инициализацией JobQueue
        application = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .arbitrary_callback_data(True)
            .build()
        )

        # Проверка инициализации JobQueue
        if not hasattr(application, 'job_queue') or application.job_queue is None:
            raise RuntimeError("JobQueue не инициализирован! Убедитесь, что установлен python-telegram-bot[job-queue]")

        # Настройка задач
        application.job_queue.run_daily(
            retrain_model_daily,
            time=datetime.strptime("00:00", "%H:%M").time(),
            name="daily_retraining"
        )

        application.job_queue.run_repeating(
            check_active_trades,
            interval=300,
            first=10,
            name="active_trades_check"
        )

        for user_id in load_allowed_users():
            user_settings = get_user_settings(user_id)
            application.job_queue.run_repeating(
                auto_search_trades,
                interval=user_settings.get('auto_interval', 3600),
                name=f"auto_search_{user_id}",
                data=user_id
            )

        # Регистрация обработчиков
        handlers = [
            CommandHandler("start", start),
            CommandHandler("idea", idea),
            # ... остальные обработчики
        ]
        
        for handler in handlers:
            application.add_handler(handler)

        # Запуск бота
        await application.initialize()
        await application.start()
        
        if application.updater:
            await application.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
                close_loop=False
            )

        logger.warning("Бот успешно запущен")

        # Основной цикл
        while True:
            await asyncio.sleep(3600)

    except Exception as e:
        logger.error(f"Ошибка запуска: {str(e)}", exc_info=True)
        await notify_admin(f"Ошибка запуска: {str(e)}")
        
    finally:
        try:
            if application.updater:
                await application.updater.stop()
            await application.stop()
            await application.shutdown()
        except Exception as e:
            logger.error(f"Ошибка завершения: {str(e)}", exc_info=True)

if __name__ == '__main__':
    asyncio.run(main())
