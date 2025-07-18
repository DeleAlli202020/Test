import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import ccxt.async_support as ccxt
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import sys
from dotenv import load_dotenv
import json
import nest_asyncio
from typing import Optional, Dict, Any
import warnings
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
load_dotenv('token.env')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
ADMIN_ID = int(os.getenv('ADMIN_ID', 0))
ALLOWED_USERS_PATH = 'allowed_users.json'
MODEL_PATH_LONG = 'model_improved1.pkl'
MODEL_PATH_SHORT = 'short_model_improved.pkl'
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", 
           "DOTUSDT", "DOGEUSDT", "POLUSDT", "TRXUSDT", "TRUMPUSDT", "AVAXUSDT", 
           "HBARUSDT", "NEARUSDT", "TONUSDT"]
TIMEFRAME = "15m"
MIN_RR = 3  # Минимальный Risk/Reward
CHECK_INTERVAL = 15 * 60  # 15 минут в секундах
LOW_RECALL_SYMBOLS = ["BTCUSDT", "BNBUSDT"]

class TradingBot:
    def __init__(self):
        self.subscribed_users = set(self.load_allowed_users())
        self.exchange = self.init_exchange()
        self.long_model_data = self.load_model_data(MODEL_PATH_LONG)
        self.short_model_data = self.load_model_data(MODEL_PATH_SHORT)
        self.bot = None

    @staticmethod
    def load_model_data(model_path: str) -> Optional[Dict[str, Any]]:
        """Загрузка данных модели с обработкой ошибок"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            model_data = joblib.load(model_path)
            if not all(key in model_data for key in ['models', 'scalers', 'active_features']):
                logger.error(f"Invalid model data structure in {model_path}")
                return None
                
            logger.info(f"Successfully loaded model data from {model_path}")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model data from {model_path}: {e}")
            return None

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Комплексная проверка качества данных"""
        if df.empty:
            logger.warning("Empty DataFrame")
            return False
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns: {required_columns}")
            return False
            
        if len(df) < 100:
            logger.warning(f"Insufficient data length: {len(df)}")
            return False
            
        if df['close'].isnull().any() or (df['close'] <= 0).any():
            logger.warning("Invalid close prices (NaN or <= 0)")
            return False
            
        if (df['high'] < df['low']).any():
            logger.warning("High prices lower than low prices")
            return False
            
        if (df['volume'] < 0).any():
            logger.warning("Negative volume values")
            return False
            
        return True
    
    def get_expected_features(self, is_short: bool) -> list:
        """Возвращает фичи в том порядке, в котором их ожидает модель"""
        model_data = self.short_model_data if is_short else self.long_model_data
        if not model_data:
            return []
        
        # Пробуем получить фичи из scaler, затем из модели
        if hasattr(model_data['scalers']['combined'], 'feature_names_in_'):
            return list(model_data['scalers']['combined'].feature_names_in_)
        elif hasattr(model_data['models']['combined'], 'feature_name_'):
            return model_data['models']['combined'].feature_name_
        else:
            return self.get_model_features(is_short)

    def init_exchange(self):
        """Инициализация биржи"""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1000,
                'options': {'adjustForTimeDifference': True}
            })
            logger.info("Binance API initialized successfully")
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize Binance API: {e}")
            raise

    def load_allowed_users(self):
        """Загрузка списка разрешенных пользователей"""
        try:
            if not os.path.exists(ALLOWED_USERS_PATH):
                # Создаем файл если не существует
                default_users = [ADMIN_ID] if ADMIN_ID != 0 else []
                with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(default_users, f)
                return default_users

            with open(ALLOWED_USERS_PATH, 'r+', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    # Если файл пустой, записываем дефолтные значения
                    default_users = [ADMIN_ID] if ADMIN_ID != 0 else []
                    f.seek(0)
                    json.dump(default_users, f)
                    return default_users
                
                try:
                    users = json.loads(content)
                    if not isinstance(users, list):
                        raise ValueError("File must contain JSON array")
                    return users
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON: {e}, resetting to default")
                    default_users = [ADMIN_ID] if ADMIN_ID != 0 else []
                    f.seek(0)
                    f.truncate()
                    json.dump(default_users, f)
                    return default_users
        except Exception as e:
            logger.error(f"Error loading allowed users: {e}")
            return [ADMIN_ID] if ADMIN_ID != 0 else []
    
    def save_allowed_users(self):
        """Сохранение списка разрешенных пользователей"""
        try:
            os.makedirs(os.path.dirname(ALLOWED_USERS_PATH), exist_ok=True)
            with open(ALLOWED_USERS_PATH, 'w', encoding='utf-8') as f:
                json.dump(list(self.subscribed_users), f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved {len(self.subscribed_users)} allowed users")
            return True
        except Exception as e:
            logger.error(f"Failed to save allowed users: {e}")
            return False
        
    async def check_symbol_availability(self):
        """Проверка доступности символов"""
        available_symbols = []
        try:
            await self.exchange.load_markets()
            logger.info("Successfully loaded Binance markets")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            return available_symbols
        
        for symbol in SYMBOLS:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                if ticker['last'] is None:
                    logger.warning(f"No price data available for {symbol}")
                    continue
                ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=10)
                if len(ohlcv) < 10:
                    logger.warning(f"Insufficient historical data for {symbol}")
                    continue
                available_symbols.append(symbol)
                logger.info(f"Symbol {symbol} is available with current price: {ticker['last']}")
            except ccxt.NetworkError as e:
                logger.warning(f"Network error checking {symbol}: {e}")
            except ccxt.ExchangeError as e:
                logger.warning(f"Exchange error checking {symbol}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error checking {symbol}: {e}")
        return available_symbols

    async def fetch_ohlcv_data(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Получение данных с обработкой ошибок"""
        for attempt in range(3):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
                if not ohlcv:
                    logger.warning(f"No OHLCV data received for {symbol} on attempt {attempt+1}")
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                
                if not TradingBot.validate_data(df):
                    logger.warning(f"Invalid data for {symbol} on attempt {attempt+1}")
                    continue
                    
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
                await asyncio.sleep(2)
                
        logger.error(f"All attempts failed for {symbol}")
        return pd.DataFrame()

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Надежный расчет ADX с обработкой ошибок"""
        try:
            # Расчет True Range
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            
            # Расчет Directional Movement
            up = high.diff()
            down = -low.diff()
            plus_dm = np.where((up > down) & (up > 0), up, 0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)
            
            # Сглаживание
            alpha = 1/window
            tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
            plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
            minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
            
            # Расчет индексов
            with np.errstate(divide='ignore', invalid='ignore'):
                plus_di = 100 * (plus_dm_smooth / tr_smooth)
                minus_di = 100 * (minus_dm_smooth / tr_smooth)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            
            # Заполнение NaN
            plus_di = plus_di.fillna(0)
            minus_di = minus_di.fillna(0)
            dx = dx.fillna(0)
            
            # Расчет ADX
            adx = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0)
            
            return adx, plus_di, minus_di
    
        except Exception as e:
            logger.error(f"Error in ADX calculation: {e}")
            return pd.Series(0, index=high.index), pd.Series(0, index=high.index), pd.Series(0, index=high.index)
        
    def init_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Инициализирует все необходимые колонки фичей"""
        required_features = [
            'price_change_1h', 'price_change_2h', 'price_change_6h', 'price_change_12h',
            'volume_score', 'volume_change', 'atr_normalized', 'obv',
            'support_level', 'resistance_level'
        ]
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0.0
                
        return df
            
    def calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех дополнительных фичей с проверкой данных"""
        if 'atr' not in df.columns:
            logger.error("ATR не рассчитан! Сначала вызовите calculate_indicators")
            return pd.DataFrame()
        try:
            if len(df) < 48:
                logger.warning("Not enough data for full feature calculation")
                return df
                
            # Копируем DataFrame чтобы не модифицировать оригинал
            df = df.copy()
            
            # Расчет всех фичей с проверкой на достаточность данных
            try:
                # Price changes
                changes = {
                    '1h': 4,
                    '2h': 8, 
                    '6h': 24,
                    '12h': 48
                }
                for tf, period in changes.items():
                    if len(df) > period:
                        df[f'price_change_{tf}'] = df['price'].pct_change(period) * 100
                    else:
                        df[f'price_change_{tf}'] = np.nan  # Лучше NaN чем 0!
                
                # Volume metrics
                if len(df) >= 6:
                    vol_mean = df['volume'].rolling(6).mean().replace(0, 1)
                    df['volume_score'] = (df['volume'] / vol_mean * 100).fillna(0)
                    df['volume_change'] = df['volume'].pct_change().fillna(0) * 100
                else:
                    df['volume_score'] = 0
                    df['volume_change'] = 0
                    
                # ATR (должен быть уже рассчитан в calculate_indicators)
                if 'atr' in df.columns:
                    df['atr_normalized'] = (df['atr'] / df['price'].replace(0, 1) * 100).fillna(0)
                else:
                    df['atr_normalized'] = 0
                    
                # OBV
                price_diff = df['price'].diff().fillna(0)
                df['obv'] = (np.sign(price_diff) * df['volume']).cumsum()
                
                # Support/Resistance
                if len(df) >= 20:
                    df['support_level'] = df['low'].rolling(20).min().fillna(df['low'].min())
                    df['resistance_level'] = df['high'].rolling(20).max().fillna(df['high'].max())
                else:
                    df['support_level'] = df['low'].min()
                    df['resistance_level'] = df['high'].max()
                    
            except Exception as e:
                logger.error(f"Error in feature calculation: {e}")
                
            return df.replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            logger.error(f"Error in calculate_additional_features: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame, is_short: bool = False) -> pd.DataFrame:
        """Расчет индикаторов для модели"""
        try:
            # Базовые индикаторы (общие для обеих моделей)
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi().fillna(50)
            macd = MACD(df['close'], window_slow=26, window_fast=12)
            df['macd'] = macd.macd().fillna(0)
            df['macd_signal'] = macd.macd_signal().fillna(0)
            df['macd_diff'] = macd.macd_diff().fillna(0)
            
            df['adx'], df['dip'], df['din'] = self.calculate_adx(df['high'], df['low'], df['close'])
            df['atr'] = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range()
            
            # EMA Cross (разные для лонг/шорт)
            df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['price'].ewm(span=50, adjust=False).mean()
            df['ema_cross'] = (df['ema_20'] < df['ema_50']).astype(int) if is_short else (df['ema_20'] > df['ema_50']).astype(int)
            
            # Volume Analysis
            df['volume_spike'] = (df['volume'] > df['volume'].rolling(50).mean() * 2).astype(int)
            df['bull_volume'] = (df['close'] > df['open']) * df['volume']
            df['bear_volume'] = (df['close'] < df['open']) * df['volume']
            
            # Support/Resistance
            df['support'] = df['low'].rolling(20).min().fillna(df['price'].min())
            df['resistance'] = df['high'].rolling(20).max().fillna(df['price'].max())
            
            # VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_signal'] = (df['price'] - df['vwap']) / df['vwap'] * 100
            
            # Bollinger Bands
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband().fillna(0)
            df['bb_lower'] = bb.bollinger_lband().fillna(0)
            df['bb_width'] = bb.bollinger_wband().fillna(0)
            
            # SuperTrend
            atr = df['atr']
            hl2 = (df['high'] + df['low']) / 2
            df['super_trend_upper'] = hl2 + (3 * atr)
            df['super_trend_lower'] = hl2 - (3 * atr)
            df['super_trend'] = 1  # Инициализация
            
            for i in df.index[1:]:
                prev_i = df.index[df.index.get_loc(i)-1]
                if df.loc[prev_i, 'close'] > df.loc[prev_i, 'super_trend_upper']:
                    df.loc[i, 'super_trend'] = 1
                elif df.loc[prev_i, 'close'] < df.loc[prev_i, 'super_trend_lower']:
                    df.loc[i, 'super_trend'] = -1
                else:
                    df.loc[i, 'super_trend'] = df.loc[prev_i, 'super_trend']
            
            # Дополнительные фичи
            df['vwap_angle'] = df['vwap'].diff(5) / 5 * 100
            df['smart_money_score'] = (df['rsi'] * 0.4 + (100 - df['rsi']) * 0.3 + df['adx'] * 0.3).clip(0, 100)
            df['sentiment'] = 50  # Заглушка
            df['price_to_resistance'] = ((df['price'] - df['resistance']) / df['price']) * 100
            df['atr_change'] = df['atr'].pct_change() * 100
            
            # Заполнение недостающих фичей
            model_features = self.get_model_features(is_short)
            for feat in model_features:
                if feat not in df.columns:
                    df[feat] = 0
                    logger.warning(f"Missing feature {feat}, filled with 0")
            
            return df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def get_model_features(self, is_short: bool) -> list:
        """Получение списка фичей для соответствующей модели"""
        model_data = self.short_model_data if is_short else self.long_model_data
        if model_data and 'active_features' in model_data:
            if 'combined' in model_data['active_features']:
                return model_data['active_features']['combined']
            elif model_data['active_features']:
                return next(iter(model_data['active_features'].values()))
        return []

    def prepare_features(self, df: pd.DataFrame, is_short: bool = False) -> pd.DataFrame:
        """Подготовка фичей с гарантией правильного порядка и наличия всех фичей"""
        try:
            # Получаем ожидаемые фичи в правильном порядке
            expected_features = self.get_expected_features(is_short)
            if not expected_features:
                logger.error("No expected features list available")
                return pd.DataFrame()
            
            # Рассчитываем все индикаторы
            df = self.calculate_indicators(df, is_short)
            if df.empty:
                return pd.DataFrame()
                
            # Рассчитываем дополнительные фичи
            df = self.calculate_additional_features(df)
            if df.empty:
                return pd.DataFrame()
            
            # Создаем финальный DataFrame с правильным порядком фичей
            final_features = pd.DataFrame(index=[0])
            
            # Словарь с осмысленными значениями по умолчанию для разных типов фичей
            DEFAULT_VALUES = {
                'price_change': 0.0,        # Для процентных изменений
                'volume': 1.0,              # Для объемов (избегаем деления на 0)
                'atr': df['close'].std(),   # Волатильность как замена ATR
                'rsi': 50.0,                # Нейтральное значение для RSI
                'macd': 0.0,                # Нейтральное значение для MACD
                'support': df['low'].min(), # Минимальный low как поддержка
                'resistance': df['high'].max() # Максимальный high как сопротивление
            }

            for feature in expected_features:
                if feature in df.columns:
                    final_features[feature] = [df[feature].iloc[-1]]
                else:
                    # Автоподбор разумного значения по умолчанию
                    default_value = next(
                        (val for key, val in DEFAULT_VALUES.items() if key in feature),
                        0.0  # Fallback значение
                    )
                    logger.warning(f"Feature {feature} not found, filling with {default_value}")
                    final_features[feature] = [default_value]
            
            # Убедимся, что порядок фичей точно соответствует ожидаемому
            final_features = final_features[expected_features]
            
            return final_features.replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    async def check_signal(self, symbol: str):
        """Проверка сигналов с надежной обработкой None"""
        try:
            df = await self.fetch_ohlcv_data(symbol, limit=100)
            if len(df) < 48:  # Минимум для всех фичей
                logger.error(f"Недостаточно данных для {symbol}: {len(df)} свечей")
                return None
                
            if not self.validate_data(df):
                logger.warning(f"Invalid data for {symbol}")
                return None

            # Проверяем LONG и SHORT сигналы
            long_signal = await self.check_model_signal(df, symbol, is_short=False)
            short_signal = await self.check_model_signal(df, symbol, is_short=True)
            
            # Возвращаем первый найденный сигнал (приоритет LONG)
            return long_signal if long_signal else short_signal
                
        except Exception as e:
            logger.error(f"Error checking signal for {symbol}: {e}")
            return None

    async def check_model_signal(self, df: pd.DataFrame, symbol: str, is_short: bool) -> Optional[Dict]:
        """Проверка сигнала модели с обработкой ошибок"""
        try:
            model_data = self.short_model_data if is_short else self.long_model_data
            if not model_data:
                logger.warning(f"No model data for {'SHORT' if is_short else 'LONG'}")
                return None
                
            model = model_data['models'].get('combined')
            scaler = model_data['scalers'].get('combined')
            
            if model is None or scaler is None:
                logger.warning(f"No model/scaler for {'SHORT' if is_short else 'LONG'}")
                return None
                
            features = self.prepare_features(df, is_short)
            if features.empty:
                logger.warning(f"Empty features for {symbol}")
                return None
                
            try:
                # Получаем feature names из scaler или модели
                # Получаем feature names из scaler или модели
                features_array = features.values.astype(np.float32)
                
                # Преобразуем данные с учетом feature names           
                # Масштабирование
                features_scaled = scaler.transform(features_array)
            
            # Предсказание
                
                # Предсказание
                proba = model.predict_proba(features_scaled)[0][1]
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                return None
        except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                return None
        # Тестовый режим (уберите после проверки)
        TEST_MODE = True
        if TEST_MODE and symbol == "BTCUSDT":
            logger.info("⚠️ ТЕСТОВЫЙ РЕЖИМ: Принудительный сигнал")
            return {
                'type': 'LONG',
                'probability': 0.9,
                'rsi': 30,
                'macd': 1.5,
                'adx': 35,
                'atr': 150,
                'support': df['close'].min(),
                'resistance': df['close'].max(),
                'model_evaluated': True
            }
        
    async def debug_symbol(self, symbol: str):
        """Выводит полную диагностику по символу с анализом LONG/SHORT сигналов"""
        try:
            # Получаем и проверяем данные
            df = await self.fetch_ohlcv_data(symbol, limit=100)
            if df.empty:
                return "❌ Нет данных для анализа"
            
            if not self.validate_data(df):
                return "⚠️ Данные не прошли валидацию"

            # Рассчитываем индикаторы
            df = self.calculate_indicators(df)
            
            # Подготавливаем фичи
            long_features = self.prepare_features(df, is_short=False)
            short_features = self.prepare_features(df, is_short=True)
            
            # Базовый анализ
            analysis = [
                f"📊 Анализ {symbol} ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                f"💰 Цена: {df['close'].iloc[-1]:.4f}",
                f"📈 RSI: {df['rsi'].iloc[-1]:.1f}",
                f"📉 MACD: {df['macd'].iloc[-1]:.4f} (Signal: {df['macd_signal'].iloc[-1]:.4f})",
                f"🌀 ADX: {df['adx'].iloc[-1]:.1f} (DI+: {df['dip'].iloc[-1]:.1f}, DI-: {df['din'].iloc[-1]:.1f})",
                f"⚡ ATR: {df['atr'].iloc[-1]:.4f} ({df['atr_normalized'].iloc[-1]:.2f}%)",
                f"📊 Объем: {df['volume'].iloc[-1]:.2f} (MA20: {df['volume'].rolling(20).mean().iloc[-1]:.2f})",
                f"🔍 Support: {df['support_level'].iloc[-1]:.4f} | Resistance: {df['resistance_level'].iloc[-1]:.4f}"
            ]

            # Анализ LONG
            if not long_features.empty and self.long_model_data:
                long_prob = self.long_model_data['models']['combined'].predict_proba(
                    self.long_model_data['scalers']['combined'].transform(long_features)
                )[0][1]
                analysis.append(f"\n🟢 LONG вероятность: {long_prob:.1%}")
                analysis.append(f"Порог: {0.35 if symbol not in LOW_RECALL_SYMBOLS else 0.316}")
                analysis.append(f"Сигнал: {'ДА' if long_prob > (0.35 if symbol not in LOW_RECALL_SYMBOLS else 0.316) else 'нет'}")

            # Анализ SHORT
            if not short_features.empty and self.short_model_data:
                short_prob = self.short_model_data['models']['combined'].predict_proba(
                    self.short_model_data['scalers']['combined'].transform(short_features)
                )[0][1]
                analysis.append(f"\n🔴 SHORT вероятность: {short_prob:.1%}")
                analysis.append(f"Порог: {0.4 if symbol not in LOW_RECALL_SYMBOLS else 0.5}")
                analysis.append(f"Сигнал: {'ДА' if short_prob > (0.4 if symbol not in LOW_RECALL_SYMBOLS else 0.5) else 'нет'}")

            # Условия входа
            conditions = [
                f"\n📌 Условия входа:",
                f"RSI {'> 70' if df['rsi'].iloc[-1] > 70 else '< 30' if df['rsi'].iloc[-1] < 30 else 'нейтральный'}",
                f"MACD {'выше сигнала' if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else 'ниже'}",
                f"Тренд: {'сильный' if df['adx'].iloc[-1] > 25 else 'слабый'}"
            ]
            
            return "\n".join(analysis + conditions)
            
        except Exception as e:
            logger.error(f"Debug error: {e}")
            return f"⚠️ Ошибка анализа: {str(e)}"

    async def send_signal_message(self, symbol: str, signal: Dict[str, Any], df: pd.DataFrame):
        """Отправка сообщения о сигнале"""
        try:
            current_price = df['price'].iloc[-1]
            atr = signal['atr']
            
            # Расчет уровней TP/SL с Risk/Reward >= 3
            if signal['type'] == 'LONG':
                sl = current_price - atr * 1.5
                tp1 = current_price + atr * 4.5  # RR 1:3
                tp2 = current_price + atr * 6.0  # RR 1:4
                rr1 = (tp1 - current_price) / (current_price - sl) if current_price != sl else 0
                rr2 = (tp2 - current_price) / (current_price - sl) if current_price != sl else 0
            else:
                sl = current_price + atr * 1.5
                tp1 = current_price - atr * 4.5  # RR 1:3
                tp2 = current_price - atr * 6.0  # RR 1:4
                rr1 = (current_price - tp1) / (sl - current_price) if sl != current_price else 0
                rr2 = (current_price - tp2) / (sl - current_price) if sl != current_price else 0

            # Определение силы тренда по ADX
            adx_strength = "слабый" if signal['adx'] < 25 else "умеренный" if signal['adx'] < 50 else "сильный"
            
            # Определение состояния RSI
            rsi_state = (
                "перепроданность" if signal['rsi'] < 30 else 
                "перекупленность" if signal['rsi'] > 70 else 
                "нейтральный"
            )
            
            # Определение направления MACD
            macd_direction = (
                "бычий" if signal['macd'] > 0 else 
                "медвежий"
            )
            
            # Определение волатильности
            volatility_level = (
                "низкая" if atr < current_price * 0.01 else 
                "средняя" if atr < current_price * 0.02 else 
                "высокая"
            )
            
            # Форматирование сообщения
            message = (
                f"🚀 **{symbol.replace('USDT', '/USDT')} — {signal['type']} Сигнал**\n"
                f"🕒 Время: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)\n"
                f"💰 Текущая цена: {current_price:.4f} USDT\n\n"
                
                f"#### 📈 Технические индикаторы:\n"
                f"- RSI: {signal['rsi']:.1f} ({rsi_state})\n"
                f"- MACD: {signal['macd']:.4f} ({macd_direction})\n"
                f"- ADX: {signal['adx']:.1f} ({adx_strength} тренд)\n"
                f"- Волатильность (ATR): {atr:.4f} ({volatility_level})\n\n"
                
                f"#### 📊 Вероятностные метрики:\n"
                f"- Модель предсказания: {signal['probability']*100:.1f}%\n"
                f"- Risk/Reward (R/R): 1:{min(rr1, rr2):.1f}\n\n"
                
                f"#### 🔍 Ключевые уровни:\n"
                f"- Ближайшая поддержка: {signal['support']:.4f} ({(current_price - signal['support'])/current_price*100:.1f}%)\n"
                f"- Ближайшее сопротивление: {signal['resistance']:.4f} ({(signal['resistance'] - current_price)/current_price*100:.1f}%)\n\n"
                
                f"#### 🎯 Цели:\n"
                f"- TP1: {tp1:.4f} (RR 1:{rr1:.1f})\n"
                f"- TP2: {tp2:.4f} (RR 1:{rr2:.1f})\n"
                f"- SL: {sl:.4f}\n\n"
                
                f"#### ⚠️ Риски:\n"
            )
            
            # Добавление предупреждений о рисках
            if signal['adx'] < 25:
                message += "- Слабый тренд → возможны ложные пробои\n"
            if signal['rsi'] > 70 and signal['type'] == 'LONG':
                message += "- RSI в зоне перекупленности → возможен откат\n"
            elif signal['rsi'] < 30 and signal['type'] == 'SHORT':
                message += "- RSI в зоне перепроданности → возможен откат\n"
            if atr < current_price * 0.01:
                message += "- Низкая волатильность → малый потенциал движения\n"
            
            # Добавляем пометку о том, оценивала ли модель сделку
            if not signal.get('model_evaluated', True):
                message += "\n⚠️ **Внимание**: Эта сделка не была оценена моделью из-за технических проблем. " \
                          "Используйте дополнительные методы анализа перед входом в сделку.\n"
            
            await self.broadcast_message(message)
            logger.info(f"Sent {signal['type']} signal for {symbol}")
        except Exception as e:
            logger.error(f"Error sending signal message: {e}")

    async def broadcast_message(self, message: str):
        """Отправка сообщения подписчикам"""
        if not hasattr(self, 'bot') or not self.bot:
            logger.error("Bot instance not available for broadcasting")
            return
            
        for user_id in self.subscribed_users:
            try:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    logger.info(f"New user: {user_id}")
    
    trading_bot.subscribed_users.add(user_id)
    trading_bot.save_allowed_users()
    
    await update.message.reply_text(
        "🚀 Welcome to Crypto Trading Bot!\n\n"
        "You will receive trading signals periodically.\n"
        "Use /status to check bot status.",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /status"""
    long_loaded = trading_bot.long_model_data is not None
    short_loaded = trading_bot.short_model_data is not None
    
    status_msg = (
        "🤖 **Bot Status**\n\n"
        f"🔄 Active symbols: {len(SYMBOLS)}\n"
        f"📊 LONG model loaded: {'✅' if long_loaded else '❌'}\n"
        f"📊 SHORT model loaded: {'✅' if short_loaded else '❌'}\n"
        f"👥 Subscribers: {len(trading_bot.subscribed_users)}\n"
        f"🕒 Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def check_all_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка всех символов на сигналы с уведомлением о результатах"""
    logger.info("Starting periodic check for all symbols")
    signals_found = 0

    if isinstance(update, Update):
        await update.message.reply_text("🔍 Начинаю проверку рынка...")
    
    for symbol in SYMBOLS:
        try:
            signal = await trading_bot.check_signal(symbol)
            if signal:
                df = await trading_bot.fetch_ohlcv_data(symbol, limit=100)
                if not df.empty:
                    await trading_bot.send_signal_message(symbol, signal, df)
                    signals_found += 1
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    # Отправляем уведомление, если не найдено ни одного сигнала
    if signals_found == 0:
        message = (
            "🔍 **Результаты проверки рынка**\n"
            f"🕒 Время: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)\n"
            f"📊 Проверено символов: {len(SYMBOLS)}\n"
            "❌ Торговых сигналов не найдено\n\n"
            "Следующая проверка через 15 минут"
        )
        await trading_bot.broadcast_message(message)
    
    logger.info(f"Completed periodic check. Found {signals_found} signals")

async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = await trading_bot.debug_symbol("BTCUSDT")
    await update.message.reply_text(f"```\n{report}\n```", parse_mode='Markdown')

async def main():
    """Основная функция"""
    global trading_bot
    trading_bot = TradingBot()
    
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        trading_bot.bot = app.bot
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("status", status))
        app.add_handler(CommandHandler("idea", check_all_symbols))
        app.add_handler(CommandHandler("debug", debug))
        # Запуск периодической проверки каждые 15 минут
        app.job_queue.run_repeating(
            lambda ctx: check_all_symbols(None, ctx),  # Без Update объекта
            interval=CHECK_INTERVAL,
            first=10
        )
        
        logger.info("Bot started")
        await app.run_polling()
    except Exception as e:
        logger.error(f"Bot failed: {e}")
    finally:
        await trading_bot.exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
