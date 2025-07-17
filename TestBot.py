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
from ta.volatility import AverageTrueRange
import sys
from dotenv import load_dotenv
import json
import nest_asyncio
from ta.volatility import BollingerBands

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
        self.long_model, self.long_scaler, self.short_model, self.short_scaler = self.load_models()

    @staticmethod
    def validate_data(df):
        """Проверка качества данных"""
        if df.empty:
            return False
        if len(df) < 20:
            return False
        if df['close'].isnull().any() or (df['close'] <= 0).any():
            return False
        if df['volume'].isnull().any() or (df['volume'] < 0).any():
            return False
        return True
        
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
            if os.path.exists(ALLOWED_USERS_PATH):
                with open(ALLOWED_USERS_PATH, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                logger.info(f"Loaded {len(users)} allowed users")
                return users
            else:
                logger.warning("Allowed users file not found, using default list")
                return [ADMIN_ID] if ADMIN_ID != 0 else []
        except Exception as e:
            logger.error(f"Failed to load allowed users: {e}")
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

    def load_models(self):
        """Загрузка моделей и скейлеров"""
        try:
            long_data = joblib.load(MODEL_PATH_LONG)
            short_data = joblib.load(MODEL_PATH_SHORT)
            
            long_model = long_data['models'].get('combined')
            long_scaler = long_data['scalers'].get('combined')
            short_model = short_data['models'].get('combined')
            short_scaler = short_data['scalers'].get('combined')
            
            logger.info("Models and scalers loaded successfully")
            return long_model, long_scaler, short_model, short_scaler
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    async def fetch_ohlcv_data(self, symbol, limit=200):
        """Получение данных с обработкой ошибок"""
        for attempt in range(3):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
                if not ohlcv:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                
                # Проверка качества данных
                if not TradingBot.validate_data(df):
                    logger.warning(f"Invalid data for {symbol} on attempt {attempt+1}")
                    continue
                    
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
                await asyncio.sleep(2)
                
        logger.error(f"All attempts failed for {symbol}")
        return pd.DataFrame()

    # В методе calculate_indicators (для обоих файлов):
    def calculate_adx(self, high, low, close, window=14):
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

    def calculate_indicators(self, df, is_short=False):
        """Расчет индикаторов для модели"""
        try:
            # Базовые индикаторы (общие для обеих моделей)
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi().fillna(50)
            macd = MACD(df['close'], window_slow=26, window_fast=12)
            df['macd'] = macd.macd().fillna(0)
            df['macd_signal'] = macd.macd_signal().fillna(0)
            df['macd_diff'] = macd.macd_diff().fillna(0)
            
            df['adx'], df['dip'], df['din'] = self.calculate_adx(df['high'], df['low'], df['close'])
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().fillna(0)
            
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
            
            # SuperTrend (исправленная версия)
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
            
            # Добавление недостающих фичей
            df['vwap_angle'] = df['vwap'].diff(5) / 5 * 100
            df['smart_money_score'] = (df['rsi'] * 0.4 + (100 - df['rsi']) * 0.3 + df['adx'] * 0.3).clip(0, 100)
            df['sentiment'] = 50  # Заглушка
            df['price_to_resistance'] = ((df['price'] - df['resistance']) / df['price']) * 100
            df['atr_change'] = df['atr'].pct_change() * 100
            
            # Убедимся, что все фичи присутствуют
            required_features = [
                'price_change_1h', 'price_change_2h', 'price_change_6h', 'price_change_12h',
                'volume_score', 'volume_change', 'atr_normalized', 'obv',
                'bb_upper', 'bb_lower', 'bb_width', 'support_level', 'resistance_level'
            ]
            
            for feat in required_features:
                if feat not in df.columns:
                    df[feat] = 0  # Заполняем нулями если фича отсутствует
                    
            # Переименуем колонки для совместимости
            df = df.rename(columns={
                'support': 'support_level',
                'resistance': 'resistance_level'
            })
            
            return df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    async def check_signal(self, symbol):
        """Проверка сигналов для символа"""
        try:
            df = await self.fetch_ohlcv_data(symbol)
            if df.empty:
                return

            # Подготовка данных для моделей
            df_long = self.calculate_indicators(df, is_short=False)
            df_short = self.calculate_indicators(df, is_short=True)
            current_price = df['price'].loc[-1]
            atr = df['atr'].loc[-1]

            # Проверка LONG сигнала
            long_signal = await self.check_long_signal(df_long, symbol)
            
            # Проверка SHORT сигнала
            short_signal = await self.check_short_signal(df_short, symbol)

            # Определение лучшего сигнала (если есть оба, выбираем с большей вероятностью)
            best_signal = None
            if long_signal and short_signal:
                if long_signal['probability'] > short_signal['probability']:
                    best_signal = long_signal
                else:
                    best_signal = short_signal
            elif long_signal:
                best_signal = long_signal
            elif short_signal:
                best_signal = short_signal

            if best_signal:
                await self.send_signal_message(symbol, best_signal, current_price, atr, df)
                
        except Exception as e:
            logger.error(f"Error checking signal for {symbol}: {e}")

    async def check_long_signal(self, df, symbol):
        """Проверка LONG сигнала"""
        try:
            if self.long_model is None:
                    return None
            
            # Проверка наличия всех фичей
            required_features = set(self.long_model.feature_names_in_)
            available_features = set(df.columns)
            missing_features = required_features - available_features
            if missing_features:
                logger.error(f"Missing features for LONG {symbol}: {missing_features}")
                return None

            # Фильтрация по условиям LONG
            last_row = df.loc[-1]
            valid_signal = (
                (25 <= last_row['rsi'] <= 75) &
                (last_row['macd'] > -0.5) &
                (last_row['adx'] > 15) &
                ((last_row['ema_cross'] == 1) | (last_row['volume_spike'] == 1)) &
                (last_row['bull_volume'] > df['volume'].rolling(20).mean().loc[-1]))
            
            if not valid_signal:
                return None

            # Подготовка фичей и предсказание
            features = self.prepare_features(df, is_short=False)
            if features.empty:
                return None

            features_scaled = self.long_scaler.transform(features)
            proba = self.long_model.predict_proba(features_scaled)[0][1]
            threshold = 0.316 if symbol in LOW_RECALL_SYMBOLS else 0.35

            if proba > threshold:
                return {
                    'type': 'LONG',
                    'probability': proba,
                    'rsi': last_row['rsi'],
                    'macd': last_row['macd'],
                    'adx': last_row['adx'],
                    'atr': last_row['atr'],
                    'support': last_row['support'],
                    'resistance': last_row['resistance']
                }
            return None
        except Exception as e:
            logger.error(f"Error checking LONG signal for {symbol}: {e}")
            return None

    async def check_short_signal(self, df, symbol):
        """Проверка SHORT сигнала"""
        try:
            if self.short_model is None:
                    return None
            
            # Проверка наличия всех фичей
            required_features = set(self.short_model.feature_names_in_)
            available_features = set(df.columns)
            missing_features = required_features - available_features
            if missing_features:
                logger.error(f"Missing features for LONG {symbol}: {missing_features}")
                return None

            # Фильтрация по условиям SHORT
            last_row = df.loc[-1]
            valid_signal = (
                (last_row['rsi'] >= 60) &
                (last_row['macd'] < 0) &
                (last_row['adx'] > 15) &
                ((last_row['ema_cross'] == 1) | (last_row['volume_spike'] == 1)) &
                (last_row['bear_volume'] > df['volume'].rolling(20).mean().loc[-1]))
            
            if not valid_signal:
                return None

            # Подготовка фичей и предсказание
            features = self.prepare_features(df, is_short=True)
            if features.empty:
                return None

            features_scaled = self.short_scaler.transform(features)
            proba = self.short_model.predict_proba(features_scaled)[0][1]
            threshold = 0.4 if symbol in LOW_RECALL_SYMBOLS else 0.5

            if proba > threshold:
                return {
                    'type': 'SHORT',
                    'probability': proba,
                    'rsi': last_row['rsi'],
                    'macd': last_row['macd'],
                    'adx': last_row['adx'],
                    'atr': last_row['atr'],
                    'support': last_row['support'],
                    'resistance': last_row['resistance']
                }
            return None
        except Exception as e:
            logger.error(f"Error checking SHORT signal for {symbol}: {e}")
            return None

    def prepare_features(self, df, is_short=False):
        """Подготовка фичей для модели"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Общие фичи для обеих моделей
            common_features = [
                'price_change_1h', 'price_change_2h', 'price_change_6h', 'price_change_12h',
                'volume_score', 'volume_change', 'atr_normalized', 'rsi', 'macd',
                'adx', 'ema_cross', 'volume_spike', 'super_trend', 'vwap_signal',
                'vwap_angle', 'bb_upper', 'bb_lower', 'bb_width', 'support_level',
                'resistance_level', 'price_to_resistance', 'atr_change'
            ]
            
            for feat in common_features:
                if feat in df.columns:
                    features[feat] = df[feat].fillna(0)
                else:
                    features[feat] = 0  # Заполняем нулями если фича отсутствует
                    logger.warning(f"Missing feature {feat}, filling with zeros")
            
            # Специфичные фичи
            if is_short:
                features['bear_volume'] = df['bear_volume'].fillna(0)
            else:
                features['bull_volume'] = df['bull_volume'].fillna(0)
            
            return features.loc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    async def send_signal_message(self, symbol, signal, current_price, atr, df):
        """Отправка сообщения о сигнале"""
        try:
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
            
            await self.broadcast_message(message)
            logger.info(f"Sent {signal['type']} signal for {symbol}")
        except Exception as e:
            logger.error(f"Error sending signal message: {e}")

    async def broadcast_message(self, message):
        """Отправка сообщения подписчикам"""
        if not hasattr(self, 'bot'):
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
    status_msg = (
        "🤖 **Bot Status**\n\n"
        f"🔄 Active symbols: {len(SYMBOLS)}\n"
        f"📊 Models loaded: {'✅' if trading_bot.long_model and trading_bot.short_model else '❌'}\n"
        f"👥 Subscribers: {len(trading_bot.subscribed_users)}\n"
        f"🕒 Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def check_all_symbols(context: ContextTypes.DEFAULT_TYPE):
    """Проверка всех символов на сигналы"""
    logger.info("Starting periodic check for all symbols")
    for symbol in SYMBOLS:
        await trading_bot.check_signal(symbol)
    logger.info("Completed periodic check for all symbols")

async def main():
    """Основная функция"""
    global trading_bot
    trading_bot = TradingBot()
    
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        trading_bot.bot = app.bot
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("status", status))
        
        # Запуск периодической проверки каждые 15 минут
        app.job_queue.run_repeating(
            check_all_symbols,
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
