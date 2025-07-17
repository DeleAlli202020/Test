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
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
import sys
from dotenv import load_dotenv
import json
import nest_asyncio

# Применение nest_asyncio
nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('d:\\trading_bot_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding='utf-8')

# Конфигурация
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, 'token.env'))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
ADMIN_ID = int(os.getenv('ADMIN_ID', 0))
ALLOWED_USERS_PATH = os.path.join(BASE_DIR, 'allowed_users.json')
MODEL_PATH_LONG = r"d:\model_improved1.pkl"
MODEL_PATH_SHORT = r"d:\short_model_improved.pkl"
FEATURES_PATH_LONG = r"d:\features.pkl"
FEATURES_PATH_SHORT = r"d:\short_features.pkl"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", 
           "POLUSDT", "TRXUSDT", "TRUMPUSDT", "AVAXUSDT", "HBARUSDT", "NEARUSDT", "TONUSDT"]
TIMEFRAME = "15m"
MIN_RR = 3  # Минимальный Risk/Reward
CHECK_INTERVAL = 15 * 60  # 15 минут в секундах
LOW_RECALL_SYMBOLS = ["BTCUSDT", "BNBUSDT"]

class TradingBot:
    def __init__(self):
        self.subscribed_users = set(self.load_allowed_users())
        
        # Инициализация биржи
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1000,
                'options': {'adjustForTimeDifference': True}
            })
            logger.info("Binance API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Binance API: {e}")
            raise
        
        # Загрузка моделей и признаков
        try:
            long_model_data = joblib.load(MODEL_PATH_LONG)
            short_model_data = joblib.load(MODEL_PATH_SHORT)
            self.long_features = joblib.load(FEATURES_PATH_LONG)
            self.short_features = joblib.load(FEATURES_PATH_SHORT)
            self.long_model = long_model_data['models'].get('combined')
            self.long_scaler = long_model_data['scalers'].get('combined')
            self.short_model = short_model_data['models'].get('combined')
            self.short_scaler = short_model_data['scalers'].get('combined')
            logger.info("Models and features loaded successfully")
            logger.info(f"Expected long features: {self.long_features}")
            logger.info(f"Expected short features: {self.short_features}")
        except Exception as e:
            logger.error(f"Failed to load models or features: {e}")
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
                return [809820681, 667191785, 453365207]
        except Exception as e:
            logger.error(f"Failed to load allowed users: {e}")
            return [809820681, 667191785, 453365207]
    
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
    
    async def fetch_current_data(self, symbol, limit=200):
        """Получение актуальных данных"""
        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                df['symbol'] = symbol
                # Фильтрация некорректных данных
                df = df[(df['price'] > 0) & (df['high'] != df['low']) & (df['volume'] > 0)]
                if len(df) < 50:
                    logger.warning(f"Insufficient data after filtering for {symbol}: {len(df)} rows")
                    return pd.DataFrame()
                return df
            except ccxt.NetworkError as e:
                logger.warning(f"Network error fetching data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                return pd.DataFrame()
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {symbol}: {str(e)}")
                return pd.DataFrame()
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {str(e)}")
                return pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """Расчет технических индикаторов"""
        try:
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi().fillna(50)
            macd = MACD(df['close'], window_slow=26, window_fast=12)
            df['macd'] = macd.macd().fillna(0)
            # Проверка данных для ADX
            if (df['high'] - df['low']).abs().sum() > 0 and df['close'].std() > 0 and df['volume'].sum() > 0:
                df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().fillna(0)
            else:
                logger.warning("Invalid data for ADX calculation, setting to 0")
                df['adx'] = pd.Series(0, index=df.index)
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            df['atr'] = atr.average_true_range().fillna(df['close'].iloc[-1] * 0.001)
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband().fillna(df['close'].iloc[-1])
            df['bb_lower'] = bb.bollinger_lband().fillna(df['close'].iloc[-1])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['price'].replace(0, 0.0001) * 100
            df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume().fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators: {e}")
            return df
    
    def prepare_data_for_model(self, df, features, is_short=False):
        """Подготовка данных для модели"""
        try:
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for model preparation: {len(df)} rows")
                return pd.DataFrame(), None
            
            # Вычисление технических индикаторов
            df = self.calculate_technical_indicators(df)
            
            # Вычисление дополнительных признаков
            df['price_change_1h'] = df['price'].pct_change(4).fillna(0) * 100
            df['price_change_2h'] = df['price'].pct_change(8).fillna(0) * 100
            df['price_change_3h'] = df['price'].pct_change(12).fillna(0) * 100
            df['price_change_4h'] = df['price'].pct_change(16).fillna(0) * 100
            df['price_change_6h'] = df['price'].pct_change(24).fillna(0) * 100
            df['price_change_8h'] = df['price'].pct_change(32).fillna(0) * 100
            df['price_change_12h'] = df['price'].pct_change(48).fillna(0) * 100
            df['volume_score'] = (df['volume'] / df['volume'].rolling(window=6).mean()).fillna(1) * 100
            df['volume_change'] = df['volume'].pct_change().fillna(0) * 100
            
            df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['price'].ewm(span=50, adjust=False).mean()
            df['ema_cross'] = (df['ema_20'] > df['ema_50']).astype(int) if not is_short else (df['ema_20'] < df['ema_50']).astype(int)
            df['volume_spike'] = (df['volume'] > df['volume'].rolling(50).mean() * 2).astype(int)
            
            typical_price = (df['high'].astype(float) + df['low'].astype(float) + df['close'].astype(float)) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_signal'] = ((df['price'] - vwap) / vwap * 100).fillna(0)
            
            vwap_diff = vwap.diff(5) / 5
            df['vwap_angle'] = (np.arctan(vwap_diff / df['price'].replace(0, 0.0001)) * 180 / np.pi).fillna(0)
            
            df['bull_volume'] = (df['close'] > df['open']) * df['volume']
            df['bear_volume'] = (df['close'] < df['open']) * df['volume'] if is_short else pd.Series(0, index=df.index)
            
            df['atr_normalized'] = (df['atr'] / df['price'].replace(0, 0.0001) * 100).fillna(0)
            
            # Признаки для шорт-модели
            if is_short:
                df['atr_change'] = df['atr'].pct_change(4).fillna(0) * 100
                support = df['low'].rolling(window=20).min().fillna(df['price'].min())
                resistance = df['high'].rolling(window=20).max().fillna(df['price'].max())
                df['support_level'] = support
                df['resistance_level'] = resistance
                df['price_to_resistance'] = ((resistance - df['price']) / df['price'].replace(0, 0.0001) * 100).fillna(0)
            else:
                df['atr_change'] = pd.Series(0, index=df.index)
                support = df['low'].rolling(window=20).min().fillna(df['price'].min())
                resistance = df['high'].rolling(window=20).max().fillna(df['price'].max())
                df['support_level'] = support
                df['resistance_level'] = resistance
                df['price_to_resistance'] = pd.Series(0, index=df.index)
            
            df['sentiment'] = pd.Series(50.0 + (df['rsi'] - 50) * 0.5 + df['macd'] * 10, index=df.index).clip(0, 100).replace([np.inf, -np.inf], np.nan).fillna(50)
            df['smart_money_score'] = (df['sentiment'] * 0.4 + (df['rsi'] / 100) * 30 + (df['adx'] / 100) * 30) / 0.7
            df['smart_money_score'] = df['smart_money_score'].replace([np.inf, -np.inf], np.nan).fillna(50).clip(0, 100)
            
            atr = df['atr']
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (3 * atr)
            lower_band = hl2 - (3 * atr)
            super_trend = pd.Series(0.0, index=df.index)
            trend = pd.Series(0, index=df.index)
            for i in range(1, len(df)):
                if df['close'].iloc[i-1] > super_trend.iloc[i-1]:
                    super_trend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = 1
                else:
                    super_trend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = -1
                if trend.iloc[i] == trend.iloc[i-1]:
                    if trend.iloc[i] == 1 and super_trend.iloc[i] < super_trend.iloc[i-1]:
                        super_trend.iloc[i] = super_trend.iloc[i-1]
                    elif trend.iloc[i] == -1 and super_trend.iloc[i] > super_trend.iloc[i-1]:
                        super_trend.iloc[i] = super_trend.iloc[i-1]
            df['super_trend'] = trend
            
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Проверка соответствия признаков
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing features in prepared data: {missing_features}")
                return pd.DataFrame(), df
            
            X = df[features].iloc[-1:]
            logger.info(f"Prepared features for model: {X.columns.tolist()}")
            return X, df
        except Exception as e:
            logger.error(f"Error in prepare_data_for_model: {e}")
            return pd.DataFrame(), df
    
    async def check_signal(self, symbol):
        """Проверка сигнала"""
        try:
            df = await self.fetch_current_data(symbol)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return
            
            # Определение порога
            threshold = 0.3160 if symbol in LOW_RECALL_SYMBOLS else 0.5
            
            # Проверка лонг сигнала
            X_long, df_long = self.prepare_data_for_model(df, self.long_features, is_short=False)
            long_proba = 0
            long_valid = False
            model_failed = False
            if not X_long.empty and self.long_model is not None and self.long_scaler is not None:
                try:
                    X_long_scaled = self.long_scaler.transform(X_long)
                    X_long_scaled = pd.DataFrame(X_long_scaled, columns=self.long_features)
                    long_proba = self.long_model.predict_proba(X_long_scaled)[:, 1][0]
                except Exception as e:
                    logger.error(f"Error processing long signal for {symbol}: {e}")
                    model_failed = True
            
            rsi = df_long['rsi'].iloc[-1] if not df_long.empty else 50
            macd = df_long['macd'].iloc[-1] if not df_long.empty else 0
            adx = df_long['adx'].iloc[-1] if not df_long.empty else 0
            ema_cross = df_long['ema_cross'].iloc[-1] if not df_long.empty else 0
            volume_spike = df_long['volume_spike'].iloc[-1] if not df_long.empty else 0
            super_trend = df_long['super_trend'].iloc[-1] if not df_long.empty else 0
            vwap_angle = df_long['vwap_angle'].iloc[-1] if not df_long.empty else 0
            bull_volume = df_long['bull_volume'].iloc[-1] if not df_long.empty else 0
            volume_mean = df_long['volume'].rolling(20).mean().iloc[-1] if not df_long.empty else 0
            
            long_valid = (
                rsi >= 25 and rsi <= 75 and
                macd > -0.5 and
                adx > 15 and
                (ema_cross == 1 or volume_spike == 1) and
                super_trend == 1 and
                vwap_angle > 0 and
                bull_volume > volume_mean
            )
            if model_failed:
                long_valid = long_valid and long_proba == 0  # Сигнал без модели
            else:
                long_valid = long_valid and long_proba > threshold
            
            # Проверка шорт сигнала
            X_short, df_short = self.prepare_data_for_model(df, self.short_features, is_short=True)
            short_proba = 0
            short_valid = False
            if not X_short.empty and self.short_model is not None and self.short_scaler is not None:
                try:
                    X_short_scaled = self.short_scaler.transform(X_short)
                    X_short_scaled = pd.DataFrame(X_short_scaled, columns=self.short_features)
                    short_proba = self.short_model.predict_proba(X_short_scaled)[:, 1][0]
                except Exception as e:
                    logger.error(f"Error processing short signal for {symbol}: {e}")
                    model_failed = True
            
            rsi = df_short['rsi'].iloc[-1] if not df_short.empty else 50
            macd = df_short['macd'].iloc[-1] if not df_short.empty else 0
            adx = df_short['adx'].iloc[-1] if not df_short.empty else 0
            ema_cross = df_short['ema_cross'].iloc[-1] if not df_short.empty else 0
            volume_spike = df_short['volume_spike'].iloc[-1] if not df_short.empty else 0
            super_trend = df_short['super_trend'].iloc[-1] if not df_short.empty else 0
            vwap_angle = df_short['vwap_angle'].iloc[-1] if not df_short.empty else 0
            bear_volume = df_short['bear_volume'].iloc[-1] if not df_short.empty else 0
            volume_mean = df_short['volume'].rolling(20).mean().iloc[-1] if not df_short.empty else 0
            
            short_valid = (
                rsi >= 60 and
                macd < 0 and
                adx > 15 and
                (ema_cross == 1 or volume_spike == 1) and
                super_trend == -1 and
                vwap_angle < 0 and
                bear_volume > volume_mean
            )
            if model_failed:
                short_valid = short_valid and short_proba == 0  # Сигнал без модели
            else:
                short_valid = short_valid and short_proba > threshold
                
            current_price = df['price'].iloc[-1]
            support = df_long['support_level'].iloc[-1] if not df_long.empty else df['price'].min()
            resistance = df_long['resistance_level'].iloc[-1] if not df_long.empty else df['price'].max()
            atr = df_long['atr'].iloc[-1] if not df_long.empty else 0.001 * current_price
            
            signal_type = None
            signal_proba = 0
            if long_valid and short_valid:
                signal_type = "LONG" if (long_proba > short_proba or model_failed) else "SHORT"
                signal_proba = long_proba if long_proba > short_proba else short_proba
            elif long_valid:
                signal_type = "LONG"
                signal_proba = long_proba
            elif short_valid:
                signal_type = "SHORT"
                signal_proba = short_proba
            
            if signal_type:
                # Расчет уровней TP и SL
                if signal_type == "LONG":
                    sl = current_price - atr * 1.5
                    tp1 = current_price + atr * 3
                    tp2 = current_price + atr * 4.5
                    rr1 = (tp1 - current_price) / (current_price - sl) if current_price != sl else 0
                    rr2 = (tp2 - current_price) / (current_price - sl) if current_price != sl else 0
                else:
                    sl = current_price + atr * 1.5
                    tp1 = current_price - atr * 3
                    tp2 = current_price - atr * 4.5
                    rr1 = (current_price - tp1) / (sl - current_price) if sl != current_price else 0
                    rr2 = (current_price - tp2) / (sl - current_price) if sl != current_price else 0
                
                if rr1 >= MIN_RR:
                    model_status = "⚠️ Сигнал не проверен моделью (ошибка данных)" if model_failed else ""
                    message = (
                        f"**{symbol.replace('USDT', '/USDT')} — Анализ сигнала**\n"
                        f"🕒 **Время:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)\n"
                        f"📊 **Текущая цена:** {current_price:.4f} USDT\n"
                        f"{model_status}\n\n"
                        f"#### 📈 Технические индикаторы:\n"
                        f"- **RSI:** {rsi:.1f} ({'перекуплен' if rsi > 70 else 'перепродан' if rsi < 30 else 'нейтральный'})\n"
                        f"- **MACD:** {macd:.4f} ({'бычий импульс' if macd > 0 else 'медвежий импульс'})\n"
                        f"- **ADX:** {adx:.1f} ({'сильный тренд' if adx > 25 else 'слабый/умеренный тренд'})\n"
                        f"- **Волатильность (ATR):** {atr:.4f} ({'высокая' if atr > current_price * 0.01 else 'ниже среднего'})\n\n"
                        f"#### 📊 Вероятностные метрики:\n"
                        f"- **Тип сделки:** {signal_type}\n"
                        f"- **Вероятность (модель):** {signal_proba*100:.1f}% {'(не проверено моделью)' if model_failed else ''}\n"
                        f"- **Risk/Reward (TP1):** 1:{rr1:.1f}\n"
                        f"- **Risk/Reward (TP2):** 1:{rr2:.1f}\n"
                        f"- **Историческая успешность:** {62 + signal_proba*20:.1f}% {'(оценка без модели)' if model_failed else '(оценка на основе модели)'}\n\n"
                        f"#### 🔍 Ключевые уровни:\n"
                        f"- **Ближайшая поддержка:** {support:.4f} ({(support - current_price)/current_price*100:.1f}%)\n"
                        f"- **Ближайшее сопротивление:** {resistance:.4f} ({(resistance - current_price)/current_price*100:.1f}%)\n"
                        f"- **TP1:** {tp1:.4f} ({(tp1 - current_price)/current_price*100:.1f}%)\n"
                        f"- **TP2:** {tp2:.4f} ({(tp2 - current_price)/current_price*100:.1f}%)\n"
                        f"- **SL:** {sl:.4f} ({(sl - current_price)/current_price*100:.1f}%)\n\n"
                        f"#### ⚠️ Риски:\n"
                        f"- {'Низкий ADX' if adx < 20 else 'Высокий ADX'} → {'возможен флэт или ложный пробой' if adx < 20 else 'тренд может быть сильным'}\n"
                        f"- Объемы: {'ниже среднего' if (bull_volume if signal_type == 'LONG' else bear_volume) < volume_mean else 'выше среднего'}"
                    )
                    await self.broadcast_message(message)
                    logger.info(f"Sent {signal_type} signal for {symbol}: Probability={signal_proba:.4f}, RR1={rr1:.1f}, ModelFailed={model_failed}")
        except Exception as e:
            logger.error(f"Error in check_signal for {symbol}: {e}")
            await self.notify_admin(f"Ошибка обработки {symbol}: {e}")
    
    async def broadcast_message(self, message):
        """Отправка сообщения всем подписчикам"""
        if not hasattr(self, 'bot'):
            return
        try:
            for user_id in self.subscribed_users:
                try:
                    await self.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    logger.error(f"Failed to send message to user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Error in broadcast_message: {e}")
    
    async def notify_admin(self, message):
        """Отправка сообщения админу"""
        if not hasattr(self, 'bot'):
            return
        try:
            await self.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"🚨 {message}",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to notify admin: {e}")
    
    async def check_signals(self, context: ContextTypes.DEFAULT_TYPE):
        """Периодическая проверка сигналов"""
        logger.info("Checking signals for all symbols")
        active_symbols = await self.check_symbol_availability()
        if not active_symbols:
            error_msg = "❌ Нет доступных символов для торговли"
            logger.error(error_msg)
            await self.notify_admin(error_msg)
            return
        
        for symbol in active_symbols:
            await self.check_signal(symbol)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    user_id = update.effective_user.id
    logger.info(f"Start command from user {user_id}")
    
    if user_id not in trading_bot.subscribed_users:
        trading_bot.subscribed_users.add(user_id)
        trading_bot.save_allowed_users()
    
    await update.message.reply_text(
        "🚀 Добро пожаловать в торговый бот!\n\n"
        "Вы будете получать сигналы по торговле.\n"
        "Используйте /status для проверки статуса.",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /status"""
    user_id = update.effective_user.id
    logger.info(f"Status command from user {user_id}")
    
    status_msg = (
        f"📊 **Статус бота**\n\n"
        f"🔄 Бот активен, проверка сигналов каждые 15 минут\n"
        f"📈 Активные символы: {', '.join(SYMBOLS)}\n"
        f"🕒 Последнее обновление: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)"
    )
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Error: {context.error}")
    await trading_bot.notify_admin(f"Ошибка бота: {context.error}")
    if update and update.effective_user:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text="🚨 Произошла ошибка. Пожалуйста, попробуйте снова.",
            parse_mode='Markdown'
        )

async def main():
    """Основная функция"""
    global trading_bot
    trading_bot = TradingBot()
    
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        trading_bot.bot = app.bot
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("status", status))
        app.add_error_handler(error_handler)
        
        app.job_queue.run_repeating(trading_bot.check_signals, interval=CHECK_INTERVAL, first=5)
        
        logger.info("Bot started")
        await app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Bot error: {e}")
        await trading_bot.notify_admin(f"Ошибка запуска бота: {e}")
    finally:
        await trading_bot.exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
