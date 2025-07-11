import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import logging
import nest_asyncio
import ccxt.async_support as ccxt
import joblib
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
import sys
from dotenv import load_dotenv
import telegram

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('forward_bot_log.txt', encoding='utf-8'),
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
MODEL_PATH = os.path.join(BASE_DIR, 'model_improved1.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'features1.pkl')

# Константы торговли
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT", 
           "SOL/USDT", "DOT/USDT", "DOGE/USDT", "POL/USDT", "TRX/USDT"]
TIMEFRAME = "15m"
INITIAL_BALANCE = 10000
POSITION_SIZE = 0.15
COMMISSION = 0.001
LOW_RECALL_SYMBOLS = ["BTC/USDT", "BNB/USDT"]

class ForwardTradingBot:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.positions = {symbol: {'amount': 0, 'buy_price': 0} for symbol in SYMBOLS}
        self.trades = []
        self.equity = [INITIAL_BALANCE]
        self.last_update_time = None
        self.active_symbols = []
        
        # Инициализация биржи
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1000,
            'options': {
                'adjustForTimeDifference': True
            }
        })
        
        # Загрузка модели
        self.load_model()
        
        # Подписчики бота
        self.subscribed_users = set()
        self.load_subscribed_users()
    
    def load_subscribed_users(self):
        try:
            if os.path.exists(ALLOWED_USERS_PATH):
                with open(ALLOWED_USERS_PATH, 'r') as f:
                    self.subscribed_users = set(json.load(f))
        except Exception as e:
            logger.error(f"Error loading subscribed users: {e}")
            self.subscribed_users = set()
    
    def save_subscribed_users(self):
        try:
            with open(ALLOWED_USERS_PATH, 'w') as f:
                json.dump(list(self.subscribed_users), f)
        except Exception as e:
            logger.error(f"Error saving subscribed users: {e}")
    
    def load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.active_features_dict = model_data['active_features']
                logger.info("Models and scalers loaded successfully")
            else:
                raise FileNotFoundError("Model file not found")
        except Exception as e:
            logger.error(f"Failed to load model data: {e}")
            raise
    
    async def check_symbol_availability(self):
        """Проверяем доступность данных для каждого символа"""
        available_symbols = []
        
        try:
            markets = await self.exchange.load_markets()
            logger.info("Successfully loaded Binance markets")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            return available_symbols
        
        for symbol in SYMBOLS:
            try:
                # Проверяем доступность тикера
                ticker = await self.exchange.fetch_ticker(symbol)
                if ticker['last'] is None:
                    logger.warning(f"No price data available for {symbol}")
                    continue
                
                # Проверяем доступность исторических данных
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

    async def fetch_current_data(self, symbol, limit=100):
        """Получаем текущие данные для символа с улучшенной обработкой ошибок"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                df['symbol'] = symbol
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

    def calculate_rsi(self, df, periods=14):
        if df.empty or len(df) < periods:
            return pd.Series(0, index=df.index)
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, df, fast=12, slow=26):
        if df.empty or len(df) < slow:
            return pd.Series(0, index=df.index)
        exp1 = df['price'].ewm(span=fast, adjust=False).mean()
        exp2 = df['price'].ewm(span=slow, adjust=False).mean()
        macd = (exp1 - exp2) / df['price'].iloc[-1] * 100
        return macd.fillna(0)

    def calculate_atr(self, df, periods=14):
        if df.empty or len(df) < periods:
            return pd.Series(0, index=df.index)
        high_low = df['high'].astype(float) - df['low'].astype(float)
        high_close = np.abs(df['high'].astype(float) - df['close'].astype(float).shift(1))
        low_close = np.abs(df['low'].astype(float) - df['close'].astype(float).shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=periods).mean()
        min_atr = df['price'].iloc[-1] * 0.001 if not df.empty else 0.00001
        return atr.fillna(min_atr)

    def calculate_vwap(self, df):
        if df.empty:
            return pd.Series(0, index=df.index)
        typical_price = (df['high'].astype(float) + df['low'].astype(float) + df['close'].astype(float)) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.fillna(0)

    def calculate_vwap_signal(self, df):
        if df.empty:
            return pd.Series(0, index=df.index)
        vwap = self.calculate_vwap(df)
        price = df['price']
        vwap_distance = (price - vwap) / vwap * 100
        return vwap_distance.fillna(0)

    def calculate_obv(self, df):
        if df.empty:
            return pd.Series(0, index=df.index)
        price_diff = df['price'].diff()
        obv = (np.sign(price_diff) * df['volume']).cumsum()
        return obv.fillna(0)

    def calculate_adx(self, df, periods=14):
        if df.empty or len(df) < periods:
            return pd.Series(0, index=df.index)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        plus_dm = high.diff().where(high.diff() > low.diff(), 0)
        minus_dm = low.diff().where(low.diff() > high.diff(), 0)
        tr = self.calculate_atr(df)
        plus_di = 100 * plus_dm.rolling(window=periods).mean() / (tr + 1e-10)
        minus_di = 100 * minus_dm.rolling(window=periods).mean() / (tr + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=periods).mean()
        return adx.fillna(0)

    def calculate_bollinger_bands(self, df, window=20, window_dev=2):
        if df.empty or len(df) < window:
            return pd.Series(0, index=df.index), pd.Series(0, index=df.index), pd.Series(0, index=df.index)
        bb = BollingerBands(close=df['price'], window=window, window_dev=window_dev)
        return bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()

    def calculate_support_resistance(self, df, window=20):
        if df.empty or len(df) < window:
            return pd.Series(0, index=df.index), pd.Series(0, index=df.index)
        support = df['low'].rolling(window=window).min()
        resistance = df['high'].rolling(window=window).max()
        return support.fillna(df['price'].min()), resistance.fillna(df['price'].max())

    def prepare_features(self, df):
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        df['price_change_1h'] = df['price'].pct_change(4) * 100
        df['price_change_2h'] = df['price'].pct_change(8) * 100
        df['price_change_3h'] = df['price'].pct_change(12) * 100
        df['price_change_4h'] = df['price'].pct_change(16) * 100
        df['price_change_6h'] = df['price'].pct_change(24) * 100
        df['price_change_8h'] = df['price'].pct_change(32) * 100
        df['price_change_12h'] = df['price'].pct_change(48) * 100
        df['volume_score'] = df['volume'] / df['volume'].rolling(window=6).mean() * 100
        df['volume_change'] = df['volume'].pct_change() * 100
        df['atr'] = self.calculate_atr(df)
        df['atr_normalized'] = df['atr'] / df['price'] * 100
        df['rsi'] = self.calculate_rsi(df)
        df['macd'] = self.calculate_macd(df)
        df['vwap_signal'] = self.calculate_vwap_signal(df)
        df['obv'] = self.calculate_obv(df)
        df['adx'] = self.calculate_adx(df)
        df['bb_upper'], df['bb_lower'], df['bb_width'] = self.calculate_bollinger_bands(df)
        df['support_level'], df['resistance_level'] = self.calculate_support_resistance(df)
        
        # Синтетические признаки
        df['sentiment'] = pd.Series(50.0 + (df['rsi'] - 50) * 0.5 + df['macd'] * 10, index=df.index).clip(0, 100)
        df['sentiment'] = df['sentiment'].replace([np.inf, -np.inf], np.nan).fillna(50)
        df['smart_money_score'] = (df['sentiment'] * 0.4 + (df['rsi'] / 100) * 30 + (df['adx'] / 100) * 30) / 0.7
        df['smart_money_score'] = df['smart_money_score'].replace([np.inf, -np.inf], np.nan).fillna(50).clip(0, 100)
        
        FEATURES = [
            'price_change_1h', 'price_change_2h', 'price_change_3h', 'price_change_4h',
            'price_change_6h', 'price_change_8h', 'price_change_12h', 'volume_score',
            'volume_change', 'atr_normalized', 'rsi', 'macd', 'vwap_signal', 'obv',
            'adx', 'bb_upper', 'bb_lower', 'bb_width', 'support_level', 'resistance_level',
            'sentiment', 'smart_money_score'
        ]
        
        df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df[FEATURES].iloc[:-2], df.iloc[:-2]

    def get_model_for_symbol(self, symbol):
        """Выбираем модель для символа"""
        base_symbol = symbol.replace('/USDT', 'USDT')
        if 'combined' in self.models:
            return self.models['combined'], self.scalers['combined'], self.active_features_dict['combined']
        elif base_symbol in self.models:
            return self.models[base_symbol], self.scalers[base_symbol], self.active_features_dict[base_symbol]
        else:
            return None, None, None

    async def execute_trades(self, context: ContextTypes.DEFAULT_TYPE):
        """Основной цикл торговли"""
        # Проверяем доступность символов перед началом
        self.active_symbols = await self.check_symbol_availability()
        if not self.active_symbols:
            error_msg = "❌ No symbols available for trading. Exiting."
            logger.error(error_msg)
            await self.notify_admin(error_msg)
            return
        
        startup_msg = f"🚀 Starting forward test with symbols: {', '.join(self.active_symbols)}"
        logger.info(startup_msg)
        await self.broadcast_message(startup_msg)
        
        while True:
            try:
                current_time = datetime.now()
                
                if self.last_update_time is None or (current_time - self.last_update_time).seconds >= 15*60:
                    self.last_update_time = current_time
                    
                    for symbol in self.active_symbols:
                        try:
                            df = await self.fetch_current_data(symbol)
                            if df.empty:
                                continue
                            
                            X, df_processed = self.prepare_features(df)
                            if X.empty:
                                continue
                            
                            model, scaler, active_features = self.get_model_for_symbol(symbol)
                            if model is None:
                                continue
                            
                            missing_features = [f for f in active_features if f not in X.columns]
                            if missing_features:
                                logger.warning(f"Missing features for {symbol}: {missing_features}")
                                continue
                            
                            threshold = 0.3160 if symbol in LOW_RECALL_SYMBOLS else 0.5
                            
                            X_scaled = scaler.transform(X[active_features])
                            X_scaled = pd.DataFrame(X_scaled, columns=active_features)
                            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                            y_pred = (y_pred_proba > threshold).astype(int)[0]
                            
                            current_price = df_processed['price'].iloc[-1]
                            rsi = df_processed['rsi'].iloc[-1]
                            macd = df_processed['macd'].iloc[-1]
                            adx = df_processed['adx'].iloc[-1]
                            
                            valid_signal = (rsi >= 30) & (rsi <= 70) & (macd > 0) & (adx > 20)
                            position = self.positions[symbol]['amount']
                            
                            # Логика входа
                            if y_pred == 1 and position == 0 and valid_signal:
                                amount = (self.balance * POSITION_SIZE) / current_price
                                cost = amount * current_price * (1 + COMMISSION)
                                
                                if cost <= self.balance:
                                    self.balance -= cost
                                    self.positions[symbol] = {
                                        'amount': amount,
                                        'buy_price': current_price
                                    }
                                    
                                    self.trades.append({
                                        'symbol': symbol,
                                        'time': current_time,
                                        'type': 'BUY',
                                        'price': current_price,
                                        'amount': amount,
                                        'balance': self.balance,
                                        'equity': self.balance + amount * current_price,
                                        'rsi': rsi,
                                        'macd': macd,
                                        'adx': adx
                                    })
                                    
                                    # Отправляем уведомление о покупке
                                    buy_msg = (
                                        f"📈 <b>BUY SIGNAL: {symbol}</b>\n\n"
                                        f"💰 <b>Price</b>: {current_price:.4f}\n"
                                        f"📊 <b>Amount</b>: {amount:.6f}\n"
                                        f"📈 <b>RSI</b>: {rsi:.1f}\n"
                                        f"📊 <b>MACD</b>: {macd:.4f}\n"
                                        f"📉 <b>ADX</b>: {adx:.1f}\n\n"
                                        f"<b>Exit Plan:</b>\n"
                                        "🛑 Stop Loss: -1.0%\n"
                                        "🎯 Take Profit: +2.0%\n"
                                        "📉 Exit on sell signal\n\n"
                                        f"<i>Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</i>"
                                    )
                                    await self.broadcast_message(buy_msg)
                                    logger.info(f"BUY {symbol} at {current_price:.4f}, amount: {amount:.6f}")
                            
                            # Логика выхода
                            elif position > 0:
                                buy_price = self.positions[symbol]['buy_price']
                                price_change = (current_price - buy_price) / buy_price * 100
                                
                                stop_condition = price_change <= -1.0
                                take_profit_condition = price_change >= 2.0
                                exit_signal = y_pred == 0
                                
                                if stop_condition or take_profit_condition or exit_signal:
                                    revenue = position * current_price * (1 - COMMISSION)
                                    self.balance += revenue
                                    
                                    exit_reason = ""
                                    if stop_condition:
                                        exit_reason = "🛑 Stop Loss Triggered"
                                    elif take_profit_condition:
                                        exit_reason = "🎯 Take Profit Triggered"
                                    else:
                                        exit_reason = "📉 Sell Signal Received"
                                    
                                    self.trades.append({
                                        'symbol': symbol,
                                        'time': current_time,
                                        'type': 'SELL',
                                        'price': current_price,
                                        'amount': position,
                                        'balance': self.balance,
                                        'equity': self.balance,
                                        'rsi': rsi,
                                        'macd': macd,
                                        'adx': adx,
                                        'profit_pct': price_change,
                                        'stop_triggered': stop_condition,
                                        'tp_triggered': take_profit_condition
                                    })
                                    
                                    # Отправляем уведомление о продаже
                                    sell_msg = (
                                        f"📉 <b>SELL SIGNAL: {symbol}</b>\n\n"
                                        f"💰 <b>Price</b>: {current_price:.4f}\n"
                                        f"📊 <b>Amount</b>: {position:.6f}\n"
                                        f"📈 <b>PnL</b>: {price_change:.2f}%\n"
                                        f"📊 <b>Reason</b>: {exit_reason}\n\n"
                                        f"<i>Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</i>"
                                    )
                                    await self.broadcast_message(sell_msg)
                                    logger.info(f"SELL {symbol} at {current_price:.4f}, PnL: {price_change:.2f}%")
                                    
                                    self.positions[symbol] = {'amount': 0, 'buy_price': 0}
                            
                            # Обновляем equity
                            total_equity = self.balance
                            for sym, pos in self.positions.items():
                                if pos['amount'] > 0:
                                    total_equity += pos['amount'] * current_price
                            self.equity.append(total_equity)

                        except Exception as e:
                            error_msg = f"⚠️ Error processing {symbol}: {e}"
                            logger.error(error_msg, exc_info=True)
                            await self.notify_admin(error_msg)
                    
                    # Отправляем периодический статус
                    status_msg = (
                        f"🔄 <b>Status Update</b>\n\n"
                        f"💰 <b>Balance</b>: {self.balance:.2f} USDT\n"
                        f"📊 <b>Equity</b>: {self.equity[-1]:.2f} USDT\n"
                        f"📈 <b>Active Positions</b>: {sum(1 for pos in self.positions.values() if pos['amount'] > 0)}\n"
                        f"<i>Last update: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</i>"
                    )
                    await self.broadcast_message(status_msg)
                    
                    try:
                        trades_df = pd.DataFrame(self.trades)
                        trades_df.to_csv("forward_trades.csv", index=False)
                    except Exception as e:
                        error_msg = f"⚠️ Error saving trades: {e}"
                        logger.error(error_msg)
                        await self.notify_admin(error_msg)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                error_msg = f"⚠️ Error in main loop: {e}"
                logger.error(error_msg, exc_info=True)
                await self.notify_admin(error_msg)
                await asyncio.sleep(60)

    def calculate_metrics(self):
        """Расчет метрик"""
        if not self.trades or len(self.trades) < 2:
            return {
                'profit': 0,
                'profit_pct': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'num_trades': 0
            }
        
        profit = self.equity[-1] - INITIAL_BALANCE
        profit_pct = profit / INITIAL_BALANCE * 100
        
        trade_profits = []
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy = self.trades[i]
                sell = self.trades[i + 1]
                trade_profit = (sell['price'] - buy['price']) * buy['amount'] * (1 - 2 * COMMISSION)
                trade_profits.append(trade_profit)
        
        win_rate = len([p for p in trade_profits if p > 0]) / len(trade_profits) if trade_profits else 0
        
        equity_series = pd.Series(self.equity)
        rolling_max = equity_series.cummax()
        drawdown = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdown.max()
        
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(365*24*4) * returns.mean() / (returns.std() + 1e-10)
        
        return {
            'profit': profit,
            'profit_pct': profit_pct,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trade_profits)
        }

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
                        parse_mode='HTML'
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
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to notify admin: {e}")

# Инициализация бота
forward_bot = ForwardTradingBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Start command from user {user_id}")
    
    # Добавляем пользователя в подписчики
    if user_id not in forward_bot.subscribed_users:
        forward_bot.subscribed_users.add(user_id)
        forward_bot.save_subscribed_users()
    
    await update.message.reply_text(
        "🚀 Welcome to Forward Trading Bot!\n\n"
        "You will receive all trading signals and updates.\n"
        "Use /status to get current trading status.",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Status command from user {user_id}")
    
    metrics = forward_bot.calculate_metrics()
    status_msg = (
        "📊 <b>Current Trading Status</b>\n\n"
        f"💰 <b>Balance</b>: {forward_bot.balance:.2f} USDT\n"
        f"📈 <b>Equity</b>: {forward_bot.equity[-1]:.2f} USDT\n"
        f"📊 <b>Profit</b>: ${metrics['profit']:.2f} ({metrics['profit_pct']:.2f}%)\n"
        f"🎯 <b>Win Rate</b>: {metrics['win_rate']:.1%}\n"
        f"📉 <b>Max Drawdown</b>: {metrics['max_drawdown']:.1%}\n"
        f"⚖️ <b>Sharpe Ratio</b>: {metrics['sharpe_ratio']:.2f}\n"
        f"🔢 <b>Total Trades</b>: {metrics['num_trades']}"
    )
    
    await update.message.reply_text(status_msg, parse_mode='HTML')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")
    await forward_bot.notify_admin(f"Bot error: {context.error}")
    if update and update.effective_user:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text="🚨 An error occurred. Please try again.",
            parse_mode='Markdown'
        )

def main():
    # Инициализация Telegram бота
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Сохраняем ссылку на бота для отправки сообщений
    forward_bot.bot = application.bot
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запуск торгового бота в фоновом режиме
    application.job_queue.run_once(
        lambda ctx: asyncio.create_task(forward_bot.execute_trades(ctx)),
        when=5
    )
    
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
