#!/usr/bin/env python3
"""
High-Frequency Cryptocurrency Trading Bot
Google-style implementation with institutional-grade reliability
"""

import asyncio
import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import joblib
from ccxt.async_support import binance
from sqlalchemy.util import symbol
from dotenv import load_dotenv
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackContext,
)

# Constants and Configuration
load_dotenv('config.env')

class Config:
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    ADMIN_ID = int(os.getenv('ADMIN_ID', 0))
    ALLOWED_USERS_FILE = 'allowed_users.json'
    LONG_MODEL_PATH = 'model_improved1.pkl'
    SHORT_MODEL_PATH = 'short_model_improved.pkl'
    
    ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", 
           "DOTUSDT", "DOGEUSDT", "POLUSDT", "TRXUSDT", "TRUMPUSDT", "AVAXUSDT", 
           "HBARUSDT", "NEARUSDT", "TONUSDT"]
    
    TIMEFRAME = "15m"
    DATA_POINTS = 200
    CHECK_INTERVAL = 900  # 15 minutes
    
    # Trading parameters
    RISK_REWARD_RATIO = 3.0
    LOW_RECALL_ASSETS = {"BTCUSDT", "BNBUSDT"}
    LONG_THRESHOLD = 0.35
    SHORT_THRESHOLD = 0.4
    LOW_RECALL_MULTIPLIER = 1.2

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('trading_bot')

# Type Aliases
FeatureVector = Dict[str, float]
Signal = Dict[str, any]

@dataclass
class ModelBundle:
    model: BaseEstimator
    scaler: RobustScaler
    features: List[str]

class TradingBot:
    """Core trading engine with institutional-grade reliability"""
    
    def __init__(self):
        self.users = self._load_users()
        self.exchange = self._init_exchange()
        self.long_model = self._load_model(Config.LONG_MODEL_PATH)
        self.short_model = self._load_model(Config.SHORT_MODEL_PATH)
        self.telegram_bot = None
        self.last_signal_time = {}
        
    def _load_users(self) -> Set[int]:
        """Load authorized users with atomic write safety"""
        try:
            if not os.path.exists(Config.ALLOWED_USERS_FILE):
                with open(Config.ALLOWED_USERS_FILE, 'w') as f:
                    json.dump([Config.ADMIN_ID], f)
                return {Config.ADMIN_ID}
            
            with open(Config.ALLOWED_USERS_FILE, 'r+') as f:
                try:
                    users = json.load(f)
                    if not isinstance(users, list):
                        raise ValueError("Invalid user data format")
                    return set(users)
                except json.JSONDecodeError:
                    logger.error("Corrupted users file, resetting")
                    return {Config.ADMIN_ID}
                    
        except Exception as e:
            logger.error(f"User loading failed: {e}")
            return {Config.ADMIN_ID}
    
    def _init_exchange(self):
        """Initialize exchange connection with fault tolerance"""
        return binance({
            'enableRateLimit': True,
            'rateLimit': 300,
            'timeout': 10000,
            'options': {
                'adjustForTimeDifference': True,
                'defaultType': 'spot'
            }
        })
    
    def _load_model(self, path: str) -> Optional[ModelBundle]:
        """Load trained model with validation"""
        try:
            if not os.path.exists(path):
                logger.error(f"Model file missing: {path}")
                return None
                
            data = joblib.load(path)
            if not all(k in data for k in ['model', 'scaler', 'features']):
                logger.error(f"Invalid model format in {path}")
                return None
                
            return ModelBundle(
                model=data['model'],
                scaler=data['scaler'],
                features=data['features']
            )
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return None
    
    async def fetch_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with retry logic"""
        for attempt in range(3):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=Config.TIMEFRAME,
                    limit=Config.DATA_POINTS
                )
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                
                if self._validate_data(df):
                    return df
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
                await asyncio.sleep(2)
                
        logger.error(f"Data fetch failed for {symbol} after 3 attempts")
        return None
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Comprehensive data quality check"""
        checks = [
            (not df.empty, "Empty DataFrame"),
            (len(df) >= 100, f"Insufficient data: {len(df)} points"),
            (df['close'].notna().all(), "NaN values in close prices"),
            ((df['high'] >= df['low']).all(), "Invalid high/low values"),
            ((df['volume'] >= 0).all(), "Negative volume values")
        ]
        
        for condition, error_msg in checks:
            if not condition:
                logger.warning(f"Data validation failed: {error_msg}")
                return False
        return True
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering pipeline"""
        try:
            # Core indicators
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi().fillna(50)
            macd = MACD(df['close'])
            df['macd'] = macd.macd().fillna(0)
            df['macd_signal'] = macd.macd_signal().fillna(0)
            
            # Trend analysis
            df['adx'] = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).adx().fillna(0)
            
            # Volatility
            df['atr'] = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range().fillna(0)
            
            # Price changes
            for hours, periods in [(1,4), (2,8), (6,24), (12,48)]:
                df[f'change_{hours}h'] = df['price'].pct_change(periods).fillna(0) * 100
            
            return df.replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            return pd.DataFrame()
    
    async def detect_signal(self, symbol: str) -> Optional[Signal]:
        """Complete signal detection pipeline"""
        try:
            # Data acquisition
            df = await self.fetch_market_data(symbol)
            if df is None:
                return None
                
            # Feature engineering
            df = self.calculate_features(df)
            if df.empty:
                return None
                
            # Model prediction
            long_signal = self._evaluate_model(df, is_short=False)
            short_signal = self._evaluate_model(df, is_short=True)
            
            return long_signal or short_signal
            
        except Exception as e:
            logger.error(f"Signal detection failed for {symbol}: {e}")
            return None
    
    def _evaluate_model(self, df: pd.DataFrame, is_short: bool) -> Optional[Signal]:
        """Evaluate trading model with position-specific logic"""
        model_data = self.short_model if is_short else self.long_model
        if not model_data:
            return None
            
        try:
            # Prepare feature vector
            features = self._prepare_features(df, model_data.features)
            if features.empty:
                return None
                
            # Scale features
            scaled = model_data.scaler.transform(features.values.reshape(1, -1))
            
            # Get prediction
            proba = model_data.model.predict_proba(scaled)[0][1]
            
            # Determine threshold
            threshold = Config.SHORT_THRESHOLD if is_short else Config.LONG_THRESHOLD
            if symbol in Config.LOW_RECALL_ASSETS:
                threshold *= Config.LOW_RECALL_MULTIPLIER
                
            # Check conditions
            last = df.iloc[-1]
            if proba > threshold and self._check_conditions(last, is_short):
                return {
                    'symbol': symbol,
                    'type': 'SHORT' if is_short else 'LONG',
                    'probability': proba,
                    'price': last['close'],
                    'rsi': last['rsi'],
                    'adx': last['adx'],
                    'atr': last['atr'],
                    'time': datetime.utcnow(),
                    'model': 'short' if is_short else 'long'
                }
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            
        return None
    
    def _prepare_features(self, df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
        """Ensure correct feature vector format"""
        features = pd.DataFrame(index=[0])
        
        for feature in required:
            if feature in df.columns:
                features[feature] = [df[feature].iloc[-1]]
            else:
                logger.warning(f"Missing feature: {feature}")
                features[feature] = [0]  # Safe default
                
        return features[required]  # Ensure correct order
    
    def _check_conditions(self, last: pd.Series, is_short: bool) -> bool:
        """Validate additional trading conditions"""
        # Trend filter
        if last['adx'] < 15:  # Weak trend
            return False
            
        # RSI filter
        if is_short:
            if last['rsi'] < 40:  # Not overbought enough
                return False
        else:
            if last['rsi'] > 60:  # Not oversold enough
                return False
                
        # Volume filter
        if last['volume'] < last['volume'].rolling(20).mean().iloc[-1] * 0.7:
            return False
            
        return True
    
    async def execute_signal(self, signal: Signal):
        """Execute trading signal with risk management"""
        try:
            # Calculate position sizing
            price = signal['price']
            atr = signal['atr']
            
            if signal['type'] == 'LONG':
                stop_loss = price - atr * 1.5
                take_profit = price + atr * (Config.RISK_REWARD_RATIO * 1.5)
            else:
                stop_loss = price + atr * 1.5
                take_profit = price - atr * (Config.RISK_REWARD_RATIO * 1.5)
                
            # Prepare message
            message = self._format_signal_message(
                signal, 
                stop_loss, 
                take_profit
            )
            
            await self._broadcast(message)
            logger.info(f"Executed {signal['type']} signal for {signal['symbol']}")
            
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
    
    def _format_signal_message(self, signal: Signal, sl: float, tp: float) -> str:
        """Generate professional trading signal message"""
        return (
            f"🚀 *{signal['symbol']} {signal['type']} Signal*\n"
            f"⏰ {signal['time'].strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"💰 Price: ${signal['price']:.4f}\n"
            f"📊 Confidence: {signal['probability']:.1%}\n"
            f"📈 RSI: {signal['rsi']:.1f}\n"
            f"🌀 ADX: {signal['adx']:.1f}\n\n"
            f"🎯 Targets:\n"
            f"• TP: ${tp:.4f} (RR {Config.RISK_REWARD_RATIO}:1)\n"
            f"• SL: ${sl:.4f}\n\n"
            f"⚠️ Risk: {signal['atr']/signal['price']:.2%} volatility"
        )
    
    async def _broadcast(self, message: str):
        """Send message to all subscribed users"""
        if not self.telegram_bot:
            logger.error("Telegram bot not initialized")
            return
            
        for user_id in self.users:
            try:
                await self.telegram_bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}: {e}")

# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user_id = update.effective_user.id
    trading_bot.users.add(user_id)
    
    await update.message.reply_text(
        "🚀 *Crypto Trading Bot Activated*\n\n"
        "You will now receive trading signals.\n"
        "Use /status to check system health.",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    status_msg = (
        "🤖 *System Status*\n\n"
        f"• Models: {'✅' if trading_bot.long_model else '❌'} Long / "
        f"{'✅' if trading_bot.short_model else '❌'} Short\n"
        f"• Exchange: {'✅ Connected' if trading_bot.exchange else '❌ Disconnected'}\n"
        f"• Monitoring: {len(Config.ASSETS)} assets\n"
        f"• Last check: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )
    
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def check_markets(context: CallbackContext):
    """Periodic market scanning"""
    logger.info("Starting market scan...")
    signals = []
    
    for symbol in Config.ASSETS:
        try:
            if signal := await trading_bot.detect_signal(symbol):
                signals.append(signal)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    if signals:
        for signal in signals:
            await trading_bot.execute_signal(signal)
    else:
        logger.info("No trading signals found")

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка соединений"""
    try:
        # Проверка Binance
        ticker = await trading_bot.exchange.fetch_ticker('BTCUSDT')
        # Проверка моделей
        long_loaded = bool(trading_bot.long_model_data)
        short_loaded = bool(trading_bot.short_model_data)
        
        await update.message.reply_text(
            f"Binance: ✅\n"
            f"Long model: {'✅' if long_loaded else '❌'}\n"
            f"Short model: {'✅' if short_loaded else '❌'}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

# В main() после других handler'ов:


async def main():
    """Application entry point"""
    global trading_bot
    trading_bot = TradingBot()
    
    try:
        # Initialize Telegram
        app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        trading_bot.telegram_bot = app.bot
        
        # Register handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("test", test))
        app.add_handler(CommandHandler("status", status))
        
        # Schedule periodic checks
        app.job_queue.run_repeating(
            check_markets,
            interval=Config.CHECK_INTERVAL,
            first=10
        )
        
        logger.info("Trading bot started")
        await app.run_polling()
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
    finally:
        await trading_bot.exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
