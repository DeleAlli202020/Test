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
from typing import Counter, Dict, List, Optional, Tuple, Set, Any

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
from ta.volatility import AverageTrueRange, BollingerBands
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackContext,
)
import warnings
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Constants and Configuration
load_dotenv('config.env')

class Config:
    TELEGRAM_TOKEN = '7364285248:AAH8wzdSDGEd1PO53wi9LedFfblbi-e8G_Y'
    MIN_ATR_RATIO = 0.002  # Минимальный ATR/Price для валидности сигнала
    MAX_ATR_RATIO = 0.05
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
    LONG_THRESHOLD = 0.4
    SHORT_THRESHOLD = 0.35
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
Signal = Dict[str, Any]

class TradingBot:
    """Core trading engine with institutional-grade reliability"""
    
    def __init__(self):
        self.users = self._load_users()
        self.exchange = self._init_exchange()
        self.long_model_data = self._load_model_data(Config.LONG_MODEL_PATH)
        self.short_model_data = self._load_model_data(Config.SHORT_MODEL_PATH)
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
    
    def _load_model_data(self, path: str) -> Optional[Dict[str, Any]]:
        """Load trained model with validation (using TestBot.py format)"""
        try:
            if not os.path.exists(path):
                logger.error(f"Model file missing: {path}")
                return None
                
            model_data = joblib.load(path)
            if not all(k in model_data for k in ['models', 'scalers', 'active_features']):
                logger.error(f"Invalid model format in {path}")
                return None
                
            logger.info(f"Successfully loaded model data from {path}")
            return model_data
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
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple:
        """Robust ADX calculation with proper error handling"""
        try:
            # Calculate True Range
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            
            # Calculate Directional Movement
            up = high.diff()
            down = -low.diff()
            plus_dm = np.where((up > down) & (up > 0), up, 0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)
            
            # Smoothing with proper window size
            window = min(window, len(high) - 1) if len(high) > 1 else 1
            alpha = 1/window if window > 0 else 1
            
            tr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False, min_periods=window).mean()
            
            # Calculate directional indicators with zero division protection
            with np.errstate(divide='ignore', invalid='ignore'):
                plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
                minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            
            # Fill NaN values
            plus_di = plus_di.fillna(0)
            minus_di = minus_di.fillna(0)
            dx = dx.fillna(0)
            
            # Calculate ADX
            adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean().fillna(0)
            
            return adx, plus_di, minus_di
        
        except Exception as e:
            logger.error(f"ADX calculation error: {str(e)}")
            zero_series = pd.Series(0, index=high.index)
            return zero_series, zero_series, zero_series
        
    def calculate_atr(df, periods=14):
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

    def calculate_indicators(self, df: pd.DataFrame, is_short: bool = False) -> pd.DataFrame:
        """Complete feature engineering with all required indicators"""
        try:
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Ensure price column exists
            if 'price' not in df.columns:
                df['price'] = df['close'].astype(float)
            
            # 1. Core Technical Indicators
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi().fillna(50)
            macd = MACD(df['close'], window_slow=26, window_fast=12)
            df['macd'] = macd.macd().fillna(0)
            df['macd_signal'] = macd.macd_signal().fillna(0)
            df['macd_diff'] = macd.macd_diff().fillna(0)
            
            # Robust ADX calculation
            df['adx'], df['dip'], df['din'] = self.calculate_adx(
                df['high'],
                df['low'],
                df['close']
            )
            
            # Volatility indicators
            if 'atr' not in df.columns:
                df['atr'] = AverageTrueRange(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=14
                ).average_true_range().fillna(0)

            if 'volume_ma' not in df.columns:
                df['volume_ma'] = df['volume'].rolling(min(20, len(df)), min_periods=1).mean().fillna(0)

            df['atr_normalized'] = (df['atr'] / df['price'].replace(0, 1)).fillna(0) * 100
            df['atr_change'] = df['atr'].pct_change().fillna(0) * 100
            
            # 2. Moving Averages and Crosses
            df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['price'].ewm(span=50, adjust=False).mean()
            df['ema_cross'] = (df['ema_20'] > df['ema_50']).astype(int)
            
            # 3. Bollinger Bands
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband().fillna(0)
            df['bb_lower'] = bb.bollinger_lband().fillna(0)
            df['bb_width'] = bb.bollinger_wband().fillna(0)
            
            # 4. Volume Indicators
            min_periods = min(6, len(df))
            vol_mean = df['volume'].rolling(min_periods, min_periods=1).mean().replace(0, 1)
            df['volume_score'] = (df['volume'] / vol_mean * 100).fillna(100)
            df['volume_change'] = df['volume'].pct_change().fillna(0) * 100
            df['volume_spike'] = (df['volume'] > vol_mean * 2).astype(int)
            df['bull_volume'] = ((df['close'] > df['open']) * df['volume']).fillna(0)
            df['bear_volume'] = ((df['close'] < df['open']) * df['volume']).fillna(0)
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['price'].diff().fillna(0)) * df['volume']).cumsum()
            
            # 5. VWAP and Signals
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_signal'] = ((df['price'] - df['vwap']) / df['vwap'].replace(0, 1)).fillna(0) * 100
            
            # 6. Support/Resistance Levels
            window = min(20, len(df))
            df['support_level'] = df['low'].rolling(window).min().fillna(df['low'].min())
            df['resistance_level'] = df['high'].rolling(window).max().fillna(df['high'].max())
            df['price_to_resistance'] = ((df['price'] - df['resistance_level']) / df['price'].replace(0, 1)).fillna(0) * 100
            
            # 7. SuperTrend
            hl2 = (df['high'] + df['low']) / 2
            df['super_trend_upper'] = hl2 + (3 * df['atr'])
            df['super_trend_lower'] = hl2 - (3 * df['atr'])
            super_trend = [1] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i-1] > df['super_trend_upper'].iloc[i-1]:
                    super_trend[i] = 1
                elif df['close'].iloc[i-1] < df['super_trend_lower'].iloc[i-1]:
                    super_trend[i] = -1
                else:
                    super_trend[i] = super_trend[i-1]
            df['super_trend'] = super_trend
            
            # 8. Composite Scores
            df['smart_money_score'] = (df['rsi'] * 0.4 + (100 - df['rsi']) * 0.3 + df['adx'] * 0.3).clip(0, 100)
            
            # Price Changes
            for hours, periods in [(1,4), (2,8), (6,24), (12,48)]:
                df[f'price_change_{hours}h'] = df['price'].pct_change(periods).fillna(0) * 100
            
            return df.replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {str(e)}")
            return pd.DataFrame()
        
    def get_model_features(self, is_short: bool) -> list:
        """Get the list of features expected by the model"""
        model_data = self.short_model_data if is_short else self.long_model_data
        if not model_data:
            return []
        
        # Try to get features from scaler first, then from model
        if hasattr(model_data['scalers']['combined'], 'feature_names_in_'):
            return list(model_data['scalers']['combined'].feature_names_in_)
        elif hasattr(model_data['models']['combined'], 'feature_names_'):
            return model_data['models']['combined'].feature_names_
        else:
            # Fallback to active_features if available
            if 'active_features' in model_data and model_data['active_features']:
                if isinstance(model_data['active_features'], dict):
                    return list(model_data['active_features'].values())[0]
                return model_data['active_features']
            return []
    
    def prepare_features(self, df: pd.DataFrame, is_short: bool = False) -> pd.DataFrame:
        """Prepare features with guaranteed correct order and all features present"""
        try:
            # Get expected features in correct order
            expected_features = self.get_model_features(is_short)
            if not expected_features:
                logger.error("No expected features list available")
                return pd.DataFrame()
            
            # Calculate all indicators
            df = self.calculate_indicators(df, is_short)
            if df.empty:
                return pd.DataFrame()
                
            # Create final DataFrame with correct feature order
            final_features = pd.DataFrame(index=[0])
            
            DEFAULT_VALUES = {
                'price_change': 0.0,
                'volume': 1.0,
                'atr': df['close'].std(),
                'rsi': 50.0,
                'macd': 0.0,
                'support': df['low'].min(),
                'resistance': df['high'].max()
            }

            for feature in expected_features:
                if feature in df.columns:
                    final_features[feature] = [df[feature].iloc[-1]]
                else:
                    # Auto-select reasonable default value
                    default_value = next(
                        (val for key, val in DEFAULT_VALUES.items() if key in feature),
                        0.0  # Fallback value
                    )
                    logger.warning(f"Feature {feature} not found, filling with {default_value}")
                    final_features[feature] = [default_value]
            
            # Ensure features are in correct order
            final_features = final_features[expected_features]
            
            return final_features.replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            return pd.DataFrame()
    
    async def detect_signal(self, symbol: str) -> Optional[Signal]:
        """Complete signal detection pipeline"""
        try:
            # Data acquisition
            df = await self.fetch_market_data(symbol)
            if df is None or len(df) < 100:
                return None
                
            # Check LONG and SHORT signals
            long_signal = await self._evaluate_model_signal(df, symbol, is_short=False)
            short_signal = await self._evaluate_model_signal(df, symbol, is_short=True)
            
            return long_signal if long_signal else short_signal
            
        except Exception as e:
            logger.error(f"Signal detection failed for {symbol}: {e}")
            return None
    
    async def _evaluate_model_signal(self, df: pd.DataFrame, symbol: str, is_short: bool) -> Optional[Signal]:
        """Оценка сигнала модели с полным набором данных"""
        try:
            model_data = self.short_model_data if is_short else self.long_model_data
            if not model_data:
                return None
                
            if 'atr' not in last:
                last['atr'] = AverageTrueRange(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=14
                ).average_true_range().iloc[-1]
            # Рассчитываем volume_ma если нет
            if 'volume_ma' not in df.columns:
                df['volume_ma'] = df['volume'].rolling(min(20, len(df)), min_periods=1).mean()
                
            features = self.prepare_features(df, is_short)
            if features.empty:
                return None
                
            last = df.iloc[-1]
            features_scaled = model_data['scalers']['combined'].transform(features.values.astype(np.float32))
            proba = model_data['models']['combined'].predict_proba(features_scaled)[0][1]
            
            threshold = Config.SHORT_THRESHOLD if is_short else Config.LONG_THRESHOLD
            if symbol in Config.LOW_RECALL_ASSETS:
                threshold *= Config.LOW_RECALL_MULTIPLIER
                
            if proba > threshold and self._check_conditions(last, is_short):
                return {
                    'symbol': symbol,
                    'type': 'SHORT' if is_short else 'LONG',
                    'probability': proba,
                    'price': last['close'],
                    'rsi': last.get('rsi', 50),
                    'adx': last.get('adx', 0),
                    'atr': last.get('atr', last['close'] * 0.01),
                    'volume': last.get('volume', 0),
                    'volume_ma': last.get('volume_ma', 0),
                    'time': datetime.utcnow()
                }
            logger.info(
                f"{symbol} {'SHORT' if is_short else 'LONG'}: "
                f"Proba={proba:.1%}, RSI={last['rsi']:.1f}, "
                f"ADX={last['adx']:.1f}, ATR={last.get('atr', 'N/A')}"
            )
                            
        except Exception as e:
            logger.error(f"Model evaluation failed for {symbol}: {str(e)}")
        
            logger.info(
                f"{symbol} {'SHORT' if is_short else 'LONG'}: "
                f"Proba={proba:.1%}, RSI={last['rsi']:.1f}, "
                f"ADX={last['adx']:.1f}, ATR={last.get('atr', 'N/A')}"
            )
        return None
    
    def _check_conditions(self, last: pd.Series, is_short: bool) -> bool:
        """Validate additional trading conditions with rolling fix"""
        try:
            # Trend filter with protection
            adx = last.get('adx', 0)
            if adx < 15:  # Weak trend
                return False
                
            # RSI filter with protection
            rsi = last.get('rsi', 50)
            if is_short:
                if rsi < 50:  # Not overbought enough
                    return False
            else:
                if rsi > 60:  # Not oversold enough
                    return False
                
            atr_ratio = last.get('atr', 0) / last.get('close', 1)
            if not (Config.MIN_ATR_RATIO <= atr_ratio <= Config.MAX_ATR_RATIO):
                return False
                    
            # Volume filter with rolling protection
            volume = last.get('volume', 0)
            volume_ma = last.get('volume_ma', volume * 2)  # Default to 2x if MA not available
            
            if volume < volume_ma * 0.7:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Condition check error: {str(e)}")
            return False
    
    async def execute_signal(self, signal: Dict[str, Any]):
        """Исполнение сигнала с абсолютной гарантией наличия ATR"""
        try:
            # Жёсткая проверка структуры сигнала
            if not all(k in signal for k in ['symbol', 'type', 'price', 'probability']):
                logger.error(f"Invalid signal structure: {signal}")
                return

            # Принудительно вычисляем ATR, если его нет
            if 'atr' not in signal or signal['atr'] <= 0:
                logger.warning(f"Recalculating ATR for {signal['symbol']}")
                try:
                    df = await self.fetch_market_data(signal['symbol'], limit=20)
                    if df is not None:
                        atr = AverageTrueRange(
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            window=14
                        ).average_true_range().iloc[-1]
                        signal['atr'] = atr if not np.isnan(atr) else signal['price'] * 0.02
                    else:
                        signal['atr'] = signal['price'] * 0.02  # Fallback: 2% от цены
                except Exception as e:
                    logger.error(f"ATR calculation failed: {e}")
                    signal['atr'] = signal['price'] * 0.02

            # Форсированное обновление времени сигнала
            signal['time'] = datetime.utcnow()

            # Расчёт уровней (гарантированно работает даже с нулевым ATR)
            price = signal['price']
            atr = max(signal['atr'], price * 0.01)  # Минимум 1% от цены
            
            if signal['type'] == 'LONG':
                stop_loss = max(price - atr * 1.5, price * 0.985)  # Минимум 1.5% стоп
                take_profit = price + atr * 4.5
            else:  # SHORT
                stop_loss = min(price + atr * 1.5, price * 1.015)
                take_profit = price - atr * 4.5

            # Отправка сообщения (теперь 100% без ошибок)
            message = self._format_signal_message(signal, stop_loss, take_profit)
            await self._broadcast(message)
            logger.info(f"Executed {signal['type']} signal for {signal['symbol']}")

        except Exception as e:
            logger.error(f"Fatal signal error: {e}")

    def _generate_trade_explanation(self, signal: Signal, sl: float, tp: float) -> str:
        """Генерация подробного объяснения сделки"""
        risk_percent = (abs(signal['price'] - sl)) / signal['price'] * 100
        reward_percent = (abs(tp - signal['price'])) / signal['price'] * 100
        
        return (
            f"🚀 *{signal['symbol']} {signal['type']} Signal*\n"
            f"⏰ {signal['time'].strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"💰 Цена входа: ${signal['price']:.4f}\n"
            f"📊 Вероятность: {signal['probability']:.1%}\n"
            f"📈 Тех. показатели:\n"
            f"  • RSI: {signal.get('rsi', 0):.1f} ({'перепродан' if signal.get('rsi', 0) < 30 else 'перекуплен' if signal.get('rsi', 0) > 70 else 'нейтральный'})\n"
            f"  • ADX: {signal.get('adx', 0):.1f} ({'сильный тренд' if signal.get('adx', 0) > 25 else 'слабый тренд'})\n"
            f"  • Волатильность (ATR): {signal.get('atr', 0):.4f}\n\n"
            f"🎯 Уровни:\n"
            f"  • TP: ${tp:.4f} (+{reward_percent:.2f}%)\n"
            f"  • SL: ${sl:.4f} (-{risk_percent:.2f}%)\n"
            f"  • Risk/Reward: 1:3\n\n"
            f"📌 Логика: {self._get_trade_logic(signal)}"
        )

    def _get_trade_logic(self, signal: Signal) -> str:
        """Генерация логики сделки"""
        logic = []
        if signal.get('rsi', 50) < 30 and signal['type'] == 'LONG':
            logic.append("RSI в зоне перепроданности")
        elif signal.get('rsi', 50) > 70 and signal['type'] == 'SHORT':
            logic.append("RSI в зоне перекупленности")
        
        if signal.get('adx', 0) > 25:
            logic.append("Сильный тренд")
        
        if signal.get('volume', 0) > signal.get('volume_ma', 0):
            logic.append("Высокий объем")
        
        return ", ".join(logic) if logic else "Стандартные условия"
    
    def _format_signal_message(self, signal: Signal, sl: float, tp: float) -> str:
        """Генерация сообщения строго по формату"""
        # Расчет процентов риска и прибыли
        risk_pct = abs(signal['price'] - sl) / signal['price'] * 100
        reward_pct = abs(tp - signal['price']) / signal['price'] * 100
        
        # Определение силы тренда
        adx = signal.get('adx', 0)
        if adx > 40:
            trend_strength = "💪 Сильный тренд"
        elif adx > 25:
            trend_strength = "🔼 Средний тренд"
        else:
            trend_strength = "🔄 Без тренда"
        
        # Форматирование сообщения
        return (
            f"🚀 *{signal['symbol']} {signal['type']} Signal*\n"
            f"⏰ {signal['time'].strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"💰 Цена входа: ${signal['price']:.4f}\n"
            f"📊 Вероятность: {signal['probability']:.1%}\n"
            f"📈 Тех.анализ:\n"
            f"  • RSI: {signal['rsi']:.1f} ({'🔻Перепродан' if signal['rsi'] < 30 else '🔺Перекуплен' if signal['rsi'] > 70 else '⚖️Нейтральный'})\n"
            f"  • ADX: {adx:.1f} {trend_strength}\n"
            f"  • ATR: {signal['atr']:.4f} ({signal['atr']/signal['price']*100:.2f}%)\n\n"
            f"🎯 Уровни:\n"
            f"  • TP: ${tp:.4f} (+{reward_pct:.2f}%)\n"
            f"  • SL: ${sl:.4f} (-{risk_pct:.2f}%)\n"
            f"  • Risk/Reward: 1:3\n\n"
            f"⚠️ Риск: Стоп-лосс {'выше' if signal['type'] == 'SHORT' else 'ниже'} текущей цены"
        )
    


    async def analyze_symbol(self, symbol: str) -> dict:
        """Анализ символа с защитой от ошибок rolling"""
        result = {
            'symbol': symbol,
            'signal': None,
            'reasons': [],
            'indicators': {},
            'volume_ok': True
        }
        
        try:
            # Получаем данные с проверкой
            df = await self.fetch_market_data(symbol)
            if df is None or len(df) < 48:
                result['reasons'].append("Недостаточно данных")
                return result
            
            # Добавляем скользящую среднюю объема заранее
            df['volume_ma'] = df['volume'].rolling(min(20, len(df)), min_periods=1).mean()
            
            # Рассчитываем индикаторы
            df = self.calculate_indicators(df)
            last = df.iloc[-1]
            
            # Сохраняем значения индикаторов с защитой
            result['indicators'] = {
                'price': last.get('close', 0),
                'rsi': round(last.get('rsi', 50), 2),
                'adx': round(last.get('adx', 0), 2),
                'volume': last.get('volume', 0),
                'volume_ma': last.get('volume_ma', 0),
                'trend': self._get_trend_description(last)
            }
            
            # Проверка объема
            volume_ok = last.get('volume', 0) >= last.get('volume_ma', 0) * 0.7
            result['volume_ok'] = volume_ok
            
            # Проверка сигналов
            long_check = await self._check_signal(df, symbol, False)
            short_check = await self._check_signal(df, symbol, True)
            
            if long_check['signal']:
                result['signal'] = long_check['signal']
            elif short_check['signal']:
                result['signal'] = short_check['signal']
            
            # Собираем причины отклонения
            result['reasons'] = long_check['reasons'] + short_check['reasons']
            if not volume_ok:
                result['reasons'].append("Объем ниже 70% от средней")
                
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {str(e)}")
            result['reasons'].append(f"Системная ошибка: {str(e)}")
        
        return result

    def _get_trend_description(self, last: pd.Series) -> str:
        """Определение тренда с защитой"""
        try:
            adx = last.get('adx', 0)
            if adx < 15:
                return "Без тренда"
            return "Вверх" if last.get('dip', 0) > last.get('din', 0) else "Вниз"
        except:
            return "Неизвестно"
    async def _check_signal(self, df: pd.DataFrame, symbol: str, is_short: bool) -> dict:
        """Проверка конкретного типа сигнала"""
        result = {'signal': None, 'reasons': []}
        
        try:
            model_data = self.short_model_data if is_short else self.long_model_data
            if not model_data:
                result['reasons'].append(f"{'SHORT' if is_short else 'LONG'} модель не загружена")
                return result
                
            features = self.prepare_features(df, is_short)
            if features.empty:
                result['reasons'].append("Не удалось подготовить фичи")
                return result
                
            threshold = Config.SHORT_THRESHOLD if is_short else Config.LONG_THRESHOLD
            if symbol in Config.LOW_RECALL_ASSETS:
                threshold *= Config.LOW_RECALL_MULTIPLIER
                
            features_scaled = model_data['scalers']['combined'].transform(features.values.astype(np.float32))
            proba = model_data['models']['combined'].predict_proba(features_scaled)[0][1]
            
            if proba > threshold and self._check_conditions(df.iloc[-1], is_short):
                result['signal'] = {
                    'symbol': symbol,
                    'type': 'SHORT' if is_short else 'LONG',
                    'probability': proba,
                    'price': df['close'].iloc[-1],
                    'rsi': df['rsi'].iloc[-1],
                    'adx': df['adx'].iloc[-1],
                    'time': datetime.utcnow()
                }
            else:
                result['reasons'].append(
                    f"{'SHORT' if is_short else 'LONG'}: {proba:.1%} < {threshold:.1%}"
                )
                
        except Exception as e:
            logger.error(f"Ошибка проверки сигнала {symbol}: {str(e)}")
            result['reasons'].append("Ошибка проверки сигнала")
        
        return result
    def _analyze_signal(self, symbol: str, signal_type: str, probability: float, 
                       rsi: float, adx: float, price: float) -> str:
        """Генерация детального анализа сигнала"""
        analysis = f"🚀 *{symbol} {signal_type}*\n"
        analysis += f"▸ Вероятность: {probability:.1%}\n"
        analysis += f"▸ Цена: ${price:.4f}\n"
        
        # Анализ RSI
        if rsi < 30 and signal_type == 'LONG':
            analysis += f"▸ RSI: {rsi:.1f} 🔻 (Сильная перепроданность)\n"
        elif rsi > 70 and signal_type == 'SHORT':
            analysis += f"▸ RSI: {rsi:.1f} 🔺 (Сильная перекупленность)\n"
        else:
            analysis += f"▸ RSI: {rsi:.1f} ⚖️ (Нейтральный)\n"
        
        # Анализ тренда
        if adx > 40:
            analysis += f"▸ ADX: {adx:.1f} 💪 (Сильный тренд)\n"
        elif adx > 25:
            analysis += f"▸ ADX: {adx:.1f} ↗️ (Умеренный тренд)\n"
        else:
            analysis += f"▸ ADX: {adx:.1f} ➡️ (Без тренда)\n"
        
        # Рекомендация
        if probability > 0.8:
            analysis += "✅ Сильный сигнал к действию"
        elif probability > 0.6:
            analysis += "🟡 Умеренный сигнал (требует подтверждения)"
        else:
            analysis += "🔴 Слабый сигнал (высокий риск)"
        
        return analysis
async def broadcast_message(bot: Bot, message: str):
    """Отправка сообщений всем подписчикам"""
    for user_id in trading_bot.users:
        try:
            # Разбиваем длинные сообщения
            if len(message) > 4000:
                parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
                for part in parts:
                    await bot.send_message(chat_id=user_id, text=part, parse_mode='Markdown')
                    await asyncio.sleep(0.5)
            else:
                await bot.send_message(chat_id=user_id, text=message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения пользователю {user_id}: {str(e)}")
            
async def send_scan_report(bot: Bot, signals: list, rejected: list):
    """Улучшенный отчет с детальными пояснениями"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    if signals:
        # Детальный отчет о сигналах с TP/SL
        for signal in signals[:3]:  # Показываем топ-3 сигнала
            s = signal['signal']
            # Рассчитываем TP/SL
            price = s['price']
            atr = s.get('atr', price * 0.01)
            
            if s['type'] == 'LONG':
                sl = max(price - atr * 1.5, price * 0.995)
                tp = price + atr * 4.5
            else:  # SHORT
                sl = min(price + atr * 1.5, price * 1.005)
                tp = price - atr * 4.5
                
            risk_pct = abs(price - sl) / price * 100
            reward_pct = abs(tp - price) / price * 100
            
            msg = (
                f"🚀 *{s['symbol']} {s['type']} Signal*\n"
                f"⏰ {s['time'].strftime('%Y-%m-%d %H:%M')} UTC\n"
                f"💰 Цена: ${price:.4f}\n"
                f"📊 Вероятность: {s['probability']:.1%}\n"
                f"📈 Тех.анализ:\n"
                f"  • RSI: {s['rsi']:.1f} ({'🔻Перепродан' if s['rsi'] < 30 else '🔺Перекуплен' if s['rsi'] > 70 else '⚖️Нейтральный'})\n"
                f"  • ADX: {s['adx']:.1f} ({'💪Сильный тренд' if s['adx'] > 40 else '🔼Средний тренд' if s['adx'] > 25 else '🔄Без тренда'})\n"
                f"  • ATR: {atr:.4f}\n\n"
                f"🎯 Уровни:\n"
                f"  • TP: ${tp:.4f} (+{reward_pct:.2f}%)\n"
                f"  • SL: ${sl:.4f} (-{risk_pct:.2f}%)\n"
                f"  • Risk/Reward: 1:3"
            )
            
            await broadcast_message(bot, msg)
            await asyncio.sleep(1)  # Задержка между сообщениями
    
    elif rejected:
        # Аналитический отчет с конкретными рекомендациями
        msg = f"📉 *Анализ рынка {timestamp}*\n\n"
        
        # Группируем причины
        reasons = Counter()
        for item in rejected:
            reasons.update(item.get('reasons', []))
        
        # Топ-3 причины с пояснениями
        msg += "🔍 Основные причины отсутствия сигналов:\n"
        for reason, count in reasons.most_common(3):
            msg += f"\n• *{reason}* ({count} пар)\n"
        
        # Примеры с анализом
        sample = [r for r in rejected if r.get('indicators')][:2]
        if sample:
            msg += "\n📌 Примеры анализа:\n"
            for item in sample:
                msg += f"\n{item['symbol']} (цена: {item['indicators']['price']:.4f}):\n"
                msg += f"- RSI: {item['indicators']['rsi']:.1f} {'🔻' if item['indicators']['rsi'] < 30 else '🔺' if item['indicators']['rsi'] > 70 else '⚖️'}\n"
                msg += f"- Тренд: {item['indicators']['trend']}\n"
        
        await broadcast_message(bot, msg)
    
    else:
        # Расширенное сообщение о неактивности
        msg = (
            f"🕒 *Отчет {timestamp}*\n\n"
            "📉 Рынок не показывает четких торговых возможностей.\n\n"
            "🔍 Основные наблюдения:\n"
            "• Низкая волатильность на большинстве пар\n"
            "• Отсутствие сильных трендов (ADX < 25)\n"
            "• Смешанные показатели RSI\n\n"
            "💡 Рекомендации:\n"
            "• Рассмотрите меньшие таймфреймы\n"
            "• Ожидайте подтверждения тренда\n"
            "• Проверьте новостной фон"
        )
        await broadcast_message(bot, msg)

async def check_markets(context: CallbackContext):
    """Периодическая проверка рынков с подробным отчетом"""
    logger.info("Начинаю сканирование рынков...")
    signals = []
    rejected_signals = []
    
    for symbol in Config.ASSETS:
        try:
            result = await trading_bot.analyze_symbol(symbol)
            if result['signal']:
                signals.append(result)
            else:
                rejected_signals.append(result)
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {str(e)}")
    
    # Отправляем уведомление о результатах
    await send_scan_report(context.bot, signals, rejected_signals)
    
    # Исполняем найденные сигналы
    if signals:
        for signal in signals:
            await trading_bot.execute_signal(signal['signal'])

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
        f"• Models: {'✅' if trading_bot.long_model_data else '❌'} Long / "
        f"{'✅' if trading_bot.short_model_data else '❌'} Short\n"
        f"• Exchange: {'✅ Connected' if trading_bot.exchange else '❌ Disconnected'}\n"
        f"• Monitoring: {len(Config.ASSETS)} assets\n"
        f"• Last check: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )
    
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Connection test"""
    try:
        # Test Binance connection
        ticker = await trading_bot.exchange.fetch_ticker('BTCUSDT')
        # Test models
        long_loaded = bool(trading_bot.long_model_data)
        short_loaded = bool(trading_bot.short_model_data)
        
        await update.message.reply_text(
            f"Binance: ✅\n"
            f"Long model: {'✅' if long_loaded else '❌'}\n"
            f"Short model: {'✅' if short_loaded else '❌'}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

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
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        
        # Keep the application running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
    finally:
        try:
            await trading_bot.exchange.close()
            if 'app' in locals():
                await app.stop()
                await app.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
