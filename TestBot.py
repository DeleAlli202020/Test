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
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
import joblib
from ccxt.async_support import binance
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
            df['atr'] = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range().fillna(0)
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
        """Model evaluation with robust ADX handling"""
        try:
            model_data = self.short_model_data if is_short else self.long_model_data
            if not model_data:
                return None
                
            model = model_data['models'].get('combined')
            scaler = model_data['scalers'].get('combined')
            
            if model is None or scaler is None:
                return None
                
            features = self.prepare_features(df, is_short)
            if features.empty:
                return None
                
            # Ensure ADX is present and valid
            if 'adx' not in features.columns:
                logger.warning("ADX feature missing in prepared features")
                features['adx'] = 0
                
            features_scaled = scaler.transform(features.values.astype(np.float32))
            proba = model.predict_proba(features_scaled)[0][1]
            
            threshold = Config.SHORT_THRESHOLD if is_short else Config.LONG_THRESHOLD
            if symbol in Config.LOW_RECALL_ASSETS:
                threshold *= Config.LOW_RECALL_MULTIPLIER
                
            last = df.iloc[-1]
            if proba > threshold and self._check_conditions(last, is_short):
                return {
                    'symbol': symbol,
                    'type': 'SHORT' if is_short else 'LONG',
                    'probability': proba,
                    'price': last['close'],
                    'rsi': last.get('rsi', 50),
                    'adx': last.get('adx', 0),  # Default to 0 if ADX missing
                    'atr': last.get('atr', 0),
                    'time': datetime.utcnow(),
                    'model': 'short' if is_short else 'long'
                }
                
        except Exception as e:
            logger.error(f"Model evaluation failed for {symbol}: {str(e)}")
            
        return None
    
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
            
            await self.broadcast_message(message)
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
    


    async def analyze_symbol(self, symbol: str) -> dict:
        """Анализ символа с подробной диагностикой"""
        result = {
            'symbol': symbol,
            'signal': None,
            'reasons': [],
            'indicators': {}
        }
        
        try:
            # Получаем данные
            df = await self.fetch_market_data(symbol)
            if df is None or len(df) < 48:
                result['reasons'].append("Недостаточно данных")
                return result
            
            # Рассчитываем индикаторы
            df = self.calculate_indicators(df)
            
            # Сохраняем значения индикаторов для отчета
            result['indicators'] = {
                'price': df['close'].iloc[-1],
                'rsi': round(df['rsi'].iloc[-1], 2),
                'adx': round(df['adx'].iloc[-1], 2),
                'volume': df['volume'].iloc[-1],
                'trend': "Вверх" if df['adx'].iloc[-1] > 25 and df['dip'].iloc[-1] > df['din'].iloc[-1] else 
                        "Вниз" if df['adx'].iloc[-1] > 25 else "Без тренда"
            }
            
            # Проверяем LONG и SHORT сигналы
            long_result = await self._evaluate_model_signal(df, symbol, False)
            short_result = await self._evaluate_model_signal(df, symbol, True)
            
            if long_result and long_result['probability'] > Config.LONG_THRESHOLD:
                result['signal'] = long_result
            elif short_result and short_result['probability'] > Config.SHORT_THRESHOLD:
                result['signal'] = short_result
            else:
                # Анализируем причины отклонения
                if long_result:
                    result['reasons'].append(
                        f"LONG: вероятность {long_result['probability']:.1%} < порога {Config.LONG_THRESHOLD:.1%}"
                    )
                if short_result:
                    result['reasons'].append(
                        f"SHORT: вероятность {short_result['probability']:.1%} < порога {Config.SHORT_THRESHOLD:.1%}"
                    )
                
                # Проверка дополнительных условий
                last = df.iloc[-1]
                if last['adx'] < 15:
                    result['reasons'].append("Слабый тренд (ADX < 15)")
                if last['volume'] < last['volume'].rolling(20).mean().iloc[-1] * 0.7:
                    result['reasons'].append("Объем ниже среднего")
                    
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {str(e)}")
            result['reasons'].append(f"Ошибка анализа: {str(e)}")
        
        return result
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
    """Отправка подробного отчета о сканировании"""
    # Отчет о найденных сигналах
    if signals:
        message = "🔍 *Результаты сканирования рынков*\n\n"
        message += f"🕒 Время: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        message += f"📊 Найдено сигналов: {len(signals)}\n\n"
        
        for signal in signals[:5]:  # Ограничим количество в сообщении
            s = signal['signal']
            message += (
                f"🚀 *{s['symbol']} {s['type']}*\n"
                f"• Вероятность: {s['probability']:.1%}\n"
                f"• Цена: {s['price']:.4f}\n"
                f"• RSI: {s['rsi']:.1f}\n"
                f"• ADX: {s['adx']:.1f}\n\n"
            )
        
        await broadcast_message(bot, message)
    
    # Отчет об отклоненных возможностях
    if rejected and len(signals) == 0:
        analysis_msg = "📊 *Анализ отклоненных сделок*\n\n"
        analysis_msg += f"Проверено пар: {len(rejected)}\n"
        analysis_msg += "Основные причины отклонения:\n"
        
        # Статистика по причинам
        reasons = {}
        for item in rejected:
            for reason in item.get('reasons', []):
                reasons[reason] = reasons.get(reason, 0) + 1
        
        # Топ-5 причин
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            analysis_msg += f"• {reason}: {count} пар\n"
        
        # Примеры отклоненных пар
        sample = [r for r in rejected if r.get('reasons')][:3]
        if sample:
            analysis_msg += "\nПримеры:\n"
            for item in sample:
                analysis_msg += f"{item['symbol']}: {', '.join(item['reasons'][:2])}\n"
        
        await broadcast_message(bot, analysis_msg)
    
    # Если вообще ничего не найдено
    if not signals and not rejected:
        await broadcast_message(bot, "🔍 Сканирование завершено. Рынок неактивен.")

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
