import joblib
import os

# Определение базовой директории (директория, где находится скрипт)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Определение путей к файлам
MODEL_PATH_LONG = os.path.join(BASE_DIR, 'model_improved1.pkl')
MODEL_PATH_SHORT = os.path.join(BASE_DIR, 'short_model_improved.pkl')
FEATURES_PATH_LONG = os.path.join(BASE_DIR, 'features.pkl')
FEATURES_PATH_SHORT = os.path.join(BASE_DIR, 'short_features.pkl')

# Списки признаков
long_features = [
    'price_change_1h', 'price_change_2h', 'price_change_3h', 'price_change_4h',
    'price_change_6h', 'price_change_8h', 'price_change_12h', 'volume_score',
    'volume_change', 'atr_normalized', 'rsi', 'macd', 'vwap_signal', 'obv',
    'adx', 'bb_upper', 'bb_lower', 'bb_width', 'support_level', 'resistance_level',
    'sentiment', 'smart_money_score', 'ema_cross', 'volume_spike', 'super_trend',
    'vwap_angle', 'bull_volume'
]

short_features = [
    'price_change_1h', 'price_change_2h', 'price_change_3h', 'price_change_4h',
    'price_change_6h', 'price_change_8h', 'price_change_12h', 'volume_score',
    'volume_change', 'atr_normalized', 'rsi', 'macd', 'vwap_signal', 'obv',
    'adx', 'bb_upper', 'bb_lower', 'bb_width', 'support_level', 'resistance_level',
    'sentiment', 'smart_money_score', 'ema_cross', 'volume_spike', 'super_trend',
    'vwap_angle', 'bear_volume', 'atr_change', 'price_to_resistance'
]

# Сохранение списков признаков
try:
    joblib.dump(long_features, FEATURES_PATH_LONG)
    joblib.dump(short_features, FEATURES_PATH_SHORT)
    print(f"Файлы признаков успешно сохранены в:\n{FEATURES_PATH_LONG}\n{FEATURES_PATH_SHORT}")
except Exception as e:
    print(f"Ошибка при сохранении файлов признаков: {e}")
