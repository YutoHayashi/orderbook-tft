import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

target_label_names = ['mid_price', 'spread']

def validate_dataframe(df: pd.DataFrame) -> bool:
    required_columns = ['timestamp', 'json_data']
    print("Validating dataframe...")
    print(f"Columns in dataframe: {df.columns.tolist()} \nRequired columns: {required_columns}")
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    print("Dataframe validation passed.")
    return True

def extract_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df['best_ask_price'] = df['asks'].apply(lambda x: x[0].get('price'))
    df['best_bid_price'] = df['bids'].apply(lambda x: x[0].get('price'))
    df['best_ask_size'] = df['asks'].apply(lambda x: x[0].get('size'))
    df['best_bid_size'] = df['bids'].apply(lambda x: x[0].get('size'))
    df['spread'] = df['best_ask_price'] - df['best_bid_price']
    return df
    
def extract_depth_features(df: pd.DataFrame, depth_level: int = 10) -> pd.DataFrame:
    df['ask_depth'] = df['asks'].apply(lambda asks: sum([ask.get('size', 0) for ask in asks[depth_level:]]))
    df['bid_depth'] = df['bids'].apply(lambda bids: sum([bid.get('size', 0) for bid in bids[depth_level:]]))
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-9)
    return df

def extract_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['mid_price_rolling_std_5'] = df['mid_price'].rolling(window=5).std()
    df['mid_price_rolling_std_10'] = df['mid_price'].rolling(window=10).std()
    df['mid_price_rolling_mean_5'] = df['mid_price'].rolling(window=5).mean()
    df['mid_price_rolling_mean_10'] = df['mid_price'].rolling(window=10).mean()
    df['momentum_5'] = df['mid_price'] - df['mid_price'].shift(5)
    df['momentum_10'] = df['mid_price'] - df['mid_price'].shift(10)
    df['spread_volatility_5'] = df['spread'].rolling(window=5).std()
    df['spread_volatility_10'] = df['spread'].rolling(window=10).std()
    return df


def extract_advanced_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced price dynamics features"""
    print("Extracting price dynamics features...")
    
    # Log price
    df['log_mid_price'] = np.log(df['mid_price'])
    
    # Multi-period returns
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['mid_price'].pct_change(period)
        df[f'log_return_{period}'] = df['log_mid_price'].diff(period)
    
    # Cumulative returns
    df['cumret_5'] = (1 + df['return_1']).rolling(5).apply(lambda x: x.prod() - 1, raw=True)
    df['cumret_10'] = (1 + df['return_1']).rolling(10).apply(lambda x: x.prod() - 1, raw=True)
    
    # Price acceleration
    df['price_acceleration'] = df['return_1'].diff()
    
    # Volatility change rate
    vol_5 = df['return_1'].rolling(5).std()
    df['vol_change_5'] = vol_5.pct_change()
    
    # Price deviation (deviation from moving average)
    for window in [5, 10, 20]:
        ma = df['mid_price'].rolling(window).mean()
        df[f'price_deviation_{window}'] = (df['mid_price'] - ma) / ma
        
    # Position relative to high/low prices
    for window in [5, 10, 20]:
        high = df['mid_price'].rolling(window).max()
        low = df['mid_price'].rolling(window).min()
        df[f'price_position_{window}'] = (df['mid_price'] - low) / (high - low + 1e-10)
    
    return df

def extract_orderbook_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Order book imbalance indicators"""
    print("Extracting order book imbalance indicators...")
    
    # Multi-level imbalance
    for level in range(1, min(6, len(df['bids'].iloc[0]))):
        df[f'imbalance_level_{level}'] = df.apply(
            lambda row: (row['bids'][level-1].get('size', 0) - row['asks'][level-1].get('size', 0)) / 
                        (row['bids'][level-1].get('size', 0) + row['asks'][level-1].get('size', 0) + 1e-10), axis=1)
    
    # Weighted imbalance (price-weighted)
    def weighted_imbalance(row):
        bid_sum = sum([bid.get('size', 0) / (bid.get('price', 1) + 1e-10) for bid in row['bids'][:5]])
        ask_sum = sum([ask.get('size', 0) / (ask.get('price', 1) + 1e-10) for ask in row['asks'][:5]])
        return (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-10)
    df['weighted_imbalance'] = df.apply(weighted_imbalance, axis=1)
    
    # Spread-normalized imbalance
    spread = df['best_ask_price'] - df['best_bid_price']
    df['spread_normalized_imbalance'] = df['depth_imbalance'] / (spread + 1e-10)
    
    # Time-varying imbalance
    df['imbalance_velocity'] = df['depth_imbalance'].diff()
    df['imbalance_acceleration'] = df['imbalance_velocity'].diff()
    
    # Imbalance persistence
    for window in [3, 5, 10]:
        df[f'imbalance_persistence_{window}'] = (
            df['depth_imbalance'].rolling(window).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5, raw=True)
        )
    
    return df

def extract_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Microstructure features"""
    print("Extracting microstructure features...")
    
    # Effective spread
    df['effective_spread'] = 2 * abs(df['mid_price'] - (df['best_bid_price'] + df['best_ask_price']) / 2)
    
    # Various spread statistics
    for window in [5, 10]:
        df[f'spread_volatility_{window}'] = df['spread'].rolling(window).std()
        df[f'spread_skew_{window}'] = df['spread'].rolling(window).skew()
        df[f'spread_kurt_{window}'] = df['spread'].rolling(window).kurt()
    
    # Price impact indicator
    volume_proxy = df['best_bid_size'] + df['best_ask_size']
    df['price_impact'] = abs(df['return_1']) / (volume_proxy + 1e-10)
    
    # Market efficiency indicator (deviation from random walk)
    def market_efficiency(returns, window):
        if len(returns) < window:
            return 0
        # Simplified Hurst exponent
        log_rs = []
        for lag in range(2, min(window//2, 10)):
            if lag >= len(returns):
                continue
            mean_return = returns.mean()
            cumdev = (returns - mean_return).cumsum()
            r = cumdev.max() - cumdev.min()
            s = returns.std()
            if s > 0:
                log_rs.append(np.log(r/s))
        if len(log_rs) < 2:
            return 0.5
        x = np.log(range(2, len(log_rs) + 2))
        slope, _, _, _, _ = stats.linregress(x, log_rs)
        return slope
    
    for window in [10, 20]:
        df[f'market_efficiency_{window}'] = df['return_1'].rolling(window).apply(
            lambda x: market_efficiency(x, window), raw=False)
    
    # Order size distribution features
    def size_distribution_features(orders):
        sizes = [order.get('size', 0) for order in orders[:5]]
        if len(sizes) == 0:
            return 0, 0, 0
        sizes = np.array(sizes)
        return np.mean(sizes), np.std(sizes), stats.skew(sizes) if len(sizes) > 2 else 0
    
    bid_stats = df['bids'].apply(size_distribution_features)
    ask_stats = df['asks'].apply(size_distribution_features)
    
    df['bid_size_mean'] = [x[0] for x in bid_stats]
    df['bid_size_std'] = [x[1] for x in bid_stats]
    df['bid_size_skew'] = [x[2] for x in bid_stats]
    df['ask_size_mean'] = [x[0] for x in ask_stats]
    df['ask_size_std'] = [x[1] for x in ask_stats]
    df['ask_size_skew'] = [x[2] for x in ask_stats]
    
    return df

def extract_temporal_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time series pattern features"""
    print("Extracting time series pattern features...")
    
    # Periodicity detection (FFT-based)
    def dominant_frequency(series, window):
        if len(series) < window or window < 4:
            return 0
        fft = np.fft.fft(series.values)
        freqs = np.fft.fftfreq(len(fft))
        # Return the strongest frequency
        # Return the strongest frequency
        magnitude = np.abs(fft[1:len(fft)//2])  # Exclude DC component
        if len(magnitude) == 0:
            return 0
        return freqs[1:len(freqs)//2][np.argmax(magnitude)]
    
    for window in [20, 40]:
        df[f'dominant_freq_{window}'] = df['return_1'].rolling(window).apply(
            lambda x: dominant_frequency(x, window), raw=False)
    
    # Trend strength
    def trend_strength(series):
        if len(series) < 3:
            return 0
        x = np.arange(len(series))
        slope, _, r_value, _, _ = stats.linregress(x, series)
        return slope * (r_value ** 2)  # Slope × R-squared
    
    for window in [5, 10, 20]:
        df[f'trend_strength_{window}'] = df['mid_price'].rolling(window).apply(
            lambda x: trend_strength(x), raw=False)
    
    # Mean reversion strength
    def mean_reversion_strength(series):
        if len(series) < 3:
            return 0
        mean_price = series.mean()
        deviations = series - mean_price
        # Autocorrelation of deviations
        if len(deviations) < 2:
            return 0
        autocorr = np.corrcoef(deviations[:-1], deviations[1:])[0,1]
        return -autocorr if not np.isnan(autocorr) else 0
    
    for window in [5, 10]:
        df[f'mean_reversion_{window}'] = df['mid_price'].rolling(window).apply(
            lambda x: mean_reversion_strength(x), raw=False)
    
    # Jump detection
    def jump_detection(returns, threshold=2):
        if len(returns) < 2:
            return 0
        std = returns.std()
        if std == 0:
            return 0
        jumps = abs(returns) > threshold * std
        return jumps.sum() / len(returns)
    
    df['jump_frequency_5'] = df['return_1'].rolling(5).apply(
        lambda x: jump_detection(x), raw=False)
    
    return df

def extract_ml_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Machine learning based features"""
    print("Extracting machine learning based features...")
    
    # Feature selection (numeric columns only, excluding rows with NaN)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols 
                    if col not in ['asks', 'bids'] + target_label_names]
        
    feature_data = df[feature_cols].dropna()
    
    print(f"    Number of features for PCA: {len(feature_cols)}, Number of valid data rows: {len(feature_data)}")
    
    if len(feature_data) > 50 and len(feature_cols) > 1:  # Only when sufficient data is available
        # PCA features
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            pca = PCA(n_components=min(5, scaled_features.shape[1]))
            pca_features = pca.fit_transform(scaled_features)
            
            # Add PCA components to original DataFrame
            pca_df = pd.DataFrame(pca_features, 
                                columns=[f'pca_{i}' for i in range(pca_features.shape[1])],
                                index=feature_data.index)
            
            for col in pca_df.columns:
                df[col] = np.nan
                df.loc[pca_df.index, col] = pca_df[col]
                
            print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
            
        except Exception as e:
            print(f"  Failed to create PCA features: {e}")
    else:
        print(f"  Skipping PCA execution: Insufficient data (rows: {len(feature_data)}, features: {len(feature_cols)})")
    
    # Interaction features (important combinations)
    interaction_pairs = [
        ('depth_imbalance', 'spread'),
        ('return_1', 'spread'),
        ('best_bid_size', 'best_ask_size'),
    ]
    
    for feature1, feature2 in interaction_pairs:
        if feature1 in df.columns and feature2 in df.columns:
            df[f'{feature1}_x_{feature2}'] = df[feature1] * df[feature2]
            df[f'{feature1}_div_{feature2}'] = df[feature1] / (df[feature2] + 1e-10)
    
    # Lag features (past values)
    important_features = ['depth_imbalance', 'spread', 'return_1']
    for feature in important_features:
        if feature in df.columns:
            for lag in [1, 2, 3]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Indicators calculated from past trends
    if 'trend_strength_5' in df.columns:
        df['future_trend_indicator'] = (
            df['trend_strength_5'].rolling(3).mean().shift(1)  # Use only information up to 1 period ago
        )
    
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("Preparing dataframe...")
    
    validate_dataframe(df)
    
    df['json_data'] = df['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df['mid_price'] = df['json_data'].apply(lambda x: x.get('mid_price'))
    df['asks'] = df['json_data'].apply(lambda x: x.get('asks'))
    df['bids'] = df['json_data'].apply(lambda x: x.get('bids'))
    df = df.drop(columns=['timestamp', 'json_data'])
    
    df = extract_base_features(df)
    df = extract_depth_features(df, depth_level=10)
    df = extract_technical_indicators(df)
    # Additional feature extraction functions
    df = extract_advanced_price_features(df)
    df = extract_orderbook_imbalance_features(df)
    df = extract_microstructure_features(df)
    df = extract_temporal_pattern_features(df)
    df = extract_ml_based_features(df)
    
    df = df.drop(columns=['asks', 'bids'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df