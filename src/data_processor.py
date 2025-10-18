import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from orderbook_snapshot_autoencoder.integration import encode_snapshot

target_label_names = ['mid_price', 'spread']

def prepare_dataframe(df: pd.DataFrame, pca_components: int = None) -> pd.DataFrame:
    print("Preparing dataframe...")
    
    df['json_data'] = df['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df['mid_price'] = df['json_data'].apply(lambda x: x.get('mid_price'))
    df['asks'] = df['json_data'].apply(lambda x: x.get('asks'))
    df['bids'] = df['json_data'].apply(lambda x: x.get('bids'))
    df['best_ask_price'] = df['asks'].apply(lambda x: x[0].get('price'))
    df['best_bid_price'] = df['bids'].apply(lambda x: x[0].get('price'))
    df['spread'] = df['best_ask_price'] - df['best_bid_price']
    
    # エンコードされたスナップショット特徴量を取得
    print("Encoding snapshots...")
    encoded_features = df['json_data'].apply(lambda x: encode_snapshot(x, pca_components).numpy())
    
    # エンコードされた特徴量をDataFrameの列として追加
    feature_dim = len(encoded_features.iloc[0])
    feature_columns = [f'encoded_feature_{i}' for i in range(feature_dim)]
    encoded_df = pd.DataFrame(encoded_features.tolist(), columns=feature_columns, index=df.index)
    
    # 元のDataFrameと結合
    df = pd.concat([df, encoded_df], axis=1)
    
    df = df.drop(columns=['timestamp', 'json_data', 'asks', 'bids', 'best_ask_price', 'best_bid_price'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df