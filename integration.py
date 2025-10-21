import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

import torch
torch.set_float32_matmul_precision('high')

from tft_trainer import load_tft_model

model_path = os.getenv('MODEL_PATH', 'models')
tft = load_tft_model(model_path)

def predict(dataloader) -> List[torch.Tensor]:
    tft.eval()
    with torch.no_grad():
        predictions = tft.predict(dataloader, mode='quantiles')
    return predictions

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from pytorch_forecasting import TimeSeriesDataSet
    
    from data_processor import prepare_dataframe
    
    # Example usage
    batch_size = 1
    
    hparams = tft.hparams
    
    dataset_parameters = hparams.get('dataset_parameters')
    max_encoder_length = dataset_parameters.get('max_encoder_length')
    max_prediction_length = dataset_parameters.get('max_prediction_length')
    time_varying_unknown_reals = dataset_parameters.get('time_varying_unknown_reals')
    
    df = pd.read_csv('csv/board_snapshots.csv')
    df = prepare_dataframe(df)
    df = df.filter(items=time_varying_unknown_reals)
    df['time_idx'] = range(len(df))
    df['series_id'] = 0
    df = df[lambda x: x.time_idx > (x.time_idx.max() - max_encoder_length)]
    
    df = pd.concat([df, pd.DataFrame({
        'time_idx': range(df['time_idx'].max() + 1, df['time_idx'].max() + 1 + max_prediction_length),
        'series_id': 0,
        **{col: np.float32(0) for col in time_varying_unknown_reals},
    })], ignore_index=True)
    
    print(f"Dataframe shape: {df.shape}")
    print(df)
    
    dataset = TimeSeriesDataSet.from_parameters(dataset_parameters, df, predict=True)
    print(f"Number of samples in the dataset: {len(dataset)}")
    
    dataloader = dataset.to_dataloader(train=False, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = predict(dataloader)
    print(f"Predictions shape: {predictions[0].shape} (batch_size, prediction_length, num_quantiles)")
    print(predictions)