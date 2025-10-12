import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

load_dotenv()

import torch
torch.set_float32_matmul_precision('high')

from trainer import TFTTrainer

def predict(x):
    model_path = os.getenv('MODEL_PATH', 'models')
    model = TFTTrainer.load_model(model_path)
    model.eval()
    with torch.no_grad():
        predictions = model.predict(x, mode="raw")
    return predictions

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from pytorch_forecasting import TimeSeriesDataSet
    
    # Example usage
    batch_size = 4
    max_encoder_length = 60
    max_prediction_length = 12
    n_time = max_encoder_length + max_prediction_length

    data = []
    for t in range(n_time):
        data.append({
            "series_id": 0,
            "time_idx": t,
            "1period_log_return": np.float32(0),
            "12period_log_return": np.float32(0),
        })
    df = pd.DataFrame(data)
    
    model_path = os.getenv('MODEL_PATH', 'models')
    model = TFTTrainer.load_model(model_path)
    hparams = model.hparams['dataset_parameters']

    dataset = TimeSeriesDataSet.from_parameters(hparams, df, predict=True)
    dataloader = dataset.to_dataloader(train=False, batch_size=batch_size, shuffle=False, num_workers=0)

    predictions = predict(dataloader)

    print("predictions:", predictions.prediction[0])
    # print("predictions shape:", preds)