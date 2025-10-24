import os
import argparse
import json
from importlib import resources

from dotenv import load_dotenv
load_dotenv()

from .tft_trainer import TFTTrainer, load_tft_model

model_path = os.getenv('MODEL_PATH', 'models')

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    
    parser.add_argument('--preset', type=str, choices=['minimum', 'dev', 'prod'], required=False, default='prod', help='Preset configuration to use.')
    parser.add_argument('--data_path', type=str, required=False, default='csv/board_snapshots.csv', help='Path to the training data.')
    parser.add_argument('--epochs', type=int, required=False, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate for the optimizer.')
    parser.add_argument('--hidden_size', type=int, required=False, help='Hidden size for the model.')
    parser.add_argument('--attention_head_size', type=int, required=False, default=4, help='Attention head size for the model.')
    parser.add_argument('--dropout', type=float, required=False, default=0.1, help='Dropout rate for the model.')
    parser.add_argument('--hidden_continuous_size', type=int, required=False, default=16, help='Hidden continuous size for the model.')
    parser.add_argument('--window_size', type=int, required=False, default=60, help='Window size for the model.')
    parser.add_argument('--max_prediction_length', type=int, required=False, default=12, help='Maximum prediction length for the model.')
    parser.add_argument('--pca_components', type=int, required=False, help='Number of PCA components to use.')
    parser.add_argument('--log_interval', type=int, required=False, default=10, help='Logging interval during training.')
    parser.add_argument('--mode', type=str, choices=['train_and_eval', 'train', 'eval'], required=False, default='train_and_eval', help='Mode: train or evaluate.')
    
    args = parser.parse_args()
    
    with resources.open_text("orderbook_tft", "presets.json") as f:
        preset = json.load(f).get(args.preset)
    
    args = {**preset, **{k: v for k, v in vars(args).items() if v is not None}}
    
    print("Using configuration:")
    for key, value in args.items():
        print(f"{key}: {value}")
    
    return args

def main() -> None:
    args = parse_args()
    mode = args.get('mode')
    
    trainer = TFTTrainer(**args, model_path=model_path)
    
    trained_model = None
    
    if mode in ['train', 'train_and_eval']:
        trained_model = trainer.train()
    
    if mode in ['eval', 'train_and_eval']:
        if trained_model is None:
            trained_model = load_tft_model(model_path)
        
        trainer.evaluate(trained_model)

if __name__ == "__main__":
    main()