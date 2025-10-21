import os
from typing import List

import pandas as pd

import torch
torch.set_float32_matmul_precision('high')

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from data_processor import target_label_names, prepare_dataframe

def create_datasets(df: pd.DataFrame,
                    target_columns: List[str],
                    max_encoder_length: int,
                    max_prediction_length: int) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    df['time_idx'] = range(len(df))
    df['series_id'] = 0
    
    train_cutoff = int(df['time_idx'].max() * 0.6)
    validation_cutoff = int(df['time_idx'].max() * 0.8)
    
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= train_cutoff],
        time_idx='time_idx',
        target=target_columns,
        group_ids=['series_id'],
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_reals=df.columns.difference(['time_idx', 'series_id']).tolist(),
        target_normalizer=MultiNormalizer([GroupNormalizer(groups=['series_id']) for _ in target_columns]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
    )
    
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[lambda x: (train_cutoff < x.time_idx) & (x.time_idx <= validation_cutoff)],
        predict=False,
        stop_randomization=True
    )
    
    testing = TimeSeriesDataSet.from_dataset(
        training,
        df[lambda x: x.time_idx > validation_cutoff]
    )
    
    print(f"Number of training samples: {len(training)}")
    print(f"Number of validation samples: {len(validation)}")
    print(f"Number of testing samples: {len(testing)}")
    
    return training, validation, testing

def load_tft_model(model_path: str) -> TemporalFusionTransformer:
    print("Loading the best model from checkpoint...")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), model_path)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the specified output path.")
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded model from {checkpoint_path}")
    
    return model

class TFTTrainer:
    def __init__(self,
                 data_path: str,
                 epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.03,
                 hidden_size: int = 64,
                 quantiles = [0.1, 0.5, 0.9],
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 16,
                 window_size: int = 60,
                 max_prediction_length: int = 12,
                 pca_components: int = 40,
                 log_interval: int = 10,
                 **kwargs):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.window_size = window_size
        self.max_prediction_length = max_prediction_length
        self.pca_components = pca_components
        self.log_interval = log_interval
        self.model_path = os.getenv('MODEL_PATH')
        
        self.df = pd.read_csv(data_path)
        self.df = prepare_dataframe(self.df, pca_components=self.pca_components)
        
        self.target_columns = target_label_names
        
        self.training_dataset, self.validation_dataset, self.testing_dataset = create_datasets(
            self.df,
            target_columns=self.target_columns,
            max_encoder_length=self.window_size,
            max_prediction_length=self.max_prediction_length
        )
    
    def train(self) -> TemporalFusionTransformer:
        print("Starting training...")
        
        train_dataloader = self.training_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        val_dataloader = self.validation_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
        
        tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=[len(self.quantiles) for _ in self.target_columns],
            loss=MultiLoss([QuantileLoss(quantiles=self.quantiles) for _ in self.target_columns]),
            log_interval=self.log_interval,
            reduce_on_plateau_patience=3,
            optimizer="Adam",
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=10,
            verbose=False,
            mode="min"
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.path.dirname(__file__), self.model_path),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        
        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stopping_callback, checkpoint_callback],
            logger=False
        )
        
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        return tft
    
    def evaluate(self, model: TemporalFusionTransformer) -> None:
        print("Evaluating model...")
        model.eval()
        
        trainer = Trainer(
            logger=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
        )
        
        test_dataloader = self.testing_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
        
        results = trainer.test(model, dataloaders=test_dataloader, verbose=False)
        
        print(f"===== Evaluation Results =====")
        for test_name, result in results[0].items():
            print(f"{test_name} results: {result}")
        print(f"==============================")