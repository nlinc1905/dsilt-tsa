from fastapi import FastAPI, File, Form, UploadFile
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd

from data_processor import Processor
from models import LogisticRegression, DeepLSTM, DeepBidirectionalLSTM, DeepTransformer


# hardcoded parameters - could use environment vars instead
CONFIG = {
    'seed': 14,
    'learning_rate': 1e-3,
    'epochs': 1,
    'lstm_hidden_layers': 1,
    'lstm_hidden_layer_units': [10],
    'transformer_embed_dim': 32,
    'transformer_blocks': 1,
    'transformer_heads': 2,
    'transformer_units': 32,
    'time_steps_trained_on': 187,
    'features_trained_on': 1,
    'best_model': "Bidirectinoal LSTM",
}


app = FastAPI()


class Prediction(BaseModel):
    filename: str
    prediction: List[float] = []
    prediction_probability: List[float] = []
    model: str


def get_preds_from_model(file, model=None):
    """
    Gets predictions for the specified model

    :param file: file from post request
    :param model: model name, defaults to None, which defaults to the best model from the
        validation data evaluation

    :return: predictions and prediction probabilities
    """
    df = pd.read_csv(file.file, header=None, low_memory=False)
    p = Processor(seed=CONFIG['seed'], raw_data_dir="data/raw", artifact_dir="data/artifacts")
    df = p.normalize_data(scaler_dir="data/artifacts/", X=df.values[:, :CONFIG['time_steps_trained_on']])
    df = p.reshape_data(X=df)

    if model == "Logistic Regression":
        lr = LogisticRegression(seed=CONFIG['seed'], checkpoint_path="model_tracking/lr.ckpt")
        lr.build(
            nbr_time_steps=CONFIG['time_steps_trained_on'],
            loss='binary_crossentropy',
            learning_rate=CONFIG['learning_rate'],
            metrics_list=['accuracy']
        )
        lr.load_saved_weights()
        return lr.make_predictions(X=df)
    
    elif model == "LSTM":
        lstm = DeepLSTM(seed=CONFIG['seed'], checkpoint_path="model_tracking/lstm.ckpt")
        lstm.build(
            nbr_time_steps=CONFIG['time_steps_trained_on'],
            nbr_features=CONFIG['features_trained_on'],
            loss='binary_crossentropy',
            learning_rate=CONFIG['learning_rate'],
            metrics_list=['accuracy'],
            nbr_hidden_layers=CONFIG['lstm_hidden_layers'],
            hidden_layer_units=CONFIG['lstm_hidden_layer_units'],
        )
        lstm.load_saved_weights()
        return lstm.make_predictions(X=df)
    
    elif model == "Transformer":
        trns = DeepTransformer(seed=CONFIG['seed'], checkpoint_path="model_tracking/trns.ckpt")
        trns.build(
            max_seq_len=CONFIG['time_steps_trained_on'],  # set the max to however many there are (don't limit it)
            nbr_features=CONFIG['features_trained_on'],
            # embed_dim determines units in dense layer for input values, and dims for positional embeddings
            embed_dim=CONFIG['transformer_embed_dim'],
            loss='binary_crossentropy',
            learning_rate=CONFIG['learning_rate'],
            metrics_list=['accuracy'],
            nbr_transformer_blocks=CONFIG['transformer_blocks'],
            nbr_attention_heads_each_block=CONFIG['transformer_heads'],
            nbr_dense_units_each_block=CONFIG['transformer_units'],
        )
        trns.load_saved_weights()
        return trns.make_predictions(X=df)
    
    # default to bidirectional LSTM because it performed the best on the validation data
    else:
        bidlstm = DeepBidirectionalLSTM(seed=CONFIG['seed'], checkpoint_path="model_tracking/bidlstm.ckpt")
        bidlstm.build(
            nbr_time_steps=CONFIG['time_steps_trained_on'],
            nbr_features=CONFIG['features_trained_on'],
            loss='binary_crossentropy',
            learning_rate=CONFIG['learning_rate'],
            metrics_list=['accuracy'],
            nbr_hidden_layers=CONFIG['lstm_hidden_layers'],
            hidden_layer_units=CONFIG['lstm_hidden_layer_units'],
        )
        bidlstm.load_saved_weights()
        return bidlstm.make_predictions(X=df)


@app.post("/", response_model=Prediction)
async def upload_and_get_preds(file: UploadFile = File(...), model_name: Optional[str] = Form(None)):
    pred_probs, preds = get_preds_from_model(file=file, model=model_name)
    return {
        "filename": file.filename,
        "prediction": list(preds),
        "prediction_probability": list(pred_probs),
        "model": model_name if model_name is not None else CONFIG['best_model'],
    }
