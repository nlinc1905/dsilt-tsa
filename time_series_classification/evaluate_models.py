import json
import random
import numpy as np
import tensorflow as tf

from data_processor import Processor
from models import LogisticRegression, DeepLSTM, DeepBidirectionalLSTM, DeepTransformer


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
}
with open("model_tracking/models.config", "w") as file:
    json.dump(CONFIG, file)

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

p = Processor(seed=CONFIG['seed'], raw_data_dir="data/raw", artifact_dir="data/artifacts")
x_train, x_test, y_train, y_test = p.prepare_training_data()

lr = LogisticRegression(seed=CONFIG['seed'], checkpoint_path="model_tracking/lr.ckpt")
lr.build(
    nbr_time_steps=x_train.shape[1],
    loss='binary_crossentropy',
    learning_rate=CONFIG['learning_rate'],
    metrics_list=['accuracy']
)
lr.train(X=x_train, Y=y_train, epochs=CONFIG['epochs'])
lr.evaluate(y_train=y_train, y_test=y_test, x_train=x_train, x_test=x_test, plot_roc=True)

lstm = DeepLSTM(seed=CONFIG['seed'], checkpoint_path="model_tracking/lstm.ckpt")
lstm.build(
    nbr_time_steps=x_train.shape[1],
    nbr_features=x_train.shape[2],
    loss='binary_crossentropy',
    learning_rate=CONFIG['learning_rate'],
    metrics_list=['accuracy'],
    nbr_hidden_layers=CONFIG['lstm_hidden_layers'],
    hidden_layer_units=CONFIG['lstm_hidden_layer_units'],
)
lstm.train(X=x_train, Y=y_train, epochs=1)
lstm.evaluate(y_train=y_train, y_test=y_test, x_train=x_train, x_test=x_test, plot_roc=True)

bidlstm = DeepBidirectionalLSTM(seed=CONFIG['seed'], checkpoint_path="model_tracking/bidlstm.ckpt")
bidlstm.build(
    nbr_time_steps=x_train.shape[1],
    nbr_features=x_train.shape[2],
    loss='binary_crossentropy',
    learning_rate=CONFIG['learning_rate'],
    metrics_list=['accuracy'],
    nbr_hidden_layers=CONFIG['lstm_hidden_layers'],
    hidden_layer_units=CONFIG['lstm_hidden_layer_units'],
)
bidlstm.train(X=x_train, Y=y_train, epochs=1)
bidlstm.evaluate(y_train=y_train, y_test=y_test, x_train=x_train, x_test=x_test, plot_roc=True)

trns = DeepTransformer(seed=CONFIG['seed'], checkpoint_path="model_tracking/trns.ckpt")
trns.build(
    max_seq_len=x_train.shape[1],  # set the max to however many there are (don't limit it)
    nbr_features=x_train.shape[2],
    # embed_dim determines units in dense layer for input values, and dims for positional embeddings
    embed_dim=CONFIG['transformer_embed_dim'],
    loss='binary_crossentropy',
    learning_rate=CONFIG['learning_rate'],
    metrics_list=['accuracy'],
    nbr_transformer_blocks=CONFIG['transformer_blocks'],
    nbr_attention_heads_each_block=CONFIG['transformer_heads'],
    nbr_dense_units_each_block=CONFIG['transformer_units'],
)

trns.train(X=x_train, Y=y_train, epochs=1)
trns.evaluate(y_train=y_train, y_test=y_test, x_train=x_train, x_test=x_test, plot_roc=True)
