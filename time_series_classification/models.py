import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional,
    Layer, Dropout, MultiHeadAttention, LayerNormalization,
    Embedding, GlobalAveragePooling2D, GlobalAveragePooling1D
)
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, roc_curve,
    confusion_matrix
)


class LogisticRegression:
    def __init__(self, seed, checkpoint_path):
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.training_hist = None

    def build(self, nbr_time_steps, loss, learning_rate, metrics_list):
        i = Input(shape=(nbr_time_steps,))
        x = Dense(1, activation='sigmoid')(i)
        self.model = Model(i, x)
        self.model.compile(
            loss=loss,
            optimizer=SGD(lr=learning_rate),
            metrics=metrics_list
        )
        print(self.model.summary())

    def load_saved_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def train(self, X, Y, epochs):
        # callback to monitor training and save weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        # split time series into train/validation sets at a chosen time step
        # here the data is simply split in half (first half train, later half validation)
        nbr_samples = X.shape[0]
        self.training_hist = self.model.fit(
            X[:-nbr_samples // 2], Y[:-nbr_samples // 2],
            epochs=epochs,
            validation_data=(X[-nbr_samples // 2:], Y[-nbr_samples // 2:]),
            callbacks=[cp_callback],
        )
        return self.training_hist

    def plot_training_metrics(self):
        if self.model or self.training_hist is None:
            raise Exception("Must train the model before plotting.")
        else:
            # plot the training loss
            plt.plot(self.training_hist.history['loss'], label='Training Loss')
            plt.plot(self.training_hist.history['val_loss'], label='Validation Loss')
            plt.title("Logistic Regression Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            # plot the training accuracy
            plt.plot(self.training_hist.history['accuracy'], label='Training Accuracy')
            plt.plot(self.training_hist.history['val_accuracy'], label='Validation Accuracy')
            plt.title("Logistic Regression Classification Accuracy by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    def make_predictions(self, X):
        pred_probs = self.model.predict(X)
        preds = np.where(pred_probs >= 0.5, 1, 0)
        return pred_probs, preds

    def evaluate(self, y_train, y_test, x_train=None, x_test=None, plot_roc=False):
        train_pred_probs, train_preds = self.make_predictions(X=x_train)
        test_pred_probs, test_preds = self.make_predictions(X=x_test)

        # print evaluation
        print("Logistic Regression Test Set Metrics")
        print("Accuracy:", accuracy_score(y_test, test_preds))
        print("F1 Score:", f1_score(y_test, test_preds))
        print("Precision:", precision_score(y_test, test_preds))
        print("Recall:", recall_score(y_test, test_preds))
        print("ROC AUC:", roc_auc_score(y_test, test_preds))
        print("Confusion Matrix Format: \n[TN, FP]\n[FN, TP]")
        print(confusion_matrix(y_test, test_preds))

        if plot_roc:
            # plot the ROC curve
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_pred_probs)
            fpr, tpr, thresholds = roc_curve(y_test, test_pred_probs)

            plt.plot(
                train_fpr, train_tpr, color='darkorange',
                label=f"Train ROC with AUC = {round(roc_auc_score(y_train, train_preds), 2)}"
            )
            plt.plot(
                fpr, tpr, color='green',
                label=f"Test ROC with AUC = {round(roc_auc_score(y_test, test_preds), 2)}"
            )
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Logistic Regression Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()


class DeepLSTM:
    def __init__(self, seed, checkpoint_path):
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.training_hist = None

    def build(self, nbr_time_steps, nbr_features, loss, learning_rate, metrics_list,
              nbr_hidden_layers=1, hidden_layer_units=[10]):
        i = Input(shape=(nbr_time_steps, nbr_features))
        for layer in range(nbr_hidden_layers):
            if layer == 0:
                x = LSTM(hidden_layer_units[layer])(i)
            else:
                x = LSTM(hidden_layer_units[layer])(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(i, x)
        self.model.compile(
            loss=loss,
            optimizer=Adam(lr=learning_rate),
            metrics=metrics_list
        )
        print(self.model.summary())

    def load_saved_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def train(self, X, Y, epochs):
        # callback to monitor training and save weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        # split time series into train/validation sets at a chosen time step
        # here the data is simply split in half (first half train, later half validation)
        nbr_samples = X.shape[0]
        self.training_hist = self.model.fit(
            X[:-nbr_samples // 2], Y[:-nbr_samples // 2],
            epochs=epochs,
            validation_data=(X[-nbr_samples // 2:], Y[-nbr_samples // 2:]),
            callbacks=[cp_callback],
        )
        return self.training_hist

    def plot_training_metrics(self):
        if self.model or self.training_hist is None:
            raise Exception("Must train the model before plotting.")
        else:
            # plot the training loss
            plt.plot(self.training_hist.history['loss'], label='Training Loss')
            plt.plot(self.training_hist.history['val_loss'], label='Validation Loss')
            plt.title("LSTM Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            # plot the training accuracy
            plt.plot(self.training_hist.history['accuracy'], label='Training Accuracy')
            plt.plot(self.training_hist.history['val_accuracy'], label='Validation Accuracy')
            plt.title("LSTM Classification Accuracy by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    def make_predictions(self, X):
        pred_probs = self.model.predict(X)
        preds = np.where(pred_probs >= 0.5, 1, 0)
        return pred_probs, preds

    def evaluate(self, y_train, y_test, x_train=None, x_test=None, plot_roc=False):
        train_pred_probs, train_preds = self.make_predictions(X=x_train)
        test_pred_probs, test_preds = self.make_predictions(X=x_test)

        # print evaluation
        print("LSTM Test Set Metrics")
        print("Accuracy:", accuracy_score(y_test, test_preds))
        print("F1 Score:", f1_score(y_test, test_preds))
        print("Precision:", precision_score(y_test, test_preds))
        print("Recall:", recall_score(y_test, test_preds))
        print("ROC AUC:", roc_auc_score(y_test, test_preds))
        print("Confusion Matrix Format: \n[TN, FP]\n[FN, TP]")
        print(confusion_matrix(y_test, test_preds))

        if plot_roc:
            # plot the ROC curve
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_pred_probs)
            fpr, tpr, thresholds = roc_curve(y_test, test_pred_probs)

            plt.plot(
                train_fpr, train_tpr, color='darkorange',
                label=f"Train ROC with AUC = {round(roc_auc_score(y_train, train_preds), 2)}"
            )
            plt.plot(
                fpr, tpr, color='green',
                label=f"Test ROC with AUC = {round(roc_auc_score(y_test, test_preds), 2)}"
            )
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('LSTM Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()


class DeepBidirectionalLSTM:
    def __init__(self, seed, checkpoint_path):
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.training_hist = None

    def build(self, nbr_time_steps, nbr_features, loss, learning_rate, metrics_list,
              nbr_hidden_layers=1, hidden_layer_units=[10]):
        """
        Builds the LSTM architecture.

        :param nbr_time_steps: Time steps in the sequence, or the number of words in a text sequence for NLP problems
        :param nbr_features: Features, or the number of latent features/embedding dimensions for NLP problems
        :param loss: Type of loss to optimize
        :param learning_rate: Controls the size of the adjustments to the model's weights in each iteration
        :param metrics_list: Any metrics to track while training (model will optimize loss, but track these too)
        :param nbr_hidden_layers: How deep to make the network
        :param hidden_layer_units: List of the number of units per layer, list length should match nbr_hidden_layers
        """
        i = Input(shape=(nbr_time_steps, nbr_features))
        for layer in range(nbr_hidden_layers):
            if layer == 0:
                x = Bidirectional(LSTM(hidden_layer_units[layer]), merge_mode="concat")(i)
            else:
                x = Bidirectional(LSTM(hidden_layer_units[layer]), merge_mode="concat")(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(i, x)
        self.model.compile(
            loss=loss,
            optimizer=Adam(lr=learning_rate),
            metrics=metrics_list
        )
        print(self.model.summary())

    def load_saved_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def train(self, X, Y, epochs):
        # callback to monitor training and save weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        # split time series into train/validation sets at a chosen time step
        # here the data is simply split in half (first half train, later half validation)
        nbr_samples = X.shape[0]
        self.training_hist = self.model.fit(
            X[:-nbr_samples // 2], Y[:-nbr_samples // 2],
            epochs=epochs,
            validation_data=(X[-nbr_samples // 2:], Y[-nbr_samples // 2:]),
            callbacks=[cp_callback],
        )
        return self.training_hist

    def plot_training_metrics(self):
        if self.model or self.training_hist is None:
            raise Exception("Must train the model before plotting.")
        else:
            # plot the training loss
            plt.plot(self.training_hist.history['loss'], label='Training Loss')
            plt.plot(self.training_hist.history['val_loss'], label='Validation Loss')
            plt.title("Bidirectional LSTM Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            # plot the training accuracy
            plt.plot(self.training_hist.history['accuracy'], label='Training Accuracy')
            plt.plot(self.training_hist.history['val_accuracy'], label='Validation Accuracy')
            plt.title("Bidirectional LSTM Classification Accuracy by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    def make_predictions(self, X):
        pred_probs = self.model.predict(X)
        preds = np.where(pred_probs >= 0.5, 1, 0)
        return pred_probs, preds

    def evaluate(self, y_train, y_test, x_train=None, x_test=None, plot_roc=False):
        train_pred_probs, train_preds = self.make_predictions(X=x_train)
        test_pred_probs, test_preds = self.make_predictions(X=x_test)

        # print evaluation
        print("Bidirectional LSTM Test Set Metrics")
        print("Accuracy:", accuracy_score(y_test, test_preds))
        print("F1 Score:", f1_score(y_test, test_preds))
        print("Precision:", precision_score(y_test, test_preds))
        print("Recall:", recall_score(y_test, test_preds))
        print("ROC AUC:", roc_auc_score(y_test, test_preds))
        print("Confusion Matrix Format: \n[TN, FP]\n[FN, TP]")
        print(confusion_matrix(y_test, test_preds))

        if plot_roc:
            # plot the ROC curve
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_pred_probs)
            fpr, tpr, thresholds = roc_curve(y_test, test_pred_probs)

            plt.plot(
                train_fpr, train_tpr, color='darkorange',
                label=f"Train ROC with AUC = {round(roc_auc_score(y_train, train_preds), 2)}"
            )
            plt.plot(
                fpr, tpr, color='green',
                label=f"Test ROC with AUC = {round(roc_auc_score(y_test, test_preds), 2)}"
            )
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Bidirectional LSTM Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()


class TransformerBlock(Layer):
    """
    Class borrowed from: https://keras.io/examples/nlp/text_classification_with_transformer/
    Note: this is only the encoder block from a full transformer
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ValueAndPositionEmbedding(Layer):
    """
    These 2 sources helped inspire this class:
        https://keras.io/examples/nlp/text_classification_with_transformer/
        https://keras.io/examples/vision/image_classification_with_vision_transformer/
    """
    def __init__(self, sequence_len, embed_dim):
        super(ValueAndPositionEmbedding, self).__init__()
        self.val_input = Dense(units=embed_dim)  # dense layer replaces token embedding from language model
        self.pos_emb = Embedding(input_dim=sequence_len, output_dim=embed_dim)

    def call(self, x):
        sequence_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=sequence_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.val_input(x)
        return x + positions


class DeepTransformer:
    def __init__(self, seed, checkpoint_path):
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.training_hist = None

    def build(self, max_seq_len, nbr_features, embed_dim, loss, learning_rate, metrics_list,
              nbr_transformer_blocks=1, nbr_attention_heads_each_block=2, nbr_dense_units_each_block=32):
        """
        Builds the transformer architecture.  For advice in choosing the number of attention heads, see:
            https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/
        Spoiler: more heads 'can' help with training, but you are likely better off using as few as necessary

        :param max_seq_len: Only the first max_seq_len items/words/time_steps in each sequence will be modeled.  This
            is equivalent to nbr_time_steps in the other models in this script, as the max_seq_len will simply be the
            number of total time steps.
        :param nbr_features: Features in the 2nd dimension of the input
        :param embed_dim: The number of latent features/embedding dimensions for the token and position embeddings.
            Setting this equal to nbr_features will ignore dimension reduction and not compress anything.
        :param loss: Type of loss to optimize
        :param learning_rate: Controls the size of the adjustments to the model's weights in each iteration
        :param metrics_list: Any metrics to track while training (model will optimize loss, but track these too)
        :param nbr_transformer_blocks: How deep to make the network
        :param nbr_attention_heads_each_block: How many heads in the multi-headed attention unit, for each transformer
        :param nbr_dense_units_each_block: How many units in the dense/feedforward part of each transformer

        :return: None
        """
        i = Input(shape=(max_seq_len, nbr_features))
        x = ValueAndPositionEmbedding(max_seq_len, embed_dim)(i)
        for layer in range(nbr_transformer_blocks):
            x = TransformerBlock(embed_dim, nbr_attention_heads_each_block, nbr_dense_units_each_block)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(20, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation='sigmoid')(x)  # replace with softmax if doing multi-class
        self.model = Model(i, x)
        self.model.compile(
            loss=loss,
            optimizer=Adam(lr=learning_rate),
            metrics=metrics_list
        )
        print(self.model.summary())

    def load_saved_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def train(self, X, Y, epochs):
        # callback to monitor training and save weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        # split time series into train/validation sets at a chosen time step
        # here the data is simply split in half (first half train, later half validation)
        nbr_samples = X.shape[0]
        self.training_hist = self.model.fit(
            X[:-nbr_samples // 2], Y[:-nbr_samples // 2],
            epochs=epochs,
            validation_data=(X[-nbr_samples // 2:], Y[-nbr_samples // 2:]),
            callbacks=[cp_callback],
        )
        return self.training_hist

    def plot_training_metrics(self):
        if self.model or self.training_hist is None:
            raise Exception("Must train the model before plotting.")
        else:
            # plot the training loss
            plt.plot(self.training_hist.history['loss'], label='Training Loss')
            plt.plot(self.training_hist.history['val_loss'], label='Validation Loss')
            plt.title("Transformer Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            # plot the training accuracy
            plt.plot(self.training_hist.history['accuracy'], label='Training Accuracy')
            plt.plot(self.training_hist.history['val_accuracy'], label='Validation Accuracy')
            plt.title("Transformer Classification Accuracy by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    def make_predictions(self, X):
        pred_probs = self.model.predict(X)
        preds = np.where(pred_probs >= 0.5, 1, 0)
        return pred_probs, preds

    def evaluate(self, y_train, y_test, x_train=None, x_test=None, plot_roc=False):
        train_pred_probs, train_preds = self.make_predictions(X=x_train)
        test_pred_probs, test_preds = self.make_predictions(X=x_test)

        # print evaluation
        print("Transformer Test Set Metrics")
        print("Accuracy:", accuracy_score(y_test, test_preds))
        print("F1 Score:", f1_score(y_test, test_preds))
        print("Precision:", precision_score(y_test, test_preds))
        print("Recall:", recall_score(y_test, test_preds))
        print("ROC AUC:", roc_auc_score(y_test, test_preds))
        print("Confusion Matrix Format: \n[TN, FP]\n[FN, TP]")
        print(confusion_matrix(y_test, test_preds))

        if plot_roc:
            # plot the ROC curve
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_pred_probs)
            fpr, tpr, thresholds = roc_curve(y_test, test_pred_probs)

            plt.plot(
                train_fpr, train_tpr, color='darkorange',
                label=f"Train ROC with AUC = {round(roc_auc_score(y_train, train_preds), 2)}"
            )
            plt.plot(
                fpr, tpr, color='green',
                label=f"Test ROC with AUC = {round(roc_auc_score(y_test, test_preds), 2)}"
            )
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Transformer Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
