"""Function approximation: MLP regression of f(x, y, z).

Put data_group3.csv in the same folder as this script and run it from there.
Pipeline: load -> explore -> split (70/15/15) -> scale -> train -> evaluate.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Config:
    """All settings in one place; change hyperparameters here."""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_group3.csv")  
    input_columns = ("x", "y", "z")
    target_column = "f"
    seed = 42
    val_split = 0.15
    test_split = 0.15
    hidden_layers = (128, 128, 128, 256, 256)
    activation = "relu"
    learning_rate = 1e-3
    batch_size = 64
    epochs = 200
    patience = 20  # early-stopping patience; the LR scheduler waits patience // 2

class DataLoader:
    """Loads, explores, splits and scales the data."""

    def __init__(self, config):
        self.config = config
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.df = None
        self.data = {}

    def load(self):
        self.df = pd.read_csv(self.config.csv_path)
        return self

    def explore(self):
        """Print basic statistics and plot the distribution of f."""
        print(self.df.shape)
        print(self.df.describe())
        print("Missing values:", int(self.df.isna().sum().sum()),
              "| Duplicate rows:", int(self.df.duplicated().sum()))
        plt.hist(self.df[self.config.target_column], bins=100)
        plt.yscale("log")
        plt.xlabel("f"); plt.ylabel("count (log scale)")
        plt.title("Distribution of f: long right tail")
        plt.show()
        return self

    def split(self):
        """70/15/15 train/validation/test split with a fixed seed (sr_no is not used)."""
        c = self.config
        x = self.df[list(c.input_columns)].values   # shape (30000, 3)
        y = self.df[c.target_column].values         # shape (30000,)
        holdout = c.val_split + c.test_split

        x_tr, x_tmp, y_tr, y_tmp = train_test_split(
            x, y, test_size=holdout, random_state=c.seed)
        x_va, x_te, y_va, y_te = train_test_split(
            x_tmp, y_tmp, test_size=c.test_split / holdout, random_state=c.seed)

        self.data = dict(x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, x_te=x_te, y_te=y_te)
        return self

    def scale(self):
        """Scalers are fitted on the training set only, so nothing leaks from val/test."""
        d = self.data
        d['x_tr'] = self.x_scaler.fit_transform(d['x_tr'])
        d['x_va'] = self.x_scaler.transform(d['x_va'])
        d['x_te'] = self.x_scaler.transform(d['x_te'])

        # f has a long right tail: arcsinh compresses it (and allows negatives), then standardise
        d['y_tr'] = self.y_scaler.fit_transform(np.arcsinh(d['y_tr']).reshape(-1, 1))
        d['y_va'] = self.y_scaler.transform(np.arcsinh(d['y_va']).reshape(-1, 1))
        d['y_te'] = self.y_scaler.transform(np.arcsinh(d['y_te']).reshape(-1, 1))
        return self

    def inverse_target(self, y_scaled):
        """Scaled values -> original units of f (used before computing metrics)."""
        return np.sinh(self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1))).ravel()

class ModelBuilder:
    """Builds the fully connected regression network."""

    def __init__(self, config):
        self.config = config

    def build(self):
        tf.keras.utils.set_random_seed(self.config.seed)
        model = tf.keras.Sequential(name="MLP")
        model.add(tf.keras.Input(shape=(len(self.config.input_columns),)))  # x, y, z
        for units in self.config.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation=self.config.activation))
        model.add(tf.keras.layers.Dense(1))  # single linear output for regression
        return model

class Trainer:
    """Compiles and trains the model; validation loss drives the callbacks."""

    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

    def train(self):
        c = self.config
        tf.keras.utils.set_random_seed(c.seed)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=c.learning_rate),
                           loss='mse', metrics=['mae'])
        callbacks = [
            # stop when val_loss stops improving and keep the best weights
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=c.patience,
                                             restore_best_weights=True),
            # halve the learning rate when val_loss stalls
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=max(1, c.patience // 2), min_lr=1e-6),
        ]
        d = self.data
        return self.model.fit(d['x_tr'], d['y_tr'], validation_data=(d['x_va'], d['y_va']),
                              batch_size=c.batch_size, epochs=c.epochs, callbacks=callbacks)

class Evaluator:
    """Metrics (in original units of f) and plots."""
    SPLIT_NAMES = {"tr": "train", "va": "validation", "te": "test"}

    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def _true_and_pred(self, split):
        """Actual and predicted f for split 'tr', 'va' or 'te', in original units."""
        d = self.loader.data
        pred = self.loader.inverse_target(self.model.predict(d[f'x_{split}'], verbose=0))
        true = self.loader.inverse_target(d[f'y_{split}'])
        return true, pred

    @staticmethod
    def _scores(true, pred):
        mse = mean_squared_error(true, pred)
        return {"MSE": mse, "RMSE": np.sqrt(mse),
                "MAE": mean_absolute_error(true, pred), "R2": r2_score(true, pred)}

    def metrics(self, split="va", bulk_limit=100):
        """Scores on all samples and on the bulk (f < bulk_limit), excluding the long tail."""
        true, pred = self._true_and_pred(split) #rows above bulk limit are just 1458 or 4.9%
        mask = true < bulk_limit
        return {"full": self._scores(true, pred),
                f"bulk (f<{bulk_limit})": self._scores(true[mask], pred[mask])}

    def print_metrics(self, split, label):
        print(f"\n=== {label} metrics ===")
        for name, scores in self.metrics(split).items():
            print(f"[{name}]")
            for k, v in scores.items():
                print(f"  {k}: {v:.6g}")

    def plot_history(self, history):
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="validation")
        plt.yscale("log")
        plt.xlabel("epoch"); plt.ylabel("MSE (scaled)")
        plt.title("Training and validation loss")
        plt.legend(); plt.show()

    def plot_pred_vs_actual(self, split):
        true, pred = self._true_and_pred(split)
        plt.scatter(true, pred, s=3, alpha=0.3)
        lims = [true.min(), true.max()]
        plt.plot(lims, lims, "r--", label="perfect prediction")
        plt.xscale("symlog"); plt.yscale("symlog")  # f is heavily skewed
        plt.xlabel("actual f"); plt.ylabel("predicted f")
        plt.title(f"Predicted vs actual f ({self.SPLIT_NAMES[split]} data)")
        plt.legend(); plt.show()

if __name__ == "__main__":
    cfg = Config()
    loader = DataLoader(cfg).load().explore().split().scale()
    model = ModelBuilder(cfg).build()
    model.summary()
    history = Trainer(cfg, model, loader.data).train()

    ev = Evaluator(model, loader)
    ev.plot_history(history)
    ev.plot_pred_vs_actual("va")
    ev.print_metrics("tr", "Train")
    ev.print_metrics("va", "Validation")

    # Test set: evaluate once, only after the configuration is final
    ev.plot_pred_vs_actual("te")
    ev.print_metrics("te", "Test")
