import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


# implemented along with ChatGPT


class _TorchMLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, num_classes):
        super().__init__()

        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)

        layers = []
        prev_dim = input_dim

        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _split_weights_and_biases(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-like MLPClassifier implemented in PyTorch (CPU-only),
    supports arbitrary hidden_layer_sizes and sklearn-style alpha.
    """

    def __init__(
        self,
        hidden_layer_sizes=(256,),
        lr=1e-3,
        alpha=1e-4,
        batch_size=8192,
        max_iter=500,
        patience=5,
        random_state=0,
        verbose=False,
    ):
        self.model_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _fit_from_tensors(self, X_t, y_t, input_dim):
        self.classes_, y_enc = np.unique(y_t.numpy(), return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y_t = torch.from_numpy(y_enc)

        self.model_ = _TorchMLPNet(
            input_dim=input_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_classes=self.n_classes_,
        )

        decay, no_decay = _split_weights_and_biases(self.model_)
        optimizer = torch.optim.Adam(
            [
                {"params": decay, "weight_decay": self.alpha},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            self.model_.train()
            perm = torch.randperm(X_t.size(0))
            total_loss = 0.0

            for i in range(0, X_t.size(0), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = X_t[idx]
                yb = y_t[idx]

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, X_t.size(0) // self.batch_size)
            if self.verbose:
                print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        return self

    def _predict_proba_batched(self, X_t):
        probs_batches = []
        for i in range(0, X_t.size(0), self.batch_size):
            xb = X_t[i : i + self.batch_size]
            logits = self.model_(xb)
            probs_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs_batches, axis=0)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        return self._fit_from_tensors(X_t, y_t, input_dim=X.shape[1])

    def fit_streaming(self, batch_iterator_fn, input_dim):
        """Fit from a callable that returns an iterator over (X_batch, y_batch)."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # build model from first batch class support
        first_X, first_y = next(batch_iterator_fn())
        first_X_t = torch.from_numpy(np.asarray(first_X, dtype=np.float32))
        first_y_t = torch.from_numpy(np.asarray(first_y, dtype=np.int64))

        self.classes_, y_enc = np.unique(first_y_t.numpy(), return_inverse=True)
        if len(self.classes_) < 2:
            self.classes_ = np.array([0, 1], dtype=np.int64)
            y_enc = first_y_t.numpy()
        self.n_classes_ = len(self.classes_)

        self.model_ = _TorchMLPNet(
            input_dim=input_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_classes=self.n_classes_,
        )
        decay, no_decay = _split_weights_and_biases(self.model_)
        optimizer = torch.optim.Adam(
            [
                {"params": decay, "weight_decay": self.alpha},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        epochs_no_improve = 0
        class_to_index = {int(c): i for i, c in enumerate(self.classes_)}

        for epoch in range(self.max_iter):
            self.model_.train()
            total_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in batch_iterator_fn():
                xb = torch.from_numpy(np.asarray(X_batch, dtype=np.float32))
                y_arr = np.asarray(y_batch, dtype=np.int64)
                y_idx = np.array([class_to_index[int(y)] for y in y_arr], dtype=np.int64)
                yb = torch.from_numpy(y_idx)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            if self.verbose:
                print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X)
            probs = self._predict_proba_batched(X_t)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
