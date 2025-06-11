import mlx.core as mx
import mlx.nn as nn
import numpy as np
from model_mlx import get_model

# Dummy DataLoader using numpy
def batch_loader(X, y, batch_size=128, shuffle=True):
    idxs = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(y), batch_size):
        batch_idx = idxs[i:i+batch_size]
        yield tuple(xx[batch_idx] for xx in X), y[batch_idx]

def binary_cross_entropy(preds, targets):
    eps = 1e-7
    preds = mx.clip(preds, eps, 1 - eps)
    return -(targets * mx.log(preds) + (1 - targets) * mx.log(1 - preds)).mean()

# Example args
n_users = 1000
n_items = 2000

# Generate synthetic data for demonstration
X = [np.random.randint(0, n_users, 10000), np.random.randint(0, n_items, 10000)]
y = np.random.randint(0, 2, 10000).astype(np.float32)

model = get_model('ncf', n_users=n_users, n_items=n_items)

optimizer = nn.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
batch_size = 256

for epoch in range(epochs):
    losses = []
    for (user_ids, item_ids), labels in batch_loader(X, y, batch_size):
        # Convert numpy -> MLX tensors
        user_ids = mx.array(user_ids, dtype=mx.int32)
        item_ids = mx.array(item_ids, dtype=mx.int32)
        labels = mx.array(labels, dtype=mx.float32)

        def loss_fn():
            preds = model(user_ids, item_ids)
            return binary_cross_entropy(preds, labels)
        
        loss, grads = mx.value_and_grad(loss_fn)()
        optimizer.step(grads)
        losses.append(float(loss))
    print(f"Epoch {epoch+1}: Loss {np.mean(losses):.4f}")

# Save model
nn.savez("ncf_mlx.npz", model)