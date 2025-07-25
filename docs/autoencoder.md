# Autoencoders: Deep Learning for Anomaly Detection

Autoencoders are neural networks trained to reconstruct their input. Anomalies are detected as points with high reconstruction error.

## Theory and Intuition
- **Encoder**: Compresses input into a lower-dimensional latent space.
- **Decoder**: Reconstructs the input from the latent space.
- **Anomalies**: Harder to reconstruct, resulting in higher error.

## Mathematical Formulation
Given input $x$, the autoencoder learns functions $f$ (encoder) and $g$ (decoder):
$$
\hat{x} = g(f(x))
$$
The objective is to minimize reconstruction loss:
$$
L(x, \hat{x}) = \|x - \hat{x}\|^2
$$
Anomaly score: $s(x) = L(x, \hat{x})$

## Architecture
- **Input layer**: Feature vector from logs
- **Hidden layers**: Compress and decompress
- **Output layer**: Same size as input

## Practical Example (Python, Keras)
```python
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=10, batch_size=32)
reconstructions = autoencoder.predict(X)
reconstruction_error = np.mean((X - reconstructions) ** 2, axis=1)
```

## Visual Intuition
- Normal data: Low reconstruction error
- Anomalies: High reconstruction error

## Why Autoencoders for Log Anomaly Detection?
- **Capture complex, non-linear patterns**
- **No need for labeled anomalies**
- **Adaptable to new log formats/features**

## Example Use Case
- Detecting novel attack patterns in security logs
- Identifying rare system failures in infrastructure logs

---

**Next: [LSTM & RNNs: Temporal Deep Learning for Logs](lstm_rnn.md) →** 