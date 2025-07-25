# LSTM & RNNs: Temporal Deep Learning for Logs

Long Short-Term Memory (LSTM) networks and Recurrent Neural Networks (RNNs) are deep learning models designed to capture temporal dependencies in sequential data, such as log streams.

## Theory and Intuition
- **RNNs**: Process sequences by maintaining a hidden state.
- **LSTMs**: Special RNNs with gates to remember/forget information, solving the vanishing gradient problem.
- **Anomalies**: Detected as unexpected events in log sequences.

## Mathematical Formulation
- **RNN cell**: $h_t = \sigma(Wx_t + Uh_{t-1} + b)$
- **LSTM cell**: Includes input, forget, and output gates:
  - $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$
  - $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$
  - $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$
  - $c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$
  - $h_t = o_t \odot \tanh(c_t)$

## Architecture
- **Input**: Sequence of log feature vectors
- **LSTM/RNN layers**: Capture temporal dependencies
- **Output**: Next event prediction or sequence reconstruction

## Practical Example (Python, Keras)
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, features)))
model.add(Dense(features))
model.compile(optimizer='adam', loss='mse')
model.fit(X_sequences, X_targets, epochs=10, batch_size=32)
predictions = model.predict(X_sequences)
reconstruction_error = np.mean((X_targets - predictions) ** 2, axis=1)
```

## Visual Intuition
- Normal sequences: Low prediction/reconstruction error
- Anomalies: High error, unexpected events

## Why LSTM/RNN for Log Anomaly Detection?
- **Capture temporal/contextual anomalies**
- **Detect rare event sequences**
- **Adapt to evolving log patterns**

## Example Use Case
- Detecting multi-step attacks in security logs
- Identifying cascading failures in distributed systems

---

**Next: [Benchmarking ML Models for Anomaly Detection](benchmarking.md) →** 