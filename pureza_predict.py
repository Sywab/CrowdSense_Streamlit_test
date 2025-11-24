import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

def predict_pureza_week():
    df = pd.read_csv('static/pureza.csv', parse_dates=['date'])
    time_slots = [
        '5am-7am entry', '5am-7am exit',
        '7am-9am entry', '7am-9am exit',
        '9am-5pm entry', '9am-5pm exit',
        '5pm-7pm entry', '5pm-7pm exit',
        '7pm-10pm entry', '7pm-10pm exit'
    ]
    scaler = MinMaxScaler()
    weekly_predictions = {}

    # Map Python weekday() to your short names
    weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    today_idx = datetime.datetime.now().weekday()
    today_short = weekday_map[today_idx]

    # Process only today's weekday data (no loop over all weekdays)
    day_data = df[df['day'] == today_short][time_slots].values
    if len(day_data) >= 4:
        # scale and create sequences from today's data only
        scaled = scaler.fit_transform(day_data)

        def create_dataset(data, time_step=3):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:i+time_step])
                y.append(data[i+time_step])
            return np.array(X), np.array(y)

        time_step = 3
        X, y = create_dataset(scaled, time_step)
        X = X.reshape(X.shape[0], time_step, len(time_slots))

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(time_step, len(time_slots))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(len(time_slots))
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=100, batch_size=2, verbose=0)

        last_sequence = scaled[-time_step:]
        last_sequence = last_sequence.reshape(1, time_step, len(time_slots))
        pred_scaled = model.predict(last_sequence, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0]
        weekly_predictions[today_short] = pred.astype(int)

        # --- Plot for today ---
        blocks = ['5am-7am', '7am-9am', '9am-5pm', '5pm-7pm', '7pm-10pm']
        hours_per_block = [2, 2, 2, 2, 8, 8, 2, 2, 3, 3]
        avg_per_hour = [val / hrs for val, hrs in zip(pred, hours_per_block)]
        entries = [avg_per_hour[i*2] for i in range(5)]
        exits = [avg_per_hour[i*2+1] for i in range(5)]

        x = np.arange(len(blocks))
        width = 0.35

        plt.figure(figsize=(10, 4))
        plt.bar(x - width/2, entries, width, label='Entries (per hour)', color='#30BCB1')
        plt.bar(x + width/2, exits, width, label='Exits (per hour)', color='#153252')
        plt.xticks(x, blocks)
        plt.xlabel('Time Block')
        plt.ylabel('Predicted Count (per hour)')
        plt.title(f'Pureza Station Predicted Entries & Exits per Hour ({today_short})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/images/pureza_predicted_bar.png')
        plt.close()

    return weekly_predictions, time_slots, today_short