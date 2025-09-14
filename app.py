import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("🚆 LRT Density Regression Network Predictor")

# Upload Excel file
uploaded_file = st.file_uploader("Upload LRT Data (Excel file)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    recto_entries = df["Recto Entry"].dropna().astype(float).tolist()

    # input x, output y(entry counts)
    X = np.arange(len(recto_entries)).reshape(-1, 1)
    y = np.array(recto_entries).reshape(-1, 1)

    # normalization
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_norm = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    # initialize network
    input_size = 1
    hidden1_size = 20
    hidden2_size = 10
    output_size = 1
    learning_rate = 0.01
    epochs = 500  # longer training for better prediction

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden1_size)
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size)
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, output_size)
    b3 = np.zeros((1, output_size))

    # activation functions
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return (x > 0).astype(float)

    # Training loop
    for epoch in range(epochs):
        hidden_input1 = np.dot(X_norm, W1) + b1
        hidden_output1 = relu(hidden_input1)

        hidden_input2 = np.dot(hidden_output1, W2) + b2
        hidden_output2 = relu(hidden_input2)

        final_output = np.dot(hidden_output2, W3) + b3
        loss = np.mean((final_output - y_norm) ** 2)

        d_loss = 2 * (final_output - y_norm) / y_norm.shape[0]

        dW3 = np.dot(hidden_output2.T, d_loss)
        db3 = np.sum(d_loss, axis=0, keepdims=True)

        d_hidden2 = np.dot(d_loss, W3.T) * relu_derivative(hidden_input2)

        dW2 = np.dot(hidden_output1.T, d_hidden2)
        db2 = np.sum(d_hidden2, axis=0, keepdims=True)

        d_hidden1 = np.dot(d_hidden2, W2.T) * relu_derivative(hidden_input1)

        dW1 = np.dot(X_norm.T, d_hidden1)
        db1 = np.sum(d_hidden1, axis=0, keepdims=True)

        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Prediction
    next_x = np.array([[len(X)]])
    next_x_norm = (next_x - X_mean) / X_std

    hidden_input1 = np.dot(next_x_norm, W1) + b1
    hidden_output1 = relu(hidden_input1)

    hidden_input2 = np.dot(hidden_output1, W2) + b2
    hidden_output2 = relu(hidden_input2)

    next_y_norm = np.dot(hidden_output2, W3) + b3
    next_y = next_y_norm * y_std + y_mean

    st.success(f"📈 Predicted next Recto Entry: {next_y.flatten()[0]:.2f}")

    # Plot actual vs predicted
    hidden_input1 = np.dot(X_norm, W1) + b1
    hidden_output1 = relu(hidden_input1)

    hidden_input2 = np.dot(hidden_output1, W2) + b2
    hidden_output2 = relu(hidden_input2)

    predicted_norm = np.dot(hidden_output2, W3) + b3
    predicted = predicted_norm * y_std + y_mean

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, y, 'bo-', label="Actual Data")
    ax.plot(X, predicted, 'r--', label="Predicted by DRN")
    ax.scatter(len(X), next_y, color='green', label="Next Predicted Point", s=100)
    ax.set_xlabel("Time Index (days)")
    ax.set_ylabel("Recto Entries")
    ax.set_title("Density Regression Network")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.info("👆 Please upload an Excel file to start.")
