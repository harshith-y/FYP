import deepxde as dde
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specify the checkpoint directory
checkpoint_path = "/Users/harshith/Documents/FYP-50001.ckpt"
model = dde.Model.load(checkpoint_path)


data = pd.read_csv("/Users/harshith/Documents/FYP/Pat2.csv") 
time = data["time"]/60
true_glucose = data["glucose"]
true_insulin = data["insulin"]

# Prepare inputs for prediction (reshape if necessary, e.g., [n, 1] for 1D time input)
inputs = time.values.reshape(-1, 1)
# Make predictions using the model
predictions = model.predict(inputs)
# Assuming the model outputs predictions in the same order as glucose and insulin
predicted_glucose = predictions[:, 0]  # Example: First column is glucose
predicted_insulin = predictions[:, 1]  # Example: Second column is insulin


# Glucose
plt.figure(figsize=(10, 6))
plt.plot(time, true_glucose, label="True Glucose (mg/dL)", color="blue")
plt.plot(time, predicted_glucose, label="Predicted Glucose (mg/dL)", linestyle="dashed", color="orange")
plt.xlabel("Time (hours)")
plt.ylabel("Glucose Concentration (mg/dL)")
plt.title("True vs Predicted Glucose Concentration")
plt.legend()
plt.grid(True)
plt.show()

# Insulin
plt.figure(figsize=(10, 6))
plt.plot(time, true_insulin, label="True Insulin Infusion Rate", color="red")
plt.plot(time, predicted_insulin, label="Predicted Insulin Infusion Rate", linestyle="dashed", color="green")
plt.xlabel("Time (hours)")
plt.ylabel("Insulin Infusion Rate")
plt.title("True vs Predicted Insulin Infusion Rate")
plt.legend()
plt.grid(True)
plt.show()

