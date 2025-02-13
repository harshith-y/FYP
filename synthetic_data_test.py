import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file_path.csv' with the path to your CSV file
data = pd.read_csv('/Users/harshith/Documents/FYP/Pat2.csv')

data['time_hours'] = data['time'] / 60

plt.figure(figsize=(10, 6))
plt.plot(data['time_hours'], data['glucose'], label='Glucose (mg/dL)', color='blue')
plt.xlabel('Time (hours)')
plt.ylabel('Glucose Concentration (mg/dL)')
plt.title('Glucose Concentration Over Time')
plt.legend()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot insulin on the primary y-axis
ax1.plot(data['time_hours'], data['insulin'], label='Insulin Infusion Rate', color='red')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Insulin Infusion Rate', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Create a secondary y-axis for carbohydrates
ax2 = ax1.twinx()
ax2.plot(data['time_hours'], data['carbohydrates'], label='Carbohydrate Intake', color='green')
ax2.set_ylabel('Carbohydrate Intake', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add title and grid
plt.title('Insulin Infusion Rate and Carbohydrate Intake Over Time')
fig.tight_layout()
plt.grid(True)
plt.show()

# Plot insulin infusion rate over time
plt.figure(figsize=(10, 6))
plt.plot(data['time_hours'], data['insulin'], label='Insulin Infusion Rate', color='red')
plt.xlabel('Time (hours)')
plt.ylabel('Insulin Infusion Rate')
plt.title('Insulin Infusion Rate Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot carbohydrate intake over time
plt.figure(figsize=(10, 6))
plt.plot(data['time_hours'], data['carbohydrates'], label='Carbohydrate Intake', color='green')
plt.xlabel('Time (hours)')
plt.ylabel('Carbohydrate Intake')
plt.title('Carbohydrate Intake Over Time')
plt.legend()
plt.grid(True)
plt.show()


fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot glucose concentration on the primary y-axis
ax1.plot(data['time_hours'], data['glucose'], label='Glucose Concentration', color='blue')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Glucose Concentration (mg/dL)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for insulin infusion rate
ax2 = ax1.twinx()
ax2.plot(data['time_hours'], data['insulin'], label='Insulin Infusion Rate', color='red')
ax2.set_ylabel('Insulin Infusion Rate', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add a title
plt.title('Glucose Concentration and Insulin Infusion Rate Over Time')
fig.tight_layout()
plt.grid(True)
plt.show()
