import tensorflow as tf
from tensorflow.keras.backend import get_value
import numpy as np
import matplotlib.pyplot as plt
from utils_inverse import GlucoseModel  # Your updated utils_inverse file
# from diabetes_PINN_main_clean import PINNModel  # Replace with actual import

# Define parameters for all patients
parameters = {
    2: {'M': 72, 'ksi': 197, 'kl': 1.94, 'Tu': 122, 'ku_Vi': 59e-3, 'Tr': 183, 'kr_Vb': 2.4e-3, 'tend': 48 * 60},
    3: {'M': 94, 'ksi': 274, 'kl': 1.72, 'Tu': 88, 'ku_Vi': 62e-3, 'Tr': 49, 'kr_Vb': 2e-3, 'tend': 48 * 60},
    4: {'M': 74, 'ksi': 191, 'kl': 1.94, 'Tu': 126, 'ku_Vi': 61e-3, 'Tr': 188, 'kr_Vb': 2.47e-3, 'tend': 48 * 60},
    5: {'M': 91, 'ksi': 282, 'kl': 1.67, 'Tu': 85, 'ku_Vi': 64e-3, 'Tr': 48, 'kr_Vb': 2.06e-3, 'tend': 48 * 60},
    6: {'M': 70, 'ksi': 203, 'kl': 1.94, 'Tu': 118, 'ku_Vi': 57e-3, 'Tr': 178, 'kr_Vb': 2.33e-3, 'tend': 48 * 60},
    7: {'M': 97, 'ksi': 267, 'kl': 1.77, 'Tu': 91, 'ku_Vi': 60e-3, 'Tr': 50, 'kr_Vb': 1.94e-3, 'tend': 48 * 60},
    8: {'M': 73, 'ksi': 200, 'kl': 1.92, 'Tu': 125, 'ku_Vi': 60e-3, 'Tr': 182, 'kr_Vb': 2.38e-3, 'tend': 48 * 60},
    9: {'M': 92, 'ksi': 272, 'kl': 1.71, 'Tu': 87, 'ku_Vi': 61e-3, 'Tr': 49, 'kr_Vb': 2.03e-3, 'tend': 48 * 60},
    10: {'M': 74, 'ksi': 191, 'kl': 1.94, 'Tu': 126, 'ku_Vi': 61e-3, 'Tr': 188, 'kr_Vb': 2.47e-3, 'tend': 48 * 60},
    11: {'M': 91, 'ksi': 282, 'kl': 1.67, 'Tu': 85, 'ku_Vi': 64e-3, 'Tr': 48, 'kr_Vb': 2.06e-3, 'tend': 48 * 60},
}

def train_inverse_model(patient_id, num_epochs=1000, learning_rate=0.01):
    """
    Train the inverse model for a given patient and plot the learning curve for ISF.
    """
    # Initialize the model for the given patient
    model = GlucoseModel(patient_id)
    true_ksi = parameters[patient_id]['ksi']
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ksi_history = []

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # Compute residual loss
            loss = model.compute_loss()

        gradients = tape.gradient(loss, [model.ksi])
        if gradients[0] is None:
            raise ValueError(f"No gradients computed for ksi at epoch {epoch}. Check the loss function.")

        optimizer.apply_gradients(zip(gradients, [model.ksi]))
        
        # Use get_value to fetch tensor values
        ksi_history.append(get_value(model.ksi))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {get_value(loss)}, Estimated ksi: {get_value(model.ksi)}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), ksi_history, label='Estimated ISF (ksi)')
    plt.axhline(y=true_ksi, color='r', linestyle='--', label='True ISF (ksi)')
    plt.xlabel('Epoch')
    plt.ylabel('Insulin Sensitivity Factor (ksi)')
    plt.title(f'Learning Curve for Patient {patient_id}')
    plt.legend()
    plt.show()

    return ksi_history, get_value(model.ksi)




# def validate_predictions(patient_id, model):
#     """
#     Validate glucose predictions using the trained model.
#     """
#     # Replace with your input data
#     inputs = np.linspace(0, parameters[patient_id]['tend'], 1000)  # Time points
#     true_glucose = np.sin(inputs / 100) * 100 + 110  # Dummy glucose data; replace with actual
    
#     # Predict glucose values
#     predicted_glucose = model.predict_glucose(inputs)  # Replace with your prediction function
    
#     # Calculate validation MSE
#     mse = np.mean((predicted_glucose - true_glucose) ** 2)
#     print(f"Validation MSE for Patient {patient_id}: {mse}")
    
#     # Plot predictions vs observed glucose
#     plt.figure(figsize=(10, 6))
#     plt.plot(inputs, predicted_glucose, label='Predicted Glucose')
#     plt.plot(inputs, true_glucose, label='Observed Glucose', linestyle='--')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Glucose (mg/dL)')
#     plt.title(f'Glucose Prediction vs Observed Data for Patient {patient_id}')
#     plt.legend()
#     plt.show()

# # Run the training and validation for each patient
# for patient_id in parameters.keys():
#     print(f"Training inverse model for Patient {patient_id}")
#     ksi_history, final_ksi = train_inverse_model(patient_id)
#     print(f"Final ksi for Patient {patient_id}: {final_ksi}")
    
#     # Validate predictions
#     model = GlucoseModel(patient_id)
#     validate_predictions(patient_id, model)



##################################################################################################

# Train the inverse model for Patient 2
patient_id = 2  # Specify the patient ID
num_epochs = 1000  # Define the number of epochs
learning_rate = 0.01  # Define the learning rate

# Initialize and train the model
print(f"Training inverse model for Patient {patient_id}")
ksi_history, final_ksi = train_inverse_model(patient_id, num_epochs=num_epochs, learning_rate=learning_rate)

# Print the final estimated ksi
print(f"Final ksi for Patient {patient_id}: {final_ksi}")