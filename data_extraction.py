import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Define a function to extract parameters from the 'reason' field
def extract_parameters_from_reason(reason_series):
    """
    Extract parameters (COB, ISF, CR, Dev, BGI) from the 'reason' field using regex.
    Args:
        reason_series (Series): The 'reason' column from the data.
    Returns:
        DataFrame: A DataFrame with extracted parameters as columns.
    """
    parameters = ["COB", "ISF", "CR", "Dev", "BGI"]
    extracted_data = {}

    for param in parameters:
        # Use regex to extract each parameter
        pattern = rf"{param}: ([\d\.\-]+)"  # Match 'Param: Value'
        extracted_data[param] = reason_series.str.extract(pattern)[0].astype(float)

    # Create a DataFrame and drop rows where all extracted parameters are NaN
    extracted_df = pd.DataFrame(extracted_data)
    extracted_df = extracted_df.dropna(how="all")  # Drop rows with no valid parameters
    return extracted_df


def process_bg_data(bg_file_path):
    """
    Process the blood glucose entries file where data is comma-separated.
    Args:
        file_path (str): Path to the blood glucose data file.
    Returns:
        DataFrame: Processed data with 'timestamp' and 'blood_glucose' columns.
    """
    try:
        # Load the data with a comma delimiter
        data = pd.read_csv(bg_file_path, header=None, names=["timestamp", "blood_glucose"], delimiter=",")
    except Exception as e:
        return None

    # Convert the timestamp to datetime with UTC
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", errors="coerce", utc=True)
        data["blood_glucose"] = pd.to_numeric(data["blood_glucose"], errors="coerce")  # Ensure BG values are numeric
    except Exception as e:
        return None

    # Drop rows with invalid timestamps or missing BG values
    processed_data = data.dropna(subset=["timestamp", "blood_glucose"])
    
    return processed_data


# Define a function to process a single patient's data
def process_patient_data(device_file_path):
    """
    Process a single patient's data file to extract relevant columns and parameters from 'reason'.
    Args:
        file_path (str): Path to the patient's data file.
    Returns:
        DataFrame: Processed data with relevant columns and timestamps.
    """
    # Load the data
    try:
        data = pd.read_csv(device_file_path)
    except Exception as e:
        print(f"Error reading {device_file_path}: {e}")
        return None

    # Convert timestamps to datetime
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")

    # Extract relevant columns
    relevant_columns = ["created_at", "openaps/iob/bolusinsulin", "openaps/iob/basaliob", "openaps/iob/iob", "openaps/enacted/IOB", "openaps/enacted/reason"]
    extracted_data = data[relevant_columns].dropna(subset=["created_at"])

    # Extract parameters from 'reason'
    if "openaps/enacted/reason" in extracted_data.columns:
        reason_params = extract_parameters_from_reason(extracted_data["openaps/enacted/reason"])
        if not reason_params.empty:  # Only add parameters if valid data exists
            extracted_data = pd.concat([extracted_data, reason_params], axis=1)
    
    return extracted_data

# Define a function to plot data
def plot_patient_data(patient_id, data, start_date=None, end_date=None):
    """
    Plot extracted data for a single patient within a date range, handling both BG and device data.
    Args:
        patient_id (str): Identifier for the patient.
        data (DataFrame): Extracted data to plot.
        start_date (str): Start date for filtering (YYYY-MM-DD). Optional.
        end_date (str): End date for filtering (YYYY-MM-DD). Optional.
    """
    # Convert start_date and end_date to datetime and localize to UTC
    if start_date:
        start_date = pd.Timestamp(start_date).tz_localize("UTC")  # Ensure timezone-awareness
    if end_date:
        end_date = pd.Timestamp(end_date).tz_localize("UTC")  # Ensure timezone-awareness

    # Filter data by date range
    if "created_at" in data.columns:  # Device data
        if start_date:
            data = data[data["created_at"] >= start_date]
        if end_date:
            data = data[data["created_at"] < end_date]
    if "timestamp" in data.columns:
        data["timestamp"] = data["timestamp"].dt.tz_localize(None)  # Remove timezone info for naive comparison
        if start_date:
            start_date = start_date.tz_localize(None)  # Convert start_date to timezone-naive
        if end_date:
            end_date = end_date.tz_localize(None)  # Convert end_date to timezone-naive

    # Filter data by date range
    if "timestamp" in data.columns:  # Blood glucose data
        if start_date:
            data = data[data["timestamp"] >= start_date]
        if end_date:
            data = data[data["timestamp"] < end_date]

    # Plot blood glucose data if available
    if "blood_glucose" in data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data["timestamp"], data["blood_glucose"], label="Blood Glucose (mg/dL)", marker='.')
        plt.title(f"Blood Glucose Over Time (Patient {patient_id})")
        plt.xlabel("Time")
        plt.ylabel("Blood Glucose (mg/dL)")
        plt.legend()
        plt.grid()
        plt.show()

    # Plot device data if available
    for col in ["openaps/iob/bolusinsulin", "openaps/iob/basaliob", "openaps/iob/iob", "openaps/enacted/IOB"]:
        if col in data.columns and not data[col].isna().all():
            col_title = col.replace("openaps/iob/", "").replace("bolusinsulin", "Bolus Insulin").replace("basaliob", "Basal IOB").replace("iob", "Total IOB")
            plt.figure(figsize=(12, 6))
            plt.plot(data["created_at"], data[col], label=col_title, marker='.')
            plt.title(f"{col_title} Over Time (Patient {patient_id})")
            plt.xlabel("Time")
            plt.ylabel("Insulin (Units)")
            plt.legend()
            plt.grid()
            plt.show()

    # Plot extracted parameters from 'reason' if available
    for param in ["COB", "ISF", "CR", "Dev", "BGI"]:
        if param in data.columns and not data[param].isna().all():
            plt.figure(figsize=(12, 6))
            plt.plot(data["created_at"], data[param], label=param, marker='.')
            plt.title(f"{param} Over Time (Patient {patient_id})")
            plt.xlabel("Time")
            plt.ylabel(param)
            plt.legend()
            plt.grid()
            plt.show()

# Process and plot data for multiple patients
def process_multiple_patients(data_dir):
    """
    Process data for all patients in a directory.
    Args:
        data_dir (str): Directory containing patient data files.
    Returns:
        dict: Dictionary with patient IDs as keys and processed DataFrames as values.
    """
    patient_data = {}
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        patient_id = os.path.splitext(file_name)[0]  # Use the file name as the patient ID
        print(f"Processing data for Patient {patient_id}...")
        data = process_patient_data(file_path)
        if data is not None and not data.empty:
            patient_data[patient_id] = data
    return patient_data

### Test the functions with a single patient's data ###
bg_file_path = "/Users/harshith/Documents/FYP/CGMInfo_from3pats/16975609/direct-sharing-31/16975609_entries__to_2020-06-09.json_csv/16975609_entries__to_2020-06-09.json.csv"
device_file_path = "/Users/harshith/Documents/FYP/CGMInfo_from3pats/16975609/direct-sharing-31/16975609_devicestatus__to_2020-06-09_csv/16975609_devicestatus__to_2020-06-09_aa.csv"

bg_data = process_bg_data(bg_file_path)
device_data = process_patient_data(device_file_path)

# Verify and plot BG data
if bg_data is not None and not bg_data.empty:
    print("Blood glucose data processed successfully!")
    print(bg_data.head())  # Display the first few rows
    plot_patient_data("16975609_BG", bg_data, start_date="2020-05-30", end_date="2020-05-31")
else:
    print("No blood glucose data available")

# Verify and plot device data
if device_data is not None and not device_data.empty:
    print("Device data processed successfully!")
    print(device_data.head())  # Display the first few rows
    plot_patient_data("16975609_Device", device_data, start_date="2020-05-20", end_date="2020-05-21")
else:
    print("No device status data available")