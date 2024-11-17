


import streamlit as st
import pandas as pd
import joblib  # Import joblib instead of pickle

# Set custom CSS to change the background color to black
st.markdown("""
    <style>
        
        body {
            background-color: black;
            color: white;
        }
        .streamlit-expanderHeader {
            color: white;
        }
        .stButton>button {
            background-color: #333;
            color: white;
        }
        .stNumberInput>div>div>input {
            background-color: #444;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #444;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Define the paths to the model files (keeping them in the same folder as the app)
model_path_density = 'PKL_FILES/alpha_rf.pkl'
model_path_velocity = 'PKL_FILES/u_rf.pkl'
model_path_Intermolecular_Free_Length = 'PKL_FILES/L_RF.pkl'
model_path_Internal_pressure = 'PKL_FILES/P-rf.pkl'
model_path_Cohesion_Energy_Density = 'PKL_FILES/CED_RF.pkl'
model_path_Gruneisen_Parameter = 'PKL_FILES/gp_rf.pkl'
model_path_Acoustic_Impendance = "PKL_FILES/z_rf.pkl"
model_path_Non_Linearity_Parameters = "PKL_FILES/ba_rf.pkl"
model_path_Solubility_Parameter = "PKL_FILES/sp_rf.pkl"

# Function to load model from joblib file
def load_model(model_path):
    try:
        # Use joblib to load the model
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please check the file path and try again.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        st.stop()

# Load models
model_density = load_model(model_path_density)
model_velocity = load_model(model_path_velocity)
model_Intermolecular_Free_Length = load_model(model_path_Intermolecular_Free_Length)
model_Internal_pressure = load_model(model_path_Internal_pressure)
model_Cohesion_Energy_Density = load_model(model_path_Cohesion_Energy_Density)
model_Gruneisen_Parameter = load_model(model_path_Gruneisen_Parameter)
model_Acoustic_Impendance = load_model(model_path_Acoustic_Impendance)
model_Non_Linearity_Parameters = load_model(model_path_Non_Linearity_Parameters)
model_Solubility_Parameter = load_model(model_path_Solubility_Parameter)

# Function to make predictions using the loaded models
def predict_seawater_properties(temperature, concentration):
    try:
        # Prepare the input data for all models
        input_data = pd.DataFrame([[temperature, concentration]], columns=['t', 'c'])

        # Make predictions
        density_prediction = model_density.predict(input_data)
        velocity_prediction = model_velocity.predict(input_data)
        intermolecular_free_length_prediction = model_Intermolecular_Free_Length.predict(input_data)
        internal_pressure_prediction = model_Internal_pressure.predict(input_data)
        cohesion_energy_density_prediction = model_Cohesion_Energy_Density.predict(input_data)
        gruneisen_parameter_prediction = model_Gruneisen_Parameter.predict(input_data)
        acoustic_impedance_prediction = model_Acoustic_Impendance.predict(input_data)
        non_linearity_parameter_prediction = model_Non_Linearity_Parameters.predict(input_data)
        solubility_parameter_prediction = model_Solubility_Parameter.predict(input_data)

        # Return all predicted properties
        return (density_prediction[0], velocity_prediction[0], intermolecular_free_length_prediction[0],
                internal_pressure_prediction[0], cohesion_energy_density_prediction[0], gruneisen_parameter_prediction[0],
                acoustic_impedance_prediction[0], non_linearity_parameter_prediction[0], solubility_parameter_prediction[0])

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return (None, None, None, None, None, None, None, None, None)

# Streamlit Interface
st.title("ThermoPredictor 🌊")
st.subheader("Predict Seawater Properties using Temperature and Concentration")

# Create columns for input and output
col1, col2 = st.columns([1, 1])

# Input section on the left side
with col1:
    st.header("Input Parameters")
    
    # Collect user inputs
    temperature = st.number_input(
        "Enter Temperature (°C)",
        min_value=295.15,  # Minimum temperature from your data (295.15°C)
        max_value=299.15,  # Maximum temperature from your data (299.15°C)
        step=0.1,
        help="Enter the temperature in degrees Celsius within the valid range of 295.15°C to 299.15°C."
    )

    concentration = st.number_input(
        "Enter Concentration (g/kg)",
        min_value=0.54687,  # Minimum concentration from your data (0.54687%),
        max_value=35.0,     # Maximum concentration from your data (35%),
        step=0.1,
    )

# Add a gap between input and output
st.markdown("<br><br>", unsafe_allow_html=True)

# Output section on the right side
with col2:
    st.header("Predicted Properties")
    
    # Predict button
    if st.button("Predict Properties"):
        # Check if the user inputs are within a reasonable range based on your data
        if not (295.15 <= temperature <= 299.15):
            st.warning("Temperature is outside the typical range in the dataset. Predictions may not be accurate.")
        if not (0.54687 <= concentration <= 35.0):
            st.warning("Concentration is outside the typical range in the dataset. Predictions may not be accurate.")
        
        # Make predictions based on user inputs
        (predicted_density, predicted_velocity, predicted_intermolecular_free_length,
         predicted_internal_pressure, predicted_cohesion_energy_density, predicted_gruneisen_parameter,
         predicted_acoustic_impedance, predicted_non_linearity_parameter, predicted_solubility_parameter) = predict_seawater_properties(temperature, concentration)
        
        # Display the predicted properties if they are valid
        if predicted_density is not None and predicted_velocity is not None and predicted_intermolecular_free_length is not None:
            st.success(f"Predicted Seawater Density: {predicted_density:.6f} kg/m³")
            st.success(f"Predicted Ultrasonic Velocity: {predicted_velocity:.6f} m/s")
            st.success(f"Predicted Intermolecular Free Length: {predicted_intermolecular_free_length:.6f} cm⁻¹")
            st.success(f"Predicted Internal Pressure: {predicted_internal_pressure:.6f} MPa")
            st.success(f"Predicted Cohesion Energy Density: {predicted_cohesion_energy_density:.6f} J/m³")
            st.success(f"Predicted Gruneisen Parameter: {predicted_gruneisen_parameter:.6f}")
            st.success(f"Predicted Acoustic Impedance: {predicted_acoustic_impedance:.6f} kg/m²s")
            st.success(f"Predicted Non-Linearity Parameter: {predicted_non_linearity_parameter:.6f}")
            st.success(f"Predicted Solubility Parameter: {predicted_solubility_parameter:.6f} (g/kg)¹²")