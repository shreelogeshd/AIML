import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Insurance Claim Prediction", layout="wide")

# Load pipeline
pipeline = joblib.load("Joblib/TrainedModel.joblib")

st.title("Car Insurance Claim Prediction")
st.markdown("### Enter vehicle and policy details to predict if a claim will occur.")

# ======= MAPPINGS =======
yes_no_map = {"Yes": 1, "No": 0}

fuel_type_map = {"CNG": 0, "Diesel": 1, "Petrol": 2}
transmission_map = {"Automatic": 0, "Manual": 1}
rear_brake_map = {"Disc": 0, "Drum": 1}
steering_type_map = {"Electric": 0, "Manual": 1, "Power": 2}

area_cluster_map = {
    'C1': 0, 'C10': 1, 'C11': 2, 'C12': 3, 'C13': 4, 'C14': 5, 'C15': 6, 'C16': 7,
    'C17': 8, 'C18': 9, 'C19': 10, 'C2': 11, 'C20': 12, 'C21': 13, 'C22': 14,
    'C3': 15, 'C4': 16, 'C5': 17, 'C6': 18, 'C7': 19, 'C8': 20, 'C9': 21
}

engine_type_map = {
    '1.0 SCe': 0, '1.2 L K Series Engine': 1, '1.2 L K12N Dualjet': 2, '1.5 L U2 CRDi': 3,
    '1.5 Turbocharged Revotorq': 4, '1.5 Turbocharged Revotron': 5, 'F8D Petrol Engine': 6,
    'G12B': 7, 'K Series Dual jet': 8, 'K10C': 9, 'i-DTEC': 10
}

model_map = {
    'M1': 0, 'M10': 1, 'M11': 2, 'M2': 3, 'M3': 4, 'M4': 5, 'M5': 6, 'M6': 7,
    'M7': 8, 'M8': 9, 'M9': 10
}

segment_map = {"A": 0, "B1": 1, "B2": 2, "C1": 3, "C2": 4, "Utility": 5}

max_torque_map= {
    '113Nm@4400rpm': 0, '170Nm@4000rpm': 1, '200Nm@1750rpm': 2, '200Nm@3000rpm': 3,
    '250Nm@2750rpm': 4, '60Nm@3500rpm': 5, '82.1Nm@3400rpm': 6, '85Nm@3000rpm': 7, 
    '91Nm@4250rpm': 8
}

max_power_map = {
    '113.45bhp@4000rpm': 0, '118.36bhp@5500rpm': 1, '40.36bhp@6000rpm': 2,
    '55.92bhp@5300rpm': 3, '61.68bhp@6000rpm': 4, '67.06bhp@5500rpm': 5,
    '88.50bhp@6000rpm': 6, '88.77bhp@4000rpm': 7, '97.89bhp@3600rpm': 8
}

# =========== UI FORM =============
with st.form("input_form"):

    st.markdown("## Policy Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1: policy_tenure = st.number_input("Policy Tenure", value=0.0)
    with col2: age_of_car = st.number_input("Age of Car", value=0.0)
    with col3: age_of_policyholder = st.number_input("Age of Policyholder", value=0.0)
    with col4: population_density = st.number_input("Population Density", value=0)

    st.markdown("## Vehicle Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1: area_cluster = st.selectbox("Area Cluster", list(area_cluster_map.keys()))
    with col2: make = st.number_input("Make", value=1)
    with col3: segment = st.selectbox("Segment", list(segment_map.keys()))
    with col4: model = st.selectbox("Model", list(model_map.keys()))

    col1, col2, col3, col4 = st.columns(4)
    with col1: fuel_type = st.selectbox("Fuel Type", list(fuel_type_map.keys()))
    with col2: max_torque = st.selectbox("Max Torque", list(max_torque_map.keys()))
    with col3: max_power = st.selectbox("Max Power", list(max_power_map.keys()))
    with col4: engine_type = st.selectbox("Engine Type", list(engine_type_map.keys()))

    col1, col2, col3, col4 = st.columns(4)
    with col1: airbags = st.number_input("Airbags", value=2)
    with col2: displacement = st.number_input("Displacement", value=1000)
    with col3: cylinder = st.number_input("Cylinder", value=4)
    with col4: gear_box = st.number_input("Gear Box", value=5)

    st.markdown("## Safety & Comfort Features")
    col1, col2, col3, col4 = st.columns(4)
    with col1: is_esc = st.selectbox("ESC", list(yes_no_map.keys()))
    with col2: is_adjustable_steering = st.selectbox("Adjustable Steering", list(yes_no_map.keys()))
    with col3: is_tpms = st.selectbox("TPMS", list(yes_no_map.keys()))
    with col4: is_parking_sensors = st.selectbox("Parking Sensors", list(yes_no_map.keys()))

    col1, col2, col3, col4 = st.columns(4)
    with col1: is_parking_camera = st.selectbox("Parking Camera", list(yes_no_map.keys()))
    with col2: rear_brakes_type = st.selectbox("Rear Brakes Type", list(rear_brake_map.keys()))
    with col3: transmission_type = st.selectbox("Transmission Type", list(transmission_map.keys()))
    with col4: steering_type = st.selectbox("Steering Type", list(steering_type_map.keys()))

    st.markdown("## Dimensions")
    col1, col2, col3, col4 = st.columns(4)
    with col1: turning_radius = st.number_input("Turning Radius", value=5.0)
    with col2: length = st.number_input("Length", value=3500)
    with col3: width = st.number_input("Width", value=1600)
    with col4: height = st.number_input("Height", value=1500)

    col1, col2 = st.columns(2)
    with col1: gross_weight = st.number_input("Gross Weight", value=1200)
    with col2: ncap_rating = st.number_input("NCAP Rating", value=0)

    st.markdown("## Additional Features")
    col1, col2, col3, col4 = st.columns(4)
    with col1: is_front_fog_lights = st.selectbox("Front Fog Lights", list(yes_no_map.keys()))
    with col2: is_rear_window_wiper = st.selectbox("Rear Wiper", list(yes_no_map.keys()))
    with col3: is_rear_window_washer = st.selectbox("Rear Washer", list(yes_no_map.keys()))
    with col4: is_rear_window_defogger = st.selectbox("Rear Defogger", list(yes_no_map.keys()))

    col1, col2, col3, col4 = st.columns(4)
    with col1: is_brake_assist = st.selectbox("Brake Assist", list(yes_no_map.keys()))
    with col2: is_power_door_locks = st.selectbox("Power Door Locks", list(yes_no_map.keys()))
    with col3: is_central_locking = st.selectbox("Central Locking", list(yes_no_map.keys()))
    with col4: is_power_steering = st.selectbox("Power Steering", list(yes_no_map.keys()))

    col1, col2, col3 = st.columns(3)
    with col1: is_driver_seat_height_adjustable = st.selectbox("Driver Seat Height Adjustable", list(yes_no_map.keys()))
    with col2: is_day_night_rear_view_mirror = st.selectbox("Day/Night Mirror", list(yes_no_map.keys()))
    with col3: is_ecw = st.selectbox("ECW", list(yes_no_map.keys()))

    col1, col2 = st.columns(2)
    with col1: is_speed_alert = st.selectbox("Speed Alert", list(yes_no_map.keys()))

    # Centered button
    st.markdown("<br>", unsafe_allow_html=True)
    centered = st.columns(3)
    with centered[1]:
        submitted = st.form_submit_button(
            "🔍 Predict Claim",
            use_container_width=True
        )

# ================= Prediction ====================
if submitted:

    row = pd.DataFrame([{
        "policy_tenure": policy_tenure,
        "age_of_car": age_of_car,
        "age_of_policyholder": age_of_policyholder,
        "area_cluster": area_cluster_map[area_cluster],
        "population_density": population_density,
        "make": make,
        "segment": segment_map[segment],
        "model": model_map[model],
        "fuel_type": fuel_type_map[fuel_type],
        "max_torque": max_torque_map[max_torque],
        "max_power": max_power_map[max_power],
        "engine_type": engine_type_map[engine_type],
        "airbags": airbags,
        "is_esc": yes_no_map[is_esc],
        "is_adjustable_steering": yes_no_map[is_adjustable_steering],
        "is_tpms": yes_no_map[is_tpms],
        "is_parking_sensors": yes_no_map[is_parking_sensors],
        "is_parking_camera": yes_no_map[is_parking_camera],
        "rear_brakes_type": rear_brake_map[rear_brakes_type],
        "displacement": displacement,
        "cylinder": cylinder,
        "transmission_type": transmission_map[transmission_type],
        "gear_box": gear_box,
        "steering_type": steering_type_map[steering_type],
        "turning_radius": turning_radius,
        "length": length,
        "width": width,
        "height": height,
        "gross_weight": gross_weight,
        "is_front_fog_lights": yes_no_map[is_front_fog_lights],
        "is_rear_window_wiper": yes_no_map[is_rear_window_wiper],
        "is_rear_window_washer": yes_no_map[is_rear_window_washer],
        "is_rear_window_defogger": yes_no_map[is_rear_window_defogger],
        "is_brake_assist": yes_no_map[is_brake_assist],
        "is_power_door_locks": yes_no_map[is_power_door_locks],
        "is_central_locking": yes_no_map[is_central_locking],
        "is_power_steering": yes_no_map[is_power_steering],
        "is_driver_seat_height_adjustable": yes_no_map[is_driver_seat_height_adjustable],
        "is_day_night_rear_view_mirror": yes_no_map[is_day_night_rear_view_mirror],
        "is_ecw": yes_no_map[is_ecw],
        "is_speed_alert": yes_no_map[is_speed_alert],
        "ncap_rating": ncap_rating
    }])

    prediction = pipeline.predict(row)[0]

    if prediction == 1:
        st.error("🚨 **Claim Likely:** This policy is predicted to result in a claim.")
    else:
        st.success("✅ **No Claim Likely:** This policy is predicted to be safe.")
