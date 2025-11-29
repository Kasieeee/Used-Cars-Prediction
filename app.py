import streamlit as st
import pandas as pd
import joblib
import os
import kagglehub

# Load model pipeline
model = joblib.load("car_price_model.pkl")


# Download latest version
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

csv_files = [
    "vw.csv", "cclass.csv", "audi.csv", "bmw.csv", "vauxhall.csv",
    "ford.csv", "hyundi.csv", "merc.csv",
    "toyota.csv", "skoda.csv", "focus.csv"
]

# Loop through and create a unique DataFrame for each CSV
for file in csv_files:
    file_path = os.path.join(path, file)
    if os.path.exists(file_path):
        # Create a valid Python variable name (replace spaces and dots)
        df_name = file.replace(".csv", "").replace(" ", "_")
        globals()[df_name] = pd.read_csv(file_path)
        print(f"✅ Loaded: {df_name} ({eval(df_name).shape[0]} rows, {eval(df_name).shape[1]} columns)")
    else:
        print(f"File not found: {file}")

make_to_df = {
    "vw": vw,
    "cclass": cclass,
    "audi": audi,
    "bmw": bmw,
    "vauxhall": vauxhall,
    "ford": ford,
    "hyundi": hyundi,
    "merc": merc,
    "toyota": toyota,
    "skoda": skoda,
    "focus": focus
}

# Add "Make" column to each and combine
combined_df = pd.concat(
    [df.assign(Make=make.capitalize()) for make, df in make_to_df.items()],
    ignore_index=True
)

print("Combined DataFrame created successfully!")
print("Shape:", combined_df.shape)

df = combined_df

# Build dictionary: Make -> list of models
make_model_map = (
    df.groupby("Make")["model"]
    .unique()
    .apply(list)
    .to_dict()
)

# Extract full list of makes
make_options = list(make_model_map.keys())

st.title("Used Car Price Predictor 🚗💷")
st.write("Select car details to estimate its used market value.")

# ----------------------------
# User selects Make
# ----------------------------
selected_make = st.selectbox("Make", make_options)

# Filter models based on selected Make
model_options = make_model_map[selected_make]

selected_model = st.selectbox("Model", model_options)

# Other categorical fields
transmission_options = df["transmission"].unique().tolist()
fuel_options = df["fuelType"].unique().tolist()

transmission = st.selectbox("Transmission", transmission_options)
fuelType = st.selectbox("Fuel Type", fuel_options)

# Numeric fields
year = st.number_input("Year", 1990, 2025, 2018)
mileage = st.number_input("Mileage (km)", 0, 300000, 50000)
tax = st.number_input("Tax (£/year)", 0, 600, 150)
mpg = st.number_input("MPG", 10, 200, 55)
engineSize = st.number_input("Engine Size (Litres)", 0.5, 6.5, 1.6)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Make": selected_make,
        "model": selected_model,
        "transmission": transmission,
        "fuelType": fuelType,
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Used Car Price: £{prediction:,.2f}")
