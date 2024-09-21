import streamlit as st
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate dummies data

np.random.seed(50)
num_sample = 5000
population_density =  np.random.randint(500, 1000, num_sample)
gdp = np.random.randint(20, 100, num_sample)
public_transport_availability = np.random.choice([0,1],num_sample)
road_infrastructure_quality = np.random.randint(1,6,num_sample)

demand_public_transit = (population_density*0.03 + gdp * 1.5 + public_transport_availability*100 + road_infrastructure_quality*50 + np.random.normal(0,50,num_sample))

# Create Datafram

df = pd.DataFrame({
    "Population_Density":population_density,
    "GDP":gdp,
    "Public_Transport_Availability":public_transport_availability,
    "Road_Infrastructure_Quality":road_infrastructure_quality,
    "Demand_Public_Transit":demand_public_transit
})

print(df)

# Features and target variables

x = df[["Population_Density","GDP","Public_Transport_Availability","Road_Infrastructure_Quality"]]
y = df["Demand_Public_Transit"]

# Split data int training and testing data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

# Train a Model

model = RandomForestRegressor(n_estimators = 100, random_state= 42)
model.fit(x_train,y_train)

# Predict on test data

y_pred  = model.predict(x_test)

# Evaluate the model

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error:{rmse}")

# Streamlit app

st.set_page_config(layout="wide")


st.title("Transportation Demand Prediction")

# Banner Image 
from PIL import Image

image = Image.open("2.png")
# Resize the image while maintaining the aspect ratio
image = image.resize((3000, 1000))  # Adjust the width and height as needed
st.image(image)



# Create input widgets

population_density_input = st.number_input("Population_Density", min_value=500,max_value=10000, value = 3000)
gdp_input = st.number_input("GDP (in 1000s)",min_value = 20, max_value = 100, value = 50)
public_transport_availability_input =  st.selectbox("Public_Transport_Availability",[0,1])
road_infrastructure_quality_input = st.slider("Road_Infrastructure_Quality",1, 5,   value = 3)

# Predict demand based on input
input_data = pd.DataFrame({
        "Population_Density":[population_density_input],
        "GDP":[gdp_input],
        "Public_Transport_Availability":[public_transport_availability_input],
        "Road_Infrastructure_Quality":[road_infrastructure_quality_input]
    })

# Prediction button

if st.button("Predict Demand"):
    prediction = model.predict(input_data)
    st.write()
    st.spinner('Making your prediction...')
    time.sleep(2)  # Simulate a delay for prediction
    predicted_demand = 100  # Placeholder for your prediction logic

    # Show prediction result
    st.success(f"Predicted Demand for Public Transit: {prediction[0]:.0f} trips/day")

    # Fun effects
    st.balloons()
    st.snow()

    # Show progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    # Show additional messages
    st.info('Thank you for using the Transportation Demand Predictor!')

    

    

    
    

    