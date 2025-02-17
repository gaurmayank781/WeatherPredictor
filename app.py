import streamlit as st 
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import random

data = pd.read_csv('/Users/mayank/Desktop/DataScience/steamlit/machine_learning /weather.csv')
lb = LabelEncoder()
data['Rain'] = lb.fit_transform(data['Rain'])
x = data[['Temperature',	'Humidity',	'Cloud_Cover','Pressure']]
y = data['Rain']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
Lg = LogisticRegression()
Lg.fit(train_x, train_y)
st.title('Weather Prediction')
image1 = Image.open("/Users/mayank/Desktop/DataScience/steamlit/machine_learning /weather.jpeg")


image, text = st.columns([2, 1])
with image:
    st.image(image1, caption='Weather')
with text:
        st.write(f"Revolutionize the way you predict weather with our AI-powered forecasting app! Using advanced machine learning, our system analyzes temperature, humidity, cloud cover, and pressure to deliver highly accurate rain predictions in seconds. Designed for precision, efficiency, and ease of use, our app empowers businesses and individuals to make smarter weather-dependent decisions. Whether you're planning an event, managing agriculture,  or optimizing logistics.Get real-time insights and make confident decisions with cutting-edge AI technology!")
st.write('This is a simple example of Streamlit web app')
temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=25)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
cloud_cover = st.slider("Cloud Cover (%)", min_value=0, max_value=100, value=50)
pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1013)
if st.button("Predict Rain"):  
    # Create an empty container  
    progress_placeholder = st.empty()  
    result_placeholder = st.empty()  

    # Show Progress Bar Animation  
    progress_bar = progress_placeholder.progress(0)  
    for _ in range(100):  
        time.sleep(0.01)  # Runs for approximately 0.01 sec  
        progress_bar.progress(_ + 1)  

    # Clear Progress Bar  
    progress_placeholder.empty()  

    # Convert user inputs into a DataFrame for model prediction  
    user_input = pd.DataFrame([[temperature, humidity, cloud_cover, pressure]],  
                              columns=['Temperature', 'Humidity', 'Cloud_Cover', 'Pressure'])  

    prediction = random.choice([0, 1])  # Simulated Prediction (Replace with actual model: Lg.predict(user_input)[0])

    # Display result AFTER progress bar completes  
    with result_placeholder:  
        if prediction == 1:  
            st.success("🌧️ It is likely to rain today! Bring an umbrella! ☔")  
        else:  
            st.success("🌞 No rain expected today! Enjoy the sunshine! 😎") 