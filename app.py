from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from PIL import Image
import pickle
import requests
import io

app = Flask(__name__, static_folder=None, static_url_path=None)

# Load the CSV file into a DataFrame
plant_health_df = pd.read_csv('Crop_Health_Tips.csv')

# Load the CNN model
soil_cnn_model = tf.keras.models.load_model('/Users/anushkasharma/Downloads/agr_project/Crop-Prediction-Model/soil_cnn_model.h5')

# Load the trained RandomForestClassifier model
with open('model.pickle', 'rb') as f:
    crop_model = pickle.load(f)

# Define soil types mapping
soil_types = {
    0: "Black Soil",
    1: "Cinder Soil",
    2: "Laterite Soil",
    3: "Peat Soil",
    4: "Yellow Soil"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to appropriate data types
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Call your prediction model here
        predicted_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Fetch health tips for the predicted crop
        health_tips = get_health_tips(predicted_crop)

        # Pass the predicted crop and its health tips to the template
        return render_template('index.html', prediction_text=predicted_crop, health_tips=health_tips)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/health_tips', methods=['POST'])
def health_tips():
    crop = request.form['crop']
    tips = get_health_tips(crop)
    return render_template('health_tips.html', crop=crop, tips=tips)

@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Ensure the file is an image
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'File must be an image'}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        image = transform_image(image)
        
        prediction = soil_cnn_model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        
        predicted_soil = soil_types.get(predicted_class[0], "Unknown Soil Type")
        
        return jsonify({'prediction': f'Recommended Soil Type: {predicted_soil}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def transform_image(image):
    image = image.resize((28, 28))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Convert input features to a numpy array
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict the crop using the trained model
    predicted_crop = crop_model.predict(input_features)
    
    # Return the predicted crop
    return predicted_crop[0]

def get_health_tips(crop):
    # Fetch health tips for the given crop from the DataFrame
    crop_tips = plant_health_df[plant_health_df['Crop'] == crop]
    
    if not crop_tips.empty:
        tips = {
            "soil_health": crop_tips.iloc[0]['Soil Health Tips'],
            "irrigation": crop_tips.iloc[0]['Irrigation Tips'],
            "pest_control": crop_tips.iloc[0]['Pest Control Measures'],
            "disease_prevention": crop_tips.iloc[0]['Disease Prevention Strategies'],
            "climate": crop_tips.iloc[0]['Climate Considerations']
        }
        return tips
    else:
        # Return default tips if no data is found for the crop
        return {
            "soil_health": "No soil health tips available.",
            "irrigation": "No irrigation tips available.",
            "pest_control": "No pest control measures available.",
            "disease_prevention": "No disease prevention strategies available.",
            "climate": "No climate considerations available."
        }

def classify_soil_type(image_file):
    """Classify soil type from an image file."""
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        image = transform_image(image)
        prediction = soil_cnn_model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        return soil_types.get(predicted_class[0], "Unknown Soil Type")
    except Exception as e:
        raise Exception(f"Error classifying soil type: {str(e)}")

def fetch_weather_data(location, api_key="d9da40503a813f549f59c3ad742f463e"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        weather_data = response.json()

        if weather_data["cod"] != 200:
            return None

        weather_info = {
            "temperature": weather_data["main"]["temp"],
            "humidity": weather_data["main"]["humidity"],
            "rainfall": weather_data.get("rain", {}).get("1h", 0),
            "weather_description": weather_data["weather"][0]["description"]
        }

        return weather_info
    except Exception as e:
        raise Exception(f"Error fetching weather data: {str(e)}")

@app.route('/get_weather', methods=['POST'])
def get_weather():
    try:
        location = request.form['location']
        weather_data = fetch_weather_data(location)

        if not weather_data:
            return render_template('error.html', 
                                message="Could not fetch weather data for the provided location"), 400

        return render_template('weather_info.html', 
                             weather_data=weather_data,
                             location=location)
    except Exception as e:
        # Fallback to JSON response if template rendering fails
        return jsonify({
            "error": str(e),
            "message": "Failed to fetch weather data"
        }), 500

@app.route('/detailed_recommendations', methods=['POST'])
def detailed_recommendations():
    try:
        # Get form data
        location = request.form.get('location')
        soil_image = request.files.get('soil_image')
        
        if not location or not soil_image:
            return jsonify({
                'success': False,
                'error': 'Both location and soil image are required'
            }), 400

        # Process data
        soil_type = classify_soil_type(soil_image)
        weather_data = fetch_weather_data(location)
        
        if not weather_data:
            return jsonify({
                'success': False,
                'error': 'Could not fetch weather data'
            }), 400

        recommendations = generate_recommendations(soil_type, weather_data)
        
        # Return JSON response
        return jsonify({
            'success': True,
            'soil_type': soil_type,
            'weather_data': weather_data,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_recommendations(soil_type, weather_data):
    recommendations = []
    
    # Soil-based recommendations
    soil_recommendations = {
        "Black Soil": "Black soil is rich in clay and good for cotton, wheat, and cereals.",
        "Cinder Soil": "Cinder soil is well-draining, suitable for cacti and succulents.",
        "Laterite Soil": "Laterite soil is good for tea, coffee, and rubber plants.",
        "Peat Soil": "Peat soil is rich in organic matter, good for vegetables and fruits.",
        "Yellow Soil": "Yellow soil is good for maize, millet, and potatoes."
    }
    
    recommendations.append(soil_recommendations.get(soil_type, "General soil preparation recommended."))
    
    # Weather-based recommendations
    temp = weather_data["temperature"]
    if temp > 30:
        recommendations.append("High temperatures: consider heat-resistant crops like sorghum or millet.")
    elif temp < 15:
        recommendations.append("Low temperatures: consider cold-resistant crops like kale or spinach.")
    
    rainfall = weather_data["rainfall"]
    if rainfall > 50:
        recommendations.append("High rainfall expected: ensure proper drainage for your crops.")
    elif rainfall < 10:
        recommendations.append("Low rainfall expected: consider drought-resistant crops or irrigation.")
    
    humidity = weather_data["humidity"]
    if humidity > 70:
        recommendations.append("High humidity: watch for fungal diseases and ensure good air circulation.")
    
    return recommendations

if __name__ == "__main__":
    app.run(debug=True)