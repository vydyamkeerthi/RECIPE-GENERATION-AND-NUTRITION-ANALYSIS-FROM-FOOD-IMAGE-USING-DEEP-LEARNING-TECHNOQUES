import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from PIL import Image
from groq import Groq

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ensure the path is correct)
try:
    model_path = 'vgg16_food101_trained.h5'  # Replace with your model path
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Groq API Setup
client = Groq(api_key="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")

# Class indices for Food-101 dataset
class_indices = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", 
    "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", 
    "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate", 
    "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", 
    "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", 
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", 
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", 
    "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", 
    "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", 
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", 
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", 
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", 
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", 
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", 
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", 
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", 
    "tuna_tartare", "waffles"
]

# Home route: Render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
        return jsonify({'error': 'Invalid file type. Only image files are allowed.'}), 400

    if model is None:
        return jsonify({'error': 'Model could not be loaded. Please try again later.'}), 500

    try:
        # Save file to a temporary path
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Load and preprocess image
        image = Image.open(filepath).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_indices[predicted_class_index]

        # Fetch ingredients and nutrition info using Groq API
        groq_response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Provide ingredients for {predicted_class}."
            }],
            model="llama-3.3-70b-versatile",
        )
        ingredients = groq_response.choices[0].message.content

        groq_response_nutrition = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Provide nutritional information for {predicted_class}."
            }],
            model="llama-3.3-70b-versatile",
        )
        nutrition = groq_response_nutrition.choices[0].message.content

        # Clean up temporary file
        os.remove(filepath)

        print(f"Prediction: {predicted_class}")
        return jsonify({
            'predicted_class': predicted_class,
            'ingredients': ingredients,
            'nutrition': nutrition
        }), 200

    except Exception as e:
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
