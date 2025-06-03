# 🍽️ AI-Powered Food Image Classification and Nutrition Dashboard

An intelligent web-based dashboard that recognizes food from images and generates instant ingredient lists and nutrition facts using deep learning and large language models.

---

## 🌟 Features

- 🔍 **Food Image Recognition**: Upload a food photo and let the AI classify it from 101 food categories.
- 🧬 **Deep Learning Model**: Powered by a custom-trained **VGG16 CNN** architecture.
- 🧠 **Smart Ingredient & Nutrition Generation**: Integrates **Groq LLM API (LLaMA 3.3 70B)** to provide human-readable food insights.
- 🎨 **Modern UI**: Built with responsive HTML + TailwindCSS frontend.
- 💬 **Interactive Output**: Displays predictions, ingredients, and nutritional content side-by-side.

---

## 🚀 How It Works

1. **Frontend**: Users upload an image via a simple dashboard UI.
2. **Backend**:
   - Image is processed and resized (224x224).
   - The image is fed into a trained VGG16 model (`vgg16_food101_trained.h5`).
   - Predicted class is passed to Groq's LLM to fetch:
     - 🧂 Ingredients
     - 🥗 Nutritional facts
3. **Results**: Displayed in real-time with smooth transitions and a modern layout.

---

## 🧠 Model Details

- Based on **VGG16** architecture (no top layers).
- Trained on [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) with:
  - 📸 Image augmentations
  - 🔢 101 output classes (softmax layer)
  - 📊 ~80 epochs, batch size 16
- Accuracy: Evaluated with custom visualization + human loop.

---

## 📁 Folder Structure

├── app.py # Flask backend for image upload and API integration
├── model.py # Model architecture and training code
├── vgg16_food101_trained.h5 # Trained model weights
├── templates/
│ └── index.html # Dashboard UI
├── static/ # (Optional) for styles/images if extended
└── api.txt # API key and class labels
