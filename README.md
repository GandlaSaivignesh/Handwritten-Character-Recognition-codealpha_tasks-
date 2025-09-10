# Handwritten-Character-Recognition-codealpha_tasks-
📌 Project Description

This project implements a Handwritten Character Recognition system using Convolutional Neural Networks (CNNs). It can classify handwritten digits (0–9) from the popular MNIST dataset. The project also provides a simple Streamlit web app to test predictions interactively.

🚀 Features

Preprocessing of image data (grayscale normalization).

CNN-based deep learning model using TensorFlow/Keras.

Achieves high accuracy on the MNIST dataset.

Scripts for training and prediction.

Streamlit demo app to upload handwritten digit images and get predictions.

📂 Project Structure
Handwritten-Character-Recognition/
│── README.md
│── requirements.txt
│── src/
│   ├── model.py       # CNN model definition
│   ├── train.py       # Training script (MNIST dataset)
│   └── predict.py     # Prediction script
│── app/
│   └── streamlit_app.py  # Web app for testing
│── outputs/           # Saved trained models

⚡ How to Run

Install dependencies:

pip install -r requirements.txt


Train the model on MNIST:

python src/train.py --epochs 5 --batch_size 32 --output_dir outputs


Predict a single handwritten digit:

python src/predict.py --model_path outputs/best_model.h5 --image sample.png


Run the Streamlit app:

streamlit run app/streamlit_app.py

📊 Dataset

MNIST: Handwritten digits dataset (0–9).

(Optional Extension) EMNIST: Handwritten letters A–Z.

🏆 Results

CNN achieves >98% accuracy on MNIST test data.

Can be extended for full A–Z handwritten alphabet classification.
