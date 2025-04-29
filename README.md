# ASL Alphabet Recognition using CNN and Streamlit

A real-time American Sign Language (ASL) alphabet recognition app built with a Convolutional Neural Network (CNN), trained on the Sign Language MNIST dataset, and deployed using a webcam interface with Streamlit.

---

## 🚀 Features

- Real-time ASL letter detection from webcam feed
- Live prediction confidence displayed on screen
- Deep CNN model with data augmentation for better accuracy
- Streamlit web app interface
- Supports exportable `.keras` model format

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy / Pandas

---

## 📁 Project Structure

```
asl-alphabet-detector/
├── app/
│   └── asl_app.py              # Streamlit webcam app
├── data/                       # ASL MNIST dataset
├── models/
│   └── sign_mnist_cnn_best.keras    # Best-performing model during training (lowest validation loss)
│   └── sign_mnist_cnn_final.keras   #	Model saved after last training epoch
├── scripts/
│   ├── train_model.py          # CNN model training script
│   └── label_map.py            # Label mapping
│   └── CS670_AI_Project_EDA_ML_Model.ipynb   # Model comparative study(does not have any role in application)
├── requirements.txt            # Pip dependencies
├── .gitignore
└── README.md
```

---

## 🗂️ File Descriptions

- **`app/asl_app.py`**  
  - Captures live webcam feed and defines a customizable Region of Interest (ROI) for hand sign detection.
  - Preprocesses frames (grayscale conversion, resizing to 28×28, normalization) to match the CNN model input.
  - Loads the trained ASL recognition model (sign_mnist_cnn_best.keras) using Streamlit caching for fast real-time inference.
  - Displays predictions with confidence scores, FPS, and provides smoothing over recent frames to stabilize output.
  - Includes interactive sidebar controls (confidence threshold, ROI size, grayscale preview) and handles webcam errors gracefully.


- **`data/`**  
  - Folder to store training (`sign_mnist_train.csv`) and testing (`sign_mnist_test.csv`) datasets.  

- **`models/sign_mnist_cnn_best.keras`**  
  - The saved Best-performing CNN model during training (lowest validation loss).  
  - Used by the Streamlit app to predict ASL letters without retraining.

- **`scripts/train_model.py`**  
  - Script to train the CNN model using the Sign Language MNIST dataset.  
  - Includes data preprocessing, augmentation, model architecture, training, evaluation, and saving the trained model to the `models/` folder.

- **`scripts/label_map.py`**  
  - Provides a mapping between numeric labels and ASL alphabets (A–Z, skipping J and Z due to dynamic gestures not represented in static images).

- **`requirements.txt`**  
  - Lists all required Python libraries to run the training script and the Streamlit app.

- **`CS670_AI_Project_EDA_ML_Model.ipynb`**  
  - This file has all the implementation history of phase 1 & phase 2.
  - Comparative study of Traditional ML & NN Deep Learning model.
  - Evaluation metrics for all the model
  - First part of phase 3 Code is here, based on this file we extented our projects application



---

## ⚙️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/vishal2609/AI-FINAL-PROJECT.git
   cd AI-FINAL-PROJECT
   ```

2. **Create virtual environment**  
   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model-(Optional)**  
    - Best model is already trained and is part of repo, but if needed you can run the script to re-train model
    - But this is optional step as model is placed in models section already.
    ```bash
    python scripts/train_model.py
    ```

6. **Run the app**  
   ```bash
   streamlit run app/asl_app.py
   ```

---

## 📦 Requirements

```txt
tensorflow
streamlit
opencv-python
numpy
pandas
scikit-learn
```

---

## 📸 Demo
  - This demo link is just to check how app will look like but it wont run as cloud app will not have access to access the webcam
![demo] ([https://asl-app-cs670.streamlit.app/](https://asl-app-cs670.streamlit.app/))

---

