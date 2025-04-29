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
│   └── asl_cnn_model.keras     # Trained model
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
  - Streamlit application script.  
  - Captures webcam feed, processes frames, loads the trained CNN model, makes real-time predictions, and displays the predicted ASL alphabet with the confidence score.

- **`data/`**  
  - Folder to store training (`sign_mnist_train.csv`) and testing (`sign_mnist_test.csv`) datasets.  
  - This folder is ignored in GitHub uploads through `.gitignore` for cleanliness.

- **`models/asl_cnn_model.keras`**  
  - The saved CNN model file after training.  
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
   git clone https://github.com/your-username/asl-alphabet-detector.git
   cd asl-alphabet-detector
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

4. **Make Download dataset**  
   Place `sign_mnist_train.csv` and `sign_mnist_test.csv` inside the `data/` directory.

5. **Train the model**  
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

![demo](https://user-images.githubusercontent.com/your-screenshot-url.png)

---

