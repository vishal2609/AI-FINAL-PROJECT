# 🤟 ASL Alphabet Recognition using CNN and Streamlit

A real-time American Sign Language (ASL) alphabet recognition app built with a Convolutional Neural Network (CNN), trained on the Sign Language MNIST dataset, and deployed using a webcam interface with Streamlit.

---

## 🚀 Features

- Real-time ASL letter detection from webcam feed
- Live prediction confidence displayed on screen
- Deep CNN model with data augmentation for better accuracy
- Streamlit web app interface
- Supports exportable `.keras` model format
- Clean project structure for easy extension

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy / Pandas

---

## 📁 Project Structure

```
asl-alphabet-detector/
├── app/
│   └── asl_app.py              # Streamlit webcam app
├── data/                       # ASL MNIST dataset (ignored in .gitignore)
├── models/
│   └── asl_cnn_model.keras     # Trained model
├── scripts/
│   ├── train_model.py          # CNN model training script
│   └── label_map.py            # Label mapping (skips J)
├── requirements.txt            # Pip dependencies
├── .gitignore
└── README.md
```

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
   source venv/bin/activate    # On macOS/Linux
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Download dataset**  
   Place `sign_mnist_train.csv` and `sign_mnist_test.csv` inside `data/`.

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

## 📜 License

MIT License. Feel free to use and build on top of this!