# 🤖 Sarcasm Detection using LSTM

This project is a **sarcasm detection model** built using **Deep Learning (LSTM)**. The model is trained on a dataset of sarcastic and non-sarcastic texts and achieves **high accuracy** in detecting sarcasm in textual data. The project includes training, testing, and deployment scripts.

## 🚀 Features
- **LSTM-based sarcasm detection model** trained on textual data.
- **Tokenizer and preprocessing pipeline** to clean and tokenize input text.
- **Deployed model for real-time sarcasm detection** using `predict_sarcasm.py`.
- **Achieves high accuracy on the dataset** (detailed below).
  
---

## 📂 Project Structure
```
├── sarcasm_detector.ipynb  # Jupyter Notebook with full training pipeline
├── tokenizer.pkl           # Tokenizer for text preprocessing
├── lstm_sarcasm_model.h5   # Trained LSTM model (saved in .h5 format)
├── predict_sarcasm.py      # Script for real-time sarcasm detection
├── train.csv               # Training dataset
├── test.csv                # Testing dataset
├── README.md               # Project documentation
```

---

## 📊 Model Performance
- **LSTM model Accuracy:** 98.95%
- **Testing Accuracy:** 95.83%
- **Precision, Recall, and F1-score:** 0.97, 0.97, 0.97  

---

## 🛠 Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AnanditaSaxenaa/sarcasm-detector.git
   cd sarcasm-detector
   ```

2. Ensure the following files are present:
   - `lstm_sarcasm_model.h5`
   - `tokenizer.pkl`
   - `predict_sarcasm.py`

---

## 🔥 Usage

### 1️⃣ Running the Model (Real-Time Sarcasm Detection)
To test sarcasm detection on custom text, run:
```sh
python predict_sarcasm.py
```
Enter a sentence when prompted, and the model will classify it as **"Sarcastic 😏"** or **"Not Sarcastic 😊"**.

---

## 📝 How It Works
1. **Data Preprocessing**:
   - Cleans text (removes URLs, punctuation, hashtags, etc.).
   - Tokenizes words and converts them to sequences.
   - Pads sequences for uniform input size.

2. **Training (Implemented in `sarcasm_detector.ipynb`)**:
   - Uses an **LSTM model** with embedding layers.
   - Trained using TensorFlow/Keras on a sarcasm-labeled dataset.

3. **Inference (Implemented in `predict_sarcasm.py`)**:
   - Loads the trained model and tokenizer.
   - Predicts sarcasm based on input text.

---

## 📌 Example Prediction
```sh
Enter a sentence (or type 'exit' to quit): "Oh great, another Monday!"
Prediction: Sarcastic 😏

Enter a sentence (or type 'exit' to quit): "I love spending time with my family."
Prediction: Not Sarcastic 😊
```

---

## 🛠 Dependencies
Ensure you have these installed:
```sh
pip install tensorflow nltk pandas numpy scikit-learn matplotlib
```

---

## 🎯 Future Improvements
- Enhance sarcasm detection using **transformer models (BERT, XLM-R)**.
- Deploy as a **web app using Flask/Streamlit**.
- Improve accuracy with **larger datasets and fine-tuning**.

---

## 🏆 Contributing
Feel free to contribute to the project by improving the model, adding new datasets, or optimizing the inference script!

---

## 📝 License
This project is open-source and available under the **MIT License**.

---
