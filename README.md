
# 🤖 AI Chatbot Using LSTM

This project is a simple AI-powered chatbot built using an LSTM-based deep learning model. It uses natural language processing (NLP) techniques to classify user input into predefined intents and generate appropriate responses.

---

## 📌 Features

- Built using **TensorFlow** and **Keras**
- Uses **LSTM** for understanding text patterns
- Trained on a custom `intents.json` file
- Saves trained model and tokenizer for future inference
- Can be extended or integrated into a web/chat interface

---

## 📁 Dataset (`intents.json`)

The dataset includes multiple user intents like:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hey", "Hello", "Is anyone there?"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["See you!", "Have a nice day!", "Bye! Come back again soon."]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "That's helpful"],
      "responses": ["You're welcome!", "No problem!", "Happy to help!"]
    }
  ]
}
```

Each intent includes:
- `tag`: Category of the intent
- `patterns`: Possible user inputs
- `responses`: Bot replies (randomly selected)

---

## 🧠 Model Architecture

- **Embedding Layer** – Converts words into vectors
- **LSTM Layer** – Captures temporal patterns in sequences
- **Dense Layers** – Fully connected layers for classification
- **Output Layer** – Uses `softmax` activation for multi-class output

---

## 🛠️ Training Pipeline

1. Load and parse `intents.json`
2. Tokenize and pad the sequences
3. Label encode the intent tags
4. Build the model:
   - `Embedding` → `LSTM` → `Dense`
5. Train the model on the prepared data
6. Save:
   - Trained model: `chat_model.h5`
   - Tokenizer: `tokenizer.pickle`
   - Label encoder: `label_encoder.pickle`

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammadUzair0786/AI-Chatbot-Using-LSTM.git
cd AI-Chatbot-Using-LSTM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training

Use the Jupyter Notebook `Chatbot_Using_LSTM.ipynb` to train the model  
or use the saved model to directly test it.

### 4. Load Model for Inference

```python
from tensorflow.keras.models import load_model
import pickle

model = load_model('chat_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)
```

---

## 🧪 Sample Output

```
You: Hi
Bot: Hello there!

You: Thanks
Bot: You're welcome!

You: Bye
Bot: Goodbye! Take care.
```

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pickle
- Jupyter Notebook

---

## 📚 Learning Outcome

✅ Learned how to:
- Structure chatbot intent data
- Preprocess text for NLP
- Apply LSTM for sequence modeling
- Save and reuse trained models

---

## 🔗 Project Link

👉 [GitHub Repository](https://github.com/MuhammadUzair0786/AI-Chatbot-Using-LSTM)

---

## 🙌 Let’s Connect

If you're working on a similar project or have suggestions, feel free to connect with me!
