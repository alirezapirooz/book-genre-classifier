# 📚✨ Book Genre Classification (9-Class NLP Project)

🚀 A deep learning NLP project that classifies book/text descriptions into 9 different genres using an LSTM-based neural network.

---

# 🔥 Project Overview

📖 This project takes a text description of a book and predicts its genre using a trained deep learning model.

🧠 It understands semantic meaning using NLP and classifies text into one of 9 genres.

---

# 🎯 Supported Genres

📌 History & Politics  
💊 Health & Wellness  
🔍 Mystery & Thriller  
🚀 Science Fiction & Fantasy  
🌍 Countries & Geography  
❤️ Romance  
🧠 Philosophy & Religion  
🔬 Science & Technology  
👶 Children & Young Adult  

---

# 🧠 Model Architecture

⚙️ Tokenization (Keras Tokenizer)  
📏 Padding (max length = 200)  
🧩 Embedding Layer  
🔁 LSTM Network  
🎯 Dense Softmax Output (9 classes)

---

# 📊 Performance

📈 Validation AUC: ~0.75 – 0.77  
📉 Accuracy: ~0.34 – 0.40  

⚠️ Note:  
Model performs better on clear genres like Mystery / Fantasy / Romance.

---

# ⚙️ Features

✨ Real-time CLI prediction system  
🔥 Top-K genre probabilities  
⚠️ Low confidence detection (Unknown handling)  
⚖️ Bias adjustment for imbalanced classes  
🧹 Clean preprocessing pipeline  
