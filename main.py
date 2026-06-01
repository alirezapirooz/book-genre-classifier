import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC


# LOAD DATA

df = pd.read_parquet("/root/alirezap/books_9genres_clean.parquet")

X = df["Description"].astype(str).values
y = np.array(df["Genres"].tolist())

print("Dataset:", X.shape, y.shape)


# SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)


# TOKENIZER

vocab_size = 10000
max_len = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def encode(X):
    return pad_sequences(
        tokenizer.texts_to_sequences(X),
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

X_train = encode(X_train)
X_val = encode(X_val)
X_test = encode(X_test)

with open("/root/alirezap/tokenizer_9genres.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# CLASS WEIGHTS 

num_classes = y.shape[1]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.argmax(y_train, axis=1)
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# MODEL

model = Sequential([
    Embedding(vocab_size, 128),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)

# CALLBACKS

checkpoint = ModelCheckpoint(
    "/root/alirezap/best_model_9genres.keras",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

# TRAIN

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop],
    verbose=2
)

# EVALUATE

loss, acc, auc = model.evaluate(X_test, y_test, verbose=1)

print("\nFINAL RESULTS")
print("Loss:", loss)
print("Accuracy:", acc)
print("AUC:", auc)

# SAVE MODEL

model.save("/root/alirezap/book_genre_model_9genres.keras")

print("DONE ✔️")