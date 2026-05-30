from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datasets import load_from_disk

import numpy as np
import pickle


# load dataset

ds = load_from_disk("/root/alireza/dataset")
df = ds["train"].to_pandas()

X = df["Description"].astype(str)

# multi-label targets
y = np.array(df["Genres"].tolist())


# split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42
)


# tokenizer

max_features = 10000
max_length = 200

tokenizer = Tokenizer(
    num_words=max_features,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_val_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_val),
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=max_length,
    padding="post",
    truncating="post"
)


# save tokenizer

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# model

model = Sequential([
    Embedding(input_dim=max_features, output_dim=128),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(18, activation="sigmoid")
])


# compile model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)


# training callbacks

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# train model

history = model.fit(
    X_train_seq,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_seq, y_val),
    callbacks=[checkpoint, early_stopping],
    verbose=2
)


# save final model

model.save("book_genre_model.keras")


# evaluate model

loss, acc = model.evaluate(
    X_test_seq,
    y_test
)

print("LOSS:", loss)
print("BINARY ACC:", acc)