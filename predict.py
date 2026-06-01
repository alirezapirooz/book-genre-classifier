import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# PATHS
# -------------------------------------------------
MODEL_PATH = "/root/alirezap/best_model_9genres.keras"
TOKENIZER_PATH = "/root/alirezap/tokenizer_9genres.pkl"

# -------------------------------------------------
# LABELS
# -------------------------------------------------
LABELS = [
    "History & Politics",
    "Health & Wellness",
    "Mystery & Thriller",
    "Science Fiction & Fantasy",
    "Countries & Geography",
    "Romance",
    "Philosophy & Religion",
    "Science & Technology",
    "Children & Young Adult"
]

MAX_LEN = 200
TOP_K = 4
THRESHOLD = 0.25

# -------------------------------------------------
# LOAD
# -------------------------------------------------
print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Model outputs :", model.output_shape[-1])
print("Labels count  :", len(LABELS))

assert model.output_shape[-1] == len(LABELS), \
    "ERROR: Model outputs and labels count do not match!"

print("\n✅ Ready!")
print("\n🔥 Genre Classifier Ready (type 'exit' to quit)\n")


# -------------------------------------------------
# ENCODE
# -------------------------------------------------
def encode(text):
    seq = tokenizer.texts_to_sequences([text])

    return pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )


# -------------------------------------------------
# PREDICT
# -------------------------------------------------
def predict(text):
    x = encode(text)

    probs = model.predict(x, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:TOP_K]

    results = [
        (LABELS[i], probs[i])
        for i in top_indices
    ]

    return probs, results


# -------------------------------------------------
# LOOP
# -------------------------------------------------
while True:

    text = input("Enter text ➜ ").strip()

    if text.lower() in ["exit", "quit"]:
        print("\nBye 👋")
        break

    if not text:
        continue

    probs, results = predict(text)

    max_prob = np.max(probs)

    print("\n📚 Predicted Genres:\n")

    if max_prob < THRESHOLD:
        print("• Unknown / Other (low confidence)")
    else:
        for label, score in results:
            print(f"• {label:<30} {score:.1%}")

    print("\n" + "-" * 50 + "\n")