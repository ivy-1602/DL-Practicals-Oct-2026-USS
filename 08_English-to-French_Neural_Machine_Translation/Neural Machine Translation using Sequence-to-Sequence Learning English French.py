# ============================================================
# PRACTICAL 9
# English-to-French Neural Machine Translation
# Sequence-to-Sequence Learning using LSTM
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import time

# ============================================================
# 1. ENGLISH-FRENCH SENTENCE PAIRS
# ============================================================

pairs = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("how are you", "comment allez vous"),
    ("i am fine", "je vais bien"),
    ("i love you", "je t aime"),
    ("see you soon", "a bientot"),
    ("see you tomorrow", "a demain"),
    ("what is your name", "comment vous appelez vous"),
    ("my name is john", "je m appelle john"),
    ("where are you", "ou etes vous"),
    ("i am at home", "je suis a la maison"),
    ("i like music", "j aime la musique"),
    ("i like books", "j aime les livres"),
    ("i like coffee", "j aime le cafe"),
    ("this is my house", "c est ma maison"),
    ("this is my book", "c est mon livre"),
    ("i am going home", "je rentre a la maison"),
    ("i want to learn", "je veux apprendre"),
    ("i want to study", "je veux etudier"),
    ("i am learning french", "j apprends le francais"),
    ("she is my friend", "elle est mon amie"),
    ("he is my brother", "il est mon frere"),
    ("we are students", "nous sommes etudiants"),
    ("they are happy", "ils sont heureux"),
    ("the weather is good", "il fait beau"),
    ("today is a good day", "aujourd hui est une bonne journee"),
    ("i am going to school", "je vais a l ecole"),
    ("we are going to school", "nous allons a l ecole")
]

print("=" * 55)
print("ENGLISH TO FRENCH SEQUENCE-TO-SEQUENCE TRANSLATION")
print("=" * 55)

# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
    return text.strip()


english = [clean(x[0]) for x in pairs]

french = [
    "<start> " + clean(x[1]) + " <end>"
    for x in pairs
]

# ============================================================
# 3. TOKENIZATION
# ============================================================

eng_tokenizer = tf.keras.preprocessing.text.Tokenizer()
fra_tokenizer = tf.keras.preprocessing.text.Tokenizer()

eng_tokenizer.fit_on_texts(english)
fra_tokenizer.fit_on_texts(french)

X = eng_tokenizer.texts_to_sequences(english)
Y = fra_tokenizer.texts_to_sequences(french)

# ============================================================
# 4. SEQUENCE PADDING
# ============================================================

max_eng = max(len(x) for x in X)
max_fra = max(len(x) for x in Y)

X = tf.keras.preprocessing.sequence.pad_sequences(
    X,
    maxlen=max_eng,
    padding="post"
)

Y = tf.keras.preprocessing.sequence.pad_sequences(
    Y,
    maxlen=max_fra,
    padding="post"
)

# Decoder input and target
decoder_input = Y[:, :-1]
decoder_target = Y[:, 1:]

print("\nEnglish vocabulary:", len(eng_tokenizer.word_index))
print("French vocabulary :", len(fra_tokenizer.word_index))
print("Maximum English length:", max_eng)
print("Maximum French length :", max_fra)

# ============================================================
# 5. MODEL PARAMETERS
# ============================================================

EMBEDDING_DIM = 32
LSTM_UNITS = 64

english_vocab = len(eng_tokenizer.word_index) + 1
french_vocab = len(fra_tokenizer.word_index) + 1

# ============================================================
# 6. ENCODER
# ============================================================

encoder_inputs = tf.keras.Input(
    shape=(max_eng,),
    name="encoder_input"
)

encoder_embedding = tf.keras.layers.Embedding(
    english_vocab,
    EMBEDDING_DIM,
    name="encoder_embedding"
)

encoder_embedded = encoder_embedding(encoder_inputs)

encoder_lstm = tf.keras.layers.LSTM(
    LSTM_UNITS,
    return_state=True,
    name="encoder_lstm"
)

encoder_output, encoder_h, encoder_c = encoder_lstm(
    encoder_embedded
)

# ============================================================
# 7. DECODER
# ============================================================

decoder_inputs = tf.keras.Input(
    shape=(max_fra - 1,),
    name="decoder_input"
)

decoder_embedding = tf.keras.layers.Embedding(
    french_vocab,
    EMBEDDING_DIM,
    name="decoder_embedding"
)

decoder_embedded = decoder_embedding(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(
    LSTM_UNITS,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)

decoder_output, _, _ = decoder_lstm(
    decoder_embedded,
    initial_state=[encoder_h, encoder_c]
)

decoder_dense = tf.keras.layers.Dense(
    french_vocab,
    activation="softmax",
    name="decoder_output"
)

outputs = decoder_dense(decoder_output)

# ============================================================
# 8. COMPLETE SEQ2SEQ MODEL
# ============================================================

model = tf.keras.Model(
    [encoder_inputs, decoder_inputs],
    outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

print("\nModel Summary:")
model.summary()

# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining model...")

start_time = time.time()

history = model.fit(
    [X, decoder_input],
    decoder_target[..., None],
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    verbose=0
)

training_time = time.time() - start_time

print("\nTraining completed!")
print("Training time:", round(training_time, 2), "seconds")

# ============================================================
# 10. TRAINING AND VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.title("Seq2Seq Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "seq2seq_loss.png",
    dpi=200
)

plt.show()

# ============================================================
# 11. ENCODER MODEL FOR INFERENCE
# ============================================================

encoder_model = tf.keras.Model(
    encoder_inputs,
    [encoder_h, encoder_c]
)

# Reverse French vocabulary
reverse_french = {
    value: key
    for key, value in fra_tokenizer.word_index.items()
}

start_token = fra_tokenizer.word_index["start"]
end_token = fra_tokenizer.word_index["end"]

# ============================================================
# 12. TRANSLATION FUNCTION
# ============================================================

def translate(sentence):

    sentence = clean(sentence)

    sequence = eng_tokenizer.texts_to_sequences(
        [sentence]
    )

    sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_eng,
        padding="post"
    )

    # Get encoder states
    state_h, state_c = encoder_model.predict(
        sequence,
        verbose=0
    )

    # Start decoder with <start>
    current_token = np.array(
        [[start_token]]
    )

    translated_words = []

    for _ in range(max_fra):

        # Decoder embedding
        embedded = decoder_embedding(
            current_token
        )

        # Decoder LSTM
        decoder_out, state_h, state_c = decoder_lstm(
            embedded,
            initial_state=[state_h, state_c]
        )

        # Predict next word
        probabilities = decoder_dense(
            decoder_out
        )

        predicted_id = int(
            tf.argmax(
                probabilities[0, -1]
            ).numpy()
        )

        predicted_word = reverse_french.get(
            predicted_id,
            ""
        )

        # Stop at <end>
        if predicted_id == end_token:
            break

        if predicted_word:
            translated_words.append(
                predicted_word
            )

        # Feed predicted word back
        current_token = np.array(
            [[predicted_id]]
        )

    return " ".join(translated_words)


# ============================================================
# 13. TEST TRANSLATIONS
# ============================================================

test_sentences = [
    "hello",
    "good morning",
    "i love you",
    "i like music",
    "i am going home",
    "we are students"
]

print("\n" + "=" * 55)
print("TRANSLATION RESULTS")
print("=" * 55)

for sentence in test_sentences:

    translation = translate(sentence)

    print(
        sentence,
        " -> ",
        translation
    )

# ============================================================
# 14. SIMPLE BLEU-LIKE SCORE
# ============================================================

def bleu_like(reference, candidate):

    reference_words = reference.split()
    candidate_words = candidate.split()

    if len(candidate_words) == 0:
        return 0.0

    matches = sum(
        1
        for word in candidate_words
        if word in reference_words
    )

    return matches / len(candidate_words)


bleu_scores = []
sentence_lengths = []

print("\n" + "=" * 55)
print("BLEU-LIKE SCORE")
print("=" * 55)

for english_sentence, french_sentence in pairs:

    prediction = translate(
        english_sentence
    )

    reference = (
        french_sentence
        .replace("<start>", "")
        .replace("<end>", "")
        .strip()
    )

    score = bleu_like(
        reference,
        prediction
    )

    bleu_scores.append(score)
    sentence_lengths.append(
        len(english_sentence.split())
    )

    print(
        f"{english_sentence:30} "
        f"BLEU-like: {score:.2f}"
    )

average_bleu = np.mean(
    bleu_scores
)

print("\nAverage BLEU-like Score:",
      round(average_bleu, 3))

# ============================================================
# 15. BLEU VS SENTENCE LENGTH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    sentence_lengths,
    bleu_scores,
    s=70
)

plt.xlabel("English Sentence Length")
plt.ylabel("BLEU-like Score")
plt.title(
    "Translation Quality vs Sentence Length"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "bleu_sentence_length.png",
    dpi=200
)

plt.show()

# ============================================================
# 16. PERPLEXITY
# ============================================================

validation_loss = history.history[
    "val_loss"
][-1]

perplexity = np.exp(
    min(validation_loss, 10)
)

print("\n" + "=" * 55)
print("EVALUATION")
print("=" * 55)

print(
    "Final Training Loss:",
    round(history.history["loss"][-1], 4)
)

print(
    "Final Validation Loss:",
    round(validation_loss, 4)
)

print(
    "Perplexity:",
    round(perplexity, 4)
)

print(
    "Average BLEU-like Score:",
    round(average_bleu, 4)
)

# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("PRACTICAL COMPLETED")
print("=" * 55)

print("Saved graphs:")
print("1. seq2seq_loss.png")
print("2. bleu_sentence_length.png")

print("\nModel:")
print("Encoder  : Embedding + LSTM")
print("Decoder  : Embedding + LSTM + Softmax")
print("Optimizer: Adam")
print("Dataset  : Small English-French demonstration pairs")
print("Training : 30 epochs")