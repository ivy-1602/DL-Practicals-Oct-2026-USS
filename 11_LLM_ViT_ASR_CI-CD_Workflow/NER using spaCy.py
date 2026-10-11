# ============================================================
# PRACTICAL 11
# Named Entity Recognition using spaCy
# ============================================================

import spacy
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD PRE-TRAINED SPACY MODEL
# ============================================================

print("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("Model loaded successfully!")

# ============================================================
# 2. BASIC PREPROCESSING
# ============================================================

def preprocess(text):
    return " ".join(text.split())


# ============================================================
# 3. NER FUNCTION
# ============================================================

def extract_entities(text):

    text = preprocess(text)

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ in ["PERSON", "ORG", "GPE"]:

            label = ent.label_

            # spaCy uses GPE for geographical/political locations
            if label == "GPE":
                label = "LOCATION"

            entities.append(
                (ent.text, label)
            )

    return entities


# ============================================================
# 4. FIVE CUSTOM NEWS SENTENCES
# ============================================================

news = [

    "Elon Musk announced a new technology project in California.",

    "Microsoft CEO Satya Nadella visited London for a technology conference.",

    "Google opened a new artificial intelligence research center in Paris.",

    "Indian Prime Minister Narendra Modi addressed students in New Delhi.",

    "Apple announced a new product during an event in Cupertino."
]


# ============================================================
# 5. EXTRACT ENTITIES
# ============================================================

print("\n")
print("=" * 60)
print("NAMED ENTITY RECOGNITION RESULTS")
print("=" * 60)

all_entities = []

for i, sentence in enumerate(news, 1):

    entities = extract_entities(sentence)

    print(f"\nNews {i}:")
    print(sentence)

    print("Entities:")

    for entity, label in entities:

        print(
            f"  {entity} --> {label}"
        )

        all_entities.append(label)


# ============================================================
# 6. ENTITY COUNT
# ============================================================

person_count = all_entities.count("PERSON")
org_count = all_entities.count("ORG")
location_count = all_entities.count("LOCATION")

print("\n")
print("=" * 60)
print("ENTITY SUMMARY")
print("=" * 60)

print("PERSON   :", person_count)
print("ORG      :", org_count)
print("LOCATION :", location_count)


# ============================================================
# 7. ENTITY DISTRIBUTION GRAPH
# ============================================================

labels = [
    "PERSON",
    "ORG",
    "LOCATION"
]

counts = [
    person_count,
    org_count,
    location_count
]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    labels,
    counts
)

plt.title(
    "Named Entity Distribution"
)

plt.xlabel("Entity Type")
plt.ylabel("Number of Entities")

plt.grid(
    axis="y",
    alpha=0.3
)

# Display values
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        str(int(height)),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "ner_entity_distribution.png",
    dpi=200
)

plt.show()


# ============================================================
# 8. PIPELINE INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("SPACY PIPELINE")
print("=" * 60)

print(nlp.pipe_names)

print("\nNER pipeline completed successfully.")