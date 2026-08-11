import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve relative to this file, not the process's current working directory,
# so retrieval works the same whether run locally, via uvicorn, or on a
# deployment platform that starts the app from a different cwd.
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def load_templates():
    if not os.path.isdir(TEMPLATES_DIR):
        raise RuntimeError(f"Templates directory not found at {TEMPLATES_DIR}")

    templates = {}
    for filename in os.listdir(TEMPLATES_DIR):
        if filename.endswith(".txt"):
            doc_type = filename.replace(".txt", "")
            with open(os.path.join(TEMPLATES_DIR, filename)) as f:
                templates[doc_type] = f.read()

    if not templates:
        raise RuntimeError(f"No .txt templates found in {TEMPLATES_DIR}")

    return templates


def retrieve(query: str):
    templates = load_templates()
    doc_types = list(templates.keys())

    # FIX: inject the doc type name into its own text, so the type name
    # itself becomes a strong matching signal (not just the sparse body text)
    enriched_texts = [
        f"{dt} {dt.replace('_', ' ')} {text}"
        for dt, text in zip(doc_types, templates.values())
    ]

    # FIX: stop_words='english' removes filler words (and, for, with)
    # so small coincidental word overlaps don't dominate the match
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(enriched_texts + [query])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])[0]
    best_idx = similarity.argmax()

    return doc_types[best_idx], templates[doc_types[best_idx]]
