from sklearn.datasets import fetch_20newsgroups
import re


def clean_text(text):
    """
    Clean dataset text by removing:
    - email headers
    - special characters
    - extra spaces
    """

    text = text.lower()

    # remove email headers
    text = re.sub(r"from:.*", "", text)
    text = re.sub(r"subject:.*", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset():

    print("Loading 20 Newsgroups dataset...")

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data

    cleaned_documents = []

    for doc in documents:

        cleaned = clean_text(doc)

        if len(cleaned) > 50:
            cleaned_documents.append(cleaned)

    print("Total cleaned documents:", len(cleaned_documents))

    return cleaned_documents