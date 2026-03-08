# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight **semantic search system** built on the **20 Newsgroups dataset**.
The system enables users to query documents using natural language and retrieves semantically similar posts.

The solution combines:

* Transformer-based **text embeddings**
* **FAISS vector database** for similarity search
* **Fuzzy clustering** for overlapping topic discovery
* A **custom semantic cache** to avoid redundant computations
* A **FastAPI service** to expose the system as an API

The project demonstrates how modern AI-powered search systems are designed.

---

# System Architecture

```
User Query
    ↓
Query Embedding (Sentence Transformer)
    ↓
Semantic Cache Lookup
    ↓
Cache Hit → Return Cached Result
    ↓
Cache Miss
    ↓
Vector Search (FAISS)
    ↓
Retrieve Most Similar Document
    ↓
Determine Dominant Cluster
    ↓
Store Result in Cache
    ↓
Return Response
```

---

# Dataset

This project uses the **Twenty Newsgroups Dataset**.

The dataset contains approximately **20,000 documents** distributed across **20 topic categories** including:

* computers
* science
* sports
* politics
* religion

The dataset is accessed using:

```
sklearn.datasets.fetch_20newsgroups()
```

To reduce noise, the following elements are removed:

* email headers
* quotes
* signatures

---

# Key Components

## 1. Embedding Model

Text documents and queries are converted into vector representations using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Features:

* 384-dimensional embeddings
* optimized for semantic similarity tasks
* lightweight and fast

---

## 2. Vector Database

The document embeddings are stored in a **FAISS index**.

FAISS enables:

* fast nearest neighbor search
* scalable similarity retrieval
* efficient vector indexing

---

## 3. Fuzzy Clustering

Instead of assigning each document to a single cluster, **Fuzzy C-Means clustering** is used.

Each document belongs to multiple clusters with different membership probabilities.

Example:

```
Document:
Cluster 2 → 0.52
Cluster 5 → 0.33
Cluster 7 → 0.15
```

This better reflects the **overlapping semantic structure** of the dataset.

---

## 4. Semantic Cache

Traditional caching only works when queries are identical.

This system implements a **semantic cache** that detects similar queries using cosine similarity between embeddings.

Example:

```
Query 1: "space shuttle launch"
Query 2: "how does the space shuttle launch"
```

Even though the wording differs, the semantic meaning is similar.

If similarity exceeds a threshold, the cached result is returned.

This reduces:

* redundant computation
* query latency

---

# API Endpoints

The system exposes a **FastAPI service**.

## POST `/query`

Send a natural language query.

Example request:

```
{
 "query": "space shuttle launch"
}
```

Example response:

```
{
 "query": "space shuttle launch",
 "cache_hit": false,
 "matched_query": null,
 "similarity_score": null,
 "result": "...retrieved document text...",
 "dominant_cluster": 6
}
```

If a similar query exists in the cache:

```
{
 "cache_hit": true
}
```

---

## GET `/cache/stats`

Returns cache statistics.

Example:

```
{
 "total_entries": 2,
 "hit_count": 1,
 "miss_count": 1,
 "hit_rate": 0.5
}
```

---

## DELETE `/cache`

Clears all cached entries.

---

# Installation

## 1. Clone Repository

```
git clone https://github.com/yourusername/semantic-search-cache.git
cd semantic-search-cache
```

---

## 2. Create Virtual Environment

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

---

# Run the Application

Start the FastAPI server:

```
uvicorn api.main:app --reload
```

Open the API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Scikit-learn
* Scikit-Fuzzy
* NumPy

---

# Possible Improvements

Future improvements could include:

* persistent cache storage
* cluster visualization
* embedding storage on disk
* distributed vector search
* query analytics

---

# Author

Teja
AI Engineer Assignment Submission
