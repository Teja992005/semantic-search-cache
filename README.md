# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight **semantic search system** built using the **20 Newsgroups dataset**.
The system allows users to search documents using natural language and retrieves semantically similar posts.

The system combines:

* Transformer-based **text embeddings**
* **FAISS vector database** for semantic similarity search
* **Fuzzy clustering** to model overlapping topics
* A custom **semantic cache** to avoid redundant computations
* A **FastAPI service** that exposes the system through REST APIs

This project demonstrates the architecture of a modern **AI-powered semantic retrieval system**.

---

# System Architecture

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
Retrieve Similar Document
↓
Determine Dominant Cluster
↓
Store Result in Cache
↓
Return API Response

---

# Dataset

The system uses the **Twenty Newsgroups Dataset** which contains approximately **20,000 documents across 20 categories**.

Example topics include:

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

This ensures the system focuses only on the **semantic content of each document**.

---

# Key Components

## 1. Embedding Model

Documents and queries are converted into vector embeddings using the **Sentence Transformers MiniLM model**.

Model used:

```
sentence-transformers/all-MiniLM-L6-v2
```

Features:

* 384-dimensional embeddings
* optimized for semantic similarity
* lightweight and efficient

---

## 2. Vector Database (FAISS)

Document embeddings are stored in a **FAISS index**, which enables fast similarity search.

FAISS provides:

* efficient nearest neighbor search
* scalable vector indexing
* high performance retrieval

---

## 3. Fuzzy Clustering

Instead of assigning each document to a single cluster, the system uses **Fuzzy C-Means clustering**.

This allows documents to belong to **multiple clusters with different membership probabilities**.

Example:

```
Document Membership

Cluster 2 → 0.52
Cluster 5 → 0.33
Cluster 7 → 0.15
```

This better represents the **overlapping semantic nature of real-world topics**.

---

## 4. Semantic Cache

Traditional caching only works when queries are identical.

This system introduces a **semantic cache** that detects similar queries using **cosine similarity between embeddings**.

Example:

Query 1

```
space shuttle launch
```

Query 2

```
how does the space shuttle launch
```

Even though the wording differs, the semantic meaning is similar.

If similarity exceeds a threshold, the cached result is returned.

Benefits:

* reduces redundant computation
* decreases query latency
* improves system efficiency

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

Returns current cache statistics.

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

## Clone Repository

```
git clone https://github.com/yourusername/semantic-search-cache.git
cd semantic-search-cache
```

---

## Create Virtual Environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

## Install Dependencies

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

This interface allows you to test the API endpoints.

---

# Docker Support

The application can also be run using **Docker** to ensure a consistent runtime environment.

## Build Docker Image

```
docker build -t semantic-search-system .
```

## Run Docker Container

```
docker run -p 8000:8000 semantic-search-system
```

After starting the container, open:

```
http://localhost:8000/docs
```

This will open the FastAPI Swagger interface.

---

# Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Scikit-learn
* Scikit-fuzzy
* NumPy
* Docker

---

# Future Improvements

Possible improvements include:

* persistent cache storage
* cluster visualization
* embedding storage on disk
* distributed vector search
* query analytics

---

# Author

Teja
AI Engineer Assignment Submission
