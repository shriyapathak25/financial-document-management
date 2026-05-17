# Financial Document Management API

This is a small FastAPI project I built to manage financial documents and search through them using semantic search (RAG).

The main idea behind this project was to create a system where users can upload financial documents, manage access using roles, and search document content using natural language queries instead of only exact keywords.

Right now this project is more of a prototype and learning project. It is not production-ready yet, but it covers authentication, role-based access, document management, vector search, and reranking.

---

# What the project does

The API allows users to:

- upload PDF and DOCX documents
- manage users and roles
- search documents using metadata
- index documents into a vector database
- perform semantic search on document content

Example:

Instead of searching something exact like:

```bash
Q4 revenue
```

Users can search naturally like:

```bash
documents talking about yearly profit growth
```

and the system tries to return the most relevant document chunks.

---

# Tech Stack

This project uses:

- FastAPI
- Qdrant
- LangChain
- HuggingFace Sentence Transformers
- Cross Encoder reranking
- JWT Authentication

---

# Project Setup

## 1. Clone the repository

```bash
git clone https://github.com/shriyapathak/financial-doc-manager
cd financial-doc-manager
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the server

```bash
uvicorn DocumentAIRAG:app --reload
```

The API will run on:

```bash
http://localhost:8000
```

Swagger documentation:

```bash
http://localhost:8000/docs
```

---

# Features

## Authentication

Users can:
- register
- login
- receive JWT tokens

---

## Role-Based Access Control

Different users have different permissions.

| Role | Permissions |
|------|-------------|
| Admin | Full access |
| Financial Analyst | Upload, edit, and view documents |
| Auditor | View and review documents |
| Client | View-only access |

---

## Document Upload

The system supports:

- PDF files
- DOCX files

Each document stores:
- title
- company name
- document type
- upload timestamp
- uploaded user

---

## Metadata Search

Users can search documents using:
- title
- company name
- document type
- uploaded user

---

## Semantic Search (RAG)

This is the main functionality of the project.

When a document gets indexed:

1. text is extracted from the file
2. text is split into chunks
3. embeddings are generated
4. vectors are stored in Qdrant

During search:

1. relevant chunks are retrieved using vector similarity
2. results are reranked using a cross encoder
3. top matching chunks are returned

---

# API Endpoints

## Auth APIs

| Method | Endpoint |
|---|---|
| POST | `/auth/register` |
| POST | `/auth/login` |

---

## User & Role APIs

| Method | Endpoint |
|---|---|
| POST | `/roles/create` |
| POST | `/users/assign-role` |
| GET | `/users/{user_id}/roles` |
| GET | `/users/{user_id}/permissions` |

---

## Document APIs

| Method | Endpoint |
|---|---|
| POST | `/documents/upload` |
| GET | `/documents` |
| GET | `/documents/search` |
| GET | `/documents/{document_id}` |
| DELETE | `/documents/{document_id}` |

---

## RAG APIs

| Method | Endpoint |
|---|---|
| POST | `/rag/index-document` |
| POST | `/rag/search` |
| GET | `/rag/context/{document_id}` |
| DELETE | `/rag/remove-document/{document_id}` |

---

# Current Limitations

This project still has some limitations because it was built mainly as a prototype.

Current limitations:
- data is stored in memory
- passwords are stored as plain text
- uploaded files are stored locally
- no OCR support for scanned PDFs
- no persistent database
- indexing is synchronous

---

# Improvements Planned

Some improvements that can be added later:

- PostgreSQL or MongoDB integration
- password hashing
- async/background indexing
- cloud storage support
- OCR support for scanned files
- better logging and monitoring
- document versioning

---

# Notes

- Smaller chunk sizes improved retrieval quality during testing
- Cross encoder reranking improved search relevance noticeably
- In-memory Qdrant was used to keep setup simple during development

---

# Author

Shriya Pathak
