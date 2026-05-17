import os
import uuid
import logging

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from jose import JWTError, jwt
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from sentence_transformers import CrossEncoder
from PyPDF2 import PdfReader
from docx import Document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Document Management")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# hardcoded temporarily to avoid wiring env config during local testing
SECRET_KEY = "mysecretkey123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# avoiding DB setup for now since auth flows are still changing frequently
users_db = {}
roles = {}
documents = {}
user_roles = {}


# cross-encoder is slower than vector similarity alone,
# but ranking quality was noticeably better for finance-related queries
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class User(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    company_name: str
    document_type: str
    uploaded_by: str
    created_at: datetime
    file_path: str


class SearchRequest(BaseModel):
    query: str


def create_token(data: dict, expires_delta: Optional[timedelta] = None):
    payload = data.copy()

    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=15)
    )

    payload.update({"exp": expire})

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        username = payload.get("sub")

        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")

        if username not in users_db:
            raise HTTPException(status_code=401, detail="User not found")

        return username

    except JWTError:
        raise HTTPException(status_code=401, detail="Token error")


PERMISSIONS = {
    "Admin": ["full_access"],

    "Financial Analyst": [
        "upload_document",
        "edit_document",
        "view_document"
    ],

    "Auditor": [
        "view_document",
        "review_document"
    ],

    "Client": [
        "view_document"
    ]
}


def check_permission(required_permission: str):

    def inner(current_user: str = Depends(get_current_user)):

        assigned_roles = user_roles.get(current_user, [])

        for role in assigned_roles:

            permissions = PERMISSIONS.get(role, [])

            if "full_access" in permissions:
                return current_user

            if required_permission in permissions:
                return current_user

        raise HTTPException(
            status_code=403,
            detail="Permission denied"
        )

    return inner


@app.post("/auth/register")
def register(user: User):

    if user.username in users_db:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    # leaving passwords unhashed for now because this is still a local prototype
    users_db[user.username] = user.password

    return {"message": "registered successfully"}


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):

    password = users_db.get(form_data.username)

    if password != form_data.password:
        raise HTTPException(
            status_code=401,
            detail="Wrong username or password"
        )

    token = create_token({
        "sub": form_data.username
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@app.post("/roles/create")
def create_role(
    role_name: str,
    _=Depends(check_permission("full_access"))
):

    roles[role_name] = role_name

    return {"message": f"Role {role_name} created"}


@app.post("/users/assign-role")
def assign_role(
    username: str,
    role: str,
    _=Depends(check_permission("full_access"))
):

    if username not in users_db:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    if role not in roles and role not in PERMISSIONS:
        raise HTTPException(
            status_code=400,
            detail="Role not found"
        )

    user_roles.setdefault(username, [])

    if role not in user_roles[username]:
        user_roles[username].append(role)

    return {"message": f"{role} assigned to {username}"}


@app.get("/users/{user_id}/roles")
def get_roles(
    user_id: str,
    _=Depends(get_current_user)
):

    return {"roles": user_roles.get(user_id, [])}


@app.get("/users/{user_id}/permissions")
def get_permissions(
    user_id: str,
    _=Depends(get_current_user)
):

    permissions = set()

    for role in user_roles.get(user_id, []):
        permissions.update(PERMISSIONS.get(role, []))

    return {"permissions": list(permissions)}


os.makedirs("uploads", exist_ok=True)


@app.post("/documents/upload")
def upload_document(
    title: str = Form(...),
    company_name: str = Form(...),
    document_type: str = Form(...),
    file: UploadFile = None,
    current_user: str = Depends(check_permission("upload_document"))
):

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="File missing"
        )

    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only pdf and docx supported"
        )

    doc_id = str(uuid.uuid4())

    file_path = f"uploads/{doc_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    doc = DocumentInfo(
        document_id=doc_id,
        title=title,
        company_name=company_name,
        document_type=document_type,
        uploaded_by=current_user,
        created_at=datetime.utcnow(),
        file_path=file_path
    )

    documents[doc_id] = doc

    return {"message": "uploaded", "document_id": doc_id}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents(
    _=Depends(check_permission("view_document"))
):

    return list(documents.values())


# static routes need to stay above /documents/{document_id}
# otherwise FastAPI treats "search" as a dynamic path param
@app.get("/documents/search")
def search_documents(
    company_name: Optional[str] = None,
    title: Optional[str] = None,
    document_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    _=Depends(check_permission("view_document"))
):

    results = list(documents.values())

    if company_name:
        results = [
            d for d in results
            if d.company_name.lower() == company_name.lower()
        ]

    if title:
        results = [
            d for d in results
            if title.lower() in d.title.lower()
        ]

    if document_type:
        results = [
            d for d in results
            if d.document_type.lower() == document_type.lower()
        ]

    if uploaded_by:
        results = [
            d for d in results
            if d.uploaded_by == uploaded_by
        ]

    return results


@app.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(
    document_id: str,
    _=Depends(check_permission("view_document"))
):

    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return documents[document_id]


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    _=Depends(check_permission("upload_document"))
):

    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    doc = documents.pop(document_id)

    # deleting the physical file too so local uploads don't pile up over time
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    return {"message": "deleted"}


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# running in-memory keeps local testing simpler without managing qdrant separately
qdrant_client = QdrantClient(":memory:")

vector_size = len(
    embedding_model.embed_query("test")
)

qdrant_client.recreate_collection(
    collection_name="financial_docs",
    vectors_config={
        "size": vector_size,
        "distance": "Cosine"
    }
)

vector_store = Qdrant(
    client=qdrant_client,
    embeddings=embedding_model,
    collection_name="financial_docs"
)


def extract_text(file_path: str):

    if file_path.endswith(".pdf"):

        # PyPDF2 struggles with scanned/image-only PDFs unless OCR is added
        reader = PdfReader(file_path)

        text_parts = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                text_parts.append(text)

        return " ".join(text_parts)

    elif file_path.endswith(".docx"):

        doc = Document(file_path)

        return " ".join(
            p.text for p in doc.paragraphs
            if p.text.strip()
        )

    raise HTTPException(
        status_code=400,
        detail="Unsupported file"
    )


@app.post("/rag/index-document")
def index_document(
    document_id: str,
    _=Depends(check_permission("upload_document"))
):

    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    doc = documents[document_id]

    if not os.path.exists(doc.file_path):
        raise HTTPException(
            status_code=500,
            detail="File missing"
        )

    try:

        text = extract_text(doc.file_path)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text found"
            )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # smaller chunks improved retrieval quality,
    # but going too small started hurting context continuity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    logger.info(
        f"{document_id} split into {len(chunks)} chunks"
    )

    if not chunks:
        return {"message": "No chunks created"}

    try:

        vector_store.add_texts(
            chunks,
            metadatas=[
                {"document_id": document_id}
                for _ in chunks
            ]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector db error: {str(e)}"
        )

    return {
        "message": "indexed",
        "chunks_added": len(chunks)
    }


@app.delete("/rag/remove-document/{document_id}")
def remove_document_vectors(
    document_id: str,
    _=Depends(check_permission("upload_document"))
):

    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    try:

        qdrant_client.delete(
            collection_name="financial_docs",

            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    return {
        "message": f"Removed vectors for {document_id}"
    }


@app.post("/rag/search")
def search_rag(
    body: SearchRequest,
    _=Depends(check_permission("view_document"))
):

    try:

        # pulling a larger candidate pool before reranking
        # improved relevance for finance-heavy documents
        hits = vector_store.similarity_search(
            body.query,
            k=20
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

    if not hits:
        return {"results": []}

    pairs = [
        (body.query, hit.page_content)
        for hit in hits
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(hits, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "results": [
            {
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "score": float(score)
            }
            for chunk, score in ranked_results[:5]
        ]
    }


@app.get("/rag/context/{document_id}")
def get_context(
    document_id: str,
    _=Depends(check_permission("view_document"))
):

    if document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    try:

        result, _ = qdrant_client.scroll(
            collection_name="financial_docs",

            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),

            limit=200,
            with_payload=True,
            with_vectors=False
        )

        chunks = [
            item.payload.get("page_content", "")
            for item in result
        ]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch context: {str(e)}"
        )

    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "chunks": chunks
    }
