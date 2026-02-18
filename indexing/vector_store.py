"""
vector_store.py — Store and query embeddings in a vector database.

Supports ChromaDB (local).
"""

from abc import ABC, abstractmethod

from ingestion.chunk_code import CodeChunk


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Store code chunks with their embeddings."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieve the top-k most similar chunks matching metadata where filter."""
        ...

    @abstractmethod
    def get_all_chunks(self) -> list[dict]:
        """Retrieve all stored chunks from the database."""
        ...


class ChromaVectorStore(BaseVectorStore):
    """Local vector store using ChromaDB."""

    def __init__(self, collection_name: str = "codebase", persist_dir: str = "./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Add code chunks and embeddings to Chroma."""
        ids = [c.id for c in chunks]
        documents = [c.to_embedding_text() for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Query Chroma for similar chunks, optionally filtering by metadata."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def get_all_chunks(self) -> list[dict]:
        """Retrieve all stored chunks from Chroma."""
        results = self.collection.get(include=["documents", "metadatas"])
        output = []
        if not results or not results["ids"]:
            return []
        for i in range(len(results["ids"])):
            output.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return output
