import os
import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify
from config import Config
from services.document_processor import process_file_to_chunks
from services.rag_service import vector_store

documents_bp = Blueprint("documents", __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@documents_bp.route("/api/documents", methods=["GET"])
def list_documents_endpoint():
    """Lists all documents in data/documents/ along with indexing metrics."""
    try:
        stats = vector_store.get_stats()
        indexed_map = {d["document_name"]: d["chunks"] for d in stats.get("documents", [])}
        
        doc_list = []
        if Config.DATA_DIR.exists():
            for f in Config.DATA_DIR.iterdir():
                if f.is_file() and allowed_file(f.name):
                    size_kb = round(f.stat().st_size / 1024, 2)
                    mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    chunks = indexed_map.get(f.name, 0)
                    doc_list.append({
                        "name": f.name,
                        "size_kb": size_kb,
                        "chunks": chunks,
                        "upload_date": mtime,
                        "status": "Indexed" if chunks > 0 else "Unindexed"
                    })
        return jsonify({"documents": doc_list}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to list documents: {str(e)}"}), 500

@documents_bp.route("/api/documents/upload", methods=["POST"])
def upload_document_endpoint():
    """Uploads a PDF or TXT document, extracts text, chunks, embeds, and indexes it."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in form data."}), 400
        
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
        
    filename = Path(file.filename).name
    if not allowed_file(filename):
        return jsonify({
            "error": f"Unsupported format. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = Config.DATA_DIR / filename
        file.save(str(file_path))
        
        # Extract and chunk
        chunks = process_file_to_chunks(
            file_path,
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        
        if not chunks:
            # Clean up empty file
            if file_path.exists():
                file_path.unlink()
            return jsonify({"error": "No readable text could be extracted from this document."}), 400

        # Replace old chunks in vector store
        vector_store.delete_document(filename)

        # Add new chunks with placeholder embeddings, then refit the embedder
        # over the entire corpus and recompute all embeddings in one shared
        # TF-IDF space. This keeps the query encoder consistent with the index
        # and makes ingestion repeatable without duplicate or stale records.
        vector_store.add_chunks(chunks, [None] * len(chunks))
        vector_store.reindex_embeddings()

        return jsonify({
            "message": f"Successfully indexed '{filename}'.",
            "document": filename,
            "chunks": len(chunks)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Document processing failed: {str(e)}"}), 500

@documents_bp.route("/api/documents/delete/<path:filename>", methods=["DELETE", "POST"])
def delete_document_endpoint(filename):
    """Deletes document file and removes its vectors from vector store."""
    try:
        clean_name = Path(filename).name
        file_path = Config.DATA_DIR / clean_name
        
        removed_chunks = vector_store.delete_document(clean_name)
        disk_deleted = False
        if file_path.exists():
            file_path.unlink()
            disk_deleted = True

        if removed_chunks > 0 or disk_deleted:
            return jsonify({
                "message": f"Deleted '{clean_name}' ({removed_chunks} chunks removed from vector store)."
            }), 200
        return jsonify({"error": f"Document '{clean_name}' not found."}), 404

    except Exception as e:
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500

@documents_bp.route("/api/documents/reindex", methods=["POST"])
def reindex_all_endpoint():
    """Reindexes all documents in data/documents/ directory."""
    try:
        vector_store.clear()
        total_chunks = 0
        total_docs = 0

        for f in Config.DATA_DIR.iterdir():
            if f.is_file() and allowed_file(f.name):
                chunks = process_file_to_chunks(f, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                if chunks:
                    vector_store.add_chunks(chunks, [None] * len(chunks))
                    total_chunks += len(chunks)
                    total_docs += 1

        # Refit the embedder over the whole corpus and recompute all embeddings.
        vector_store.reindex_embeddings()

        return jsonify({
            "message": "Reindexing complete.",
            "documents_indexed": total_docs,
            "total_chunks": total_chunks
        }), 200

    except Exception as e:
        return jsonify({"error": f"Reindexing failed: {str(e)}"}), 500
