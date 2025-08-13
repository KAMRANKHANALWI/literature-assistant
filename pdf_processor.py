# pdf_processor.py
# Convert PDF documents to text chunks and store in vector database for retrieval

import os
import glob
import warnings
import chromadb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure environment for optimal performance
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

print("🔄 Starting PDF processing pipeline...")

# Configuration settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 400
RESEARCH_FOLDER = "research_papers"

# Initialize embedding model for vector generation
print("🔄 Loading sentence embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize document processing components
print("🔄 Initializing document processing pipeline...")
converter = DocumentConverter()
chunker = HybridChunker(max_tokens=MAX_TOKENS, merge_peers=True)

# --------------------------------------------------------------
# STEP 1: SCAN FOR PDF FILES
# --------------------------------------------------------------
print(f"\n📂 Scanning for PDF files in '{RESEARCH_FOLDER}' folder...")

# Create research folder if it doesn't exist
if not os.path.exists(RESEARCH_FOLDER):
    os.makedirs(RESEARCH_FOLDER)
    print(f"📂 Created '{RESEARCH_FOLDER}' folder - Add your PDF files here!")
    exit()

# Find all PDF files in the folder (including subfolders)
pdf_files = glob.glob(os.path.join(RESEARCH_FOLDER, "**/*.pdf"), recursive=True)
print(f"📚 Found {len(pdf_files)} PDF files to process")

if len(pdf_files) == 0:
    print("❌ No PDF files found! Add PDFs to the research_papers folder.")
    exit()

# --------------------------------------------------------------
# STEP 2: PROCESS PDFs INTO CHUNKS WITH METADATA
# --------------------------------------------------------------
print(f"\n🔧 Processing PDFs into semantic chunks...")

all_documents = []
processing_stats = {"successful": 0, "failed": 0, "total_chunks": 0}

for i, pdf_path in enumerate(pdf_files, 1):
    try:
        filename = os.path.basename(pdf_path)
        print(f"📄 {i}/{len(pdf_files)}: {filename}")

        # Convert PDF to structured document representation
        result = converter.convert(pdf_path)

        # Create semantic chunks from document
        chunk_iter = chunker.chunk(dl_doc=result.document)
        chunks = list(chunk_iter)

        # Process each chunk and extract metadata
        for chunk_id, chunk in enumerate(chunks):
            # Extract page numbers from chunk provenance data
            page_numbers = []
            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                page_numbers.append(prov.page_no)

            # Extract section title from chunk metadata
            title = "No Title"
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                title = chunk.meta.headings[0]

            # Create LangChain document with comprehensive metadata
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "filename": filename,
                    "filepath": pdf_path,
                    "chunk_id": chunk_id,
                    "total_chunks": len(chunks),
                    "page_numbers": (
                        str(sorted(set(page_numbers))) if page_numbers else "[]"
                    ),
                    "title": title,
                    "text_length": len(chunk.text),
                    "source_type": "pdf",
                },
            )
            all_documents.append(doc)

        processing_stats["successful"] += 1
        processing_stats["total_chunks"] += len(chunks)
        print(f"   ✅ Created {len(chunks)} chunks")

    except Exception as e:
        print(f"   ❌ Error processing {filename}: {str(e)}")
        processing_stats["failed"] += 1

# Display processing summary
print(f"\n📊 PROCESSING SUMMARY:")
print(f"✅ Successful files: {processing_stats['successful']}/{len(pdf_files)}")
print(f"❌ Failed files: {processing_stats['failed']}")
print(f"📄 Total chunks created: {processing_stats['total_chunks']}")

if len(all_documents) == 0:
    print("❌ No documents processed successfully!")
    exit()

# --------------------------------------------------------------
# STEP 3: CREATE VECTOR DATABASE WITH EMBEDDINGS
# --------------------------------------------------------------
print(f"\n🔮 Creating vector database with {len(all_documents)} documents...")

# Create data directory for database storage
os.makedirs("data", exist_ok=True)

# Initialize persistent Chroma database
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# Create vector store with embeddings - this generates embeddings for all chunks
vectorstore = Chroma.from_documents(
    documents=all_documents,
    embedding=embedding_model,
    client=chroma_client,
    collection_name="research_papers",
    persist_directory="data/chroma_db",
)

print(f"✅ Vector database created successfully!")
print(f"📊 Total documents in database: {vectorstore._collection.count()}")
print(f"🗃️ Database location: data/chroma_db")

print(f"\n🎯 Ready! Run search_interface.py to test search functionality.")
