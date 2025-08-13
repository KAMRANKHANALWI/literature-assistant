# search_interface.py
# Command-line search interface for testing vector database functionality

import warnings
import chromadb
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure environment for clean output
warnings.filterwarnings("ignore", message=".*pin_memory.*")

print("🔍 Starting search interface...")

# Configuration - must match pdf_processor.py settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------
# STEP 1: CONNECT TO EXISTING DATABASE
# --------------------------------------------------------------
print("📄 Connecting to vector database...")

try:
    # Initialize same embedding model used for creating database
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Connect to existing Chroma database
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="research_papers",
        embedding_function=embedding_model,
        persist_directory="data/chroma_db",
    )

    total_docs = vectorstore._collection.count()
    print(f"✅ Connected! Database contains {total_docs} document chunks")

except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("💡 Run pdf_processor.py first to create the database!")
    exit()


# --------------------------------------------------------------
# STEP 2: SEARCH FUNCTION
# --------------------------------------------------------------
def search_documents(query, num_results=3):
    """
    Execute semantic search query against vector database.

    Args:
        query: Search query string
        num_results: Number of top results to return

    Returns:
        List of (document, similarity_score) tuples
    """
    try:
        # Perform semantic similarity search with scores
        results_with_scores = vectorstore.similarity_search_with_score(
            query, k=num_results
        )
        return results_with_scores
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def display_search_results(query, results):
    """Display search results in formatted output for easy reading."""
    print(f"\n🔍 Search Query: '{query}'")
    print("-" * 50)

    if not results:
        print("❌ No results found")
        return

    for i, (doc, score) in enumerate(results, 1):
        # Convert distance score to similarity percentage
        similarity = round(1 - score, 4)
        filename = doc.metadata.get("filename", "Unknown")
        title = doc.metadata.get("title", "No Title")
        pages = doc.metadata.get("page_numbers", "[]")

        print(f"\n{i}. SIMILARITY: {similarity:.3f}")
        print(f"   📁 FILE: {filename}")
        print(f"   📄 SECTION: {title}")
        print(f"   📖 PAGES: {pages}")
        print(f"   📝 PREVIEW: {doc.page_content[:150]}...")


# --------------------------------------------------------------
# STEP 3: DEMONSTRATION SEARCHES
# --------------------------------------------------------------
print("\n🧪 Testing search functionality with sample queries...")

# Example search queries for biomedical research
demo_queries = ["treatment outcomes", "diagnostic accuracy", "molecular mechanisms"]

for query in demo_queries:
    results = search_documents(query, num_results=2)
    display_search_results(query, results)

# --------------------------------------------------------------
# STEP 4: INTERACTIVE SEARCH MODE
# --------------------------------------------------------------
print("\n" + "=" * 50)
print("🎯 INTERACTIVE SEARCH MODE")
print("=" * 50)
print("💡 Enter search queries to find relevant sections from your documents")

while True:
    # Get user input
    query = input("\n🔍 Enter your search query (or 'quit' to exit): ").strip()

    # Check for exit commands
    if query.lower() in ["quit", "exit", "q"]:
        print("👋 Goodbye!")
        break

    # Validate input
    if not query:
        print("❌ Please enter a search query")
        continue

    print(f"\n📄 Searching for: '{query}'")

    # Perform search
    results = search_documents(query, num_results=5)

    if not results:
        print("❌ No results found. Try different keywords.")
        continue

    # Display formatted search results
    print(f"✅ Found {len(results)} relevant results:")
    print("-" * 50)

    for i, (doc, score) in enumerate(results, 1):
        similarity = round(1 - score, 4)
        filename = doc.metadata.get("filename", "Unknown")
        title = doc.metadata.get("title", "No Title")
        pages = doc.metadata.get("page_numbers", "[]")

        print(f"\n{i}. SIMILARITY: {similarity:.3f}")
        print(f"   📁 FILE: {filename}")
        print(f"   📄 SECTION: {title}")
        print(f"   📖 PAGES: {pages}")
        print(f"   📝 CONTENT: {doc.page_content[:200]}...")

    # Option to view full content of specific result
    show_details = input(
        f"\n💡 Show full content for result (1-{len(results)}) or Enter to continue: "
    ).strip()

    if show_details.isdigit():
        idx = int(show_details) - 1
        if 0 <= idx < len(results):
            doc, score = results[idx]
            print(f"\n📖 FULL CONTENT:")
            print("=" * 50)
            print(doc.page_content)
            print("=" * 50)

print(f"\n🎯 Search testing complete! Run chat_app.py for the web interface.")
