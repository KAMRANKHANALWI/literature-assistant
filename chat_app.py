# chat_app.py
# Streamlit web interface for interactive document chat using RAG

import streamlit as st
import chromadb
import os
import warnings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Configure environment for optimal performance
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Configuration - must match pdf_processor.py settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------
# INITIALIZE MODELS AND DATABASE
# --------------------------------------------------------------


@st.cache_resource
def load_models():
    """Initialize language model and embedding model with Streamlit caching for performance."""
    llm = ChatGroq(
        model="llama3-8b-8192", temperature=0.1, api_key=os.getenv("GROQ_API_KEY")
    )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return llm, embedding_model


@st.cache_resource
def connect_database():
    """Connect to vector database with error handling and user feedback."""
    try:
        # Connect to existing Chroma database
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        _, embedding_model = load_models()

        vectorstore = Chroma(
            client=chroma_client,
            collection_name="research_papers",
            embedding_function=embedding_model,
            persist_directory="data/chroma_db",
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.info("💡 Run pdf_processor.py first to create the database!")
        st.stop()


# --------------------------------------------------------------
# SEARCH AND CONTEXT RETRIEVAL
# --------------------------------------------------------------


def search_relevant_documents(query, vectorstore, num_results=5):
    """
    Search vector database for documents relevant to user query.

    Args:
        query: User's search query
        vectorstore: Vector database instance
        num_results: Number of documents to retrieve

    Returns:
        Tuple of (formatted_context_string, list_of_search_results)
    """
    # Perform semantic similarity search
    results = vectorstore.similarity_search_with_score(query, k=num_results)

    # Build context string for LLM and results for UI display
    context_parts = []
    search_results = []

    for doc, score in results:
        # Extract metadata for source attribution
        filename = doc.metadata.get("filename", "unknown")
        page_numbers = doc.metadata.get("page_numbers", "[]")
        title = doc.metadata.get("title", "No Title")

        # Build source citation
        source_parts = [filename]
        if page_numbers != "[]":
            # Parse page numbers from string format
            page_nums = (
                page_numbers.strip("[]").replace("'", "").replace(" ", "").split(",")
            )
            if page_nums and page_nums[0]:
                source_parts.append(f"p. {', '.join(page_nums)}")

        # Format source attribution
        source = f"Source: {' - '.join(source_parts)}"
        if title != "No Title":
            source += f"\nTitle: {title}"

        # Add to context with source attribution
        context_parts.append(f"{doc.page_content}\n{source}")

        # Prepare for UI display
        search_results.append(
            {
                "content": doc.page_content,
                "filename": filename,
                "title": title,
                "pages": page_numbers,
                "similarity": round(1 - score, 4),
            }
        )

    return "\n\n".join(context_parts), search_results


# --------------------------------------------------------------
# CHAT RESPONSE GENERATION
# --------------------------------------------------------------


def generate_chat_response(chat_history, context):
    """
    Generate streaming chat response using retrieved document context.

    Args:
        chat_history: List of previous messages in conversation
        context: Retrieved document context from vector search

    Returns:
        Generated response string
    """
    llm, _ = load_models()

    # Create system prompt with retrieved context
    system_prompt = f"""You are a research assistant that answers questions based on scientific literature.
    Use only the information provided in the context below. If the context doesn't contain 
    relevant information for the question, clearly state this limitation.
    
    Context from research papers:
    {context}
    """

    # Build complete message history with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)

    try:
        # Generate streaming response for real-time display
        response_stream = llm.stream(messages)

        # Display response with typing effect
        full_response = ""
        response_placeholder = st.empty()

        for chunk in response_stream:
            if hasattr(chunk, "content"):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

        # Remove typing cursor and display final response
        response_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
        return "Please check your GROQ_API_KEY configuration in the .env file."


# --------------------------------------------------------------
# STREAMLIT WEB APPLICATION
# --------------------------------------------------------------


def main():
    """
    Main Streamlit application for interactive document chat.

    Sets up the web interface, handles chat state management,
    and orchestrates the retrieval-augmented generation pipeline.
    """
    # Configure Streamlit page
    st.set_page_config(page_title="Research Assistant", page_icon="🧬", layout="wide")

    st.title("🧬 Biomedical Research Assistant")

    # Connect to database
    vectorstore = connect_database()

    # Initialize chat message history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display welcome message for new sessions
    if not st.session_state.messages:
        st.markdown(
            """
        <div style="background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px; 
                    border-left: 4px solid #0f52ba;">
            <h4 style="color: #0f52ba; margin-top: 0;">💡 Ask questions about your research papers</h4>
            <p style="color: #666;">Search across molecular biology, clinical studies, genomics, and more...</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Display chat message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if user_query := st.chat_input("Ask about your research papers..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Search for relevant context and generate response
        with st.status("🔍 Searching documents...", expanded=False) as status:
            context, search_results = search_relevant_documents(user_query, vectorstore)

            # Display retrieved sources for transparency
            st.write("**📄 Found relevant sections:**")
            for result in search_results:
                with st.expander(
                    f"📁 {result['filename']} (similarity: {result['similarity']:.3f})"
                ):
                    st.write(f"**Section:** {result['title']}")
                    st.write(f"**Pages:** {result['pages']}")
                    st.write(result["content"])

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_chat_response(st.session_state.messages, context)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# Run the Streamlit application
if __name__ == "__main__":
    main()
