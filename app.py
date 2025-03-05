import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Configure page layout
st.set_page_config(
    page_title="Thirukural Wisdom Finder",
    page_icon="📜",
    layout="wide"
)

# Initialize session state for "Learn More" visibility
if "show_docs" not in st.session_state:
    st.session_state.show_docs = False

# Sidebar for app details
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/%E0%AE%A4%E0%AE%BF%E0%AE%B0%E0%AF%81%E0%AE%95%E0%AF%8D%E0%AE%95%E0%AF%81%E0%AE%B1%E0%AE%B3%E0%AF%8D_%E0%AE%A4%E0%AF%86%E0%AE%B3%E0%AE%BF%E0%AE%B5%E0%AF%81.pdf/page1-856px-%E0%AE%A4%E0%AE%BF%E0%AE%B0%E0%AF%81%E0%AE%95%E0%AF%8D%E0%AE%95%E0%AF%81%E0%AE%B1%E0%AE%B3%E0%AF%8D_%E0%AE%A4%E0%AF%86%E0%AE%B3%E0%AE%BF%E0%AE%B5%E0%AF%81.pdf.jpg")
    st.title("📜 Thirukural Wisdom Finder")
    st.markdown("Search and explore Thirukurals with deep contextual understanding.")
    st.markdown("📚 **Dataset Source:** [Thirukural Dataset on Hugging Face](https://huggingface.co/datasets/Selvakumarduraipandian/Thirukural)")
    
    # Toggle "Learn More"
    if st.button("🔗 Learn More"):
        st.session_state.show_docs = not st.session_state.show_docs
        st.rerun()  # Forces UI to refresh

# Load dataset
@st.cache_data
def load_thirukural_dataset():
    try:
        dataset = load_dataset("Selvakumarduraipandian/Thirukural", split="train")
        return dataset.to_pandas()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# Generate embeddings and FAISS index
@st.cache_data
def generate_and_store_embeddings(df):
    EMBEDDINGS_FILE = "thirukural_embeddings.json"
    FAISS_INDEX_FILE = "thirukural_faiss_index.idx"
    
    # Check if embeddings and index already exist
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load FAISS index
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        return np.array(data["embeddings"]), data["texts"], faiss_index

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("⏳ Generating embeddings, please wait...")
        
        # Create embedding text
        df["EmbeddingText"] = df.apply(create_embedding_text, axis=1)
        
        # Generate embeddings
        embeddings = model.encode(df["EmbeddingText"].tolist(), show_progress_bar=True)
        
        # Convert to float32 for FAISS
        embeddings_float32 = embeddings.astype(np.float32)
        
        # Create FAISS index
        dimension = embeddings_float32.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        faiss_index.add(embeddings_float32)
        
        # Save embeddings
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "embeddings": embeddings.tolist(),
                "texts": df["EmbeddingText"].tolist()
            }, f, ensure_ascii=False)
        
        # Save FAISS index
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)

        return np.array(embeddings), df["EmbeddingText"].tolist(), faiss_index
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([]), [], None

# Create text for embeddings
def create_embedding_text(row):
    return f"""
    Kural: {row['Kural']}
    English Couplet: {row['Couplet']}
    Chapter: {row['Chapter']}
    Section: {row['Section']}
    Adhigaram: {row['Adhigaram']}
    Transliteration: {row['Transliteration']}
    Meaning: {row['Vilakam']}
    Kalaingar Urai: {row['Kalaingar_Urai']}
    Parimezhalagar Urai: {row['Parimezhalagar_Urai']}
    M. Varadharajanar: {row['M_Varadharajanar']}
    Solomon Pappaiya: {row['Solomon_Pappaiya']}
    """.strip()

# Search Thirukural using FAISS
def search_thirukural(query, df, embeddings, texts, faiss_index, top_k=5):
    if embeddings.size == 0 or faiss_index is None:
        st.error("🚨 No embeddings or FAISS index found. Generate embeddings first.")
        return []

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode([query])[0].astype(np.float32)
        
        # Search using FAISS
        distances, indices = faiss_index.search(query_emb.reshape(1, -1), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = df.iloc[idx]
            results.append({
                "Kural": row["Kural"],
                "English Couplet": row["Couplet"],
                "Chapter": row["Chapter"],
                "Section": row["Section"],
                "Adhigaram": row["Adhigaram"],
                "Transliteration": row["Transliteration"],
                "Meaning": row["Vilakam"],
                "Kalaingar_Urai": row["Kalaingar_Urai"],
                "Parimezhalagar_Urai": row["Parimezhalagar_Urai"],
                "M_Varadharajanar": row["M_Varadharajanar"],
                "Solomon_Pappaiya": row["Solomon_Pappaiya"],
                "Distance": round(float(dist), 4),  # Using distance instead of similarity
            })

        return results
    except Exception as e:
        st.error(f"Error in search: {e}")
        return []

# Load dataset and generate embeddings
df = load_thirukural_dataset()
embeddings, texts, faiss_index = generate_and_store_embeddings(df)

# Main UI layout
st.header("🔍 Search for Thirukural Wisdom")

# Search input
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Enter a wisdom-related query / scene from your life experiences / topic:",
        placeholder="e.g., honesty, leadership, friendship"
    )
with col2:
    num_results = st.slider("Results:", min_value=1, max_value=10, value=5)

if st.button("🔎 Search"):
    if query.strip():
        with st.spinner("Fetching the most relevant Thirukurals..."):
            results = search_thirukural(query, df, embeddings, texts, faiss_index, top_k=num_results)

            if not results:
                st.error("❌ No results found. Try a different query.")
            else:
                st.subheader("📜 Top Matching Thirukurals")
                for i, result in enumerate(results):
                    with st.expander(f"{i+1}. {result['English Couplet']} (Distance: {result['Distance']})"):
                        st.markdown(f"**📖 Original (Tamil):**  \n{result['Kural']}", unsafe_allow_html=True)
                        st.write(f"**🔤 Transliteration:** {result['Transliteration']}")
                        st.write(f"**📝 Meaning:** {result['Meaning']}")
                        st.write(f"**📚 Chapter:** {result['Chapter']} | **Section:** {result['Section']} | **Adhigaram:** {result['Adhigaram']}")
                        st.write(f"**📌 Kalaingar Urai:** {result['Kalaingar_Urai']}")
                        st.write(f"**📌 Parimezhalagar Urai:** {result['Parimezhalagar_Urai']}")
                        st.write(f"**📌 M. Varadharajanar Urai:** {result['M_Varadharajanar']}")
                        st.write(f"**📌 Solomon Pappaiya Urai:** {result['Solomon_Pappaiya']}")

    else:
        st.warning("⚠️ Please enter a query.")

# Documentation Section (conditionally displayed)
if st.session_state.show_docs:
    st.markdown("---")
    st.header("📚 About This Application")

    with st.expander("📖 Basic Overview", expanded=True):
        st.write("""
            This application allows users to search and explore the ancient Tamil wisdom text 
            **Thirukural** using modern NLP techniques. Key features include:
            
            - Semantic search using sentence embeddings
            - Multiple interpretations from different commentators
            - Contextual analysis across 133 chapters (Adhigarams)
            - Cross-referencing with transliterations and meanings
            
            The system uses **cosine similarity** to find kurals that are contextually closest 
            to your query, whether you search in English or Tamil.
            """)

    with st.expander("💻 Technical Architecture", expanded=True):
        st.write("""
        **System Architecture Diagram**
        """)

        # Graphviz DOT code
        dot_code = """
        digraph G {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor=lightblue];
            A [label="User Query"];
            B [label="Text Preprocessing"];
            C [label="Sentence Embedding Generation"];
            D [label="Cosine Similarity Calculation"];
            E [label="Ranked Results"];
            F [label="UI Rendering"];
            A -> B;
            B -> C;
            C -> D;
            D -> E;
            E -> F;

            subgraph cluster_embedding {
                label="Embedding Layer";
                C1 [label="Pre-trained Transformer (all-MiniLM-L6-v2)"];
                C -> C1;
            }

            subgraph cluster_data {
                label="Data Layer";
                D1 [label="Thirukural Dataset"];
                D2 [label="Stored Embeddings"];
                D1 -> D;
                D2 -> D;
            }
        }
        """
        st.graphviz_chart(dot_code)

    with st.expander("🔍 Example Use Cases", expanded=True):
        st.write("Try these sample queries:")
        st.code('''
    "How to handle conflicts in teamwork?"
    "Principles of good governance"
    "தோற்றத்தின் மீது தொடர்பு"  # Tamil query example
    ''', language='python')
    
    with st.expander("💻 Sample Output Format", expanded=True):
        st.write("Sample Output Format:")

        st.markdown("""
        ```json
        {
          "Kural": "அஃகா மிகு தண்ணீர்க் கெட்டு நீரார்...",
          "English Couplet": "Like water on a lotus leaf...",
          "Similarity": 0.89,
          "Meanings": {
            "Kalaingar": "...",
            "Parimezhalagar": "..."
          }
        }
        ```
        """, unsafe_allow_html=True)

    with st.expander("🔗 Sources & References", expanded=True):
        st.write("""
    1. **Thirukural Dataset**  
       [Hugging Face Dataset](https://huggingface.co/datasets/Selvakumarduraipandian/Thirukural)  
       by Selvakumar Duraipandian

    2. **Embedding Model**  
       [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
       from Sentence Transformers

    3. **Semantic Search Paper**  
       [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)  
       Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
    """)
