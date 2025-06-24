import streamlit as st
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import faiss
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Updated LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    """Application configuration class"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    EMBEDDING_MODEL = "text-embedding-3-large"
    DEFAULT_LLM_MODEL = "deepseek/deepseek-chat"

    # Available working models only
    AVAILABLE_MODELS = {
        "deepseek/deepseek-chat": "DeepSeek Chat (Main)",
        "deepseek/deepseek-r1": "DeepSeek R1 (Reasoning)"
    }

    FAISS_INDEX_DIR = "faiss_index"
    HADITH_CHUNKS_PATH = "DB/hadith_chunks.parquet"
    HADITH_EMBEDDINGS_PATH = "DB/hadith_embeddings.parquet"

    MAX_RETRIEVAL_DOCS = 5
    DEFAULT_TEMPERATURE = 0.1

    @classmethod
    def validate(cls):
        """Configuration validation"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        if not os.path.exists(cls.FAISS_INDEX_DIR):
            raise ValueError(f"Index directory {cls.FAISS_INDEX_DIR} not found")
        if not os.path.exists(cls.HADITH_CHUNKS_PATH):
            raise ValueError(f"Hadith chunks file {cls.HADITH_CHUNKS_PATH} not found")


# Initialize configuration
try:
    Config.validate()
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()


# Simplified CSS - keeping only what works
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --surface-primary: rgba(18, 18, 26, 0.9);
        --surface-secondary: rgba(30, 30, 45, 0.8);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.8);
        --text-tertiary: rgba(255, 255, 255, 0.6);
        --accent-primary: #6366f1;
        --accent-secondary: #06b6d4;
        --border-light: rgba(255, 255, 255, 0.1);
        --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    .custom-header {
        background: var(--surface-primary);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-light);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-soft);
    }

    .app-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 400;
        line-height: 1.6;
    }

    .search-section {
        background: var(--surface-primary);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-light);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-soft);
    }

    .result-card {
        background: var(--surface-primary);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-light);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 2rem;
    }

    .arabic-text {
        font-family: 'Amiri', 'Traditional Arabic', serif !important;
        font-size: 1.4rem !important;
        line-height: 2 !important;
        text-align: right !important;
        direction: rtl !important;
        color: var(--text-primary) !important;
        background: var(--surface-secondary) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border-right: 4px solid var(--accent-secondary) !important;
        margin: 1.5rem 0 !important;
    }

    .source-item {
        background: var(--surface-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--accent-primary);
    }

    .metadata-grid {
        background: var(--surface-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 3px solid var(--accent-secondary);
    }

    .metadata-item {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }

    .metadata-label {
        font-weight: 600;
        color: var(--accent-secondary);
        min-width: 140px;
        flex-shrink: 0;
    }

    .metadata-value {
        color: var(--text-secondary);
        flex: 1;
    }

    .settings-section {
        background: var(--surface-secondary);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-soft);
    }

    /* Fix expander styling */
    .streamlit-expanderHeader {
        background: var(--surface-secondary) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-light) !important;
    }

    .streamlit-expanderContent {
        background: var(--surface-secondary) !important;
        border-radius: 0 0 8px 8px !important;
        border: 1px solid var(--border-light) !important;
        border-top: none !important;
    }

    @media (max-width: 768px) {
        .app-title {
            font-size: 2rem;
        }

        .metadata-item {
            flex-direction: column;
            gap: 0.25rem;
        }

        .metadata-label {
            min-width: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)


class MetadataManager:
    """Class for working with hadith metadata"""

    def __init__(self):
        self.hadith_chunks_df = None
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from parquet file"""
        try:
            self.hadith_chunks_df = pd.read_parquet(Config.HADITH_CHUNKS_PATH)
            logger.info(f"Loaded {len(self.hadith_chunks_df)} metadata records")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def get_chunk_metadata(self, uid_chunked: str) -> Dict[str, Any]:
        """Get complete metadata for chunk by uid_chunked"""
        try:
            row = self.hadith_chunks_df[self.hadith_chunks_df['uid_chunked'] == uid_chunked]
            if row.empty:
                logger.warning(f"Metadata for uid_chunked {uid_chunked} not found")
                return {}
            return row.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error getting metadata for {uid_chunked}: {e}")
            return {}


class HadithRetriever:
    """Class for working with vector search on hadiths"""

    def __init__(self):
        self.embeddings = None
        self.vectorstores = {}
        self.metadata_manager = MetadataManager()
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding model via OpenAI API"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
            )
            logger.info("Embedding model successfully initialized via OpenAI API")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    @st.cache_resource
    def load_faiss_index(_self, collection_name: str) -> Optional[FAISS]:
        """Load FAISS index for specified collection with caching"""
        try:
            index_path = os.path.join(Config.FAISS_INDEX_DIR, collection_name)
            if not os.path.exists(index_path):
                logger.error(f"Index for collection '{collection_name}' not found: {index_path}")
                return None

            vectorstore = FAISS.load_local(
                index_path,
                _self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Index for collection '{collection_name}' successfully loaded")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading index for collection '{collection_name}': {e}")
            return None

    def search_multiple_sources(self, query: str, collection_names: List[str], max_docs_per_source: int = 3) -> Dict[
        str, List[Document]]:
        """Search across multiple collections"""
        results = {}

        for collection_name in collection_names:
            vectorstore = self.load_faiss_index(collection_name)
            if vectorstore:
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": max_docs_per_source})
                    docs = retriever.get_relevant_documents(query)

                    # Enrich with metadata
                    enriched_docs = []
                    for doc in docs:
                        uid_chunked = doc.metadata.get('uid_chunked')
                        if uid_chunked:
                            full_metadata = self.metadata_manager.get_chunk_metadata(uid_chunked)
                            if full_metadata:
                                enriched_doc = Document(
                                    page_content=doc.page_content,
                                    metadata=full_metadata
                                )
                                enriched_docs.append(enriched_doc)
                            else:
                                enriched_docs.append(doc)
                        else:
                            enriched_docs.append(doc)

                    results[collection_name] = enriched_docs
                    logger.info(f"Found {len(enriched_docs)} documents in {collection_name}")
                except Exception as e:
                    logger.error(f"Error searching in {collection_name}: {e}")
                    results[collection_name] = []

        return results


class HadithQAChain:
    """Class for working with LLM and generating answers"""

    def __init__(self, model_name: str = None, temperature: float = None):
        self.model_name = model_name or Config.DEFAULT_LLM_MODEL
        self.temperature = temperature or Config.DEFAULT_TEMPERATURE
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM model through OpenRouter"""
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=Config.OPENROUTER_API_KEY,
                openai_api_base=Config.OPENROUTER_BASE_URL,
                temperature=self.temperature,
                max_tokens=2000
            )
            logger.info(f"LLM model {self.model_name} successfully initialized")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def generate_answer(self, question: str, documents: List[Document], collection_name: str) -> str:
        """Generate answer for specific collection"""
        try:
            context = "\n\n".join([f"Source {i + 1}: {doc.page_content}" for i, doc in enumerate(documents)])

            prompt = f"""You are an expert Islamic scholar specializing in hadith analysis. Based on the provided hadith sources from {collection_name}, provide a comprehensive and scholarly answer to the user's question.

Context from {collection_name}:
{context}

Question: {question}

Instructions:
1. Provide an accurate, scholarly response based solely on the provided hadiths
2. Maintain academic rigor and respectful tone
3. Reference specific hadiths when making points
4. If information is insufficient, state this clearly
5. Structure your response logically with clear reasoning
6. Use formal academic language appropriate for Islamic scholarship

Answer:"""

            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Error generating answer for {collection_name}: {e}")
            return f"Error generating answer: {e}"

    def generate_comparative_analysis(self, question: str, collection_results: Dict[str, str]) -> str:
        """Generate comparative analysis across collections"""
        try:
            collections_summary = ""
            for collection, answer in collection_results.items():
                collections_summary += f"\n\n**{collection.replace('_', ' ').title()}:**\n{answer}\n"

            prompt = f"""You are a senior Islamic scholar conducting a comparative analysis of hadith interpretations across different canonical collections. 

Question analyzed: {question}

Results from different collections:
{collections_summary}

Task: Provide a comprehensive comparative analysis that:
1. Identifies common themes and unanimous positions across collections
2. Notes any differences or unique perspectives from specific collections
3. Explains the significance of any variations in a scholarly context
4. Synthesizes the findings into coherent conclusions
5. Addresses the reliability and consistency of the teachings
6. Provides academic insights into the methodological differences between collections

Structure your analysis with clear sections and maintain the highest level of academic rigor expected in Islamic scholarship.

Comparative Analysis:"""

            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Error generating comparative analysis: {e}")
            return f"Error generating comparative analysis: {e}"


def get_available_collections() -> List[str]:
    """Get list of available hadith collections"""
    try:
        index_dir = Path(Config.FAISS_INDEX_DIR)
        if not index_dir.exists():
            return []
        collections = [d.name for d in index_dir.iterdir() if d.is_dir()]
        return sorted(collections)
    except Exception as e:
        logger.error(f"Error getting collections list: {e}")
        return []


def get_collection_display_info() -> Dict[str, Dict[str, str]]:
    """Get display information for collections"""
    return {
        'sahih_bukhari': {'title': 'Sahih al-Bukhari', 'author': 'Imam Muhammad ibn Ismail al-Bukhari'},
        'bukhari': {'title': 'Sahih al-Bukhari', 'author': 'Imam Muhammad ibn Ismail al-Bukhari'},
        'sahih_muslim': {'title': 'Sahih Muslim', 'author': 'Imam Muslim ibn al-Hajjaj'},
        'muslim': {'title': 'Sahih Muslim', 'author': 'Imam Muslim ibn al-Hajjaj'},
        'sunan_abudawud': {'title': 'Sunan Abu Dawud', 'author': 'Imam Abu Dawud as-Sijistani'},
        'abudawud': {'title': 'Sunan Abu Dawud', 'author': 'Imam Abu Dawud as-Sijistani'},
        'jami_tirmidhi': {'title': 'Jami at-Tirmidhi', 'author': 'Imam Muhammad at-Tirmidhi'},
        'tirmidhi': {'title': 'Jami at-Tirmidhi', 'author': 'Imam Muhammad at-Tirmidhi'},
        'sunan_nasai': {'title': 'Sunan an-Nasai', 'author': 'Imam Ahmad an-Nasai'},
        'nasai': {'title': 'Sunan an-Nasai', 'author': 'Imam Ahmad an-Nasai'},
        'sunan_ibnmajah': {'title': 'Sunan Ibn Majah', 'author': 'Imam Ibn Majah al-Qazwini'},
        'ibnmajah': {'title': 'Sunan Ibn Majah', 'author': 'Imam Ibn Majah al-Qazwini'},
        'ahmed': {'title': 'Musnad Ahmad', 'author': 'Imam Ahmad ibn Hanbal'},
        'aladab_almufrad': {'title': 'Al-Adab Al-Mufrad', 'author': 'Imam al-Bukhari'},
        'bulugh_almaram': {'title': 'Bulugh al-Maram', 'author': 'Ibn Hajar al-Asqalani'},
        'darimi': {'title': 'Sunan ad-Darimi', 'author': 'Imam Abdullah ibn Abd al-Rahman ad-Darimi'},
        'malik': {'title': 'Muwatta Malik', 'author': 'Imam Malik ibn Anas'},
        'mishkat_almasabih': {'title': 'Mishkat al-Masabih', 'author': 'Al-Khatib al-Tabrizi'},
        'nawawi40': {'title': 'Forty Hadith an-Nawawi', 'author': 'Imam an-Nawawi'},
        'qudsi40': {'title': 'Forty Hadith Qudsi', 'author': 'Various Scholars'},
        'riyad_assalihin': {'title': 'Riyad as-Salihin', 'author': 'Imam an-Nawawi'},
        'shahwaliullah40': {'title': 'Forty Hadith Shah Waliullah', 'author': 'Shah Waliullah'},
        'shamail_muhammadiyah': {'title': 'Ash-Shama\'il al-Muhammadiyah', 'author': 'Imam at-Tirmidhi'},
    }


def display_collection_selector(available_collections: List[str]) -> List[str]:
    """Display collection selector using native Streamlit components"""
    collection_info = get_collection_display_info()

    st.markdown("### 📚 Select Hadith Collections")
    st.markdown("Choose one or more authentic hadith collections for comprehensive analysis")

    # Initialize session state
    if 'selected_collections' not in st.session_state:
        st.session_state.selected_collections = []

    # Use multiselect for better UX
    options = []
    labels = []
    for collection in available_collections:
        info = collection_info.get(collection, {'title': collection.replace('_', ' ').title()})
        options.append(collection)
        labels.append(f"{info['title']} ({collection})")

    # Create mapping
    collection_mapping = dict(zip(labels, options))

    selected_labels = st.multiselect(
        "Select collections:",
        labels,
        default=[labels[i] for i, collection in enumerate(options) if
                 collection in st.session_state.selected_collections],
        help="Select multiple collections for comparative analysis"
    )

    # Update session state
    st.session_state.selected_collections = [collection_mapping[label] for label in selected_labels]

    # Show simple count without chips
    if st.session_state.selected_collections:
        st.success(f"Selected {len(st.session_state.selected_collections)} collection(s)")

    return st.session_state.selected_collections


def display_search_progress(step: int, total: int, current_action: str):
    """Display search progress using native Streamlit components"""
    progress = step / total

    st.markdown("### 🔍 Search Progress")
    st.progress(progress)

    steps = [
        "🔄 Initializing semantic search",
        "📚 Loading collection indices",
        "🔍 Searching for relevant hadiths",
        "🤖 Generating scholarly analysis",
        "📊 Creating comparative study",
        "✅ Finalizing results"
    ]

    for i, step_text in enumerate(steps):
        if i < step:
            st.markdown(f"✅ {step_text}")
        elif i == step:
            st.markdown(f"⏳ {step_text}")
        else:
            st.markdown(f"⏸️ {step_text}")

    if current_action:
        st.info(f"Current: {current_action}")


def format_metadata_display(metadata: Dict[str, Any]) -> None:
    """Format and display metadata in a structured way"""

    # Technical fields to exclude from display - expanded list
    technical_fields = {
        'uid_chunked', 'chunk_id', 'Unnamed: 0', 'Unnamed: 0.1',
        'embedding', 'vector', 'index', 'id',
        # Additional technical fields to exclude
        'book_id', 'chapter_id', 'collection_set', 'chapter_file_name',
        'source_path', 'uid', 'chunk_index', 'chunk_total', 'text_ar_chunk'
    }

    # Group metadata by categories
    primary_info = {}
    content_info = {}
    reference_info = {}

    for key, value in metadata.items():
        # Convert key to lowercase for comparison
        key_lower = key.lower()

        # Skip technical fields
        if key in technical_fields or key_lower in technical_fields:
            continue

        # Skip empty or null values
        if pd.isna(value) or value is None or value == "" or value == "None":
            continue

        # Clean up the value
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ['nan', 'none', 'null']:
            continue

        # Categorize fields
        if key in ['collection_book', 'author_en', 'author_ar']:
            primary_info[key] = value_str
        elif key in ['text_en', 'text_ar']:
            content_info[key] = value_str
        elif key in ['hadith_id', 'hadith_id_in_book', 'book_title_en', 'book_title_ar',
                     'chapter_title_en', 'chapter_title_ar', 'volume']:
            reference_info[key] = value_str

    # Display metadata in organized sections
    if primary_info:
        st.markdown("**📚 Collection Information:**")
        st.markdown('<div class="metadata-grid">', unsafe_allow_html=True)
        for key, value in primary_info.items():
            display_name = {
                'collection_book': 'Collection',
                'author_en': 'Author (English)',
                'author_ar': 'Author (Arabic)'
            }.get(key, key.replace('_', ' ').title())

            st.markdown(f'''
            <div class="metadata-item">
                <span class="metadata-label">{display_name}:</span>
                <span class="metadata-value">{value}</span>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Reference Information
    if reference_info:
        st.markdown("**🔗 Reference Information:**")
        st.markdown('<div class="metadata-grid">', unsafe_allow_html=True)
        for key, value in reference_info.items():
            display_name = {
                'hadith_id': 'Hadith ID',
                'hadith_id_in_book': 'Hadith Number in Book',
                'book_title_en': 'Book Title (English)',
                'book_title_ar': 'Book Title (Arabic)',
                'chapter_title_en': 'Chapter Title (English)',
                'chapter_title_ar': 'Chapter Title (Arabic)',
                'volume': 'Volume'
            }.get(key, key.replace('_', ' ').title())

            st.markdown(f'''
            <div class="metadata-item">
                <span class="metadata-label">{display_name}:</span>
                <span class="metadata-value">{value}</span>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def format_source_document(doc: Document, index: int, collection_name: str) -> None:
    """Format and display source document using Streamlit components"""
    metadata = doc.metadata
    content = doc.page_content

    def safe_get(key, default="Not specified"):
        value = metadata.get(key, default)
        if pd.isna(value) or value is None or value == "":
            return default
        return str(value)

    # Get texts properly - prefer metadata over content
    english_text = safe_get('text_en', content)
    arabic_text = safe_get('text_ar', 'Arabic text not available')

    # Create container for source
    st.markdown(f"""
    <div class="source-item">
        <h4>📖 {safe_get('collection_book')} - Hadith #{safe_get('hadith_id_in_book')}</h4>
    </div>
    """, unsafe_allow_html=True)

    # English text
    st.markdown("**English Translation:**")
    st.markdown(english_text)

    # Arabic text (only if available and not just "Arabic text not available")
    if arabic_text != 'Arabic text not available' and arabic_text.strip():
        st.markdown("**Arabic Text:**")
        st.markdown(f"""
        <div class="arabic-text">
            {arabic_text}
        </div>
        """, unsafe_allow_html=True)

    # Display metadata in an expandable section
    with st.expander("📋 View Hadith Details", expanded=False):
        format_metadata_display(metadata)

    st.markdown("---")


def main():
    """Main Streamlit application function"""

    # Page configuration
    st.set_page_config(
        page_title="AskSunna - Islamic Knowledge Search",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    inject_custom_css()

    # Custom header
    st.markdown('''
    <div class="custom-header">
        <h1 class="app-title">AskSunna</h1>
        <p class="app-subtitle">Advanced Islamic Knowledge Search Engine<br>
        Explore authentic hadith collections with AI-powered semantic search and scholarly analysis</p>
    </div>
    ''', unsafe_allow_html=True)

    # Sidebar for model settings
    with st.sidebar:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Model Configuration")

        selected_model = st.selectbox(
            "AI Model",
            options=list(Config.AVAILABLE_MODELS.keys()),
            format_func=lambda x: Config.AVAILABLE_MODELS[x],
            index=0,
            help="Choose the AI model for analysis"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=Config.DEFAULT_TEMPERATURE,
            step=0.1,
            help="Controls creativity vs consistency (0.0 = more consistent, 1.0 = more creative)"
        )

        max_docs = st.slider(
            "Max documents per collection",
            min_value=1,
            max_value=10,
            value=Config.MAX_RETRIEVAL_DOCS,
            help="Number of most relevant documents to retrieve from each collection"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Search Settings")

        include_comparison = st.checkbox(
            "Generate Comparative Analysis",
            value=True,
            help="Create comparative analysis when multiple collections are selected"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize system components
    if 'hadith_retriever' not in st.session_state:
        with st.spinner("🔄 Initializing AI systems..."):
            try:
                st.session_state.hadith_retriever = HadithRetriever()
                st.success("✅ System initialized successfully")
            except Exception as e:
                st.error(f"❌ Initialization error: {e}")
                st.stop()

    # Initialize QA chain with current settings
    if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
        st.session_state.qa_chain = HadithQAChain(selected_model, temperature)
        st.session_state.current_model = selected_model

    # Get available collections
    available_collections = get_available_collections()
    if not available_collections:
        st.error("❌ No hadith collections found!")
        st.stop()

    # Collection selection
    selected_collections = display_collection_selector(available_collections)

    # Search interface
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Search Query")

    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "Enter your question about Islamic teachings:",
            placeholder="What do the hadiths say about prayer and its importance in daily life?",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("🔎 Search Hadiths", type="primary", use_container_width=True)

    # Add cancel button if search is in progress
    if st.session_state.get('search_in_progress', False):
        if st.button("❌ Cancel Search", type="secondary", use_container_width=True):
            st.session_state.search_in_progress = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Process search
    if search_button and question.strip() and selected_collections:
        st.session_state.search_in_progress = True

        # Create progress placeholder
        progress_placeholder = st.empty()

        try:
            # Step 1: Initialize
            with progress_placeholder.container():
                display_search_progress(0, 6, "Preparing search systems")
            time.sleep(0.5)

            # Step 2: Load indices
            with progress_placeholder.container():
                display_search_progress(1, 6, f"Loading {len(selected_collections)} collection indices")
            time.sleep(0.5)

            # Step 3: Search
            with progress_placeholder.container():
                display_search_progress(2, 6, "Performing semantic search")

            search_results = st.session_state.hadith_retriever.search_multiple_sources(
                question, selected_collections, max_docs
            )

            # Step 4: Generate answers
            with progress_placeholder.container():
                display_search_progress(3, 6, "Generating scholarly analysis")

            results = {}
            collection_answers = {}

            for collection_name, documents in search_results.items():
                if documents and st.session_state.get('search_in_progress', True):
                    answer = st.session_state.qa_chain.generate_answer(
                        question, documents, collection_name
                    )
                    results[collection_name] = (answer, documents)
                    collection_answers[collection_name] = answer

            # Step 5: Comparative analysis
            comparative_analysis = None
            if include_comparison and len(collection_answers) > 1 and st.session_state.get('search_in_progress', True):
                with progress_placeholder.container():
                    display_search_progress(4, 6, "Creating comparative analysis")

                comparative_analysis = st.session_state.qa_chain.generate_comparative_analysis(
                    question, collection_answers
                )

            # Step 6: Finalize
            with progress_placeholder.container():
                display_search_progress(5, 6, "Finalizing results")
            time.sleep(0.5)

            progress_placeholder.empty()
            st.session_state.search_in_progress = False

            if results:
                # Display comparative analysis first
                if comparative_analysis:
                    st.markdown("## 📊 Comparative Scholarly Analysis")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(comparative_analysis)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")

                # Display individual collection results
                st.markdown("## 📚 Individual Collection Results")

                collection_info = get_collection_display_info()

                for collection_name, (answer, sources) in results.items():
                    info = collection_info.get(collection_name, {'title': collection_name})

                    st.markdown(f"### 📖 {info['title']}")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(
                        f"**Found {len(sources)} sources** • **Model:** {Config.AVAILABLE_MODELS[selected_model]}")
                    st.markdown("---")
                    st.markdown(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Sources section - Display each source individually
                    if sources:
                        st.markdown(f"#### 📋 Sources from {info['title']}")
                        for i, doc in enumerate(sources):
                            with st.container():
                                format_source_document(doc, i, collection_name)

            else:
                st.warning("⚠️ No relevant sources found in the selected collections.")

        except Exception as e:
            st.error(f"❌ An error occurred during search: {e}")
            logger.error(f"Search error: {e}")
            st.session_state.search_in_progress = False

    elif search_button and not selected_collections:
        st.warning("⚠️ Please select at least one collection to search.")

    elif search_button and not question.strip():
        st.warning("⚠️ Please enter a search question.")

    # Footer
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; padding: 2rem; color: var(--text-tertiary);">
        <h4 style="color: var(--text-secondary);">🌟 AskSunna - Islamic Knowledge Search Engine</h4>
        <p><strong>Powered by:</strong> OpenAI Embeddings • DeepSeek Models • FAISS Vector Search • Advanced NLP</p>
        <p><em>Comprehensive hadith database with semantic search and scholarly comparative analysis</em></p>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()