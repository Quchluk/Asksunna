import streamlit as st
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import faiss
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Обновленные LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


# Конфигурация
class Config:
    """Класс для хранения конфигурации приложения"""
    # Для эмбеддингов используем OpenAI напрямую
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Для LLM используем OpenRouter
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_MODEL = "deepseek/deepseek-chat"

    # Пути согласно ТЗ
    FAISS_INDEX_DIR = "/Users/Tosha/Desktop/Projects/asksunna/asksunna/faiss_index"
    HADITH_CHUNKS_PATH = "/Users/Tosha/Desktop/Projects/asksunna/asksunna/DB/hadith_chunks.parquet"
    HADITH_EMBEDDINGS_PATH = "/Users/Tosha/Desktop/Projects/asksunna/asksunna/DB/hadith_embeddings.parquet"

    MAX_RETRIEVAL_DOCS = 5
    LLM_TEMPERATURE = 0.1

    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY не найден в .env файле")
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY не найден в .env файле")
        if not os.path.exists(cls.FAISS_INDEX_DIR):
            raise ValueError(f"Директория с индексами {cls.FAISS_INDEX_DIR} не найдена")
        if not os.path.exists(cls.HADITH_CHUNKS_PATH):
            raise ValueError(f"Файл с чанками хадисов {cls.HADITH_CHUNKS_PATH} не найден")


# Инициализация конфигурации
try:
    Config.validate()
except ValueError as e:
    st.error(f"Ошибка конфигурации: {e}")
    st.stop()


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
        """
        Get complete metadata for chunk by uid_chunked

        Args:
            uid_chunked: unique chunk identifier

        Returns:
            Dictionary with complete metadata
        """
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
        self.vectorstore = None
        self.retriever = None
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
            logger.info("Эмбеддинг модель успешно инициализирована через OpenAI API")
        except Exception as e:
            logger.error(f"Ошибка инициализации эмбеддинг модели: {e}")
            raise

    @st.cache_resource
    def load_faiss_index(_self, collection_name: str) -> Optional[FAISS]:
        """
        Загрузка FAISS индекса для указанного сборника с кэшированием

        Args:
            collection_name: Название сборника хадисов

        Returns:
            FAISS vectorstore или None при ошибке
        """
        try:
            index_path = os.path.join(Config.FAISS_INDEX_DIR, collection_name)

            if not os.path.exists(index_path):
                logger.error(f"Индекс для сборника '{collection_name}' не найден: {index_path}")
                return None

            # Загрузка FAISS индекса
            vectorstore = FAISS.load_local(
                index_path,
                _self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info(f"Индекс для сборника '{collection_name}' успешно загружен")
            return vectorstore

        except Exception as e:
            logger.error(f"Ошибка загрузки индекса для сборника '{collection_name}': {e}")
            return None

    def setup_retriever(self, collection_name: str) -> bool:
        """
        Настройка ретривера для указанного сборника

        Args:
            collection_name: Название сборника хадисов

        Returns:
            True если успешно, False при ошибке
        """
        try:
            self.vectorstore = self.load_faiss_index(collection_name)

            if self.vectorstore is None:
                return False

            # Создание ретривера
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": Config.MAX_RETRIEVAL_DOCS}
            )

            logger.info(f"Ретривер для сборника '{collection_name}' готов к работе")
            return True

        except Exception as e:
            logger.error(f"Ошибка настройки ретривера: {e}")
            return False

    def search_similar_chunks(self, query: str) -> List[Document]:
        """
        Поиск похожих чанков для запроса с обогащением метаданными

        Args:
            query: Пользовательский запрос

        Returns:
            Список релевантных документов с полными метаданными
        """
        if not self.retriever:
            logger.error("Ретривер не инициализирован")
            return []

        try:
            docs = self.retriever.get_relevant_documents(query)

            # Обогащение документов полными метаданными
            enriched_docs = []
            for doc in docs:
                uid_chunked = doc.metadata.get('uid_chunked')
                if uid_chunked:
                    full_metadata = self.metadata_manager.get_chunk_metadata(uid_chunked)
                    if full_metadata:
                        # Создаем новый документ с полными метаданными
                        enriched_doc = Document(
                            page_content=doc.page_content,
                            metadata=full_metadata
                        )
                        enriched_docs.append(enriched_doc)
                    else:
                        enriched_docs.append(doc)
                else:
                    enriched_docs.append(doc)

            logger.info(f"Найдено {len(enriched_docs)} релевантных документов с полными метаданными")
            return enriched_docs

        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            return []


class HadithQAChain:
    """Класс для работы с LLM и генерации ответов"""

    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self._init_llm()

    def _init_llm(self):
        """Инициализация LLM модели через OpenRouter"""
        try:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                openai_api_key=Config.OPENROUTER_API_KEY,
                openai_api_base=Config.OPENROUTER_BASE_URL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=1500
            )
            logger.info("LLM модель успешно инициализирована через OpenRouter")
        except Exception as e:
            logger.error(f"Ошибка инициализации LLM: {e}")
            raise

    def setup_qa_chain(self, retriever):
        """
        Настройка QA цепочки с кастомным промптом

        Args:
            retriever: Ретривер для поиска документов
        """
        try:
            # Custom prompt for working with hadiths
            prompt_template = """
You are an expert on Islamic hadiths. Use the provided hadith fragments to give an accurate and well-founded answer to the user's question.

Context (hadith fragments):
{context}

Question: {question}

Instructions:
1. Answer accurately and exclusively based on the provided hadiths
2. If there is insufficient information for a complete answer, honestly state this
3. When possible, indicate collection names and hadith numbers
4. Be respectful, accurate, and academic in your approach
5. Answer in the language the question was asked in
6. Structure your answer logically and clearly
7. When quoting, indicate the source in parentheses

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Создание QA цепочки
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            logger.info("QA цепочка успешно настроена")

        except Exception as e:
            logger.error(f"Ошибка настройки QA цепочки: {e}")
            raise

    def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Получение ответа на вопрос

        Args:
            question: Вопрос пользователя

        Returns:
            Словарь с ответом и источниками
        """
        if not self.qa_chain:
            logger.error("QA цепочка не инициализирована")
            return {"answer": "Ошибка: QA цепочка не готова", "sources": []}

        try:
            result = self.qa_chain.invoke({"query": question})

            return {
                "answer": result["result"],
                "sources": result["source_documents"]
            }

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return {"answer": f"Ошибка при генерации ответа: {e}", "sources": []}


def get_available_collections() -> List[str]:
    """
    Получение списка доступных сборников хадисов

    Returns:
        Список названий сборников
    """
    try:
        index_dir = Path(Config.FAISS_INDEX_DIR)
        if not index_dir.exists():
            return []

        collections = [d.name for d in index_dir.iterdir() if d.is_dir()]
        return sorted(collections)

    except Exception as e:
        logger.error(f"Ошибка получения списка сборников: {e}")
        return []


def format_source_document(doc: Document, index: int) -> str:
    """
    Format source document for display with complete metadata in English

    Args:
        doc: Document from search
        index: Document number

    Returns:
        Formatted string
    """
    metadata = doc.metadata
    content = doc.page_content  # Show full content, not truncated

    # Safe value retrieval with None and NaN handling
    def safe_get(key, default="Not specified"):
        value = metadata.get(key, default)
        if pd.isna(value) or value is None or value == "":
            return default
        return str(value)

    # Determine if content is Arabic or English
    arabic_text = ""
    english_text = ""

    # Check if the main content is Arabic (contains Arabic characters)
    if any('\u0600' <= char <= '\u06FF' for char in content[:100]):
        arabic_text = content
        english_text = safe_get('text_en', 'English translation not available')
    else:
        english_text = content
        arabic_text = safe_get('text_ar', safe_get('text_ar_chunk', 'Arabic text not available'))

    return f"""
**📖 Source {index + 1}**

**🔍 Basic Information:**
- **Collection:** {safe_get('collection_book')}
- **Author:** {safe_get('author_en')} ({safe_get('author_ar')})
- **Book:** {safe_get('book_title_en')} ({safe_get('book_title_ar')})
- **Chapter:** {safe_get('chapter_title_en')} ({safe_get('chapter_title_ar')})

**📋 Identifiers:**
- **Hadith ID:** {safe_get('hadith_id')}
- **Number in Book:** {safe_get('hadith_id_in_book')}
- **Chunk ID:** {safe_get('uid_chunked')}
- **Chunk Number:** {safe_get('chunk_index')} of {safe_get('chunk_total')}

**👤 Narrator:** {safe_get('narrator_en')}

**📄 English Text:**
{english_text}

**🌐 Arabic Text:**
{arabic_text}

---
"""


def display_search_statistics(sources: List[Document]):
    """
    Display statistics for found sources

    Args:
        sources: List of found documents
    """
    if not sources:
        return

    # Collect statistics
    collections = {}
    authors = {}
    books = {}

    for doc in sources:
        metadata = doc.metadata

        collection = metadata.get('collection_book', 'Unknown')
        author = metadata.get('author_en', 'Unknown')
        book = metadata.get('book_title_en', 'Unknown')

        collections[collection] = collections.get(collection, 0) + 1
        authors[author] = authors.get(author, 0) + 1
        books[book] = books.get(book, 0) + 1

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📚 By Collections")
        for collection, count in collections.items():
            st.text(f"• {collection}: {count}")

    with col2:
        st.subheader("✍️ By Authors")
        for author, count in authors.items():
            st.text(f"• {author}: {count}")

    with col3:
        st.subheader("📖 By Books")
        for book, count in list(books.items())[:5]:  # Show top 5
            book_short = book[:30] + "..." if len(book) > 30 else book
            st.text(f"• {book_short}: {count}")


def main():
    """Основная функция Streamlit приложения"""

    # Настройка страницы
    st.set_page_config(
        page_title="Hadith QA Retriever",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Hadith QA Retriever")
    st.markdown("*Search for answers in the hadith corpus using AI with complete metadata*")

    # Инициализация компонентов в сессии
    if 'hadith_retriever' not in st.session_state:
        try:
            with st.spinner("🔄 Initializing system..."):
                st.session_state.hadith_retriever = HadithRetriever()
                st.session_state.qa_chain = HadithQAChain()
                st.success("✅ System successfully initialized!")
        except Exception as e:
            st.error(f"❌ Initialization error: {e}")
            st.stop()

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Collection selection
        collections = get_available_collections()
        if not collections:
            st.error("❌ No hadith collections found!")
            st.stop()

        selected_collection = st.selectbox(
            "Select hadith collection:",
            collections,
            help="Choose a collection to search in"
        )

        # Search settings
        st.subheader("🔍 Search Parameters")
        max_docs = st.slider("Maximum documents", 1, 10, Config.MAX_RETRIEVAL_DOCS)
        Config.MAX_RETRIEVAL_DOCS = max_docs

        st.markdown("---")
        st.subheader("📊 System Information")
        st.info(f"**Total collections:** {len(collections)}")
        st.info(f"**Embeddings:** {Config.EMBEDDING_MODEL}")
        st.info(f"**LLM:** {Config.LLM_MODEL}")

        st.markdown("---")
        st.subheader("📚 Available Collections")
        for i, collection in enumerate(collections, 1):
            icon = "🎯" if collection == selected_collection else "📘"
            st.markdown(f"{icon} {collection}")

    # Main interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("🔍 Ask Questions About Hadiths")

        # Example questions
        example_questions = [
            "What do hadiths say about prayer?",
            "Which hadiths mention righteousness?",
            "What is said about fasting in hadiths?",
            "Hadiths about relationships with parents",
            "What is said about knowledge and learning?"
        ]

        selected_example = st.selectbox(
            "Or choose an example:",
            [""] + example_questions,
            help="Select a ready-made question or enter your own"
        )

        # Question input field
        question = st.text_area(
            "Enter your question:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="For example: What do hadiths say about prayer?"
        )

        # Search button
        search_button = st.button("🔎 Find Answer", type="primary", use_container_width=True)

    with col2:
        st.header("📊 Status")
        st.success(f"**Selected:** {selected_collection}")
        st.info(f"**Max results:** {max_docs}")

        if st.button("🔄 Reload Indexes", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared!")

    # Process query
    if search_button and question.strip():
        with st.spinner("🔄 Searching for answer in hadith corpus..."):
            try:
                # Setup retriever for selected collection
                if not st.session_state.hadith_retriever.setup_retriever(selected_collection):
                    st.error(f"❌ Error loading index for collection '{selected_collection}'")
                    st.stop()

                # Setup QA chain
                st.session_state.qa_chain.setup_qa_chain(st.session_state.hadith_retriever.retriever)

                # Get answer
                result = st.session_state.qa_chain.get_answer(question)

                # Display results
                st.markdown("---")
                st.header("💬 Answer")

                # Answer in a styled container with better visibility
                with st.container():
                    st.markdown(
                        f"""
                        <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; color: #000000; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='color: #000000; line-height: 1.6;'>
                        {result["answer"]}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Statistics and sources
                if result["sources"]:
                    st.markdown("---")
                    st.header("📊 Statistics of Found Sources")
                    display_search_statistics(result["sources"])

                    st.markdown("---")
                    st.header("📖 Detailed Sources")
                    st.markdown(f"*Found {len(result['sources'])} relevant hadith fragments*")

                    for i, doc in enumerate(result["sources"]):
                        with st.expander(
                                f"Source {i + 1}: {doc.metadata.get('collection_book', 'Unknown')} - "
                                f"Hadith #{doc.metadata.get('hadith_id_in_book', 'N/A')}",
                                expanded=i == 0  # First source open by default
                        ):
                            st.markdown(format_source_document(doc, i))
                else:
                    st.warning("⚠️ No relevant sources found")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                logger.error(f"Error in main process: {e}")

    elif search_button and not question.strip():
        st.warning("⚠️ Please enter a question")

    # Футер
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>📚 Hadith QA Retriever</h4>
        <p><strong>Powered by:</strong> OpenAI Embeddings • OpenRouter LLM • FAISS Vector Search</p>
        <p><em>Complete hadith metadata database with semantic search</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()