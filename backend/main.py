# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Импорты из вашего Streamlit приложения
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from typing import List, Dict, Any, Optional
import os
import logging
import pandas as pd

# Добавим импорты для наших сервисов
from services.graph_service import GraphService
from tests import GraphData, GraphNode, GraphEdge, SearchResponse, CollectionInfo, HadithDetail, NarratorInfo

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
class Config:
    """Класс для хранения конфигурации приложения"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_MODEL = "deepseek/deepseek-chat"

    FAISS_INDEX_DIR = "/faiss_index"
    HADITH_CHUNKS_PATH = "/DB/hadith_chunks.parquet"
    HADITH_EMBEDDINGS_PATH = "/DB/hadith_embeddings.parquet"

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

    def load_faiss_index(self, collection_name: str) -> Optional[FAISS]:
        """Load FAISS index for specified collection"""
        try:
            index_path = os.path.join(Config.FAISS_INDEX_DIR, collection_name)

            if not os.path.exists(index_path):
                logger.error(f"Index for collection '{collection_name}' not found: {index_path}")
                return None

            vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info(f"Index for collection '{collection_name}' successfully loaded")
            return vectorstore

        except Exception as e:
            logger.error(f"Error loading index for collection '{collection_name}': {e}")
            return None

    def setup_retriever(self, collection_name: str) -> bool:
        """Setup retriever for specified collection"""
        try:
            self.vectorstore = self.load_faiss_index(collection_name)

            if self.vectorstore is None:
                return False

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": Config.MAX_RETRIEVAL_DOCS}
            )

            logger.info(f"Retriever for collection '{collection_name}' ready")
            return True

        except Exception as e:
            logger.error(f"Error setting up retriever: {e}")
            return False

    def search_similar_chunks(self, query: str) -> List[Document]:
        """Search for similar chunks with metadata enrichment"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return []

        try:
            docs = self.retriever.get_relevant_documents(query)

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

            logger.info(f"Found {len(enriched_docs)} relevant documents with full metadata")
            return enriched_docs

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

class HadithQAChain:
    """Class for working with LLM and generating answers"""

    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM model via OpenRouter"""
        try:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                openai_api_key=Config.OPENROUTER_API_KEY,
                openai_api_base=Config.OPENROUTER_BASE_URL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=1500
            )
            logger.info("LLM model successfully initialized via OpenRouter")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def setup_qa_chain(self, retriever):
        """Setup QA chain with custom prompt"""
        try:
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

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            logger.info("QA chain successfully configured")

        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise

    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer to question"""
        if not self.qa_chain:
            logger.error("QA chain not initialized")
            return {"answer": "Error: QA chain not ready", "sources": []}

        try:
            result = self.qa_chain.invoke({"query": question})

            return {
                "answer": result["result"],
                "sources": result["source_documents"]
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {"answer": f"Error generating answer: {e}", "sources": []}

# Pydantic модели для API
class SearchRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = ["all"]
    max_results: Optional[int] = 5

class SearchResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    summary: str
    statistics: Dict[str, Any]

class CollectionInfo(BaseModel):
    name: str
    display_name: str
    description: str
    hadith_count: int

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    collection: Optional[str] = None
    tooltip: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    color: str
    type: Optional[str] = None

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    summary: str

# Инициализация FastAPI
app = FastAPI(
    title="AskSunna API",
    description="Islamic Knowledge Graph API for Hadith Search",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для компонентов
hadith_retriever = None
qa_chain = None
metadata_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global hadith_retriever, qa_chain, metadata_manager

    try:
        Config.validate()

        logger.info("Initializing components...")
        hadith_retriever = HadithRetriever()
        qa_chain = HadithQAChain()
        metadata_manager = MetadataManager()

        logger.info("✅ All components initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

# Статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Main page"""
    return FileResponse('static/index.html')

@app.get("/api/collections", response_model=List[CollectionInfo])
async def get_collections():
    """Get list of available collections"""
    try:
        collections = get_available_collections()

        collection_info = []
        for collection in collections:
            display_names = {
                "bukhari": "Сахих аль-Бухари",
                "muslim": "Сахих Муслим",
                "tirmidhi": "Джами ат-Тирмизи",
                "abudawud": "Сунан Абу Дауд",
                "nasai": "Сунан ан-Насаи",
                "ibnmajah": "Сунан Ибн Маджах"
            }

            descriptions = {
                "bukhari": "Один из самых авторитетных сборников хадисов",
                "muslim": "Второй по значимости сборник достоверных хадисов",
                "tirmidhi": "Сборник хадисов с подробными комментариями",
                "abudawud": "Сборник хадисов по исламскому праву",
                "nasai": "Сборник хадисов с акцентом на достоверность",
                "ibnmajah": "Один из шести канонических сборников хадисов"
            }

            # Подсчет хадисов в коллекции
            hadith_count = get_collection_hadith_count(collection)

            collection_info.append(CollectionInfo(
                name=collection,
                display_name=display_names.get(collection, collection.title()),
                description=descriptions.get(collection, f"Коллекция {collection}"),
                hadith_count=hadith_count
            ))

        return collection_info

    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=SearchResponse)
async def search_hadith(request: SearchRequest):
    """Search hadiths by query"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Determine collection for search
        if "all" in request.collections or not request.collections:
            collections = get_available_collections()
            if not collections:
                raise HTTPException(status_code=500, detail="No collections available")
            selected_collection = collections[0]
        else:
            selected_collection = request.collections[0]

        # Setup retriever
        if not hadith_retriever.setup_retriever(selected_collection):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to setup retriever for collection '{selected_collection}'"
            )

        # Setup QA chain
        qa_chain.setup_qa_chain(hadith_retriever.retriever)

        # Get answer
        result = qa_chain.get_answer(
                 # main.py (продолжение вашего кода)

                 @ app.post("/api/search", response_model=SearchResponse)
        async

        def search_hadith(request: SearchRequest):

            """Search hadiths by query"""
        try:
            if not request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            # Determine collection for search
            if "all" in request.collections or not request.collections:
                collections = get_available_collections()
                if not collections:
                    raise HTTPException(status_code=500, detail="No collections available")
                selected_collection = collections[0]
            else:
                selected_collection = request.collections[0]

            # Setup retriever
            if not hadith_retriever.setup_retriever(selected_collection):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to setup retriever for collection '{selected_collection}'"
                )

            # Setup QA chain
            qa_chain.setup_qa_chain(hadith_retriever.retriever)

            # Get answer and sources
            result = qa_chain.get_answer(request.query)

            # Process sources for response
            processed_sources = []
            for doc in result.get("sources", []):
                source_data = {
                    "uid_chunked": doc.metadata.get("uid_chunked", ""),
                    "content": doc.page_content,
                    "collection": doc.metadata.get("collection", ""),
                    "book": doc.metadata.get("book", ""),
                    "hadith_number": doc.metadata.get("hadith_number", ""),
                    "narrator": doc.metadata.get("narrator", ""),
                    "arabic_text": doc.metadata.get("arabic_text", ""),
                    "grade": doc.metadata.get("grade", ""),
                    "similarity_score": getattr(doc, 'similarity_score', 0.0)
                }
                processed_sources.append(source_data)

            # Generate summary
            summary = generate_search_summary(request.query, processed_sources)

            # Generate statistics
            stats = generate_search_statistics(processed_sources, request.collections)

            return SearchResponse(
                answer=result.get("answer", ""),
                sources=processed_sources,
                summary=summary,
                statistics=stats
            )

        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/graph", response_model=GraphData)
    async def get_graph_data(
            query: str = Query(..., description="Search query"),
            collections: List[str] = Query(default=["all"], description="Collections to search in"),
            max_nodes: int = Query(default=20, description="Maximum number of nodes")
    ):
        """Get graph data for visualization"""
        try:
            # Perform search first
            search_request = SearchRequest(query=query, collections=collections)
            search_result = await search_hadith(search_request)

            # Generate graph from search results
            graph_service = GraphService()
            graph_data = graph_service.build_graph_from_search(
                query=query,
                sources=search_result.sources,
                summary=search_result.summary,
                max_nodes=max_nodes
            )

            return graph_data

        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/hadith/{uid_chunked}")
    async def get_hadith_details(uid_chunked: str):
        """Get detailed information about specific hadith"""
        try:
            metadata = metadata_manager.get_chunk_metadata(uid_chunked)

            if not metadata:
                raise HTTPException(status_code=404, detail="Hadith not found")

            return {
                "uid_chunked": uid_chunked,
                "content": metadata.get("content", ""),
                "arabic_text": metadata.get("arabic_text", ""),
                "collection": metadata.get("collection", ""),
                "book": metadata.get("book", ""),
                "chapter": metadata.get("chapter", ""),
                "hadith_number": metadata.get("hadith_number", ""),
                "narrator": metadata.get("narrator", ""),
                "grade": metadata.get("grade", ""),
                "themes": metadata.get("themes", []),
                "keywords": metadata.get("keywords", []),
                "related_hadiths": get_related_hadiths(uid_chunked)
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting hadith details: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/narrator/{narrator_name}")
    async def get_narrator_info(narrator_name: str):
        """Get information about hadith narrator"""
        try:
            narrator_info = get_narrator_details(narrator_name)

            if not narrator_info:
                raise HTTPException(status_code=404, detail="Narrator not found")

            return narrator_info

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting narrator info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stats")
    async def get_statistics():
        """Get general statistics about collections"""
        try:
            stats = {
                "total_hadiths": len(metadata_manager.hadith_chunks_df),
                "collections": {},
                "top_narrators": get_top_narrators(),
                "theme_distribution": get_theme_distribution()
            }

            # Get stats per collection
            for collection in get_available_collections():
                collection_df = metadata_manager.hadith_chunks_df[
                    metadata_manager.hadith_chunks_df['collection'] == collection
                    ]
                stats["collections"][collection] = {
                    "name": get_collection_display_name(collection),
                    "hadith_count": len(collection_df),
                    "book_count": collection_df['book'].nunique() if 'book' in collection_df.columns else 0,
                    "narrator_count": collection_df['narrator'].nunique() if 'narrator' in collection_df.columns else 0
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Helper functions
    def get_available_collections() -> List[str]:
        """Get list of available FAISS collections"""
        try:
            collections = []
            if os.path.exists(Config.FAISS_INDEX_DIR):
                for item in os.listdir(Config.FAISS_INDEX_DIR):
                    index_path = os.path.join(Config.FAISS_INDEX_DIR, item)
                    if os.path.isdir(index_path):
                        collections.append(item)
            return sorted(collections)
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    def get_collection_hadith_count(collection: str) -> int:
        """Get number of hadiths in collection"""
        try:
            if metadata_manager and metadata_manager.hadith_chunks_df is not None:
                collection_df = metadata_manager.hadith_chunks_df[
                    metadata_manager.hadith_chunks_df['collection'] == collection
                    ]
                return len(collection_df)
            return 0
        except Exception as e:
            logger.error(f"Error counting hadiths for {collection}: {e}")
            return 0

    def get_collection_display_name(collection: str) -> str:
        """Get display name for collection"""
        display_names = {
            "bukhari": "Сахих аль-Бухари",
            "muslim": "Сахих Муслим",
            "tirmidhi": "Джами ат-Тирмизи",
            "abudawud": "Сунан Абу Дауд",
            "nasai": "Сунан ан-Насаи",
            "ibnmajah": "Сунан Ибн Маджах"
        }
        return display_names.get(collection, collection.title())

    def generate_search_summary(query: str, sources: List[Dict]) -> str:
        """Generate summary for search results"""
        if not sources:
            return "По вашему запросу не найдено релевантных хадисов."

        collections_found = set(source.get("collection", "") for source in sources)
        narrators_found = set(source.get("narrator", "") for source in sources if source.get("narrator"))

        summary = f"Найдено {len(sources)} релевантных хадисов "
        summary += f"из {len(collections_found)} коллекций "
        summary += f"от {len(narrators_found)} рассказчиков."

        return summary

    def generate_search_statistics(sources: List[Dict], requested_collections: List[str]) -> Dict[str, Any]:
        """Generate statistics for search results"""
        stats = {
            "total_results": len(sources),
            "collections_used": {},
            "top_narrators": {},
            "average_similarity": 0.0,
            "grade_distribution": {}
        }

        if not sources:
            return stats

        # Collection distribution
        for source in sources:
            collection = source.get("collection", "unknown")
            stats["collections_used"][collection] = stats["collections_used"].get(collection, 0) + 1

        # Narrator distribution
        for source in sources:
            narrator = source.get("narrator", "")
            if narrator:
                stats["top_narrators"][narrator] = stats["top_narrators"].get(narrator, 0) + 1

        # Grade distribution
        for source in sources:
            grade = source.get("grade", "unknown")
            stats["grade_distribution"][grade] = stats["grade_distribution"].get(grade, 0) + 1

        # Average similarity
        similarities = [source.get("similarity_score", 0.0) for source in sources]
        stats["average_similarity"] = sum(similarities) / len(similarities) if similarities else 0.0

        return stats

    def get_related_hadiths(uid_chunked: str, limit: int = 5) -> List[Dict]:
        """Get related hadiths for given hadith"""
        try:
            # This would use similarity search or topic modeling
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting related hadiths: {e}")
            return []

    def get_narrator_details(narrator_name: str) -> Dict[str, Any]:
        """Get detailed information about narrator"""
        try:
            if not metadata_manager or metadata_manager.hadith_chunks_df is None:
                return {}

            narrator_df = metadata_manager.hadith_chunks_df[
                metadata_manager.hadith_chunks_df['narrator'] == narrator_name
                ]

            if narrator_df.empty:
                return {}

            collections = narrator_df['collection'].unique().tolist()
            hadith_count = len(narrator_df)

            return {
                "name": narrator_name,
                "hadith_count": hadith_count,
                "collections": collections,
                "biography": f"Один из передатчиков хадисов, упоминается в {len(collections)} коллекциях.",
                "reliability": "Высокая",  # This would come from narrator database
                "period": "7-8 век н.э.",  # This would come from narrator database
                "top_themes": get_narrator_themes(narrator_name)
            }

        except Exception as e:
            logger.error(f"Error getting narrator details: {e}")
            return {}

    def get_narrator_themes(narrator_name: str) -> List[str]:
        """Get main themes for narrator's hadiths"""
        # This would analyze themes from narrator's hadiths
        return ["Поклонение", "Этика", "Семейная жизнь"]

    def get_top_narrators(limit: int = 10) -> List[Dict]:
        """Get top narrators by hadith count"""
        try:
            if not metadata_manager or metadata_manager.hadith_chunks_df is None:
                return []

            narrator_counts = metadata_manager.hadith_chunks_df['narrator'].value_counts().head(limit)

            return [
                {"name": narrator, "hadith_count": count}
                for narrator, count in narrator_counts.items()
                if narrator and str(narrator) != 'nan'
            ]

        except Exception as e:
            logger.error(f"Error getting top narrators: {e}")
            return []

    def get_theme_distribution() -> Dict[str, int]:
        """Get distribution of themes across all hadiths"""
        # This would analyze themes from all hadiths
        # For now, return sample data
        return {
            "Поклонение": 450,
            "Этика и мораль": 380,
            "Семейная жизнь": 320,
            "Торговля": 280,
            "Правовые вопросы": 250,
            "История": 200
        }

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

def get_collection_display_name(collection: str) -> str:
    """Get display name for collection"""
    display_names = {
        "bukhari": "Сахих аль-Бухари",
        "muslim": "Сахих Муслим",
        "tirmidhi": "Джами ат-Тирмизи",
        "abudawud": "Сунан Абу Дауд",
        "nasai": "Сунан ан-Насаи",
        "ibnmajah": "Сунан Ибн Маджах"
    }
    return display_names.get(collection, collection.title())