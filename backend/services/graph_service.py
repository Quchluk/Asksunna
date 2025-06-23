from typing import List, Dict, Any, Set, Tuple
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphService:
    """Service for building and managing graph data structures"""

    def __init__(self):
        self.color_scheme = {
            "summary": "#7b68ee",
            "bukhari": "#00d8c9",
            "muslim": "#00d4ff",
            "tirmidhi": "#ff6bcb",
            "abudawud": "#ffb347",
            "nasai": "#c89bf7",
            "ibnmajah": "#38b2ac",
            "narrator": "#ff6bcb",
            "theme": "#00d4ff",
            "connection": "rgba(255,255,255,0.3)"
        }

    def build_graph_from_search(
            self,
            query: str,
            sources: List[Dict[str, Any]],
            summary: str,
            max_nodes: int = 20
    ) -> GraphData:
        """Build graph data from search results"""

        try:
            nodes = []
            edges = []

            # Add summary node
            summary_node = self._create_summary_node(query, summary)
            nodes.append(summary_node)

            # Group sources by collection
            collections_map = self._group_sources_by_collection(sources[:max_nodes])

            # Add collection nodes
            collection_nodes = self._create_collection_nodes(collections_map)
            nodes.extend(collection_nodes)

            # Connect summary to collections
            for collection_id in collections_map.keys():
                edges.append(self._create_edge("summary", collection_id, self.color_scheme["summary"]))

            # Add hadith nodes and connect to collections
            hadith_nodes, hadith_edges = self._create_hadith_nodes_and_edges(collections_map)
            nodes.extend(hadith_nodes)
            edges.extend(hadith_edges)

            # Add narrator nodes and connections
            narrator_nodes, narrator_edges = self._create_narrator_nodes_and_edges(sources)
            nodes.extend(narrator_nodes)
            edges.extend(narrator_edges)

            # Add similarity connections between hadiths
            similarity_edges = self._create_similarity_edges(sources)
            edges.extend(similarity_edges)

            # Generate graph summary
            graph_summary = self._generate_graph_summary(len(nodes), len(edges), collections_map)

            return GraphData(
                nodes=nodes,
                edges=edges,
                summary=graph_summary
            )

        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return GraphData(nodes=[], edges=[], summary="Ошибка построения графа")

    def _create_summary_node(self, query: str, summary: str) -> GraphNode:
        """Create central summary node"""

        # Generate title based on query
        title = self._generate_summary_title(query)

        return GraphNode(
            id="summary",
            label=title,
            type="summary",
            tooltip=f"Обобщённый ответ: {summary[:100]}...",
            metadata={
                "query": query,
                "full_summary": summary,
                "node_type": "central"
            }
        )

    def _group_sources_by_collection(self, sources: List[Dict]) -> Dict[str, List[Dict]]:
        """Group sources by collection"""
        collections = defaultdict(list)

        for source in sources:
            collection = source.get("collection", "unknown")
            collections[collection].append(source)

        return dict(collections)

    def _create_collection_nodes(self, collections_map: Dict[str, List[Dict]]) -> List[GraphNode]:
        """Create collection nodes"""
        nodes = []

        for collection_id, hadiths in collections_map.items():
            display_name = self._get_collection_display_name(collection_id)

            node = GraphNode(
                id=collection_id,
                label=display_name,
                type="collection",
                collection=collection_id,
                tooltip=f"{display_name}: {len(hadiths)} хадисов",
                metadata={
                    "hadith_count": len(hadiths),
                    "collection_id": collection_id,
                    "description": self._get_collection_description(collection_id)
                }
            )
            nodes.append(node)

        return nodes

    def _create_hadith_nodes_and_edges(self, collections_map: Dict[str, List[Dict]]) -> Tuple[
        List[GraphNode], List[GraphEdge]]:
        """Create hadith nodes and their edges to collections"""
        nodes = []
        edges = []

        for collection_id, hadiths in collections_map.items():
            collection_color = self.color_scheme.get(collection_id, self.color_scheme["connection"])

            for i, hadith in enumerate(hadiths):
                hadith_id = f"hadith_{collection_id}_{i}"

                # Create hadith node
                label = self._truncate_text(hadith.get("content", ""), 80)

                node = GraphNode(
                    id=hadith_id,
                    label=label,
                    type="hadith",
                    collection=collection_id,
                    tooltip=self._create_hadith_tooltip(hadith),
                    metadata={
                        **hadith,
                        "hadith_id": hadith_id,
                        "display_content": label
                    }
                )
                nodes.append(node)

                # Create edge to collection
                edge = self._create_edge(collection_id, hadith_id, collection_color)
                edges.append(edge)

        return nodes, edges

    def _create_narrator_nodes_and_edges(self, sources: List[Dict]) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Create narrator nodes and connect to hadiths"""
        nodes = []
        edges = []

        # Get unique narrators
        narrators = {}
        hadith_narrator_map = {}

        for i, source in enumerate(sources):
            narrator = source.get("narrator", "").strip()
            if narrator and narrator != "nan":
                collection = source.get("collection", "unknown")
                hadith_id = f"hadith_{collection}_{i}"

                # Track narrator
                if narrator not in narrators:
                    narrators[narrator] = {
                        "hadiths": [],
                        "collections": set()
                    }

                narrators[narrator]["hadiths"].append(hadith_id)
                narrators[narrator]["collections"].add(collection)
                hadith_narrator_map[hadith_id] = narrator

        # Create narrator nodes
        for narrator_name, data in narrators.items():
            narrator_id = f"narrator_{narrator_name.replace(' ', '_')}"

            node = GraphNode(
                id=narrator_id,
                label=narrator_name,
                type="narrator",
                tooltip=f"{narrator_name}: {len(data['hadiths'])} хадисов в {len(data['collections'])} коллекциях",
                metadata={
                    "narrator_name": narrator_name,
                    "hadith_count": len(data["hadiths"]),
                    "collections": list(data["collections"])
                }
            )
            nodes.append(node)

            # Create edges to hadiths
            for hadith_id in data["hadiths"]:
                edge = self._create_edge(
                    hadith_id,
                    narrator_id,
                    self.color_scheme["narrator"],
                    edge_type="narrator"
                )
                edges.append(edge)

        return nodes, edges

    def _create_similarity_edges(self, sources: List[Dict]) -> List[GraphEdge]:
        """Create edges between similar hadiths"""
        edges = []

        # Simple similarity based on common keywords or narrator
        for i, source1 in enumerate(sources):
            collection1 = source1.get("collection", "unknown")
            hadith_id1 = f"hadith_{collection1}_{i}"

            for j, source2 in enumerate(sources[i + 1:], i + 1):
                collection2 = source2.get("collection", "unknown")
                hadith_id2 = f"hadith_{collection2}_{j}"

                # Check similarity criteria
                similarity = self._calculate_similarity(source1, source2)

                if similarity > 0.3:  # Threshold for creating edge
                    edge = self._create_edge(
                        hadith_id1,
                        hadith_id2,
                        self.color_scheme["connection"],
                        edge_type="similarity"
                    )
                    edges.append(edge)

        return edges

    def _calculate_similarity(self, source1: Dict, source2: Dict) -> float:
        """Calculate similarity between two hadiths"""
        similarity = 0.0

        # Same narrator
        if source1.get("narrator") == source2.get("narrator") and source1.get("narrator"):
            similarity += 0.4

        # Same collection
        if source1.get("collection") == source2.get("collection"):
            similarity += 0.2

        # Similar content length (basic heuristic)
        len1 = len(source1.get("content", ""))
        len2 = len(source2.get("content", ""))
        if len1 > 0 and len2 > 0:
            length_similarity = 1 - abs(len1 - len2) / max(len1, len2)
            similarity += length_similarity * 0.2

        return min(similarity, 1.0)

    def _create_edge(self, source: str, target: str, color: str, edge_type: str = None) -> GraphEdge:
        """Create graph edge"""
        return GraphEdge(
            source=source,
            target=target,
            color=color,
            type=edge_type
        )

    def _generate_summary_title(self, query: str) -> str:
        """Generate title for summary node based on query"""
        query_lower = query.lower()

        if "намерени" in query_lower:
            return "Намерение - основа оценки деяний"
        elif "молитв" in query_lower or "намаз" in query_lower:
            return "Молитва - столп религии"
        elif "пост" in query_lower:
            return "Пост в исламе"
        elif "закят" in query_lower:
            return "Закят - обязательная милостыня"
        elif "хадж" in query_lower:
            return "Паломничество - хадж"
        else:
            return f"Ответ на: {query[:40]}..."

    def _get_collection_display_name(self, collection_id: str) -> str:
        """Get display name for collection"""
        names = {
            "bukhari": "Сахих аль-Бухари",
            "muslim": "Сахих Муслим",
            "tirmidhi": "Джами ат-Тирмизи",
            "abudawud": "Сунан Абу Дауд",
            "nasai": "Сунан ан-Насаи",
            "ibnmajah": "Сунан Ибн Маджах"
        }
        return names.get(collection_id, collection_id.title())

    def _get_collection_description(self, collection_id: str) -> str:
        """Get description for collection"""
        descriptions = {
            "bukhari": "Один из самых авторитетных сборников хадисов",
            "muslim": "Сборник достоверных хадисов имама Муслима",
            "tirmidhi": "Сборник с подробными комментариями",
            "abudawud": "Сборник хадисов по исламскому праву",
            "nasai": "Сборник с акцентом на достоверность",
            "ibnmajah": "Один из шести канонических сборников"
        }
        return descriptions.get(collection_id, f"Коллекция {collection_id}")

    def _create_hadith_tooltip(self, hadith: Dict) -> str:
        """Create tooltip text for hadith"""
        collection = self._get_collection_display_name(hadith.get("collection", ""))
        narrator = hadith.get("narrator", "Неизвестен")
        number = hadith.get("hadith_number", "")

        tooltip = f"{collection}"
        if number:
            tooltip += f", Хадис {number}"
        tooltip += f"\nРассказчик: {narrator}"

        return tooltip

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + "..."

    def _generate_graph_summary(self, node_count: int, edge_count: int, collections_map: Dict) -> str:
        """Generate summary for the graph"""
        collection_count = len(collections_map)
        total_hadiths = sum(len(hadiths) for hadiths in collections_map.values())

        summary = f"Граф содержит {node_count} узлов и {edge_count} связей. "
        summary += f"Найдено {total_hadiths} хадисов из {collection_count} коллекций."

        return summary