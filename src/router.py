from typing import Any, Callable, Dict, List


class DualIndexRouter:
    """
    Kapselt den Zugriff auf zwei Vektorräume und ermöglicht
    unterbrechungsfreie Cutover- und Rollback-Operationen.
    """

    def __init__(
        self,
        index_v1: Any,
        embed_v1: Callable[[str], Any],
        index_v2: Any,
        embed_v2: Callable[[str], Any],
        active_version: str = "v1",
    ):
        self.indices = {
            "v1": (index_v1, embed_v1),
            "v2": (index_v2, embed_v2),
        }
        self.active_version = active_version

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        index, embed_fn = self.indices[self.active_version]
        query_vector = embed_fn(query)
        return index.search(query_vector, limit=top_k)

    def cutover(self, target_version: str) -> None:
        if target_version not in self.indices:
            raise ValueError(f"Unbekannte Index-Version: {target_version}")
        self.active_version = target_version
