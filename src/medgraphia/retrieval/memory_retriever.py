"""Per-user conversational memory retriever, wrapping the QAText graph in graph/queries.py."""

from __future__ import annotations

from dataclasses import dataclass, field

from medgraphia.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QAMemory:
    qa_id: str
    question: str
    answer: str
    created_at: str


@dataclass
class MemoryRetrievalResult:
    memories: list[QAMemory] = field(default_factory=list)

    @property
    def qa_ids(self) -> list[str]:
        return [m.qa_id for m in self.memories]


class MemoryRetriever:
    """
    Usage::

        retriever = MemoryRetriever()
        result = await retriever.retrieve(user_id="doctor_01", cuis=["D003920"])
        for m in result.memories:
            print(m.question, "->", m.answer)
    """

    def __init__(self, limit: int = 5, half_life_days: float = 30.0) -> None:
        self._limit = limit
        self._half_life_days = half_life_days

    @classmethod
    def from_settings(cls) -> MemoryRetriever:
        from medgraphia.config import get_settings

        cfg = get_settings()
        return cls(limit=cfg.qa_memory_limit, half_life_days=cfg.qa_memory_half_life_days)

    async def retrieve(self, user_id: str | None, cuis: list[str]) -> MemoryRetrievalResult:
        if not user_id or user_id == "anonymous" or not cuis:
            return MemoryRetrievalResult()

        from medgraphia.graph.queries import get_user_qa_memories

        try:
            rows = await get_user_qa_memories(
                user_id=user_id, cuis=cuis, limit=self._limit, half_life_days=self._half_life_days
            )
        except Exception as exc:
            logger.warning("qa_memory_retrieve_failed", error=str(exc))
            return MemoryRetrievalResult()

        memories = [
            QAMemory(
                qa_id=r["qa_id"], question=r["question"], answer=r["answer"], created_at=r["created_at"]
            )
            for r in rows
        ]
        logger.info("qa_memory_retrieved", user_id=user_id, count=len(memories))
        return MemoryRetrievalResult(memories=memories)
