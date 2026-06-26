from __future__ import annotations

from uuid import UUID

from services.retrieval import RetrievalService


def test_retrieval_point_id_is_qdrant_compatible_uuid() -> None:
    point_id = RetrievalService._point_id("provider_strategy.md", 0)

    assert str(UUID(point_id)) == point_id
