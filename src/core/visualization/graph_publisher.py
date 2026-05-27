"""In-process publisher for operation-scoped visualization deltas."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from src.core.visualization.graph_event import VisualGraphDelta


class GraphDeltaPublisher:
    """Publish graph deltas to WebSocket subscribers grouped by operation."""

    def __init__(self, *, queue_size: int = 512) -> None:
        self._queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[VisualGraphDelta]]] = defaultdict(set)
        self._dropped_counts: dict[str, int] = defaultdict(int)

    def subscribe(self, operation_id: str) -> asyncio.Queue[VisualGraphDelta]:
        queue: asyncio.Queue[VisualGraphDelta] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[operation_id].add(queue)
        return queue

    def unsubscribe(self, operation_id: str, queue: asyncio.Queue[VisualGraphDelta]) -> None:
        subscribers = self._subscribers.get(operation_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(operation_id, None)

    def publish_nowait(self, delta: VisualGraphDelta) -> None:
        for queue in list(self._subscribers.get(delta.operation_id, set())):
            if queue.full():
                try:
                    queue.get_nowait()
                    self._dropped_counts[delta.operation_id] += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(delta)
            except asyncio.QueueFull:
                self._dropped_counts[delta.operation_id] += 1

    def dropped_count(self, operation_id: str) -> int:
        return self._dropped_counts.get(operation_id, 0)


graph_delta_publisher = GraphDeltaPublisher()


__all__ = ["GraphDeltaPublisher", "graph_delta_publisher"]
