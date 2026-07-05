import asyncio
import json
import logging
from typing import Dict, Set

logger = logging.getLogger("sse_manager")

class SSEManager:
    def __init__(self):
        # Maps job_id -> set of asyncio.Queues
        self._listeners: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to progress events for a job. Returns a queue."""
        async with self._lock:
            if job_id not in self._listeners:
                self._listeners[job_id] = set()
            queue = asyncio.Queue()
            self._listeners[job_id].add(queue)
            logger.info(f"Subscribed client to SSE job stream for job: {job_id}. Current listeners: {len(self._listeners[job_id])}")
            return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        """Unsubscribe from progress events."""
        async with self._lock:
            if job_id in self._listeners:
                self._listeners[job_id].discard(queue)
                if not self._listeners[job_id]:
                    del self._listeners[job_id]
                logger.info(f"Unsubscribed client from SSE job stream for job: {job_id}")

    async def publish(self, job_id: str, event_type: str, data: dict = None):
        """Publish a progress event to all listeners for a job."""
        event_dict = {
            "type": event_type,
            "job_id": job_id,
            **(data or {})
        }
        event_str = json.dumps(event_dict)
        
        async with self._lock:
            queues = self._listeners.get(job_id, set()).copy()
            
        if queues:
            logger.info(f"Publishing event '{event_type}' for job {job_id} to {len(queues)} listener(s)")
            for q in queues:
                # Use thread-safe loop helper if events are published from a background thread
                try:
                    q.put_nowait(event_str)
                except Exception as e:
                    logger.warning(f"Failed to put event in queue: {e}")

    def publish_threadsafe(self, job_id: str, event_type: str, data: dict = None):
        """Publish an event thread-safely from outside the async loop (e.g. background threads)."""
        event_dict = {
            "type": event_type,
            "job_id": job_id,
            **(data or {})
        }
        event_str = json.dumps(event_dict)
        
        # Get active running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, try to find one or log it
            logger.warning("No running asyncio event loop found to publish threadsafe event.")
            return

        async def _put():
            async with self._lock:
                queues = self._listeners.get(job_id, set()).copy()
            for q in queues:
                await q.put(event_str)
                
        asyncio.run_coroutine_threadsafe(_put(), loop)

sse_manager = SSEManager()
