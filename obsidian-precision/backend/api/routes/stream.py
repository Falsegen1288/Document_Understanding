import asyncio
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.services.sse_manager import sse_manager

logger = logging.getLogger("stream_route")

router = APIRouter()

@router.get("/stream/{job_id}")
async def stream_job_events(job_id: str):
    """
    Establish a Server-Sent Events (SSE) progress connection.
    Clients receive page-by-page progress increments and full extraction bboxes in real-time.
    """
    logger.info(f"SSE client requesting progress stream for job: {job_id}")
    
    async def event_generator():
        # Register a local async queue for this SSE listener session
        queue = await sse_manager.subscribe(job_id)
        try:
            while True:
                try:
                    # Wait for an event to be published
                    event_data = await queue.get()
                    yield f"data: {event_data}\n\n"
                    
                    # If job completes or fails, we can close the SSE connection gracefully
                    # Wait, let's parse a quick check on string format
                    if '"type": "job_complete"' in event_data or '"type": "job_failed"' in event_data:
                        logger.info(f"SSE stream completed for job: {job_id}. Closing connection.")
                        break
                        
                except asyncio.CancelledError:
                    logger.info(f"SSE client cancelled connection for job: {job_id}")
                    break
        finally:
            # Clean up queue subscription when connection drops
            await sse_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
