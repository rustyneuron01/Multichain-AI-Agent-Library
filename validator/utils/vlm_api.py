import asyncio
import aiohttp
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
from fiber.logging_utils import get_logger

logger = get_logger(__name__)

# Constants for rate limiting
MAX_REQUESTS_PER_MINUTE = 10000  # Reduced from 10000
MAX_TOKENS_PER_MINUTE = 1000000  # Reduced from 2000000
BATCH_SIZE = 30  # Increased from 10
RETRY_ATTEMPTS = 3
COOLDOWN_AFTER_RATE_LIMIT = 15  # seconds
SLEEP_ON_BATCH_FAILURE = 1  # seconds
API_TIMEOUT = 45  # seconds

@dataclass
class StatusTracker:
    """Tracks API request statistics and rate limits."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: float = 0
    
    def log_status(self):
        """Log current status"""
        logger.info(
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Rate Limits: {self.num_rate_limit_errors}"
        )

@dataclass
class VLMRequest:
    """Represents a single VLM API request."""
    task_id: int
    messages: List[Dict]
    max_tokens: int = 1000
    temperature: float = 0.2
    model: str = "gpt-4o"
    attempts_left: int = RETRY_ATTEMPTS
    metadata: Dict = field(default_factory=dict)
    result: Optional[str] = None

class VLMProcessor:
    """Manages batched VLM API requests with rate limiting and retries."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.status = StatusTracker()
        self.retry_queue = asyncio.Queue()
        self.request_semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        self.last_request_time = 0
        self.available_request_capacity = MAX_REQUESTS_PER_MINUTE
        self.available_token_capacity = MAX_TOKENS_PER_MINUTE
        self._task_id_counter = 0
        self._task_id_lock = asyncio.Lock()
        
    async def _get_next_task_id(self) -> int:
        """Get next unique task ID."""
        async with self._task_id_lock:
            self._task_id_counter += 1
            return self._task_id_counter
            
    async def _update_capacity(self):
        """Update available capacity based on time elapsed."""
        now = time.time()
        time_passed = now - self.last_request_time
        if time_passed >= 60:  # Reset after a minute
            self.available_request_capacity = MAX_REQUESTS_PER_MINUTE
            self.available_token_capacity = MAX_TOKENS_PER_MINUTE
            self.last_request_time = now
        elif time_passed > 0:  # Partial replenishment
            request_replenishment = (MAX_REQUESTS_PER_MINUTE * time_passed / 60)
            token_replenishment = (MAX_TOKENS_PER_MINUTE * time_passed / 60)
            self.available_request_capacity = min(
                MAX_REQUESTS_PER_MINUTE,
                self.available_request_capacity + request_replenishment
            )
            self.available_token_capacity = min(
                MAX_TOKENS_PER_MINUTE,
                self.available_token_capacity + token_replenishment
            )
            self.last_request_time = now

    async def _process_request(
        self,
        session: aiohttp.ClientSession,
        request: VLMRequest
    ) -> Optional[str]:
        """Process a single VLM request with retries and rate limiting."""
        try:
            async with self.request_semaphore:  # Limit concurrent requests
                await self._update_capacity()
                
                # Check if we need to cool down after rate limit
                seconds_since_rate_limit = time.time() - self.status.time_of_last_rate_limit_error
                if seconds_since_rate_limit < COOLDOWN_AFTER_RATE_LIMIT:
                    await asyncio.sleep(COOLDOWN_AFTER_RATE_LIMIT - seconds_since_rate_limit)
                
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=request.model,
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    ),
                    timeout=API_TIMEOUT
                )
                content = response.choices[0].message.content
                self.status.num_tasks_succeeded += 1
                return content
                
        except asyncio.TimeoutError as e:
            logger.error(f"Request {request.task_id} timed out after {API_TIMEOUT}s: {str(e)}")
            self.status.num_tasks_failed += 1
            if request.attempts_left > 0:
                request.attempts_left -= 1
                await self.retry_queue.put(request)
            return None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Request {request.task_id} failed: {error_msg}\n"
                f"Error type: {type(e).__name__}\n"
                f"Model: {request.model}\n"
                f"Max tokens: {request.max_tokens}\n"
                f"Temperature: {request.temperature}"
            )
            
            self.status.num_tasks_failed += 1
            
            if "rate limit" in error_msg.lower():
                self.status.time_of_last_rate_limit_error = time.time()
                self.status.num_rate_limit_errors += 1
                logger.warning(f"Rate limit error encountered. Total: {self.status.num_rate_limit_errors}")
                if request.attempts_left > 0:
                    request.attempts_left -= 1
                    await self.retry_queue.put(request)
            elif request.attempts_left > 0:
                request.attempts_left -= 1
                await self.retry_queue.put(request)
                
            return None

    async def process_batch(
        self,
        requests: List[VLMRequest]
    ) -> List[Optional[str]]:
        """Process a batch of VLM requests with improved concurrency control."""
        if not requests:
            return []

        async with aiohttp.ClientSession() as session:
            # Process initial requests
            tasks = []
            for request in requests:
                await self._update_capacity()
                if self.available_request_capacity >= 1:
                    self.available_request_capacity -= 1
                    request.attempts_left -= 1
                    self.status.num_tasks_started += 1
                    self.status.num_tasks_in_progress += 1
                    tasks.append(asyncio.create_task(self._process_request(session, request)))
                else:
                    await asyncio.sleep(0.1)

            # Process retry queue
            retry_tasks = []
            while not self.retry_queue.empty():
                request = await self.retry_queue.get()
                await self._update_capacity()
                if self.available_request_capacity >= 1:
                    self.available_request_capacity -= 1
                    self.status.num_tasks_started += 1
                    self.status.num_tasks_in_progress += 1
                    retry_tasks.append(asyncio.create_task(self._process_request(session, request)))
                else:
                    await self.retry_queue.put(request)
                    break

            # Wait for all tasks
            all_results = await asyncio.gather(*tasks, *retry_tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(all_results[:len(requests)]):  # Only take initial request results
                if isinstance(result, Exception):
                    logger.error(f"Batch request failed: {str(result)}")
                    self.status.num_tasks_failed += 1
                    processed_results.append(None)
                else:
                    processed_results.append(result)
                self.status.num_tasks_in_progress -= 1
            
            return processed_results

    async def get_reference_counts_batch(
        self,
        frames: List[Dict[str, Any]],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Get reference counts for multiple frames in parallel."""
        requests = []
        for frame_data in frames:
            messages = [
                {"role": "system", "content": "You are an expert at counting objects in soccer match frames."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_data['encoded_image']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            request = VLMRequest(
                task_id=await self._get_next_task_id(),
                messages=messages,
                metadata={"frame_id": frame_data.get("frame_id")}
            )
            requests.append(request)

        # Process in batches
        all_results = []
        for i in range(0, len(requests), BATCH_SIZE):
            batch = requests[i:i + BATCH_SIZE]
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)

        return all_results

    async def validate_bbox_content_batch(
        self,
        images: List[Dict[str, Any]],
        expected_class: str
    ) -> List[float]:
        """Validate multiple bounding boxes in parallel."""
        requests = []
        for img_data in images:
            prompt = (
                f"This image supposedly has a {expected_class} in a soccer context. "
                "Rate probability [0.0..1.0]. Return only the numeric probability."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data['encoded_image']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            request = VLMRequest(
                task_id=await self._get_next_task_id(),
                messages=messages,
                max_tokens=10,
                metadata={"bbox_id": img_data.get("bbox_id")}
            )
            requests.append(request)

        # Process in batches
        all_results = []
        for i in range(0, len(requests), BATCH_SIZE):
            batch = requests[i:i + BATCH_SIZE]
            batch_results = await self.process_batch(batch)
            
            # Convert results to floats
            processed_results = []
            for result in batch_results:
                try:
                    if result is not None:
                        value = float(result.strip())
                        processed_results.append(max(0.0, min(1.0, value)))
                    else:
                        processed_results.append(0.0)
                except (ValueError, TypeError):
                    processed_results.append(0.0)
            
            all_results.extend(processed_results)

        return all_results

    async def validate_keypoints_batch(
        self,
        frames: List[Dict[str, Any]],
        prompt: str
    ) -> List[float]:
        """Validate keypoints for multiple frames in parallel."""
        requests = []
        for frame_data in frames:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing soccer pitch keypoint placements. You will compare keypoint placements between a reference image and a predicted image."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                "Compare these two images:\n"
                                "1. Reference image showing correct keypoint placements on a soccer pitch\n"
                                "2. Predicted image showing keypoint placements that need validation\n\n"
                                "Rate how well the predicted keypoints match the reference keypoints from 0.0 (completely wrong) to 1.0 (perfect match).\n"
                                "Consider:\n"
                                "- Number of keypoints (should match reference)\n"
                                "- Position accuracy (keypoints should be in similar relative positions)\n"
                                "- Coverage of important areas (corners, center, penalty areas)\n\n"
                                "Return ONLY a single number between 0.0 and 1.0"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_data['reference_image']}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_data['keypoint_image']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            request = VLMRequest(
                task_id=await self._get_next_task_id(),
                messages=messages,
                max_tokens=10,
                metadata={"frame_id": frame_data.get("frame_id")}
            )
            requests.append(request)

        # Process in batches
        all_results = []
        for i in range(0, len(requests), BATCH_SIZE):
            batch = requests[i:i + BATCH_SIZE]
            logger.info(f"Processing keypoint batch {i//BATCH_SIZE + 1} with {len(batch)} requests")
            batch_results = await self.process_batch(batch)
            
            # Convert results to floats
            processed_results = []
            for result in batch_results:
                try:
                    if result is not None:
                        # Clean any markdown formatting and extract just the number
                        result = result.replace('```', '').strip()
                        # Find the first number in the response
                        import re
                        numbers = re.findall(r'0\.\d+|\d+\.?\d*', result)
                        if numbers:
                            value = float(numbers[0])
                            processed_results.append(max(0.0, min(1.0, value)))
                        else:
                            logger.error(f"No valid number found in result: {result}")
                            processed_results.append(0.0)
                    else:
                        processed_results.append(0.0)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing keypoint result: {str(e)}, raw result: {result}")
                    processed_results.append(0.0)
            
            all_results.extend(processed_results)

        return all_results 
        return all_results 