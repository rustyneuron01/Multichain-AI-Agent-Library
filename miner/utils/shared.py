import asyncio

# Keep only one lock for tracking miner availability
miner_lock = asyncio.Lock() 