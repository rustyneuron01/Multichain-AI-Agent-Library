import asyncio

class AsyncBarrier:
    """
    An asyncio-based barrier to synchronize multiple asynchronous tasks.
    Ensures that all parties reach the barrier before proceeding.
    """

    def __init__(self, parties: int):
        if parties <= 0:
            raise ValueError("The number of parties must be greater than 0.")
        
        self.parties = parties
        self.count = 0
        self.condition = asyncio.Condition()
        self.generation = 0 
        self.timeout_event = asyncio.Event() 

    async def wait(self, timeout: float = 60.0):
        async with self.condition:
            gen = self.generation
            self.count += 1

            if self.count == self.parties:
                self.generation += 1
                self.count = 0
                self.condition.notify_all()
            else:
                try:
                    await asyncio.wait_for(self._wait_for_generation(gen), timeout=timeout)
                except asyncio.TimeoutError:
                    self.generation += 1
                    self.count = 0
                    self.condition.notify_all()

    async def _wait_for_generation(self, gen: int):
        """
        Helper function to wait until the barrier generation changes.
        """
        while gen == self.generation:
            await self.condition.wait()
