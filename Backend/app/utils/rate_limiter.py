import threading,time
from collections import deque

class RateLimiter:
     """Thread-safe sliding-window rate limiter. Blocks the caller until
     it's safe to make another request, instead of firing and hoping."""
     def __init__(self,max_requested:int,period:float = 60.0):
          self.max_requested = max_requested
          self.period= period
          self.timestamps = deque()
          self.lock = threading.Lock()

     def acquire(self):
          while True:
               with self.lock:
                    now = time.monotonic()
                    while self.timestamps and now - self.timestamps[0] > self.period:
                         self.timestamps.popleft()
                    
                    if len(self.timestamps) < self.max_requested:
                         self.timestamps.append(now)
                         return
                    
                    sleep_time = self.period - (now-self.timestamps[0]) + 0.1

                    print(f"[RateLimiter] {self.max_requested}/{self.period}s reached — sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)