import torch
from queue import Queue
from threading import Thread
import time
import asyncio
from typing import Dict, Any


class BatchProcessor:
    def __init__(self, tokenizer, model, batch_size=8, max_wait=0.1):
        self.queue = Queue()
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.active = True
        self._start_processor()
    
    def _start_processor(self):
        def processor():
            while self.active:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
        Thread(target=processor, daemon=True).start()
    
    def _collect_batch(self):
        batch = []
        start_time = time.time()

        while len(batch) < self.batch_size and (time.time() - start_time) < self.max_wait:
            try:
                batch.append(self.queue.get_nowait())
            except:
                time.sleep(0.001)
        return batch
    
    def _process_batch(self, batch: list):
        try:
            texts = [item["text"] for item in batch]
            inputs = self.tokenizer(
                texts,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = 512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            for i, item in enumerate(batch):
                label = "positive" if probs[i][1] >= 0.5 else "negative"
                item["future"].set_result({
                    "label": label,
                    "score": probs[i][1].item()
                })
        except Exception as e:
            for item in batch:
                item["future"].set_exception(e)
    
    async def predict(self, text: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.queue.put({"text": text, "future": future})
        return await future