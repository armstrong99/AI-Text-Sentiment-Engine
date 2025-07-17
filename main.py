from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import HTTPException
from contextlib import asynccontextmanager
from batch_processor.schemas import PredictionRequest, PredictionResponse
from batch_processor.index import BatchProcessor
from sse_starlette.sse import EventSourceResponse
from config import MODEL_NAME, BATCH_SIZE, MAX_WAIT
from data.preprocess_data import load_and_tokenize_dataset
from data.fine_tune_model import CustomModelTrainer
from utils import timeit
import torch
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Place any startup code here
    # Limit to 1-3 threads max to prevent overconsumption, pytorch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(3)

    # train model every 6 hours
    result = load_and_tokenize_dataset("data/data.csv")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # train the model
    trainer = CustomModelTrainer(model, 
                                 result['tokenizer'], 
                                 tokenized_val_dataset=result["validation"], 
                                  tokenized_train_dataset=result["train"])
    trainer.train_model()
    trainer.save_model()

 
    print("Loading fine tuned model...") 

    model.eval() # Set to evaluation mode

    #init batch processor
    app.state.processor = BatchProcessor(
         tokenizer=result['tokenizer'],
         model=model,
         batch_size=BATCH_SIZE,
         max_wait=MAX_WAIT
    )
    yield

    # Place any shutdown code here
    app.state.processor.active = False
    print("shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

        if not request.text:
            raise HTTPException(status_code=400, detail="Input text is required")
        
        return await app.state.processor.predict(request.text)

@app.get("/stream")
async def stream(text: str):
    async def event_generator():
        words = text.split()
        for word in words:
             yield {"data": word}
             await asyncio.sleep(0.3)
    return EventSourceResponse(event_generator())
@app.get("/health")
def health_check():
    return {"status": "ok"}