import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditScoringRequest, CreditScoringResponse
from src.predict import CreditScoringModel
import mlflow.sklearn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreditRiskAPI")

# Global Variable
model_engine = None


# --- Lifespan Logic (Replaces @app.on_event) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Everything before 'yield' runs on startup.
    Everything after 'yield' runs on shutdown.
    """
    global model_engine
    try:
        model_uri = "models:/CreditRisk_LogisticRegression/Latest"
        model_engine = CreditScoringModel(model_uri)
        logger.info(f"Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        model_engine = None

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="Bati Bank Credit Scoring API",
    description="Microservice for predicting credit risk.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    status = "healthy" if model_engine else "degraded"
    return {"status": status}
