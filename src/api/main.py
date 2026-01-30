import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditScoringRequest, CreditScoringResponse
from src.predict import CreditScoringModel
import mlflow.sklearn
