# ML Cloud Deployment Project

## Overview
This project demonstrates deployment of a machine learning model on cloud using Docker.

## Tech Stack
- FastAPI
- Docker
- Render
- Scikit-learn

## Features
- Breast Cancer Prediction
- REST API
- Cloud deployment
- Frontend UI

## Live Demo
https://ml-api-project-cquw.onrender.com/docs

## Run Locally
uvicorn app:app --reload

## Docker Run
docker build -t ml-api .
docker run -p 8000:8000 ml-api
