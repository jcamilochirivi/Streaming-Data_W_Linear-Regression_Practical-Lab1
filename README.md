# Practical Lab 1: Streaming Data for Predictive Maintenance with Linear Regression-Based Alerts

Student: Juan Camilo Chirivi - ID 9115141

### Project Architecture

The main workflow is:
Historical CSV → Neon PostgreSQL → Query Training Data → Linear Regression → Residual Analysis → Threshold Discovery → Synthetic Testing Data → Streaming → Alert/Error Detection → Event Log

The project separates database and streaming functionality into Python
modules under src/, while the complete analysis and experiment are
demonstrated in the Jupyter Notebook.

### The original dataset used for training is:

data/training/RMBR4-2_export_test.csv
The original dataset contains timestamped industrial current
measurements.

## Setup Instructions

### Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

### Install dependencies
python -m pip install -r requirements.txt

### Create the .env file
DATABASE_URL=postgresql://neondb_owner:npg_9B8WZtnileQr@ep-orange-poetry-b5hl667n-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require

### Ejecute the jupyter notebook