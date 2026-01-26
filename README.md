# Setting up MLFlow server

1. Ensure you have uv installed and have used it to setup the env for this repo:
```
cd path\to\root
uv sync
```

2. Run the following command:
```
uvx mlflow server
```
This will setup the mlflow server on https://localhost:5000

# Running a test workflow

1. Navigate to the project root and run the python `main.py` module
```
cd path\to\root
python -m src.main
```
 