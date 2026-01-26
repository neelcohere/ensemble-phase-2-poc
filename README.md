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

1. Ensure you have an API key defined for the model you intend to use in a `.env` file in the root:
```
cd path\to\root
touch .env
```
For example, this project defaults to the Cohere chat API, which requires a `COHERE_API_KEY` to be set as an env var.

2. Navigate to the project root and run the python `main.py` module
```
cd path\to\root
uv run ensemble-phase-2-poc
```

# Running unit tests
```
uv run test/test_import.py
```

# Contributing
1. Install all dependencies with `uv sync`
2. Make your changes
3. Use `ruff` to lint

```
ruff check              # run lint checks
ruff check --fix        # run lint checks and auto fix
```