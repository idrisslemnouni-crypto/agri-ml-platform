FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip install --no-deps .
EXPOSE 8002
CMD ["python", "-m", "uvicorn", "agri_platform.api:app", "--host", "0.0.0.0", "--port", "8002"]
