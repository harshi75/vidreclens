FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/vidreclens/__init__.py ./vidreclens/__init__.py
COPY src/vidreclens/serve ./vidreclens/serve
COPY bundle ./bundle
ENV PYTHONPATH=/app
ENV VIDRECLENS_BUNDLE=/app/bundle
EXPOSE 8000
CMD ["uvicorn", "--factory", "vidreclens.serve.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
