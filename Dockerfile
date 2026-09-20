# Match the Python minor version you use locally. The model is deserialised by
# this interpreter, and scikit-learn objects do not promise to survive a version
# jump.
FROM python:3.14-slim

# PYTHONUNBUFFERED: without it stdout is buffered and container logs appear late
# or not at all, which makes every debugging session harder.
# PYTHONDONTWRITEBYTECODE: no .pyc files to bake into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, source later: this layer is only rebuilt when
# requirements.txt changes, so editing a line of Python does not reinstall
# scikit-learn. --no-cache-dir keeps pip's download cache out of the image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the model
COPY build/model/ ./model/
ENV MODEL_PATH=/app/model

# The source has to ship: cloudpickle stored the preprocessing function by
# reference, so the loading process must be able to import src.inference.pipeline.
COPY src/ ./src/


# Documentation only — it does not publish anything by itself. The mapping
# happens at run time.
EXPOSE 8000

RUN useradd --create-home --uid 1000 appuser
USER appuser

# 0.0.0.0 and not localhost: inside a container "localhost" means this container
# only, and the port mapping from outside would reach nothing. No --reload: it
# watches the filesystem and has no place in an image.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]