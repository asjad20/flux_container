# Keep your base image (it has the right CUDA drivers)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# --- ADD THIS BLOCK ---
# Force upgrade PyTorch to 2.5.1 before installing other requirements.
# This fixes the 'infer_schema' error by updating the torch library registry.
RUN pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# ----------------------

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY Airborne-1_Beaded/ ./Airborne-1_Beaded/
COPY Airborne-2/ ./Airborne-2/
COPY gemini_images/ ./gemini_images/
COPY Hercule/ ./Hercule/
COPY Speed_Rope/ ./Speed_Rope/

CMD ["python", "-u", "handler.py"]