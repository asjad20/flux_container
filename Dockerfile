FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY Airborne-1_Beaded/ ./Airborne-1_Beaded/
COPY Airborne-2/ ./Airborne-2/
COPY gemini_images/ ./gemini_images/
COPY Hercule/ ./Hercule/
COPY Speed_Rope/ ./Speed_Rope/

CMD ["python", "-u", "handler.py"]