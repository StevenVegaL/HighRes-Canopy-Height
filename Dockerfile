# Imagen base con Python (CPU)
FROM python:3.10-slim

# Evitar prompts interactivos en apt
ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema para numpy, pillow, matplotlib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiamos solo requirements primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos TODO el proyecto dentro del contenedor
COPY . .

# Variables útiles para Streamlit en modo servidor
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Puerto por defecto de Streamlit
EXPOSE 8501

# Comando de arranque: tu landing de Streamlit
ENTRYPOINT ["streamlit", "run", "app/streamlit_landing_CHM_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
