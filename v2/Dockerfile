FROM python:3.7-slim-stretch

# Información del desarrollador
LABEL maintainer="Francisco Solano <fransol0728@correo.ugr.es>"

# Directorio de trabajo
WORKDIR /

# Copiamos el fichero de dependencias
COPY requirements.txt ./

# Copiamos los modelos de predicción
COPY exp_smoothing_humidity.pkl ./
COPY exp_smoothing_temperature.pkl ./

# Copiamos el código del microservicio
COPY microservicio.py ./

# Instalamos las dependencias necesarias para el proyecto
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Levantamos el servidor
CMD gunicorn --bind 0.0.0.0:5000 microservicio:app
