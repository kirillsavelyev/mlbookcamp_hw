# svizor/zoomcamp-model:3.9.12-slim was created by Dockerfile
# FROM python:3.9.12-slim
# WORKDIR /app
# COPY ["model1.bin", "dv.bin", "./"]

FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["predict.py", "./"]
EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

# run it by CLI
# docker build -t zoomcamp_05_hw .
# docker run -it --rm -p 9696:9696 zoomcamp_05_hw