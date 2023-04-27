FROM python:3.9
#FROM tiangolo/uvicorn-gunicorn:python3.11-slim

LABEL maintainer="Sebastian Ramirez <tiangolo@gmail.com>"

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
