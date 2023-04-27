FROM tiangolo/uvicorn-gunicorn:python3.10

LABEL maintainer="Sebastian Ramirez <tiangolo@gmail.com>"

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./FASTAPI/app /FASTAPI/app
