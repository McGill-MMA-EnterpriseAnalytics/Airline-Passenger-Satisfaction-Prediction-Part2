# Basic nginx dockerfile starting with Ubuntu 20.04
#FROM ubuntu:20.04
#RUN apt-get -y update
#RUN apt-get -y install nginx


FROM tiangolo/uvicorn-gunicorn:python3.10

LABEL maintainer="Sebastian Ramirez <tiangolo@gmail.com>"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app
