# Basic nginx dockerfile starting with Ubuntu 20.04
#FROM ubuntu:20.04
#RUN apt-get -y update
#RUN apt-get -y install nginx

FROM tiangolo/uvicorn-gunicorn:python3.10

LABEL maintainer="Sebastian Ramirez <tiangolo@gmail.com>"

RUN pip install --upgrade pip

#WORKDIR /FASTAPI
COPY requirements.txt ./FASTAPI/app/requirements.txt
RUN pip freeze > requirements.txt
RUN pip install --no-cache-dir -r ./FASTAPI/app/requirements.txt

#COPY ./app
