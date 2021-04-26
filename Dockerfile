# FROM ubuntu:16.04
FROM python:3.6

# RUN apt-get update -y && \
#     apt-get install -y python-pip python-dev && \
#     pip install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 install --upgrade pip

RUN pip3 install wheel

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD [ "application.py" ]