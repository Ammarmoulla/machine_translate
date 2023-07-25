#build
FROM python:3.10.4-slim-bullseye as build

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN  apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

WORKDIR /content/machine_translate/

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "sh", "start.sh" ]