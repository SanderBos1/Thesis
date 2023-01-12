ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=11.0.5
ARG PYTHON_VERSION=3.9.8

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /
ARG PYSPARK_VERSION=3.3.1
RUN apt-get update
RUN apt-get install default-jdk -y
RUN apt-get update && apt-get install procps -y

ARG spark_version=3.3.1
ARG hadoop_version=3

RUN apt-get update -y && \
    apt-get install -y curl && \
    curl https://archive.apache.org/dist/spark/spark-${spark_version}/spark-${spark_version}-bin-hadoop${hadoop_version}.tgz -o spark.tgz && \
    tar -xf spark.tgz && \
    mv spark-${spark_version}-bin-hadoop${hadoop_version} /usr/bin/ && \
    mkdir /usr/bin/spark-${spark_version}-bin-hadoop${hadoop_version}/logs && \
    rm spark.tgz



RUN adduser sander
USER sander
WORKDIR /home/sander

ENV VIRTUAL_ENV=/home/sander/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PATH="/home/sander/.local/bin:$PATH"

ENV SPARK_HOME /usr/bin/spark-${spark_version}-bin-hadoop${hadoop_version}

COPY --chown=sander:sander . .

RUN pip install --upgrade pip
RUN pip install pyspark==${PYSPARK_VERSION}
RUN pip install numpy
RUN pip install statsmodels
RUN pip install matplotlib
RUN pip install pandas

ENV PYSPARK_PYTHON /home/sander/opt/venv/bin/python

#uncomment the following line for debugging
#ENTRYPOINT ["tail", "-f", "/dev/null"]
CMD ["python", "./Main.py"]


