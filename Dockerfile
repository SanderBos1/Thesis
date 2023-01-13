FROM apache/spark-py:latest

USER root

RUN apt-get install python3-venv -y

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PATH="/.local/bin:$PATH"
ENV PYSPARK_PYTHON /opt/venv/bin/python

RUN pip3 install pyspark
RUN pip3 install numpy
RUN pip3 install statsmodels
RUN pip3 install matplotlib
RUN pip3 install pandas

COPY  . .

#uncomment the following line for debugging
#ENTRYPOINT ["tail", "-f", "/dev/null"]
CMD ["python", "./Main.py"]


