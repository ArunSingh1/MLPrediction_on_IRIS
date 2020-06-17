FROM python:3 
RUN apt-get update -y 
RUN apt-get install -y python3-pip 
COPY . /home/support/Documents/OBSOLETE/Arun/ALML/MLPrediction_on_IRIS
WORKDIR /home/support/Documents/OBSOLETE/Arun/ALML/MLPrediction_on_IRIS
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]
