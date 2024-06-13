FROM ubuntu:22.04
LABEL maintainer="Vlad"
RUN apt-get update -y
COPY . /opt/mlhw
WORKDIR /opt/mlhw
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD ["python3", "app.py"]
