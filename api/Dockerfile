FROM python:3.8.13 as base

ADD requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
#RUN apt install git
#RUN git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.0
#RUN ImageMagick-7.1.0/configure
#RUN make
#RUN ImageMagick-7.1.0/configure --with-modules
#RUN make install

ENV PYTHONPATH=$PYTHONPATH:/src/

ADD ./ /src/

WORKDIR /src

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "--timeout=600", "app:app", "--threads=10"]
