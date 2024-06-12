FROM paddlepaddle/paddle:2.4.2-gpu-cuda11.7-cudnn8.4-trt8.4

RUN apt update
RUN python -m pip install --upgrade pip
RUN apt install libvips-dev --no-install-recommends -y

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install Pillow==9.5.0

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN pip install "python-doctr[torch]"
RUN pip install rapidfuzz==2.15.1
RUN apt install libvips-dev -y --upgrade
RUN pip install pyvips==2.2.1

WORKDIR /app
COPY . /app/

CMD ["uvicorn", "application.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]
