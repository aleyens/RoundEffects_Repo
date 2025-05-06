FROM nvcr.io/nvidia/jax:23.10-py3 

# Setup env
ENV LANG C.UTF-8
# ENV LC_ALL C.UTF-8
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONFAULTHANDLER 1

# COPY files

COPY . .

RUN pip install -r requirements.txt

