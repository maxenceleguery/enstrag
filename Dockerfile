FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /app/

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends wget libgl1 libglib2.0-0 \
    && apt-get -y autoremove && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /app/* \
    # install conda
    && mkdir -p /opt/conda \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /opt/conda/miniconda.sh \
    && apt-get remove -y wget \
    && bash /opt/conda/miniconda.sh -b -p /opt/miniconda \
    && rm -rf /opt/conda/* \
    && . /opt/miniconda/bin/activate \
    && conda clean --all \
    && echo '#!/bin/bash\n/opt/miniconda/bin/conda run --no-capture-output -n ragenv python3 "$@"' > /usr/bin/condapython3 \
    && chmod +x /usr/bin/condapython3

FROM base AS final

WORKDIR /app/

COPY ./rag_env.yaml /tmp
RUN . /opt/miniconda/bin/activate \
    && conda env update --name ragenv --file /tmp/rag_env.yaml \
    && conda clean --all \
    && rm -rf /root/.cache/*
COPY ./enstrag ./enstrag
COPY ./pyproject.toml ./pyproject.toml
RUN condapython3 -m pip install --default-timeout=100 . \
    && condapython3 -m spacy download en_core_web_sm
    #&& rm -rf ./enstrag

EXPOSE 8000

CMD ["condapython3", "-m", "enstrag", "-v"]