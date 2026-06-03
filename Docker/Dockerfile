FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/IBCDFO:/workspace/IBCDFO/pounders/py

WORKDIR /workspace/IBCDFO

# PyROL/Trilinos is a native extension. On Linux/Python 3.11, pip should use
# the manylinux wheel; the build tools are kept here as a fallback and for
# scientific packages that may need native compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gfortran \
        git \
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel numpy

# Install PyROL first and require a binary wheel. If this fails, the selected
# Python/platform combination does not have a usable PyROL wheel.
RUN python -m pip install --only-binary=:all: pyroltrilinos==0.5.3 \
    && python -c "import ROL; from ROL.numpy_vector import NumpyVector; print('ROL import check passed')"

RUN python -m pip install \
        ipykernel \
        jupyterlab \
        matplotlib \
        pandas \
        pygsti \
        scipy \
        seaborn \
        tqdm

COPY . /workspace/IBCDFO

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
