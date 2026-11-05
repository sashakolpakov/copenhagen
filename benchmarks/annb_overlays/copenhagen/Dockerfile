FROM ann-benchmarks

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopenblas-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

COPY ann_benchmarks/algorithms/copenhagen/vendor/copenhagen /opt/copenhagen

RUN pip install pybind11 numpy setuptools wheel
RUN cd /opt/copenhagen && python setup.py build_ext --inplace
ENV PYTHONPATH=/opt/copenhagen
RUN python -c "from python.core import CopenhagenIndex; print(CopenhagenIndex)"
