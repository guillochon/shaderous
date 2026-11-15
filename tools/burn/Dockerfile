# Training image for the burning surrogate (tools/burn), for GPU pods on Runpod. Built and
# pushed to ghcr.io/guillochon/shaderous-burn by .github/workflows/burn-image.yml on every
# push to main that touches tools/burn.
#
# Slim on purpose: a pod waits for the whole image to download before it starts, and hosts
# that had never pulled Runpod's own PyTorch base (10.6 GB compressed, with the CUDA toolkit,
# Jupyter and more) took over 10 minutes. Training needs only Python and the CUDA 12.8 torch
# wheel, which carries its own CUDA libraries; the host needs a driver for CUDA 12.8 or later
# (gpu.minCudaVersion "12.8" on the pod).
#
# Everything a training run needs except its data is baked in: torch, pynucastro, Numba and
# SciPy for the reference burner, the code, the generated network (aprox13.py, from
# build_net.py) and the shipped surrogate (a teacher for distillation).
#
# Data goes in /opt/burn/data: a network volume mounted there, or generated in the pod
# (gen_data.py). Models are written to /opt/burn, on the container disk. There is no SSH or
# Jupyter: commands go in through Runpod's exec (exec-pod), and the container just stays up.
FROM python:3.12-slim

RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir pynucastro==3.2.0 numba scipy

WORKDIR /opt/burn
COPY *.py aprox13_surrogate.pt ./
# The network, then one import of the burner, which reads it.
RUN python build_net.py > /dev/null && python -c "import burner, train" && mkdir -p data

ENV PYTHONPATH=/opt/burn PYTHONIOENCODING=utf-8
CMD ["sleep", "infinity"]
