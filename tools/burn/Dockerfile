# Training image for the burning surrogate (tools/burn), for GPU pods on Runpod. Built and
# pushed to ghcr.io/guillochon/shaderous-burn by .github/workflows/burn-image.yml on every
# push to main that touches tools/burn.
#
# Everything a training run needs except its data is baked in: PyTorch on Runpod's own base
# (whose layers its hosts are likely to have cached already, and whose /start.sh keeps the pod
# up), pynucastro, Numba and SciPy for the reference burner, the code, the generated network
# (aprox13.py, from build_net.py) and the shipped surrogate (a teacher for distillation).
#
# Data goes in /opt/burn/data: a network volume mounted there, or generated in the pod
# (gen_data.py). Models are written to /opt/burn, on the container disk.
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

RUN pip install --no-cache-dir pynucastro==3.2.0 numba scipy

WORKDIR /opt/burn
COPY *.py aprox13_surrogate.pt ./
# The network, then one import of the burner, which reads it.
RUN python build_net.py > /dev/null && python -c "import burner, train" && mkdir -p data

ENV PYTHONPATH=/opt/burn PYTHONIOENCODING=utf-8
