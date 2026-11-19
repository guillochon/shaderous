# Training image for the burning surrogate (tools/burn), for GPU pods on Runpod. Built from the
# repository's root and pushed to ghcr.io/guillochon/shaderous-burn by
# .github/workflows/burn-image.yml on every push to main that touches tools/burn or the
# equation-of-state table it heats the gas with.
#
# Slim on purpose: a pod waits for the whole image to download before it starts, and hosts
# that had never pulled Runpod's own PyTorch base (10.6 GB compressed, with the CUDA toolkit,
# Jupyter and more) took over 10 minutes. Training needs only Python and the CUDA 12.8 torch
# wheel, which carries its own CUDA libraries; the host needs a driver for CUDA 12.8 or later
# (gpu.minCudaVersion "12.8" on the pod).
#
# Everything a training run needs except its data is baked in: torch, pynucastro, Numba and
# SciPy for the reference burner, the code, the generated network (aprox13.py, from
# build_net.py), the shipped surrogate (a teacher for distillation), and the equation of
# state's table with its Python reader (tools/eos/helm.py and the lookup it shares,
# tools/eos/table.py), kept at the repository's own
# relative paths so that the code finds them as it does in a checkout.
#
# Data goes in /opt/shaderous/tools/burn/data: sent in (runpodctl), a network volume mounted
# there, or generated in the pod (gen_data.py). Models are written beside it, on the container
# disk. There is no SSH or
# Jupyter: commands go in through Runpod's exec (exec-pod), and the container just stays up.
FROM python:3.12-slim

# torch and its CUDA libraries in several layers, not one: a host downloads up to three
# layers at once, and a single 4 GB layer came down in one stream for over 10 minutes. pip
# resolves the exact set first; each group is then installed in a layer of its own.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --dry-run --quiet --report /tmp/torch.json torch==2.8.0 --index-url $TORCH_INDEX \
 && python -c "import json; p=[i['metadata'] for i in json.load(open('/tmp/torch.json'))['install']]; \
g=lambda f: ' '.join(f\"{m['name']}=={m['version']}\" for m in p if f(m['name'].lower())); \
open('/tmp/g1','w').write(g(lambda n: n.startswith('nvidia-cudnn'))); \
open('/tmp/g2','w').write(g(lambda n: n.startswith(('nvidia-cublas', 'nvidia-cusparse')))); \
open('/tmp/g3','w').write(g(lambda n: n.startswith('nvidia-') and not n.startswith(('nvidia-cudnn', 'nvidia-cublas', 'nvidia-cusparse')))); \
open('/tmp/g4','w').write(g(lambda n: not n.startswith('nvidia-') and n != 'torch'))"
RUN pip install --no-cache-dir --no-deps $(cat /tmp/g1) --index-url $TORCH_INDEX
RUN pip install --no-cache-dir --no-deps $(cat /tmp/g2) --index-url $TORCH_INDEX
RUN pip install --no-cache-dir --no-deps $(cat /tmp/g3) --index-url $TORCH_INDEX
RUN pip install --no-cache-dir --no-deps $(cat /tmp/g4) --index-url $TORCH_INDEX
RUN pip install --no-cache-dir torch==2.8.0 --index-url $TORCH_INDEX \
 && pip install --no-cache-dir pynucastro==3.2.0 numba scipy

WORKDIR /opt/shaderous/tools/burn
COPY src/eos/helmholtz_table.bin src/eos/helmholtz_meta.ts /opt/shaderous/src/eos/
COPY tools/eos/helm.py tools/eos/table.py /opt/shaderous/tools/eos/
COPY tools/burn/*.py tools/burn/aprox13_surrogate.pt ./
# The network, then one import of the burner, which reads it.
RUN python build_net.py > /dev/null && python -c "import burner, train" && mkdir -p data

ENV PYTHONPATH=/opt/shaderous/tools/burn PYTHONIOENCODING=utf-8
CMD ["sleep", "infinity"]
