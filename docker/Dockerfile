# Base: PyTorch, containing CUDA Toolkit, Transformer Engine
FROM nvcr.io/nvidia/pytorch:25.10-py3
SHELL ["/bin/bash", "-lc"]
USER root

ARG REPO=https://github.com/vuiseng9/fp4-training
ARG BRANCH=qtrain-revref

WORKDIR /workspace
RUN git clone $REPO && cd fp4-training && git checkout $BRANCH && git submodule update --init --recursive

WORKDIR /workspace/fp4-training/custom 
RUN make re

RUN TE_DIR="$(python -c 'import pathlib, transformer_engine as te; print(pathlib.Path(te.__file__).parent)')" \
 && echo "Transformer Engine at: $TE_DIR" \
 && cd "$TE_DIR" \
 && git apply /workspace/fp4-training/utils/fix_te_linear_v2.8.patch

WORKDIR /workspace/fp4-training/
RUN mkdir -p .vscode
RUN ln -sv $(realpath vscode/launch.json) .vscode/.

WORKDIR /workspace/fp4-training
ENV PYTHONPATH=/workspace/fp4-training
CMD ["/bin/bash"]
