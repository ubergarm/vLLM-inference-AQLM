FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv

CMD /bin/bash
