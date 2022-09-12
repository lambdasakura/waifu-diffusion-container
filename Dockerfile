FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
VOLUME ["/root/.cache"]

RUN apt-get update && apt-get install -y wget git git-lfs libglib2.0-0 libsm6 libxrender1 libxext-dev 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH

 RUN git clone https://github.com/DamascusGit/stable-diffusion && \
    cd stable-diffusion && \
    conda init bash && \
    conda env create -f environment.yaml && \
    echo "conda activate ldm" >> ~/.bashrc
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]
RUN pip install diffusers==0.3.0 transformers scipy ftfy
COPY ./patch/safety_checker.py /opt/miniconda3/envs/ldm/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py
WORKDIR /usr/src
# ENTRYPOINT ["/bin/bash"]
CMD python main.py
# for JupyterLab
# EXPOSE 8888
# RUN conda install jupyterlab
# SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]
# RUN ipython kernel install --user --name ldm
# RUN mkdir -p /usr/src/
# WORKDIR /usr/src/
# CMD ["jupyter-lab","--allow-root", "--ip=0.0.0.0","--port=8888","--no-browser","--notebook-dir=/usr/src/"]
