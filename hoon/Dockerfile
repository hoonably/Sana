FROM nvcr.io/nvidia/pytorch:24.06-py3

# Ensure conda is available in shell
ENV PATH=/opt/conda/bin:$PATH

# Install system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /Sana

# Install Miniconda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Create conda env with Python 3.10
RUN conda create -n sana python=3.10 -y

# Register conda.sh in bashrc so that 'conda activate' works in shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate sana" >> /root/.bashrc

# Install packages into the conda env
RUN /opt/conda/bin/conda run -n sana pip install --upgrade pip && \
    /opt/conda/bin/conda run -n sana pip install \
        torch torchvision torchaudio \
        pyrallis timm diffusers mmcv==2.0.0 opencv-python-headless==4.7.0.72 \
        xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Default shell opens with 'sana' env activated
CMD ["bash"]
