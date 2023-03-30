FROM tensorflow/tensorflow:latest-gpu

# Install Conda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda clean --all && \
    ln -s /usr/local/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /usr/local/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set up environment
ENV PATH /opt/conda/bin:$PATH

# Create conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate conda environment
SHELL ["conda", "run", "-n", "ic", "/bin/bash", "-c"]

# Set up working directory
WORKDIR /app

# Copy source code
COPY . .

# Start the application
CMD ["python", "server.py"]

