FROM condaforge/mambaforge-pypy3

COPY . .
RUN ls -la
RUN mamba env create --name ic -f requirements.yml && \
conda clean -afy
