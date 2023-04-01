# FROM continuumio/miniconda3:latest
FROM tensorflow/serving:latest:gpu

WORKDIR /app

COPY environment.yml .

RUN conda install -c conda-forge mamba
RUN mamba env create --file environment.yml

CMD ["conda", "env", "list" ]

