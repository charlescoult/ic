# Training Runs

Application stack for model training queue and execution
Once started, this application stack allows for asynchronous task requests to be queued through a POST request to the endpoint `localhost:9090/`

## Application Flow
* `POST` request is received at the base `localhost:9090` or `localhost:9090/request` enpoints with JSON run configuration payload
* Request is passed to RabbitMQ `tasks` queue to await distribution to an available worker by RabbitMQ broker
* Worker receives request from RabbitMQ broker and processes them one at a time

## Installation

### Pure Host
* NVIDIA display driver
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
* [CUDNN](https://developer.nvidia.com/cudnn)
* `Anaconda`, `Miniconda` or `Mamba` package manager

### Individual Docker Containers
* NVIDIA display driver
* [NVIDIA Container Toolkit (nvidia-docker)](https://github.com/NVIDIA/nvidia-docker)
* Docker

### Docker Compose
* NVIDIA display driver
* [NVIDIA Container Toolkit (nvidia-docker)](https://github.com/NVIDIA/nvidia-docker)
* Docker Compose

### Note regarding [NVidia Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
In order for the `ic-train-worker` docker container to be built, the `nvidia-container-runtime` must be used.

* Ensure `nvidia-container-toolkit` is installed
* Ensure `nvidia-container-runtime` is installed as well
* Ensure the `nvidia` runtime is available and set to be used as default (`/etc/docker/daemon.json`):
    ```json
    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "args": [],
                "path": "nvidia-container-runtime"
            }
        }
    }
    ```
* When starting the `docker-compose.yml` configuration, be sure to set `DOCKER_BUILDKIT=0` to disable the new and expermental Docker BuildKit (which doesn't want to use the default runtime, `nvidia` that we set above for some reason - [see issue here](https://github.com/docker/compose/issues/9681))
    ```bash
    DOCKER_BUILDKIT=0 docker compose up
    ```

## Running

### Pure Host
* Install and activate environment
    * `conda install mamba` - mamba runs much faster than conda
    * `mamba env create -f environment.ic-train.yml`
        * `environment.ic` - contains all necessary packages to run all three services
    * `. activate ic-train` or `conda activate ic-train`
* Server
    * `python server.py`
        * `-d` - hot-reload 'development' mode
* RabbitMQ
    * `rabbitmq-server`
* Worker
    * `python tasks_worker.py`

### Individual Docker Containers
* Build
    * Server
        * `docker build --rm -t ic-train-server -f Docker.ic-train-server`
    * RabbitMQ
        * `docker build --rm -t rabbitmq -f Dockerfile.rabbitmq`
    * Worker
        * `docker build --rm -t ic-train-worker -f Docker.ic-train-worker`
            * May need `DOCKER_BUILDKIT=0` if `docker build` is configured to run using the new Docker Buildkit
* Run
    * Server
        * `docker run -p 9090:9090 ic-train-server`
    * RabbitMQ
        * `docker run -p 5672:5672 rabbitmq`
    * Worker
        * `docker run --gpus all ic-train-worker`

### Docker Compose
* `DOCKER_BUILDKIT=0 docker compose up`
    * `--build` - force rebuild, if necessary

