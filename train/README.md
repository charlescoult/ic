# Training Runs

Model training


## Installation

* Docker
* Docker Compose

### Note regarding (NVidia Container Toolkit Installation)[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html]
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
* When starting the `docker-compose.yml` configuration, be sure to set `DOCKER_BUILDKIT=0` to disable the new and expermental docker BuildKit (which doesn't want to use the default runtime, `nvidia` that we set above)
    ```bash
    DOCKER_BUILDKIT=0 docker compose up
    ```

