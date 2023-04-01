# Training Runs

Model training



## Note regarding (NVidia Container Toolkit Installation)[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html]
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

