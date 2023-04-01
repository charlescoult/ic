

from tasks import app

def add_worker():
    worker = app.Worker(
        include = ['tasks.run_request_task'],
        # only want one run running at a time due to GPU constrainsts
        # TODO start multiple depending on GPU and CPU memory usage
        concurrency = 1,
    )
    worker.start()

if __name__ == '__main__':
    add_worker()
