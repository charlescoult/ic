from tasks import app

def add_worker():
    return app.Worker(
        include = ['tasks'],
        # only want one run running at a time due to GPU constrainsts
        # TODO start multiple depending on GPU and CPU memory usage
        concurrency = 1,
    )

if __name__ == '__main__':
    print("Starting worker.")
    add_worker().start()
