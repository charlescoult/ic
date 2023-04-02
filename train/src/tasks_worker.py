# from tasks import app
import os
from celery import Celery
import time
import argparse

from run.metadata import RunMeta
from config.runs_config import load_runs_config, RUNS_CONFIG_DEFAULT
from run.run import start_run

# Get RabbitMQ hostname
rabbitmq_host = os.getenv( 'RABBITMQ_HOST', 'localhost' )

# Set XLA_FLAGS environment variable so that nvvm/libdevice will be found...
conda_prefix = os.getenv( 'CONDA_PREFIX' )
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={conda_prefix}'

# Use the default runs_config if no other is specified
# - needs to be in global context for task(s) to utilize
runs_config = RUNS_CONFIG_DEFAULT

# Define the Celery application connection to broker
app = Celery(
    'tasks',
    broker = f'pyamqp://guest@{rabbitmq_host}//'
)

# For debugging environment variables
def print_environ():
    for key, value in os.environ.items():
        print(f"{key}={value}")

# starts a run request
def run_request( request ):
    run = RunMeta(
        request,
        **runs_config,
    )
    print( run )
    start_run( run )

@app.task
def run_request_task(
    request,
):
    print("Request Task in worker ")
    print_environ()
    print( request )

    run_request(
        request,
    )

    print("Completed")

# Add a worker to the Celery application
def add_worker():
    return app.Worker(
        include = ['tasks'], # only want one run running at a time due to GPU constrainsts
        # TODO start multiple depending on GPU and CPU memory usage
        concurrency = 1,
        pool = 'solo',
    )

# Parse CLI args
def parse_args(
):
    parser = argparse.ArgumentParser(
        prog = '',
    )
    parser.add_argument(
        '-t',
        '--test',
        action = 'store_true',
    )
    parser.add_argument(
        '-r',
        '--runs_config',
    )
    return parser.parse_args()

if __name__ == '__main__':
    print("Starting worker.")

    args = parse_args()

    if ( args.runs_config ):
        print( 'Loading runs_config from %s' % args.runs_config )
        runs_config = load_runs_config( args.runs_config )
    else:
        print( 'Using default runs_config' )

    if( args.test ):
        print( 'Using testing runs_dir' )
        runs_config['runs_dir'] = '/media/data/test_runs'

    print( runs_config )

    add_worker().start()
