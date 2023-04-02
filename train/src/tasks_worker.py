# from tasks import app
import os
from celery import Celery
import time
import argparse

from run.metadata import RunMeta
from config.runs_config import load_runs_config, RUNS_CONFIG_DEFAULT
from run.run import start_run

rabbitmq_host = os.getenv( 'RABBITMQ_HOST', 'localhost' )

runs_config = RUNS_CONFIG_DEFAULT

app = Celery(
    'tasks',
    broker = f'pyamqp://guest@{rabbitmq_host}//'
)

def run_request( request ):
    run = RunMeta(
        request,
        **runs_config,
    )
    print( run )

    time.sleep(5)
    start_run( run )

@app.task
def run_request_task(
    request,
):
    print("Request Task in worker ")
    print( request )

    run_request(
        request,
    )

    print("Completed")

def add_worker():
    return app.Worker(
        include = ['tasks'], # only want one run running at a time due to GPU constrainsts
        # TODO start multiple depending on GPU and CPU memory usage
        concurrency = 1,
        pool = 'solo',
    )

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
