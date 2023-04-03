import os
import argparse
import json
import datetime
import time as t

from flask import Flask, request, g

from tasks import handle_new_run_request
from tasks import app as celery_app

from config.runs_config import load_runs_config, RUNS_CONFIG_DEFAULT

app = Flask( __name__ )

@app.route( '/request', methods = [ 'POST' ] )
def run_request():

    print(app.url_map)

    args = datetime.datetime.now()

    run_request = request.get_json()

    handle_new_run_request( run_request )

    return {
        'processed_request': args,
        'run_request': run_request,
    }

@app.route('/')
@app.route('/status')
def get_status():
    # return json.dumps( g.runs_config, indent = 3 )
    i = celery_app.control.inspect()

    return {
        'queued_tasks': i.reserved(),
        'registered_workers': i.registered(),
        'active_tasks': i.active(),
        'scheduled_tasks': i.scheduled(),
    }



def parse_args():

    parser = argparse.ArgumentParser(
        prog = 'Training server',
        description = 'Post a JSON RunMeta run descriptor to train a model from it',
    )

    parser.add_argument(
        '-p',
        '--port',
        default = '9090',
    )

    # run this server in develop mode (hot-reload, not optimized)
    parser.add_argument(
        '-d',
        '--develop',
        action = 'store_true',
    )

    # runs data directory and HDF information
    # - use default if none specified
    parser.add_argument(
        '-r',
        '--runs_config',
    )

    args = parser.parse_args()

    return args

def load_runs_config( args ):
    if ( args.runs_config ):
        # from file
        runs_config = load_runs_config( args.runs_config )
    else:
        # default
        runs_config = RUNS_CONFIG_DEFAULT

    g.runs_config = runs_config

@app.before_request
def initialize_app():

    # parse args, again
    # It's annoying that this has to be run at the start of every request...
    # and not just once at the start of the server
    args = parse_args()

    # load runs_config to g
    load_runs_config( args )

def start_server(
    port,
    develop,
):

    if ( develop ):
        # Serve development via Flask's run command
        print( "Running in Development mode" )
        app.run(
            host = '0.0.0.0',
            port = port,
            debug = True,
        )
    else:
        # Serve production via waitress
        print( "Running in Production mode" )
        from waitress import serve
        serve(
            app,
            host = "0.0.0.0",
            port = port,
        )

if __name__ == '__main__':

    args = parse_args()

    start_server(
        args.port,
        args.develop,
    )

