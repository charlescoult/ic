import os
import argparse
import json
from multiprocessing import Queue
import pickle

from flask import Flask, g

from tasks import process_run_request

from run.runs_config import load_runs_config, RUNS_CONFIG_DEFAULT

app = Flask( __name__ )
app.config['APP_STARTED'] = True

# client = MongoClient( "mongo:27017" )

@app.route('/')
def run_request():
    print( g.runs_config )
    return json.dumps( g.runs_config, indent = 3 )

def parse_args():

    parser = argparse.ArgumentParser(
        prog = 'Training server',
        description = 'Post a JSON RunMeta run descriptor to train a model from it',
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

def save_queue():
    queue_fn = os.path.join( g.runs_config['runs_dir'], g.runs_config['queue_fn'] )
    print(g.queue)
    with open( queue_fn, 'wb' ) as f:
        print(g.queue)
        f.write( pickle.dumps( g.queue ) )

def load_queue():
    print(g.runs_config)
    queue_fn = os.path.join( g.runs_config['runs_dir'], g.runs_config['queue_fn'] )
    if ( os.path.isfile( queue_fn ) ):
        with open( queue_fn, 'rb' ) as f:
            print('opening')
            g.queue = pickle.loads( f.read() )
    else:
        g.queue = Queue()
        save_queue()

def load_runs_config( args ):
    if ( args.runs_config ):
        # from file
        runs_config = load_runs_config( args.runs_config )
    else:
        # default
        runs_config = RUNS_CONFIG_DEFAULT

    print( runs_config )
    g.runs_config = runs_config

@app.before_first_request
def initialize_app():

    print( 'Initialization' )

    # parse args, again
    args = parse_args()

    # load runs_config to g
    load_runs_config( args )

    # load queue to g
    load_queue()

def start_server(
    develop,
    port = 9090,
):
    if ( develop ):
        # Serve development via Flask's run command
        print( "Running in Development mode" )
        app.run(
            host = '0.0.0.0',
            port = os.environ.get(
                "FLASK_SERVER_PORT",
                port,
            ),
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
        args.develop,
    )
