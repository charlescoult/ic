import os
import argparse
import json

from flask import Flask, g
from pymongo import MongoClient

from run.runs_config import load_runs_config, RUNS_CONFIG_DEFAULT

app = Flask( __name__ )

# client = MongoClient( "mongo:27017" )

@app.route('/')
def todo():
    '''
    try:
        client.admin.command( 'ismaster' )
    except:
        return "MongoDB server not available"
    '''
    print( g.runs_config )
    return json.dumps( g.runs_config, indent = 3 )

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Training server',
        description = 'Post a JSON RunMeta run descriptor to train a model from it',
    )

    parser.add_argument(
        '-d',
        '--develop',
        action = 'store_true',
    )
    parser.add_argument(
        '-r',
        '--runs_config',
    )

    args = parser.parse_args()

    return args

@app.before_first_request
def initialize_app(
):
    print( 'Before first request' )
    args = parse_args()

    # load runs_config
    if ( args.runs_config ):
        # from file
        runs_config = load_runs_config( args.runs_config )
    else:
        # default
        runs_config = RUNS_CONFIG_DEFAULT

    g.runs_config = runs_config

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
