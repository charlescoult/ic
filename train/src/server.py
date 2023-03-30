import os
import argparse

from flask import Flask
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
    return "Hello World"

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

if __name__ == '__main__':

    args = parse_args()

    # load runs_config
    if ( args.runs_config ):
        # from file
        runs_config = load_runs_config( args.runs_config )
    else:
        # default
        runs_config = RUNS_CONFIG_DEFAULT

    if ( args.develop ):
        # Serve development via Flask's run command
        print( "Running in Development mode" )
        app.run(
            host = '0.0.0.0',
            port = os.environ.get(
                "FLASK_SERVER_PORT",
                9090,
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
            port = 9090,
        )
