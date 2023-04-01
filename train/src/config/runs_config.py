import argparse
import json

RUNS_CONFIG_DEFAULT = {
    'runs_dir': '/media/data/runs',
    'runs_hdf': 'runs.h5',
    'runs_hdf_key': 'runs',
}

RUNS_CONFIG_TEST = {
    **RUNS_CONFIG_DEFAULT,
    'runs_dir': '/media/data/test_runs',
}

def load_runs_config(
    filepath,
):
    with open( filepath, 'r' ) as file:
        config = json.load( file )

    return config

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Test for loading runs_config file',
    )
    parser.add_argument(
        'runs_config',
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    print( RUNS_CONFIG_TEST )

    print( load_runs_config( args.runs_config ) )


