import argparse
import json
import os

from run import start_run
from metadata import RunMeta
from runs_config import load_runs_config, RUNS_CONFIG_DEFAULT

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Training Run',
        description = 'Input JSON run config file.',
    )
    parser.add_argument(
        'config_file',
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
    args = parser.parse_args()

    return args

def make_dir_if_not_exist( runs_dir ):
    if ( os.path.exists( runs_dir ) ):
        if ( not os.path.isdir( runs_dir ) ):
            raise Exception( "File exists at runs_dir" )
    else:
        os.makedirs( runs_dir )

if __name__ == '__main__':

    args = parse_args()

    # Defaults for runs_config HDF file
    runs_config = RUNS_CONFIG_DEFAULT

    if ( args.runs_config ):
        print( 'Loading runs_config from %s' % args.runs_config )
        runs_config = load_runs_config( args.runs_config )
    elif ( args.test ):
        print( 'Using testing runs_config' )
        runs_config['runs_dir'] = '/media/data/test_runs'
    else:
        print( 'Using default runs_config' )

    # load the specified run config file
    with open( args.config_file ) as f:
        run = json.load( f )

    make_dir_if_not_exist( runs_config['runs_dir'] )

    run = RunMeta(
        run,
        runs_dir = runs_config['runs_dir'],
        runs_hdf = runs_config['runs_hdf'],
        runs_hdf_key = runs_config['runs_hdf_key'],
    )

    print(run)

    start_run( run )

