import json
import argparse

from tasks import handle_new_run_request
from config import grid_search

def queue_up(
    configs,
):
    for config in configs:
        handle_new_run_request(
            config
        )

def queue_up_from_config(
    config_filename,
):
    with open( config_filename ) as config_file:
        config = json.load( config_file )
    group_type = config.get('_run_group_type')
    if( group_type == 'grid' ):
        configs = grid_search.get_permutations( config )
    else:
        configs = [ config ]

    return configs

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Queue up a group of runs from a config file',
    )

    parser.add_argument(
        'config_filename',
    )

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    configs = queue_up_from_config(
        args.config_filename,
    )

    queue_up(
        configs,
    )
