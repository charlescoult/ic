import os
import json
import argparse
import requests

from tasks import handle_new_run_request
from config import grid_search

# Adds a request to the `tasks` Celery queue for each
# run configuration file in run_configs
def queue_up(
    run_configs,
    use_api = True,
):

    for run_config in run_configs:
        if use_api:
            response = requests.post(
                'http://localhost:9090/',
                data = run_config,
                headers = {
                    'Content-Type': 'application/json'
                },
            )
            print(response)
        else:
            handle_new_run_request(
                run_config
            )

# Converts a config file (single run config or a group of configs)
# to a list of configs to run
def get_run_configs(
    config
):
    group_type = config.get('_run_group_type')
    if( group_type == 'grid' ):
        run_configs = grid_search.get_permutations( config )
    else:
        run_configs = [ config ]

    return run_configs

# loads a config from file and runs it/them
def get_run_configs_from_file(
    config_filename,
):
    with open( config_filename ) as config_file:
        config = json.load( config_file )

    run_configs = get_run_configs( config )

    return run_configs

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Queue up a run or a group of runs from a config file',
        description = 'config_filename can be a file that describes a single run or a group run, as indicated by the \'_run_group_type\' property',
    )

    parser.add_argument(
        'config_filename',
    )

    parser.add_argument(
        '-l',
        '--local',
        help = 'Bypass the API endpoint and add directly to the celery RabbitMQ connection defined',
    )

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    configs = get_run_configs_from_file(
        args.config_filename,
    )

    print(configs)

    queue_up(
        configs,
    )

