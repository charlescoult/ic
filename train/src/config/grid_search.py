import itertools
import json

# Takes in a grid-search JSON configuration file
# and yeilds a list of each resulting run configuration
# - Not the most efficient or prettiest code but hey, it works.
def get_permutations( grid_config ):
    if not isinstance( grid_config, dict ):
        # single variable base-case
        return grid_config
    else:
        dict_permutations = [ {} ]
        for key, value in grid_config.items():
            if ( isinstance( value, list ) ):
                list_permutations = []
                for item in value:
                    for dict_item in dict_permutations:
                        list_permutations.append( {
                            **dict_item,
                            key: item,
                        } )
                dict_permutations = list_permutations
            else:
                # dict or item (key, value)
                value_permutations = get_permutations( value )
                if ( isinstance( value_permutations, list ) ):
                    list_permutations = []
                    for item in value_permutations:
                        for dict_item in dict_permutations:
                            list_permutations.append( {
                                **dict_item,
                                key: item,
                            } )
                    dict_permutations = list_permutations
                else:
                    for dict_item in dict_permutations:
                        dict_item[ key ] = get_permutations( value )

        return dict_permutations

def get_grid_configs(
    filename,
):
    with open( filename ) as file:
        config = json.load( file )

    return get_permutations( config )

example_config = {
    "batch_size": [
        16,
        32,
        64,
        128
    ],
    "max_epochs": 30,
    "model": {
        "base": [
            "Inception_v3_iNaturalist",
            "Xception"
        ],
        "classifier": {
            "dropout": [
                0.0,
                0.3,
                0.5
            ],
            "output_normalize": False
        },
        "learning_rate": [
            0.0001,
            0.0005,
            0.001
        ],
        "label_smoothing": 0.1
    },
    "dataset": {
        "data_augmentation": False,
        "downsample": None,
        "source": "flowers",
        "split_test": 0.05,
        "split_val": 0.2
    },
    "callbacks": {
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 8,
            "restore_best_weights": True,
            "start_from_epoch": 5
        }
    }
}

# Test example using above config
if __name__ == '__main__':

    configs = get_permutations( example_config )

    for config in configs:
        print( json.dumps( config, indent = 3 ) )


    print( len( configs ) )

