import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create Wandb config file', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-P',
        '--project', 
        type=str, 
        required=True,
        help='Project')
    parser.add_argument(
        '-E',
        '--entity', 
        type=str, 
        required=True,
        help='Entity')
    parser.add_argument(
        '--path', 
        type=str, 
        default='..',
        help='Path to config file')
    args = parser.parse_args()

    data = {
        "project": args.project,
        "entity": args.entity
    }
    with open(args.path, 'w+') as outfile:
        json.dump(data, outfile)