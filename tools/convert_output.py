import numpy as np
import argparse
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file')
    parser.add_argument('-o', '--output-file', default=None)

    args = parser.parse_args()

    assert args.input_file.endswith('.json') or args.input_file.endswith('.pkl')

    if args.output_file is None:
        args.output_file = (
            args.input_file.replace('.json', '.pkl')
            if args.input_file.endswith('.json') else
            args.input_file.replace('.pkl', '.json')
        )

    return args


def pkl_to_json(pkl_data):
    output = [
        (bbox.tolist(),
         [dict(size=rle['size'], counts=rle['counts'].decode('utf-8'))
          for rle in masks
         ])
        for (_, bbox), (_, masks) in pkl_data
    ]
    return output


def json_to_pkl(json_data):
    output = [
        (np.array(bbox) if bbox else np.zeros((0,5), dtype=np.float32),
         [dict(size=rle['size'], counts=rle['counts'].encode('utf-8'))
          for rle in masks
         ])
        for bbox, masks in json_data
    ]
    output = [
        ((np.zeros((0,5), dtype=np.float32), bbox),
         ([], masks))
        for bbox, masks in output
    ]
    return output


if __name__ == '__main__':
    args = parse_args()

    if args.input_file.endswith('.json'):
        json_input = json.load(open(args.input_file, 'r'))
        pkl_output = json_to_pkl(json_input)
        pickle.dump(pkl_output, open(args.output_file, 'wb'))

    elif args.input_file.endswith('.pkl'):
        pkl_input = pickle.load(open(args.input_file, 'rb'))
        json_output = pkl_to_json(pkl_input)
        json.dump(json_output, open(args.output_file, 'w'))
