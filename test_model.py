# -*- coding: utf-8 -*-

import os
import run_model
import datetime
import argparse
from logger import create_logger


def main():
    parser = argparse.ArgumentParser(
        description='Testing model ... '
    )

    id_process = os.getpid()
    time_current = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tag_model = '_PID=' + str(id_process) + '_TIME=' + time_current
    path_track = os.path.abspath('./testing/track' + tag_model + '/')

    if not os.path.exists(path_track):
        os.makedirs(path_track)

    file_log = os.path.join(path_track, 'log.txt')
    file_results = os.path.join(path_track, 'results.txt')

    #
    # The directory from which to find the data. default in data_processers is './data/'
    parser.add_argument(
        '-fd', '--FileData', required=False,
        help='Path of the data-set'
    )
    #
    parser.add_argument(
        '-fp', '--FilePretrain', required=True,  # TODO change to True
        help='File of pretrained model'
    )
    parser.add_argument(
        '-mt', '--MapTest', required=True, choices=['map_1', 'map_2', 'map_3'], type=str,
        help='Test Map'
    )

    parser.add_argument(
        '-sr', '--SaveResults', required=False, action='store_true',
        help='Save Results (True/False, default is false)'
    )

    parser.add_argument(
        '-sb', '--SizeBeam', required=False, choices=range(1, 20), default=4, type=int,
        help='Size of Beam (Integer, default is 4)'
    )

    parser.add_argument(
        '-lnf', '--LengthNormalizationFactor', required=False, default=0.5,
        help='Length Normalization Factor [0.5-0.7] (0.5 is the default)'
    )

    args = parser.parse_args()
    args.id_process = id_process
    args.time_current = time_current

    assert (args.FilePretrain is not None)
    args.PathPretrain = os.path.abspath(args.FilePretrain)

    args.MapTest = str(args.MapTest)

    assert isinstance(args.LengthNormalizationFactor, float)
    assert 0.5 <= args.LengthNormalizationFactor <= 0.7  # according to 'Google's neural machine translation system: Bridging the gap between human and machine translation'

    assert isinstance(args.SizeBeam, int), "Size of Beam is not an int"



    args.SaveResultsPath = file_results
    create_logger(file_results, 'testing results log')
    logger = create_logger(file_log, 'testing log')
    logger.info(args)

    run_model.test_model(args)


if __name__ == "__main__": main()
