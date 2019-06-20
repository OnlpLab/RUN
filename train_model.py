# -*- coding: utf-8 -*-


import numpy
import os
import run_model
import datetime
import argparse
from logger import create_logger


def main():
    parser = argparse.ArgumentParser(
        description='Trainning model ... '
    )
    #

    # The directory from which to find the data. default in data_processors is './data/'
    parser.add_argument(
        '-fd', '--FileData', required=False,
        help='Path of the dataset'
    )
    #
    parser.add_argument(
        '-d', '--DimModel', required=False,
        help='Dimension of LSTM model '
    )
    parser.add_argument(
        '-s', '--Seed', required=False,
        help='Seed of random state'
    )

    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', required=False,
        help='Max epoch number of training'
    )

    parser.add_argument(
        '-do', '--DropOut', required=False,
        help='Drop-out rate'
    )
    #
    parser.add_argument(
        '-m1', '--Map1', required=False,
        help='First Train Map'
    )
    parser.add_argument(
        '-m2', '--Map2', required=False,
        help='Second Train Map'
    )

    parser.add_argument(
        '-vl', '--Validation', required=False, action='store_true',
        help='validate trained model (True/False, default is false)'
    )

    parser.add_argument(
        '-sb', '--SizeBeam', required=False, choices=range(1, 20), default=4,
        help='Validation mode: Size of Beam (Integer, default is 4)'
    )

    parser.add_argument(
        '-lnf', '--LengthNormalizationFactor', required=False, default=0.5,
        help='Validation mode: Length Normalization Factor [0.5-0.7] (0.5 is the default)'
    )


    args = parser.parse_args()

    if args.MaxEpoch is None:
        args.MaxEpoch = numpy.int32(20)

    else:
        args.MaxEpoch = numpy.int32(args.MaxEpoch)

    if args.DimModel is None:
        args.DimModel = numpy.int32(100)
    else:
        args.DimModel = numpy.int32(args.DimModel)
    if args.Seed is None:
        args.Seed = numpy.int32(90001)
    else:
        args.Seed = numpy.int32(args.Seed)
    if args.DropOut is None:
        args.DropOut = numpy.float32(0.9)
    else:
        args.DropOut = numpy.float32(args.DropOut)
    #
    if args.Map1 is None:
        args.Map1 = 'map_2'
    else:
        args.Map1 = str(args.Map1)
    if args.Map2 is None:
        args.Map2 = 'map_3'
    else:
        args.Map2 = str(args.Map2)

    assert isinstance(args.LengthNormalizationFactor, float)
    assert 0.5 <= args.LengthNormalizationFactor <= 0.7

    assert isinstance(args.SizeBeam, int), "Size of Beam is not an int"


    id_process = os.getpid()
    time_current = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #
    tag_model = '_PID=' + str(id_process) + '_TIME=' + time_current

    #
    path_track = './tracks/track' + tag_model + '/'

    file_log = os.path.join(path_track + 'log.txt')

    if not os.path.exists(path_track):
        os.makedirs(path_track)

    args.path_save = os.path.abspath(path_track)

    logger = create_logger(file_log, 'trainer log')
    logger.info(args)

    run_model.train_model(args)


if __name__ == "__main__":
    main()
    print("END")
