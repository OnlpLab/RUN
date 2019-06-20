# -*- coding: utf-8 -*-

import pickle
import numpy
import logging


class DataProcess(object):
    """
    This class process raw data into a format that the model knows how to use
    """

    def __init__(self, path_rawdata=None):

        if path_rawdata:
            self.path_rawdata = path_rawdata
        else:
            self.path_rawdata = './data/'

        with open(self.path_rawdata + 'dataset.pickle', 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')

        with open(self.path_rawdata + 'valselect.pickle', 'rb') as f:
            devset = pickle.load(f, encoding='latin1')

        with open(self.path_rawdata + 'dictionary.pickle', 'rb') as f:
            stats = pickle.load(f, encoding='latin1')
        with open(self.path_rawdata + 'maps.pickle', 'rb') as f:
            self.maps = pickle.load(f, encoding='latin1')

        self.lang2idx = stats['word2ind']
        self.dim_lang = stats['volsize']

        self.dim_world = 2 * 3 + 4 + 3 + 6
        self.dim_action = 6

        self.names_map = raw_data.keys()

        self.dict_data = {
            'all': {},
            'train': {},
            'dev': {}
        }
        self.dict_data_list = {'all':{}}

        for name_map in self.names_map:
            self.dict_data['all'][name_map] = []
            self.dict_data['train'][name_map] = []
            self.dict_data['dev'][name_map] = []
            self.dict_data_list['all'][name_map]=[]
            
            for idx_data, data in enumerate(raw_data[name_map]):
                if data['id'] not in self.dict_data_list['all'][name_map]:
                   self.dict_data_list['all'][name_map].append(data['id'])
                self.dict_data['all'][name_map].append(data)
                if idx_data in devset[name_map]:
                    self.dict_data['dev'][name_map].append(data)
                else:
                    self.dict_data['train'][name_map].append(data)

        self.seq_lang_numpy = None
        self.seq_world_numpy = None
        self.seq_action_numpy = None

    def get_pos(self, idx_data, name_map, tag_split):
        one_data = self.dict_data[tag_split][name_map][idx_data]
        path_one_data = one_data['route_no_face']
        return path_one_data[0], path_one_data[-1]

    def process_one_data(self, idx_data, name_map, tag_split):
        one_data = self.dict_data[tag_split][name_map][idx_data]
        data=Data(one_data)
        list_word_idx = [
            self.lang2idx[w.lower()] for w in data.instruction_after_entity_extraction if w.lower() in self.lang2idx
        ]

        self.seq_lang_numpy = numpy.array(
            list_word_idx, dtype=numpy.int32
        )
        self.seq_world_numpy = numpy.zeros(
            (len(data.route), self.dim_world)
        )

        for idx_world, world_current in enumerate(data.route): #metadata about the relative position of the instruction in the navigation story (e.g., if a first instruction then direction actions are most likely needed)
            bow_basic = numpy.zeros(4)
            max_line = numpy.amax([x['line_number'] for x in self.dict_data['all'][name_map] if x['id'] == data.id])

            ratio = data.line_number / max_line

            if ratio <= 0.25:
                bow_basic[0] = 1
            elif ratio <= 0.5:
                bow_basic[1] = 1
            elif ratio <= 0.75:
                bow_basic[2] = 1
            else:
                bow_basic[3] = 1

            self.seq_world_numpy[idx_world, :] = numpy.concatenate((bow_basic,
                                                                    data.list_world_state_turn[idx_world],
                                                                    data.list_world_state_walking[idx_world]))

        #
        self.seq_action_numpy = numpy.zeros(
            (len(data.actions),),
            dtype=numpy.int32
        )
        for idx_action, one_hot_vec_action in enumerate(data.actions):
            self.seq_action_numpy[idx_action] = numpy.argmax(
                one_hot_vec_action
            )

        return data

    def get_world_state_for_basic(self, tag_split, name_map, line_number):  #metadata about the relative position of the instruction in the navigation story (e.g., if a first instruction then direction actions are most likely needed)
        bow_basic = numpy.zeros(4)
        one_data = self.dict_data[tag_split][name_map][line_number]
        id_task = one_data['id']
        line_number = one_data['line_number']
        max_line = numpy.amax([x['line_number'] for x in self.dict_data['all'][name_map] if x['id'] == id_task])

        ratio = (line_number) / (max_line)

        if ratio <= 0.25:
            bow_basic[0] = 1
        elif ratio <= 0.5:
            bow_basic[1] = 1
        elif ratio <= 0.75:
            bow_basic[2] = 1
        else:
            bow_basic[3] = 1
        return bow_basic


class Data(object):
    def __init__(self, data):
        self.instruction = data['instruction']
        self.id = data['id']
        self.instruction_after_entity_extraction = data['list_of_apples']
        self.route = data['route_no_face']
        self.actions = data['target_no_face']
        self.line_number = data['line_number']
        self.entities_streets=data['map_count_var_streets']
        self.entities_not_streets=data['map_count_var']
        self.list_world_state_turn=data['world_state_turn']
        self.list_world_state_walking=data['world_state_walking']
