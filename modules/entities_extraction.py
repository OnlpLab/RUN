import csv
import difflib
import difflib
import itertools
import math
import mkl
import nltk
import numpy
import numpy as np
import os
import pickle
import random
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from execution_system import ExecutionSystem
from execution_system import MovementType, Turn

BLOCK_AROUND_AGENT_X = 58
BLOCK_AROUND_AGENT_Y = 38
THERSHOLD_SCORE = 90

'''

- fuzzy-search
- this pre-processes the instructions  - just to make it faster.

'''


def get_data():
    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    full_path_map = parent_directory + "/data/maps.pickle"
    map = pickle.load(open(full_path_map, "rb"))
    full_path_data_set = parent_directory + "/data/dataset.pickle"
    data_set = pickle.load(open(full_path_data_set, "rb"))
    return map, data_set


def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        # numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring


def extract_entities(map_current, instruction_to_extract):
    street_dic_words = ['street', 'st.', 'st', 'avn', 'avn.', 'avenue', 'west', 'north', 'south', 'east']
    map_streets_variables = {}
    x = instruction_to_extract['route'][0].x
    y = instruction_to_extract['route'][0].y
    words = instruction_to_extract['list_of_words']
    words_con = ' '.join(words)
    query = text2int(words_con)

    grids = [grid for grid in map_current['grids'] if
             (x - grid['x'] <= BLOCK_AROUND_AGENT_X or grid['x'] - x <= BLOCK_AROUND_AGENT_X) and
             (y - grid['y'] <= BLOCK_AROUND_AGENT_Y or grid['y'] - y <= BLOCK_AROUND_AGENT_Y)]

    counter_street = 0
    streets_names = list(set(
        [' '.join(nltk.word_tokenize(street_name['street_name'].lower())) for grid in grids for grid_street in
         map_current['grids_to_streets'] for street_name in grid_street['streets'] if
         grid_street['x'] == grid['x'] and grid_street['y'] == grid['y']]))

    street_map_sim = {}
    street_map_unique = []
    for street_name in streets_names:
        if street_name not in street_map_unique:
            street_map_unique.append(street_name)
        split_names = street_name.split()

        lists_names_without_endings_list = list(itertools.permutations(split_names))
        for list_current in lists_names_without_endings_list:
            string_remove_con = ' '.join(list_current)
            if string_remove_con not in street_map_sim:
                street_map_sim[string_remove_con] = street_name
            if string_remove_con not in street_map_unique:
                street_map_unique.append(string_remove_con)
        for list_current in lists_names_without_endings_list:
            list_current = list(list_current)
            for word_index, word in enumerate(list_current):

                if word in street_dic_words:
                    del list_current[word_index]

                string_remove_con = ' '.join(list_current)
                if string_remove_con not in street_map_sim:
                    street_map_sim[string_remove_con] = street_name
                if string_remove_con not in street_map_unique:
                    street_map_unique.append(string_remove_con)

    score = 100

    while score >= THERSHOLD_SCORE:
        found_list = process.extract(query, street_map_unique, scorer=fuzz.token_set_ratio, limit=10)
        found_list = sorted(found_list, key=lambda x: (x[1], len(x[0])), reverse=True)
        found_name, score = found_list[0]
        matcher = difflib.SequenceMatcher(a=found_name, b=query)
        match = matcher.find_longest_match(0, len(matcher.a), 0, len(matcher.b))
        start_index, end_index = match.b, match.b + match.size
        original_name = street_map_sim[found_name]
        if original_name in map_streets_variables:
            new_name = map_streets_variables[original_name]

        else:
            new_name = counter_street
            map_streets_variables[original_name] = new_name
            counter_street += 1

        query = query[0:start_index] + ' YYYY' + str(new_name) + ' ' + query[end_index:len(query)]

    # entities not streets
    map_variables = {}
    counter_entity = 0
    tags_names = list(set(
        [' '.join(nltk.word_tokenize(tag['name'].lower())) for grid in grids for tag in grid['tags'] if
         tag['name'] != '']))

    score = 100
    while score >= THERSHOLD_SCORE:

        found_list_first = process.extract(query, tags_names, scorer=fuzz.partial_ratio, limit=3)
        found_list_scores = [(n, s) for n, s in found_list_first if s >= 82]

        found_list = sorted(found_list_scores, key=lambda x: (len(x[0]), x[1]), reverse=True)
        if len(found_list) <= 0:
            break
        found_name, score = found_list[0]
        matcher = difflib.SequenceMatcher(a=found_name, b=query)
        match = matcher.find_longest_match(0, len(matcher.a), 0, len(matcher.b))
        start_index, end_index = match.b, match.b + match.size

        if found_name in map_variables:
            new_name = map_variables[found_name]

        else:
            new_name = counter_entity
            map_variables[found_name] = new_name
            counter_entity += 1

        query = query[0:start_index] + ' XXXX' + str(new_name) + ' ' + query[end_index:len(query)]

    instruction_to_extract['list_of_words_with_entity_extraction'] = nltk.word_tokenize(query)
    instruction_to_extract['map_variable_to_entity_streets'] = map_streets_variables
    instruction_to_extract['map_variable_to_entity_not_streets'] = map_variables
    return instruction_to_extract


if __name__ == "__main__":
    print ("start extracting entities")
    maps, data_set = get_data()
    for map_name, map_data in data_set.items():

        for index_data, data in enumerate(map_data):
            current_map = maps[map_name]
            new_instruction = extract_entities(current_map, data)
