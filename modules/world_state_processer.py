import numpy as np
import math
import nltk
import difflib

MAX_GRID_DISTANCE = 85

"""
WorldStateProcesser constructs the world-state as a concatenation of current position (get_world_state_for_walking) and forward position (get_world_state_for_turn). 
Each position is represented as a BOW. 
"""


class WorldStateProcesser(object):
    def __init__(self, map_current):
        """
        init exe world-state processor
        :param map_current: chosen OSS map
        """

        self.map = map_current

    def get_world_state_for_walking(self, current_position, map_count_var, map_count_var_streets):

        bow_tags = np.zeros(3)
        bow_tags_here = np.zeros(3)
        if current_position.side == -1:
            return np.concatenate((bow_tags_here, bow_tags))
        assert (current_position.side == 1 or current_position.side == 2)

        if not (len(map_count_var) <= 0 or map_count_var == None or bool(map_count_var) == False or not map_count_var):

            street_to_grid = [element for element in self.map['streets'][current_position.street]['grids'] if
                              element['x'] == current_position.x and element['y'] == current_position.y]

            tags_here = [grid['tags'] for grid in self.map['grids'] if
                         current_position.x - 5 <= grid['x'] <= current_position.x + 5
                         and current_position.y - 5 <= grid['y'] <= current_position.y + 5]

            for tags in tags_here:
                for tag in tags:
                    street_name = nltk.word_tokenize(tag['name'])
                    tag_name_toke_con = ' '.join(street_name)
                    if tag_name_toke_con is None or tag_name_toke_con == '':
                        continue
                    if tag_name_toke_con in map_count_var.keys():
                        number_var = map_count_var[tag_name_toke_con]
                        while len(bow_tags_here) <= number_var:
                            number_var -= 1
                        bow_tags_here[number_var] = 1
                    else:

                        for key in map_count_var.keys():

                            seq = difflib.SequenceMatcher(None, tag_name_toke_con.lower(), key.lower())
                            d = seq.ratio() * 100
                            if (
                                    d > 77 or key.lower() in tag_name_toke_con.lower() or tag_name_toke_con.lower() in key.lower()):
                                number_var = map_count_var[key]

                                bow_tags_here[number_var] = 1

            grid_order = street_to_grid[0]['grid_order']

            grids_ahead = []
            if current_position.side == 1:
                grids_x_y = [(element['x'], element['y']) for element in
                             self.map['streets'][current_position.street]['grids'] if
                             element['grid_order'] <= grid_order and math.fabs(
                                 element['grid_order'] - grid_order) < MAX_GRID_DISTANCE]

                for grid in self.map['grids']:
                    for grid_in_street_only_x_y in grids_x_y:
                        x_check = grid_in_street_only_x_y[0]
                        y_check = grid_in_street_only_x_y[1]
                        if x_check - 5 <= grid['x'] <= x_check + 5 and grid['y'] >= y_check - 5 and grid[
                            'y'] <= y_check + 5:
                            grids_ahead.append(grid['id'])


            else:

                grids_x_y = [(element['x'], element['y']) for element in
                             self.map['streets'][current_position.street]['grids'] if
                             element['grid_order'] >= grid_order and math.fabs(
                                 element['grid_order'] - grid_order) < MAX_GRID_DISTANCE]

                for grid in self.map['grids']:
                    for grid_in_street_only_x_y in grids_x_y:
                        x_check = grid_in_street_only_x_y[0]
                        y_check = grid_in_street_only_x_y[1]
                        if x_check - 5 <= grid['x'] <= x_check + 5 and grid['y'] >= y_check - 5 and grid[
                            'y'] <= y_check + 5:
                            grids_ahead.append(grid['id'])

            for grid_idx, grid_id in enumerate(grids_ahead):
                for tag in self.map['grids'][grid_id]['tags']:
                    street_name = nltk.word_tokenize(tag['name'])
                    tag_name_toke_con = ' '.join(street_name)
                    if tag_name_toke_con is None or tag_name_toke_con == '':
                        continue
                    if tag_name_toke_con in map_count_var.keys():
                        number_var = map_count_var[tag_name_toke_con]
                        while len(bow_tags) <= number_var:
                            number_var -= 1
                        if bow_tags[number_var] != 1:
                            bow_tags[number_var] = 1
                    else:

                        for key in map_count_var.keys():

                            seq = difflib.SequenceMatcher(None, tag_name_toke_con.lower(), key.lower())
                            d = seq.ratio() * 100
                            if (
                                    d > 77 or key.lower() in tag_name_toke_con.lower() or tag_name_toke_con.lower() in key.lower()):
                                number_var = map_count_var[key]
                                if bow_tags[number_var] != 1:
                                    bow_tags[number_var] = 1
        bow_streets = np.zeros(3)
        bow_streets_here = np.zeros(3)
        if not (len(map_count_var_streets) <= 0 or map_count_var_streets == None or bool(
                map_count_var_streets) == False or not map_count_var_streets):

            street_to_grid = [element for element in self.map['streets'][current_position.street]['grids'] if
                              element['x'] == current_position.x and element['y'] == current_position.y]

            streets_here = [grid['streets'] for grid in self.map['grids_to_streets'] if
                            grid['x'] == current_position.x and grid['y'] == current_position.y]

            for streets in streets_here:
                for street in streets:
                    street_name = street['street_name']

                    if street_name is None or street_name == '':
                        continue
                    if street_name in map_count_var_streets.keys():
                        number_var = map_count_var_streets[street_name]
                        while len(bow_streets_here) <= number_var:
                            number_var -= 1
                        bow_streets_here[number_var] = 1

            grid_order = street_to_grid[0]['grid_order']

            grids_ahead = []
            if current_position.side == 1:
                grids_x_y = [(element['x'], element['y']) for element in
                             self.map['streets'][current_position.street]['grids'] if
                             element['grid_order'] <= grid_order and math.fabs(
                                 element['grid_order'] - grid_order) < MAX_GRID_DISTANCE]

                for grid in self.map['grids_to_streets']:
                    for grid_in_street_only_x_y in grids_x_y:
                        x_check = grid_in_street_only_x_y[0]
                        y_check = grid_in_street_only_x_y[1]
                        if grid['x'] == x_check and grid['y'] == y_check:
                            for street_found in grid['streets']:
                                grids_ahead.append(street_found['street_id'])

            else:

                grids_x_y = [(element['x'], element['y']) for element in
                             self.map['streets'][current_position.street]['grids']
                             if
                             element['grid_order'] >= grid_order and math.fabs(
                                 element['grid_order'] - grid_order) < MAX_GRID_DISTANCE]

                for grid in self.map['grids_to_streets']:
                    for grid_in_street_only_x_y in grids_x_y:
                        x_check = grid_in_street_only_x_y[0]
                        y_check = grid_in_street_only_x_y[1]
                        if grid['x'] == x_check and grid['y'] == y_check:
                            for street_found in grid['streets']:
                                grids_ahead.append(street_found['street_id'])

            for street_idx, street_id in enumerate(grids_ahead):
                street_name = self.map['streets'][street_id]['street_name']
                if street_name is None or street_name == '':
                    continue
                if street_name in map_count_var_streets.keys():
                    number_var = map_count_var_streets[street_name]
                    while len(bow_streets) <= number_var:
                        number_var -= 1
                    if bow_streets[number_var] != 1:
                        bow_streets[number_var] = 1

        return np.concatenate((bow_tags_here, bow_tags, bow_streets_here, bow_streets))

    def get_world_state_for_turn(self, current_position, map_count_var_streets):

        bow_turn_street = np.zeros(3)

        assert (current_position.side == 1 or current_position.side == 2)

        streets_to_grid = [element for element in self.map['grids_to_streets'] if
                           element['x'] == current_position.x and element['y'] == current_position.y]

        street_1 = None
        street_2 = None
        street_1_current_side_vertical = None
        street_1_not_side_vertical = None
        street_2_current_side_vertical = None
        street_2_not_side_vertical = None

        for street in streets_to_grid[0]['streets']:
            street_number = street['street_id']
            street_name = street['street_name']
            street_azimuth = street['azimuth']
            if street_azimuth > 180:
                street_azimuth = street_azimuth - 180

            check_if_first = False

            if street_number == current_position.street:
                check_if_first = True

            grid_order = [element['grid_order'] for element in self.map['streets'][street_number]['grids'] if
                          element['x'] == current_position.x and element['y'] == current_position.y][0]
            street_to_grid = [element for element in self.map['streets'][street_number]['grids'] if
                              MAX_GRID_DISTANCE > math.fabs(element['grid_order'] - grid_order) > 3
                              and (element['x'] != current_position.x or element['y'] != current_position.y)]

            street_to_grid_sorted = sorted(street_to_grid, key=lambda k: math.fabs(k['grid_order'] - grid_order))
            side_1_name = None
            side_2_name = None
            side_1_number = None
            side_2_number = None
            side_1_vertical_name = None
            side_2_vertical_name = None

            for grid_sort in street_to_grid_sorted:
                x_vertical = grid_sort['x']
                y_vertical = grid_sort['y']
                grid_order_vertical = grid_sort['grid_order']

                streets_to_grid_vertical = [element['streets'] for element in self.map['grids_to_streets'] if
                                            element['x'] == x_vertical and element['y'] == y_vertical
                                            ][0]

                assert (len(streets_to_grid_vertical) <= 2)
                if len(streets_to_grid_vertical) <= 1:
                    continue
                streets_to_grid_vertical = [element for element in streets_to_grid_vertical if
                                            element['street_id'] != street_number]
                if len(streets_to_grid_vertical) <= 0:
                    continue
                if current_position.side == 1:

                    if grid_order_vertical < grid_order and side_1_name is None:
                        side_1_vertical_name = streets_to_grid_vertical[0]['street_name']
                    elif grid_order_vertical > grid_order and side_2_name is None:
                        side_2_vertical_name = streets_to_grid_vertical[0]['street_name']
                else:
                    if grid_order_vertical > grid_order and side_1_name is None:
                        side_1_vertical_name = streets_to_grid_vertical[0]['street_name']
                    elif grid_order_vertical < grid_order and side_2_name is None:
                        side_2_vertical_name = streets_to_grid_vertical[0]['street_name']

            if street_1 is None and check_if_first == True:
                street_1 = street_name

                street_1_current_side_vertical = side_1_vertical_name
                street_1_not_side_vertical = side_2_vertical_name


            elif street_2 is None:
                street_2 = street_name

                street_2_current_side_vertical = side_1_vertical_name
                street_2_not_side_vertical = side_2_vertical_name

        if street_1 in map_count_var_streets:
            number_var = map_count_var_streets[street_1]
            while number_var >= len(bow_turn_street):
                number_var -= 1
            bow_turn_street[number_var] = 6

        if street_1_not_side_vertical in map_count_var_streets:
            number_var = map_count_var_streets[street_1_not_side_vertical]
            while number_var >= len(bow_turn_street):
                number_var -= 1
            bow_turn_street[number_var] = 4

        if street_1_current_side_vertical in map_count_var_streets:
            number_var = map_count_var_streets[street_1_current_side_vertical]
            while number_var >= len(bow_turn_street):
                number_var -= 1
            bow_turn_street[number_var] = 5

        if street_2 in map_count_var_streets:
            number_var = map_count_var_streets[street_2]
            while number_var >= len(bow_turn_street):
                number_var -= 1
            bow_turn_street[number_var] = 3
        if street_2_current_side_vertical in map_count_var_streets:
            number_var = map_count_var_streets[street_2_current_side_vertical]
            while number_var >= len(bow_turn_street):
                number_var -= 1
            bow_turn_street[number_var] = 2
        if street_2_not_side_vertical in map_count_var_streets:
            number_var = map_count_var_streets[street_2_not_side_vertical]
            while number_var >= len(bow_turn_street):
                number_var -= 1
            bow_turn_street[number_var] = 1
        return bow_turn_street
