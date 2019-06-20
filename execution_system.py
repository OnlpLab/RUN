import math
from enum import Enum
import copy

MAX_GRID_DISTANCE = 5


class ExecutionSystem(object):
    """
    This is the execution system for the OSS map.
    The execution system can execute 5 types of actions (defined in the MovementType class).
    """

    def __init__(self, map):
        self.map = map

    @staticmethod
    def normalize_azimuth(azimuth):

        if azimuth < 0:
            azimuth = azimuth + 360
        elif azimuth >= 360:
            azimuth = azimuth - 360

        if azimuth < 180:
            side = 1
        else:
            side = 2
        if azimuth > 180:
            azimuth -= 180

        return azimuth, side

    @staticmethod
    def azimuth_for_turn(azimuth, turn_direction):
        if turn_direction == Turn.RIGHT:
            azimuth = azimuth + 90
        elif turn_direction == Turn.LEFT:
            azimuth = azimuth - 90
        elif turn_direction == Turn.BEHIND:
            azimuth = azimuth + 180
        elif turn_direction == Turn.RIGHT_DIAGONAL:
            azimuth = azimuth + 45
        else:
            azimuth = azimuth - 45
        return azimuth

    def execute_movement(self, current_position, action_type, movement=-1):
        assert isinstance(current_position, Position), \
            'current position is not a Position type. It is of type:{}'.format(type(current_position))
        valid_action = self.validate_step(current_position, action_type, movement)
        if not valid_action:
            return False, current_position

        assert (
                action_type == MovementType.WALK or action_type == MovementType.END or action_type == MovementType.TURN
                or action_type == MovementType.FACE or action_type == MovementType.VERIFY_STREET
        )
        if action_type == action_type.FACE:
            current_position = Position((current_position.x, current_position.y, movement))
            return True, current_position
        elif action_type == action_type.END:
            current_position = Position((current_position.x, current_position.y, current_position.direction))
            return True, current_position
        elif action_type == action_type.WALK:
            return self.execute_walk(current_position, movement)

        elif action_type == action_type.VERIFY_STREET:
            if movement == current_position.street:
                return True, current_position
            else:
                return False, current_position
        else:  # TURN
            return self.execute_turn(current_position, movement)

    def execute_walk(self, current_position, number_of_steps):
        assert isinstance(current_position, Position), 'current position is not an Position type'

        street_to_grid = [element for element in self.map['streets'][current_position.street]['grids'] if
                          element['x'] == current_position.x and element['y'] == current_position.y]

        assert (len(street_to_grid) == 1), (
            current_position, len(street_to_grid), current_position.street, current_position.side)
        number_grid_order = street_to_grid[0]['grid_order']

        if current_position.side == 1:
            position = self.map['streets'][current_position.street]['grids'][int(number_grid_order - number_of_steps)]
        else:
            if not (int(number_grid_order + number_of_steps) + 1 <= len(
                    self.map['streets'][current_position.street]['grids'])):
                number_of_steps -= 1
            position = self.map['streets'][current_position.street]['grids'][int(number_grid_order + number_of_steps)]

        current_position = Position((position['x'], position['y'], current_position.direction))
        return True, current_position

    def execute_turn(self, current_position, turn_direction):
        assert isinstance(current_position, Position), 'current position is not an Position type'

        assert (
                turn_direction == Turn.RIGHT or turn_direction == Turn.LEFT
                or turn_direction == Turn.RIGHT_DIAGONAL
                or turn_direction == Turn.LEFT_DIAGONAL
                or turn_direction == Turn.BEHIND), 'incorrect turn: '.format(turn_direction)

        grid_to_street = [element for element in self.map['grids_to_streets'] if
                          element['x'] == current_position.x and element['y'] == current_position.y]

        assert len(grid_to_street) > 0
        street_grid = [element for element in grid_to_street[0]['streets'] if
                       element['street_id'] == current_position.street]
        assert len(street_grid) > 0

        assert (current_position.side == 1 or current_position.side == 2), 'direction is -1 in TURN action'
        if current_position.side == 1:
            azimuth = street_grid[0]['azimuth']
        else:
            azimuth = street_grid[0]['azimuth'] + 180

        azimuth = self.azimuth_for_turn(azimuth, turn_direction)

        azimuth, side = self.normalize_azimuth(azimuth)

        closest_street = -1
        closest_azimuth_delta = 360
        for street in grid_to_street[0]['streets']:
            delta = math.fabs(azimuth - street['azimuth'])

            if closest_azimuth_delta > delta:
                closest_street = street
                closest_azimuth_delta = delta

        direction = (closest_street['street_id'], side)
        if direction == current_position.direction:
            side = side % 2 + 1
            direction = (closest_street['street_id'], side)
        current_position = Position((current_position.x, current_position.y, direction))
        return True, current_position

    def validate_step(self, current_position, movement_type, movement=-1):
        """
        check-
        1. if turn - that there is a path to turn to.
        2. if turn - direction is known
        3. if face - that street exists.
        4. if face - can't be the same direction as before (must change something)
        5. if walk - that it is not the end of te path.
        6. if walk - direction is known
        """

        assert (
                movement_type == MovementType.WALK or movement_type == MovementType.END or movement_type == MovementType.TURN
                or movement_type == MovementType.FACE or movement_type == MovementType.VERIFY_STREET
        )

        assert isinstance(current_position, Position), 'current position is not an Position type'

        if movement_type == MovementType.FACE:
            street_to_face = movement[0]
            return self.check_valid_face(current_position, street_to_face)

        if movement_type == MovementType.END:
            return True

        if current_position.side == -1:  # only END and FACE can hadel unknown directions
            return False

        if movement_type == MovementType.TURN:
            assert (movement == Turn.LEFT or movement == Turn.RIGHT or movement == Turn.BEHIND)
            if movement == Turn.LEFT:
                return self.check_valid_turn(current_position, Turn.LEFT)

            elif movement == Turn.RIGHT:
                return self.check_valid_turn(current_position, Turn.RIGHT)

            else:  # turn behind
                return self.check_valid_turn(current_position, Turn.BEHIND)

        else:  # WALK:
            return self.check_valid_walk(current_position)

    def check_valid_face(self, current_position, street_number):
        assert isinstance(current_position, Position), 'current position is not an Position type'

        grid_to_street = [element for element in self.map['grids_to_streets'] if
                          element['x'] == current_position.x and element['y'] == current_position.y]
        street_grid = [element for element in grid_to_street[0]['streets'] if element['street_id'] == street_number]
        if len(street_grid) > 0:
            return True
        else:
            return False

    def check_valid_turn(self, current_position, turn_direction):

        assert isinstance(current_position, Position), 'current position is not an Position type'
        assert (turn_direction == Turn.RIGHT or turn_direction == Turn.LEFT or turn_direction == Turn.RIGHT_DIAGONAL
                or turn_direction == Turn.LEFT_DIAGONAL or turn_direction == Turn.BEHIND)

        grid_to_street = [element for element in self.map['grids_to_streets'] if
                          element['x'] == current_position.x and element['y'] == current_position.y]

        street_grid = [element for element in grid_to_street[0]['streets'] if
                       element['street_id'] == current_position.street]
        assert (len(street_grid) >= 1)
        if current_position.side == 1:
            azimuth = street_grid[0]['azimuth']
        else:
            azimuth = street_grid[0]['azimuth'] + 180

        azimuth = self.azimuth_for_turn(azimuth, turn_direction)

        azimuth, _ = self.normalize_azimuth(azimuth)

        closest_azimuth_delta = 360
        for street in grid_to_street[0]['streets']:
            delta = math.fabs(azimuth - street['azimuth'])
            if closest_azimuth_delta > delta:
                closest_azimuth_delta = delta
        if closest_azimuth_delta <= 45:
            return True
        else:
            return False

    def check_valid_walk(self, current_position, number_of_steps=-1):
        assert isinstance(current_position, Position), 'current position is not an Position type'

        if current_position.side == -1:
            return False
        street_grids = [element['grids'] for element in self.map['streets'] if
                        element['street_id'] == current_position.street]
        len_grids = len(street_grids[0])
        grid_order = [element['grid_order'] for element in street_grids[0] if
                      element['x'] == current_position.x and element['y'] == current_position.y]

        if current_position.side == 2 and grid_order[0] >= len_grids - number_of_steps:
            return False
        elif current_position.side == 1 and grid_order[0] <= number_of_steps - 1:
            return False
        else:
            return True

    def oracle_evaluation(self, list_gen_paths, list_ref_position):
        for path in list_gen_paths:
            is_path_correct = self.evaluate(path, list_ref_position)
            if is_path_correct is True:
                return True
        return False

    def evaluate(self, list_gen_position, list_ref_position):

        distance = self.distance(list_gen_position[-1], list_ref_position[-1])

        # get only route points that constract the shape of the route
        route_gen = self.get_main_route(list_gen_position)
        route_ref = self.get_main_route(list_ref_position)

        # check routes are the same

        if route_gen != route_ref:
            return False

        # check last point distance and direction should be the same
        if distance <= MAX_GRID_DISTANCE:  # MAX_DISTANCE=5
            return True
        return False


    def evaluate_multi_sentence(self, list_gen_position, list_ref_position):

        distance = self.distance(list_gen_position[-1], list_ref_position[-1])

        # get only route points that constract the shape of the route
        route_gen = self.get_main_route(list_gen_position)
        route_ref = self.get_main_route(list_ref_position)

        #pop last direction
        del route_gen[-1]
        del route_ref[-1]


        # check routes are the same


        if (route_gen != route_ref):
            return False

        # check last point distance and direction should be the same
        if (distance <= MAX_GRID_DISTANCE):  # MAX_DISTANCE=5
            return True
        return False


    @staticmethod
    def clean_list_from_doubles(list):
        list_clean = []
        for index_point, point in enumerate(list):

            if index_point == 0:
                list_clean.append(copy.deepcopy(point))
            elif index_point > 0 and point != list[index_point - 1]:
                list_clean.append(copy.deepcopy(point))
        return list_clean

    def get_main_route(self, route_current):
        route_no_look_around = self.get_route_no_looking_around_on_the_route(route_current)
        route_main_no_walk_in_last_direction = self.get_route_directions_only(route_no_look_around)

        return route_main_no_walk_in_last_direction

    @staticmethod
    def get_route_no_looking_around_on_the_route(route_current):
        route = []
        prev_point = None
        for index_position, position in enumerate(route_current):
            if prev_point is None or prev_point.x != position.x or prev_point.y != position.y:
                route.append(copy.deepcopy(position))
                prev_point = position

        position = route_current[-1]
        if prev_point.street != position.street or prev_point.side != position.side:
            route.append(position)
        return route

    @staticmethod
    def get_route_directions_only(route):
        route_sides = []
        prev_point = None
        for index_position, position in enumerate(route):
            if prev_point is None or prev_point.street != position.street or prev_point.side != position.side:
                route_sides.append(copy.deepcopy((position.street, position.side)))
                prev_point = position
        return route_sides

    @staticmethod
    def distance(current_position, true_position):
        # assert isinstance(current_position, Position), 'current position is not an Position type'
        # assert isinstance(true_position, Position), 'target position is not an Position type'

        return math.sqrt(
            math.pow((current_position.y - true_position.y), 2) + math.pow((current_position.x - true_position.x), 2))

    def check_valid_turn(self, current_position, movement):

        assert (movement == Turn.RIGHT or movement == Turn.LEFT or movement == Turn.RIGHT_DIAGONAL
                or movement == Turn.LEFT_DIAGONAL or movement == Turn.BEHIND)

        grid_to_street = [element for element in self.map['grids_to_streets'] if
                          element['x'] == current_position.x and element['y'] == current_position.y]

        street_grid = [element for element in grid_to_street[0]['streets'] if
                       element['street_id'] == current_position.street]
        assert (len(street_grid) >= 1), (current_position, grid_to_street)
        if current_position.side == 1:
            azimuth = street_grid[0]['azimuth']
        else:
            azimuth = street_grid[0]['azimuth'] + 180
        if movement == Turn.RIGHT:
            azimuth = azimuth + 90
        elif movement == Turn.LEFT:
            azimuth = azimuth - 90
        elif movement == Turn.RIGHT_DIAGONAL:
            azimuth = azimuth + 45
        elif movement == Turn.RIGHT_DIAGONAL:
            azimuth = azimuth - 45
        else:
            azimuth = azimuth + 180

        if azimuth < 0:
            azimuth = azimuth + 360
        elif azimuth >= 360:
            azimuth = azimuth - 360
        if azimuth > 180:
            azimuth -= 180

        closest_street = -1
        closest_azimuth_delta = 360
        for street in grid_to_street[0]['streets']:
            delta = math.fabs(azimuth - street['azimuth'])
            if closest_azimuth_delta > delta:
                closest_street = street
                closest_azimuth_delta = delta
        if closest_azimuth_delta <= 45:
            return True
        else:
            return False

    def check_valid_walk(self, current_position):

        if current_position.side == -1:
            return False
        street_grids = [element['grids'] for element in self.map['streets'] if
                        element['street_id'] == current_position.street]
        len_grids = len(street_grids[0])
        grid_order = [element['grid_order'] for element in street_grids[0] if
                      element['x'] == current_position.x and element['y'] == current_position.y]

        if current_position.side == 2 and grid_order[0] >= len_grids - 1:
            return False
        elif current_position.side == 1 and grid_order[0] <= 0:
            return False
        else:
            return True


class Position:
    def __init__(self, position):
        self.x = int(position[0])
        self.y = int(position[1])
        self.direction = position[2]
        assert (len(self.direction) == 2)
        self.street = int(self.direction[0])
        self.side = int(self.direction[1])
        assert (self.side == 1 or self.side == 2 or self.side == -1)

    def __str__(self):
        return "position x: {}, y: {}, street: {}, side: {}".format(self.x, self.y, self.street, self.side)

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.street == other.street and self.side == other.side:
            return True
        return False


class MovementType(Enum):
    """
       1. FACE(street_id,end_street)- The face is a turn that is unrelated to the current direction of the agent.
                                      It takes the id of the street and the end it is faceing.
                                      If the end of the street is unknown then it is -1.
       2. WALK(X)- moves the agent in the direction it is facing.
                   X is the number of grids the agent will move in the current direction.
       3. TURN(turn_direction)- turns the agent based on the current direction of the agent and the turn_direction (type Turn class).
       4. VERIFY_STREET(street_id) - verification of the street the agent is facing.
       5. END - stop movement action.
       """
    FACE = 1
    WALK = 2
    TURN = 3
    VERIFY_STREET = 4
    END = 5


class Turn(Enum):
    LEFT = 0
    RIGHT = 1
    BEHIND = 2
    RIGHT_DIAGONAL = 3
    LEFT_DIAGONAL = 4
    UNKNOWN = 5
