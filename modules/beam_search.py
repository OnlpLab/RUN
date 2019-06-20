# -*- coding: utf-8 -*-

import copy
import numpy
import pickle
import torch
import torch.nn.functional as F
from enum import Enum
from torch.autograd import Variable

from execution_system import ExecutionSystem
from execution_system import MovementType, Turn
from modules.world_state_processer import WorldStateProcesser

LENGTH_NORMALIZATION_CONST = 5
SOS=5


class BeamSearchNeuralNavigation(object):
    """
    This is a beam search code.
    It is used during testing and validation stage.
    """

    def __init__(self, settings,data_process):
        assert (settings['size_beam'] >= 1)
        self.size_beam = settings['size_beam']
        #
        assert (
                settings['path_model'] is None or settings['trained_model'] is None
        )
        #
        if settings['path_model'] is not None:
            with open(settings['path_model'], 'rb') as f:
                self.model = pickle.load(f)
        else:
            assert (settings['trained_model'] is not None)
            self.model = settings['trained_model']
            self.encoder_f = self.model['encoder_f']
            self.encoder_b = self.model['encoder_b']
            self.decoder = self.model['decoder']
            self.dim_model = self.model['dim_model']
            self.dim_lang = self.model['dim_lang']
            print("got model")

        self.drop_out_rate = self.model['drop_out_rate']
        assert (
            self.drop_out_rate <= numpy.float32(1.0)
        )
        self.data_process = data_process

        self.scope_att = None
        self.scope_att_times_W = None
        #
        self.beam_list = []
        self.finish_list = []

        self.dim_lang = settings['dim_lang']
        self.map = settings['map']
        self.executer = ExecutionSystem(self.map)
        self.world_state_processer = WorldStateProcesser(self.map)
        self.length_normalization_factor = settings['length_normalization_factor']



    def refresh_state_multi_sentence(self):
        self.scope_att = None
        self.scope_att_times_W = None
        #
        self.beam_list = []
         
 
        self.finish_list = []


    def refresh_state(self):
        self.scope_att = None
        self.scope_att_times_W = None
        #
        self.beam_list = []
        self.finish_list = []

    def set_encoder(self, seq_lang, input_variable_world, data):
        self.data=data
        self.input_variable_world = input_variable_world[0][0].unsqueeze(0).unsqueeze(0)
        input_variable_x_length = len(seq_lang)

        ht_enc_forwards = Variable(torch.DoubleTensor(numpy.zeros((input_variable_x_length, self.dim_model))))
        ht_enc_backwards = Variable(torch.DoubleTensor(numpy.zeros((input_variable_x_length, self.dim_model))))

        ht_enc_forward, ct_enc_forward = self.encoder_f.initHidden
        ht_enc_backward, ct_enc_backward = self.encoder_b.initHidden

        Emb_lang_sparse = torch.DoubleTensor(numpy.identity(self.dim_lang))

        xt_lang_forward = self.encoder_f.Emb_enc_forward[seq_lang]
        xt_lang_backward = self.encoder_b.Emb_enc_backward[seq_lang]

        for ei in range(input_variable_x_length):
            ht_enc_forward, ct_enc_forward = self.encoder_f(
                xt_lang_forward[ei], ht_enc_forward, ct_enc_forward)
            ht_enc_forwards[ei] = ht_enc_forward

        for ei in reversed(range(input_variable_x_length)):
            ht_enc_backward, ct_enc_backward = self.encoder_b(
                xt_lang_backward[ei], ht_enc_backward, ct_enc_backward)
            ht_enc_backwards[ei] = ht_enc_backward

        embedded = Variable(Emb_lang_sparse[seq_lang])

        self.embedded = embedded
        self.ht_enc_forwards = ht_enc_forwards
        self.ht_enc_backwards = ht_enc_backwards

        self.hidden, self.ctm1 = self.decoder.initHidden

    def init_beam(self, pos_start, pos_end, cost=0.00):

        item = {
            'hidden': self.hidden.clone(),
            'ctm1': self.ctm1.clone(),
            'feat_current_position': self.input_variable_world,
            #
            'pos_current': pos_start,
            'pos_destination': pos_end,
            'list_pos': [copy.deepcopy(pos_start)],
            #
            'list_idx_action': [],
            'continue': True,
            #
            'cost': cost,
            'final_cost': 0
        }

        self.beam_list.append(item)

    def decode_step(self, feat_current_position, hidden, ctm1, actions):

        xt_action = torch.bmm(feat_current_position, self.decoder.Emb_dec.unsqueeze(0))  # dot

        self.decoder.scope_att = torch.cat((self.embedded, self.ht_enc_forwards, self.ht_enc_backwards,
                                            xt_action.squeeze().expand(self.ht_enc_forwards.size()[0], 100)), dim=1)

        self.decoder.scope_att_times_W = torch.bmm(self.decoder.scope_att.unsqueeze(0),
                                                   self.decoder.W_att_scope.unsqueeze(0))

        if len(actions) <= 0:
            ni = SOS
        else:
            ni = actions[-1]

        decoder_input = Variable(torch.LongTensor([[ni]]))

        xt_input = self.decoder.embedding(decoder_input).view(1, 1, -1)

        ctm1, hidden, zt = self.decoder(xt=xt_input,
                                        htm1=hidden,
                                        ctm1=ctm1
                                        )

        out_put = torch.cat((hidden, zt), dim=2)

        output_weighted = torch.bmm(out_put, self.decoder.W_out_hz.unsqueeze(0))

        post_transform = torch.bmm(output_weighted, self.decoder.W_out.unsqueeze(0))

        log_probt = F.log_softmax(post_transform[0][0], dim=0)

        return xt_action, hidden, ctm1, log_probt.data.numpy()

    def validate_step(self, idx_action, pos_current):
        # check-
        # 1. if turn - that there is a path to turn to.
        # 2. if turn - direction is known
        # 3. if face - that street exists.
        # 4. if face - can't be the same direction as before (must change something)
        # 5. if walk - that it is not the end of te path.
        # 6. if walk - direction is known

        assert (
            idx_action >= 0 and idx_action <= 5
        )

        if pos_current.side == -1:
            return False
        if idx_action == 0:  # turn left
            return self.executer.check_valid_turn(pos_current, Turn.LEFT)
        elif idx_action == 1:  # turn right
            return self.executer.check_valid_turn(pos_current, Turn.RIGHT)
        elif idx_action == 2:  # turn behind
            return self.executer.check_valid_turn(pos_current, Turn.BEHIND)
        elif idx_action == 3:  # walk1
            return self.executer.check_valid_walk(pos_current)
        else:
            return True

    def take_one_step(self, current_position, idx_action):

        if idx_action >= 0 and idx_action <= 2:
            type = MovementType.TURN
            movement = Turn(idx_action)

        elif idx_action == 3:
            movement = 1
            type = MovementType.WALK
        else:
            return current_position
        _, pos_next = self.executer.execute_movement(current_position, type, movement)
        return pos_next

        #

    def get_feat_current_position(self, pos_current): #world-state
        #
        world_state_turn = Variable(torch.DoubleTensor(self.world_state_processer.get_world_state_for_turn
                                                       (pos_current,
                                                        self.data.entities_streets)).unsqueeze(0).unsqueeze(0)) #forward world-state

        world_state_walk = Variable(torch.DoubleTensor(self.world_state_processer.get_world_state_for_walking
                                                       (pos_current, self.data.entities_not_streets,
                                                        self.data.entities_streets)).unsqueeze(0).unsqueeze(0)) #current world-state
        world_state_basic = Variable(torch.DoubleTensor(self.data_process.get_world_state_for_basic
                                                        ('all', self.name_map, self.idx_data)).unsqueeze(0).unsqueeze(0)) #metadata

        all_world = torch.cat((world_state_basic, world_state_turn, world_state_walk), dim=2)
        return all_world

    def beam_score_avg(self, score, seq_length):
        return score / seq_length

    def beam_score_complex(self, score, seq_length): #from  'Google’s neural machine translation system: Bridging the gap between human and machine translation.'
        length_penalty = (LENGTH_NORMALIZATION_CONST + seq_length) / (LENGTH_NORMALIZATION_CONST + 1)
        return score / (length_penalty ** self.length_normalization_factor)

    def beam_score_normalization(self, score, seq_length):

            return self.beam_score_complex(score, seq_length)


    def search_func(self):
        counter, max_counter = 0, 100
        max_options = 50
        while (len(self.finish_list) < max_options) and (counter < max_counter):

            new_list = []
            for item in self.beam_list:
                xt, hidden, ctm1, log_probt = self.decode_step(
                    feat_current_position=item['feat_current_position'],
                    hidden=item['hidden'], ctm1=item['ctm1'], actions=item['list_idx_action']) #decode
                top_k_list = range(log_probt.shape[0])
                for top_idx_action in top_k_list:
                    if self.validate_step(top_idx_action, item['pos_current']):  # check if the action is moving forward that there is a path forward. If a turn: that there is an option to turn
                        new_item = {
                            'hidden': hidden.clone(),
                            'ctm1': ctm1.clone(),

                            'list_idx_action': [
                                idx for idx in item['list_idx_action']
                            ],
                            'list_pos': [copy.deepcopy(pos) for pos in item['list_pos']
                            ]
                        }
                        new_item['list_idx_action'].append(
                            top_idx_action
                        )
                        #
                        new_item['pos_current'] = copy.deepcopy(
                            self.take_one_step(
                                item['pos_current'], top_idx_action
                            )
                        )

                        #
                        new_item['pos_destination'] = copy.deepcopy(
                            item['pos_destination']
                        )
                        #
                        new_item['feat_current_position'] = self.get_feat_current_position(
                            new_item['pos_current']
                        )

                        #
                        new_item['list_pos'].append(
                            copy.deepcopy(new_item['pos_current'])
                        )
                        #
                        if top_idx_action == 4:

                            new_item['continue'] = False
                        else:
                            new_item['continue'] = True
                        #

                        new_item['cost'] = item['cost'] + (-1.0) * log_probt[top_idx_action]

                        new_item['final_cost'] = self.beam_score_normalization(new_item['cost'],
                                                                               len(new_item['list_idx_action']))

                        new_list.append(new_item)

            new_list = sorted(
                new_list, key=lambda x: x['final_cost']
            )
            if len(new_list) > self.size_beam:  # as deep as the beam search size
                new_list = new_list[:self.size_beam]

            self.beam_list = []  # clean previous list
            while len(new_list) > 0:
                pop_item = new_list.pop(0)
                if pop_item['continue']:
                    self.beam_list.append(pop_item)
                else:
                    self.finish_list.append(pop_item)
                    if self.size_beam ==0: #greedy found
                        return
            counter += 1
            #
        #
        while len(self.beam_list) > 0:
            self.finish_list.append(self.beam_list.pop(0))

        if len(self.finish_list) > 0:
            self.finish_list = sorted(
                self.finish_list, key=lambda x: x['final_cost']
            )

    def get_route_generated(self):
        return self.finish_list[0]['list_pos']

    def get_actions_generated(self):
        return self.finish_list[0]['list_idx_action']

    def get_all_routes_in_beam(self):
        return [x['list_pos'] for x in self.finish_list]


