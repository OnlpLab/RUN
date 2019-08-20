# -*- coding: utf-8 -*-

import logging
import time
import copy
import numpy
import modules.data_processors as data_processers
import modules.beam_search as beam_search
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.encoder import EncoderRNN_Backward, EncoderRNN_Forward
from torch import optim
import torch.nn.functional as F
from copy import deepcopy
from modules.attention_decoder import AttnDecoderRNN

SOS = 5


def trainer(encoder_f, encoder_b, attn_decoder1, encoder_optimizer_f, encoder_optimizer_b, decoder_optimizer,
            seq_lang, input_variable_world, target_variable, hidden_size, model_settings):
    loss = 0

    input_variable_x_length = len(seq_lang)
    target_variable_length = len(target_variable)

    ht_enc_forwards = Variable(torch.DoubleTensor(numpy.zeros((input_variable_x_length, hidden_size))))
    ht_enc_backwards = Variable(torch.DoubleTensor(numpy.zeros((input_variable_x_length, hidden_size))))

    (ht_enc_forward, ct_enc_forward) = encoder_f.initHidden

    (ht_enc_backward, ct_enc_backward) = encoder_b.initHidden
    criterion = nn.NLLLoss()

    Emb_lang_sparse = torch.DoubleTensor(
        numpy.identity(model_settings['dim_lang']))

    xt_lang_forward = encoder_f.Emb_enc_forward[seq_lang]
    xt_lang_backward = encoder_b.Emb_enc_backward[seq_lang]
    xt_world = torch.bmm(input_variable_world, attn_decoder1.Emb_dec.unsqueeze(0))  # dot

    for ei in range(input_variable_x_length):
        ht_enc_forward, ct_enc_forward = encoder_f(
            xt_lang_forward[ei], ht_enc_forward, ct_enc_forward)
        ht_enc_forwards[ei] = ht_enc_forward

    for ei in reversed(range(input_variable_x_length)):
        ht_enc_backward, ct_enc_backward = encoder_b(
            xt_lang_backward[ei], ht_enc_backward, ct_enc_backward)
        ht_enc_backwards[ei] = ht_enc_backward

    embedded = Variable(Emb_lang_sparse[seq_lang])

    hidden, ctm1 = attn_decoder1.initHidden

    decoder_input = Variable(torch.LongTensor([[SOS]]))

    for di in range(target_variable_length):
        attn_decoder1.scope_att = torch.cat(
            (embedded, ht_enc_forwards, ht_enc_backwards, xt_world[0][di].expand(ht_enc_forwards.size()[0], 100)),
            dim=1)

        attn_decoder1.scope_att_times_W = torch.bmm(attn_decoder1.scope_att.unsqueeze(0),
                                                    attn_decoder1.W_att_scope.unsqueeze(0))

        xt_input = attn_decoder1.embedding(decoder_input).view(1, 1, -1)

        ctm1, hidden, zt_dec = attn_decoder1(xt=xt_input,
                                             htm1=hidden,
                                             ctm1=ctm1
                                             )

        output = torch.cat((hidden, zt_dec), dim=2)

        output_weighted = torch.bmm(output, attn_decoder1.W_out_hz.unsqueeze(0))

        post_transform = torch.bmm(output_weighted, attn_decoder1.W_out.unsqueeze(0))

        output_actions_and_probs_log = F.log_softmax(post_transform[0], dim=1)

        loss += criterion(output_actions_and_probs_log, target_variable[di])
        topv, topi = output_actions_and_probs_log.data.topk(1)
        ni = topi[0][0]
        decoder_input = Variable(torch.LongTensor([[ni]]))

    encoder_optimizer_f.zero_grad()
    encoder_optimizer_b.zero_grad()
    decoder_optimizer.zero_grad()

    loss.backward()

    encoder_optimizer_b.step()
    encoder_optimizer_f.step()
    decoder_optimizer.step()

    return loss / target_variable_length


def train_model(args):
    """
    this function is called to train model
    """
    logger = logging.getLogger("trainer log")
    logger.info("start train_model")

    start_time = time.time()

    random_seed = numpy.asscalar(args.Seed)

    torch.manual_seed(random_seed)

    logger.info("reading and processing data ... ")

    data_process = data_processers.DataProcess(
        path_rawdata=args.FileData
    )

    #
    logger.info("building model ... ")

    model_settings = {
        'dim_lang': data_process.dim_lang,
        'dim_world': data_process.dim_world,
        'dim_action': data_process.dim_action,
        'dim_model': args.DimModel,
        'drop_out_rate': args.DropOut
    }

    logger.info("model setting")

    hidden_size = numpy.asscalar(model_settings['dim_model'])

    encoder_f = EncoderRNN_Forward(model_settings=model_settings)

    encoder_b = EncoderRNN_Backward(model_settings=model_settings)
    attn_decoder1 = AttnDecoderRNN(model_settings)

    learning_rate = 1e-4
    weight_decay = 1e-8
    encoder_optimizer_f = optim.Adam(encoder_f.parameters(), lr=learning_rate, weight_decay=weight_decay)
    encoder_optimizer_b = optim.Adam(encoder_b.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(attn_decoder1.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epi in range(args.MaxEpoch):

        print("training epoch ", epi)

        err = 0.0
        num_steps = 0
        train_start = time.time()

        # goes per each instruction
        for name_map in [args.Map1, args.Map2]:
            max_steps = len(
                data_process.dict_data['train'][name_map]
            )
            # goes per each instruction
            for idx_data, data in enumerate(data_process.dict_data['train'][name_map]):
                data_process.process_one_data(
                    idx_data, name_map, 'train'
                )

                input_variable_x = torch.LongTensor(data_process.seq_lang_numpy.tolist())

                input_variable_world = Variable(torch.DoubleTensor(data_process.seq_world_numpy).unsqueeze(0))
                target_variable = Variable(
                    torch.LongTensor(data_process.seq_action_numpy.tolist()).unsqueeze(0).transpose(-1, 0))

                loss = trainer(encoder_f, encoder_b, attn_decoder1, encoder_optimizer_f, encoder_optimizer_b,
                               decoder_optimizer,
                               input_variable_x, input_variable_world, target_variable, hidden_size, model_settings)

                err += loss

            num_steps += max_steps

        train_end = time.time()

        train_err = err / num_steps
        logger.info("Train_err: {}, Time it took: {}".format(train_err, train_end - train_start))

        save_file = args.path_save + 'model' + '.pkl'

        bs_settings = {'path_model': None, 'trained_model': {'encoder_f': encoder_f, 'encoder_b': encoder_b,
                                                             'decoder': attn_decoder1,
                                                             'drop_out_rate': model_settings['drop_out_rate'],
                                                             'dim_model': numpy.asscalar(model_settings['dim_model']),
                                                             'dim_lang': model_settings['dim_lang']},
                       'dim_lang': data_process.dim_lang, 'map': None, 'size_beam': args.SizeBeam,
                       'length_normalization_factor': args.LengthNormalizationFactor}

        torch.save(bs_settings['trained_model'], save_file)

        #########validation##########

        if not args.Validation:
            continue

        dev_start = time.time()

        cnt_all = 0
        cnt_success = 0
        num_steps = 0

        for name_map in [args.Map1, args.Map2]:
            max_steps = len(
                data_process.dict_data['dev'][name_map]
            )
            #
            bs_settings['map'] = data_process.maps[
                name_map
            ]
            bs_settings['lang2idx'] = data_process.lang2idx
            bs = beam_search.BeamSearchNeuralNavigation(
                bs_settings, data_process
            )
            bs.name_map = name_map
            #
            for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
                one_sample = data_process.process_one_data(
                    idx_data, name_map, 'dev'
                )
                input_variable_x = torch.LongTensor(data_process.seq_lang_numpy.tolist())
                cnt_all += 1

                input_variable_world = Variable(torch.DoubleTensor(data_process.seq_world_numpy).unsqueeze(0))

                bs.idx_data = idx_data
                bs.set_encoder(
                    input_variable_x,
                    input_variable_world,
                    one_sample
                )

                pos_start, pos_end = data_process.get_pos(
                    idx_data, name_map, 'dev'
                )

                bs.init_beam(
                    copy.deepcopy(pos_start), copy.deepcopy(pos_end)
                )
                bs.search_func()
                gen_top_path = bs.get_route_generated()
                ref_path = one_sample.route
                #
                if bs.executer.evaluate(gen_top_path, ref_path):
                    cnt_success += 1
                    print("success rate for now: {}. success:{}. all:{}".format(round(1.0 * cnt_success / cnt_all, 4),
                                                                                cnt_success, cnt_all))

                bs.refresh_state()

            num_steps += max_steps

        dev_end = time.time()

        success_rate = round(1.0 * cnt_success / num_steps, 4)

        logger.info("success_rate: {}. Time it took: {}".format(success_rate, dev_end - dev_start))
    end_time = time.time()

    logger.info("Total time it took: {}".format(end_time - start_time))

    logger.info("finished training")


def test_model(args):
    """
    this function is called to test model
    """

    results_logs = logging.getLogger("testing results log")
    logger = logging.getLogger("testing log")
    logger.info("start train_model")

    logger.info("reading and processing data ... ")

    data_process = data_processers.DataProcess(
        path_rawdata=args.FileData
    )

    bs_settings = {'size_beam': args.SizeBeam, 
                   'length_normalization_factor': args.LengthNormalizationFactor, 'path_model': None,
                   'trained_model': torch.load(args.PathPretrain), 'dim_lang': data_process.dim_lang,
                   'map': data_process.maps[args.MapTest], 'lang2idx': data_process.lang2idx}

    #

    #
    logger.info("building model ... ")

    name_map = args.MapTest
    cnt_all = 0
    cnt_success = 0

    num_steps = len(
        data_process.dict_data['dev'][name_map]
    ) + len(
        data_process.dict_data['train'][name_map]
    )
    bs = beam_search.BeamSearchNeuralNavigation(
        bs_settings, data_process
    )
    #
    bs_results = []
    #
    for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
        one_sample = data_process.process_one_data(
            idx_data, name_map, 'dev'
        )
        cnt_all += 1

        input_variable_x = torch.LongTensor(data_process.seq_lang_numpy.tolist())

        input_variable_world = Variable(torch.DoubleTensor(data_process.seq_world_numpy).unsqueeze(0))
        bs.idx_data = idx_data
        bs.set_encoder(
            input_variable_x,
            input_variable_world,
            one_sample
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'dev'
        )
        bs.init_beam(
            deepcopy(pos_start), deepcopy(pos_end)
        )
        bs.name_map = name_map

        bs.search_func()

        gen_top_path = bs.get_route_generated()
        ref_path = one_sample.route
        #
        is_the_route_correct = bs.executer.evaluate(gen_top_path, ref_path)
        if is_the_route_correct:
            cnt_success += 1
        logger.info(
            "success rate for now: {}. success:{}. all:{}".format(round(1.0 * cnt_success / cnt_all, 4), cnt_success,
                                                                  cnt_all))

        result = {
            'success': is_the_route_correct,
            'id': one_sample.id,
            'path_ref': ref_path,
            'path_gen': gen_top_path,
            'actions ref': data_process.seq_action_numpy,
            'actions gen': bs.get_actions_generated(),
            'instruction': one_sample.instruction,
            'instructions after entities extractions': one_sample.instruction_after_entity_extraction

        }
        bs_results.append(result)
        #
        bs.refresh_state()
        #
    #
    #
    for idx_data, data in enumerate(data_process.dict_data['train'][name_map]):
        one_sample = data_process.process_one_data(
            idx_data, name_map, 'train'
        )
        cnt_all += 1

        input_variable_x = torch.LongTensor(data_process.seq_lang_numpy.tolist())

        input_variable_world = Variable(torch.DoubleTensor(data_process.seq_world_numpy).unsqueeze(0))

        bs.set_encoder(
            input_variable_x,
            input_variable_world,
            one_sample
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'train'
        )
        bs.init_beam(
            deepcopy(pos_start), deepcopy(pos_end)
        )
        bs.search_func()

        gen_top_path = bs.get_route_generated()
        ref_path = one_sample.route

        is_the_route_correct = bs.executer.evaluate(gen_top_path, ref_path)
        if is_the_route_correct:
            cnt_success += 1
        logger.info(
            "success rate for now: {}. success:{}. all:{}".format(round(1.0 * cnt_success / cnt_all, 4), cnt_success,
                                                                  cnt_all))

        result = {
            'success': is_the_route_correct,
            'id': one_sample.id,
            'path_ref': ref_path,
            'path_gen': gen_top_path,
            'actions ref': data_process.seq_action_numpy,
            'actions gen': bs.get_actions_generated(),
            'instruction': one_sample.instruction,
            'instructions after entities extractions': one_sample.instruction_after_entity_extraction
        }
        bs_results.append(result)

        bs.refresh_state()

    success_rate = round(1.0 * cnt_success / num_steps, 4)

    if args.SaveResults:
        results_logs.info(bs_results)

    logger.info("the success_rate is {}: ".format(success_rate))

    logger.info("finished testing")







def test_model_multi_sentence(args):
    """
    this function is called to test model
    """

    results_logs = logging.getLogger("testing results log")
    logger = logging.getLogger("testing log")
    logger.info("start train_model")

    logger.info("reading and processing data ... ")

    data_process = data_processers.DataProcess(
        path_rawdata=args.FileData
    )

    bs_settings = {'size_beam': args.SizeBeam,
                   'length_normalization_factor': args.LengthNormalizationFactor, 'path_model': None,
                   'trained_model': torch.load(args.PathPretrain), 'dim_lang': data_process.dim_lang,
                   'map': data_process.maps[args.MapTest], 'lang2idx': data_process.lang2idx}

    #

    #
    logger.info("building model ... ")

    name_map = args.MapTest
    cnt_all = 0
    cnt_success = 0

    num_steps = len(data_process.dict_data_list['all'][name_map])
    bs = beam_search.BeamSearchNeuralNavigation(
        bs_settings, data_process
    )
    #
    bs_results = []
    #
    for idx,story_id in enumerate(data_process.dict_data_list['all'][name_map]):
        list_of_indexes=[]
        for idx_data, data in enumerate(data_process.dict_data['all'][name_map]): #
    
            if data['id']==story_id:
                list_of_indexes.append(idx_data)



        first=list_of_indexes[0]
        bs.data=data_process.process_one_data( first, name_map, 'all'  )
        bs.name_map=name_map
        bs.idx_data=first

        pos_start, _ = data_process.get_pos(first, name_map, 'all')
        last=list_of_indexes[-1]
        _, pos_end = data_process.get_pos(last, name_map, 'all' )

        input_variable_world=bs.get_feat_current_position(pos_start)
 
        ref_path_multi_sentence = []
        gen_top_path_multi_sentence=[]
        for idx_data in list_of_indexes:
            one_sample=data_process.process_one_data( idx_data, name_map, 'all'  )
            input_variable_x = torch.LongTensor(data_process.seq_lang_numpy.tolist())
            bs.set_encoder( input_variable_x, input_variable_world,one_sample )
            bs.init_beam( copy.deepcopy(pos_start), copy.deepcopy(pos_end))


            bs.search_func()

            ref_path = one_sample.route
            gen_top_path = bs.get_route_generated()


            ref_path_multi_sentence+=ref_path[:]
            gen_top_path_multi_sentence+=gen_top_path[:]

            pos_start=gen_top_path[-1]
            input_variable_world=bs.get_feat_current_position(pos_start)

            bs.refresh_state_multi_sentence()

        is_the_route_correct = bs.executer.evaluate_multi_sentence(gen_top_path_multi_sentence, ref_path_multi_sentence)
        cnt_all+=1
        if is_the_route_correct:
            cnt_success += 1
        logger.info( "success rate for now: {}. success:{}. all:{}".format(round(1.0 * cnt_success / cnt_all, 4), cnt_success,cnt_all))
 
        


        result = {
            'success': is_the_route_correct,
            'id': id,
            'path_ref': ref_path_multi_sentence,
            'path_gen': gen_top_path_multi_sentence,
            'last instruction': one_sample.instruction

        }
        bs_results.append(result)
        #
        bs.refresh_state()
        #
    #
    #

    success_rate = round(1.0 * cnt_success / num_steps, 4)

    if args.SaveResults:
        results_logs.info(bs_results)

    logger.info("the success_rate is {}: ".format(success_rate))

    logger.info("finished testing")


