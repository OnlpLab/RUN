from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.utils import sample_weights
import numpy


class AttnDecoderRNN(nn.Module):
    def __init__(self, model_settings):
        super(AttnDecoderRNN, self).__init__()
        self.dim_world = model_settings['dim_world']
        self.dim_lang = model_settings['dim_lang']
        self.dim_model = numpy.asscalar(model_settings['dim_model'])
        self.dim_action = model_settings['dim_action']
        self.dropout_p = numpy.asscalar(model_settings['drop_out_rate'])
        self.ren_gen = torch.rand(self.dim_model)
        self.embedding = nn.Embedding(self.dim_action, self.dim_model)

        self.drop_out_layer_gen = Variable(self.ren_gen < self.dropout_p).double()

        # params:

        self.Emb_dec = sample_weights(self.dim_world, self.dim_model)

        self.W_att_target = sample_weights(self.dim_model, self.dim_model)

        self.W_att_scope = sample_weights(self.dim_lang + 3 * self.dim_model,
                                          self.dim_model)

        self.b_att = nn.Parameter(torch.DoubleTensor(numpy.zeros((self.dim_model, 1))))

        self.W_out_hz = sample_weights(self.dim_lang + self.dim_model * 4,
                                       self.dim_model)
        self.W_out = sample_weights(self.dim_model, self.dim_action)

        self.W_dec = sample_weights(self.dim_lang + 5 * self.dim_model, 4 * self.dim_model)

        self.b_dec = nn.Parameter(torch.DoubleTensor(numpy.zeros(4 * self.dim_model)))

    def forward(self, xt, htm1, ctm1):
        xt = xt.type(torch.DoubleTensor)
        seq_lang_lenght = self.scope_att.size()[0]
        htm1_weighted = torch.bmm(htm1.expand(1, seq_lang_lenght, 100), self.W_att_target.unsqueeze(0))

        htm1_attention = htm1_weighted + self.scope_att_times_W

        htm1_attention_scaled = torch.tanh(htm1_attention)

        htm1_attention_with_b = torch.bmm(htm1_attention_scaled, self.b_att.unsqueeze(0))

        current_att_weight = self.softmax(htm1_attention_with_b)

        zt = torch.bmm(current_att_weight.squeeze().unsqueeze(0).unsqueeze(0),
                       self.scope_att.unsqueeze(0))  # attention output

        input_for_decoder = torch.cat((xt, htm1, zt),
                                      dim=2)  # implemented a LSTM (could use the built-in method but we experimented different LSTM implementations)

        input_for_decoder_weighted = torch.bmm(input_for_decoder, self.W_dec.unsqueeze(0))

        input_for_decoder_weighted_with_b = self.b_dec + input_for_decoder_weighted.squeeze()

        gate_input = torch.sigmoid(input_for_decoder_weighted_with_b[:self.dim_model])
        gate_forget = torch.sigmoid(input_for_decoder_weighted_with_b[self.dim_model:2 * self.dim_model])
        gate_output = torch.sigmoid(input_for_decoder_weighted_with_b[2 * self.dim_model:3 * self.dim_model])
        gate_pre_c = torch.tanh(input_for_decoder_weighted_with_b[3 * self.dim_model:])

        ct = gate_forget * ctm1 + gate_input * gate_pre_c
        ht = gate_output * torch.tanh(ct)

        ht_dropout = ht * self.drop_out_layer_gen

        return ct, ht_dropout, zt

    @property
    def initHidden(self):
        array = numpy.zeros((1, 1, self.dim_model))
        return (Variable(torch.DoubleTensor(array)),
                Variable(torch.DoubleTensor(array)))

    def softmax(self, x):
        x = x.squeeze()
        exp_x = torch.exp(x - torch.max(x).expand_as(x))
        return exp_x / (torch.sum(exp_x).expand_as(x))
