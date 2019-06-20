from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from  modules.utils import sample_weights
import numpy


class EncoderRNN_Forward(nn.Module):
    def __init__(self, model_settings):
        super(EncoderRNN_Forward, self).__init__()
        self.dim_lang = model_settings['dim_lang']
        self.dim_model = numpy.asscalar(model_settings['dim_model'])

        # params
        self.Emb_enc_forward = sample_weights(self.dim_lang, self.dim_model)
        self.b_enc_forward = nn.Parameter(torch.DoubleTensor(numpy.zeros(4 * self.dim_model, )))
        self.W_enc_forward = sample_weights(2 * self.dim_model, 4 * self.dim_model)

    def forward(self, xt, htm1, ctm1): #implemented a LSTM (could use the built-in method but we experimented different LSTM implementations)
        input_encoder = torch.cat((xt.squeeze(), htm1.squeeze()))

        input_weighted = torch.bmm(input_encoder.unsqueeze(0).unsqueeze(0), self.W_enc_forward.unsqueeze(0))

        post_transform = self.b_enc_forward + input_weighted.squeeze()

        gate_input = torch.sigmoid(post_transform[:self.dim_model])
        gate_forget = torch.sigmoid(post_transform[self.dim_model:2 * self.dim_model])
        gate_output = torch.sigmoid(post_transform[2 * self.dim_model:3 * self.dim_model])
        gate_pre_c = torch.tanh(post_transform[3 * self.dim_model:])

        ct = gate_forget * ctm1 + gate_input * gate_pre_c
        ht = gate_output * torch.tanh(ct)

        return ht, ct

    @property
    def initHidden(self):
        array = numpy.zeros((1, 1, self.dim_model))

        return (Variable(torch.DoubleTensor(array)),
                Variable(torch.DoubleTensor(array)))


class EncoderRNN_Backward(nn.Module):
    def __init__(self, model_settings):
        super(EncoderRNN_Backward, self).__init__()
        self.dim_lang = model_settings['dim_lang']
        self.dim_model = numpy.asscalar(model_settings['dim_model'])

        # params
        self.Emb_enc_backward = sample_weights(self.dim_lang, self.dim_model)
        self.b_enc_backward = nn.Parameter(torch.DoubleTensor(numpy.zeros(4 * self.dim_model, )))
        self.W_enc_backward = sample_weights(2 * self.dim_model, 4 * self.dim_model)

    def forward(self, xt, htm1, ctm1): #implemented a LSTM (could use the built-in method but we experimented different LSTM implementations)
        input_encoder = torch.cat((xt.squeeze(), htm1.squeeze()))

        input_weighted = torch.bmm(input_encoder.unsqueeze(0).unsqueeze(0), self.W_enc_backward.unsqueeze(0))

        post_transform = self.b_enc_backward + input_weighted.squeeze()

        gate_input = torch.sigmoid(post_transform[:self.dim_model])
        gate_forget = torch.sigmoid(post_transform[self.dim_model:2 * self.dim_model])
        gate_output = torch.sigmoid(post_transform[2 * self.dim_model:3 * self.dim_model])
        gate_pre_c = torch.tanh(post_transform[3 * self.dim_model:])

        ct = gate_forget * ctm1 + gate_input * gate_pre_c
        ht = gate_output * torch.tanh(ct)

        return ht, ct

    @property
    def initHidden(self):
        array = numpy.zeros((1, 1, self.dim_model))

        return (Variable(torch.DoubleTensor(array)),
                Variable(torch.DoubleTensor(array)))
