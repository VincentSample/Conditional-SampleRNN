import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np
import sys
    
class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels, nb_classes,
                 weight_norm):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels  
        ns_frame_samples = map(int, np.cumprod(frame_sizes))   #Frame size [16,4] Cumprod [16,64]
        # First frame_size = 16 , n_frame_samples = 16, Second frame_size = 4, n_frame_samples = 64
        # list of two frame_level RNNS
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, nb_classes, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ])
        #And one bottom tier with same frame_size as the top
        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

    #Number of samples taken into account in bottom frame_level_rnn
    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, nb_classes, weight_norm):
        super().__init__()
        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim
        self.nb_classes = nb_classes

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))

        self.input_expand = torch.nn.Conv1d(
            in_channels=n_frame_samples + self.nb_classes ,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.input_expand.weight)
        init.constant_(self.input_expand.bias, 0)
        if weight_norm:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        for i in range(n_rnn):
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal]
            )
            init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.upsampling = nn.LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )
        init.uniform_(
            self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        )
        init.constant_(self.upsampling.bias, 0)
        if weight_norm:
            self.upsampling.conv_t = torch.nn.utils.weight_norm(
                self.upsampling.conv_t
            )

    def forward(self, prev_samples, upper_tier_conditioning, hidden):
        (batch_size, _, _) = prev_samples.size()

        input = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)
        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()

        (output, hidden) = self.rnn(input, hidden)

        output = self.upsampling(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return (output, hidden)


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.input = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=frame_size,
            bias=False
        )
        init.kaiming_uniform_(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)

        self.hidden = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.hidden.weight)
        init.constant_(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)

        self.output = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        nn.lecun_uniform(self.output.weight)
        init.constant_(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()

        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        return F.log_softmax(x.view(-1, self.q_levels)) \
                .view(batch_size, -1, self.q_levels)


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
        (output, new_hidden) = rnn(
            prev_samples, upper_tier_conditioning, self.hidden_states[rnn]
        )
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model, nb_classes):
        super().__init__(model)
        self.nb_classes = nb_classes
    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()
        len_input_sequence = input_sequences.shape[1] - self.nb_classes
        input_sequences , cond = utils.size_splits(input_sequences,[len_input_sequence, self.nb_classes],1) #cond is one hot vector
        (batch_size, _) = input_sequences.size()
        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * utils.linear_dequantize(
                input_sequences[:, from_index : to_index],
                self.model.q_levels
            )

            #Duplicate cond so that each sample of size (rnn.n_frame_samples) is concatenated with the one hot vector
            
            number_of_repeat = int(prev_samples.shape[1]/rnn.n_frame_samples)
            cond_multi = cond[:,None,:]
            cond_multi = cond_multi.repeat(1,number_of_repeat,1)
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )
            
            cond_multi = cond_multi.float()
            prev_samples = torch.cat([prev_samples,cond_multi],2)
            
            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, upper_tier_conditioning
            )
        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences \
            [:, self.model.lookback - bottom_frame_size :]

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model, nb_classes, cuda=False, cond=False):
        super().__init__(model)
        self.cuda = cuda
        self.nb_classes = nb_classes
        self.cond = cond
    def __call__(self, n_seqs, seq_len):
        torch.backends.cudnn.enabled = True

        self.reset_hidden_states()

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        
        sequences = torch.LongTensor(n_seqs*self.nb_classes, self.model.lookback + seq_len) \
                         .fill_(utils.q_zero(self.model.q_levels))
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue
                
                prev_samples = torch.autograd.Variable(
                    2 * utils.linear_dequantize(
                        sequences[:, i - rnn.n_frame_samples : i],
                        self.model.q_levels
                    ).unsqueeze(1),
                    volatile=True
                )
                if self.cond == False: #no conditiong,generate all classes
                    one_hot_tensor = torch.tensor([]).float()
                    for label in range(self.nb_classes):
                        for nb_sample in range(n_seqs):
                            one_hot_vec =  utils.one_hot(torch.tensor([label]),self.nb_classes).float()
                            one_hot_tensor = torch.cat([one_hot_tensor,one_hot_vec])
                else:   #use custom one hot vector
                    one_hot_tensor = torch.tensor([]).float()
                    one_hot_vec = torch.tensor([self.cond]).float()
                    for label in range(self.nb_classes):
                        for nb_sample in range(n_seqs):
                            one_hot_tensor = torch.cat([one_hot_tensor,one_hot_vec])
                one_hot_tensor = one_hot_tensor[:,None,:]
                prev_samples = torch.cat([prev_samples,one_hot_tensor],2)
                if self.cuda:
                    prev_samples = prev_samples.cuda()

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                           .unsqueeze(1)

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning
                )

            prev_samples = torch.autograd.Variable(
                sequences[:, i - bottom_frame_size : i],
                volatile=True
            )
            if self.cuda:
                prev_samples = prev_samples.cuda()
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                                      .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback :]
