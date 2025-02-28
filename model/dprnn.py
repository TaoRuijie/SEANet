import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

class dprnn(nn.Module):
    def __init__(self, N = 256, L = 40, B = 64, H = 128, K = 100, R = 6):
        '''
        Module list: Encoder - Decoder - Extractor
        '''
        super(dprnn, self).__init__()
        self.N, self.L, self.B, self.H, self.K, self.R = N, L, B, H, K, R
        
        self.encoder   = Encoder(L, N)
        self.separator = Extractor(N, L, B, H, K, R)
        self.decoder_s = Decoder(N, L)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, visual, M = 1):
        mixture_w = self.encoder(mixture)
        est_speech = self.separator(mixture_w, visual, M)
        mixture_w   = mixture_w.repeat(self.R, 1, 1)
        est_speech  = torch.cat((est_speech), dim = 0)    
        est_speech  = self.decoder_s(mixture_w, est_speech)

        T_ori = mixture.size(-1)
        T_res = est_speech.size(-1)
        est_speech = F.pad(est_speech, (0, T_ori - T_res))

        return est_speech

class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)      # B, 1, L  = [2, 1, 64000]
        mixture_w = F.relu(self.conv1d_U(mixture)) # B, D, L' = [2, 256, 3199]
        return mixture_w

class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source

class Extractor(nn.Module):
    def __init__(self, N, L, B, H, K, R):
        '''
        Module list: VisualConv1D - RNN - Cross - Adder
        '''
        super(Extractor, self).__init__()
        self.N, self.L, self.B, self.H, self.K, self.R = N, L, B, H, K, R
        self.layer_norm         = nn.GroupNorm(1, N, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        self.v_ds               = nn.Linear(512, N, bias=False)
        stacks = []
        for x in range(5):
            stacks +=[VisualConv1D(V = N)]
        self.v_conv = nn.Sequential(*stacks)
        self.av_conv            = nn.Conv1d(B+N, B, 1, bias=False)
        self.rnn_s, self.rnn_n, self.cross = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        for i in range(R):
            self.rnn_s.append(RNN(B, H))
        self.adder_s = Adder(N, B)   

    def forward(self, x, visual, M):        
        M, N, P = x.size()
        visual = self.v_ds(visual)
        visual = visual.transpose(1,2)
        visual = self.v_conv(visual)
        visual = F.interpolate(visual, (P), mode='linear')
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)
        x = torch.cat((x, visual),1)
        x = self.av_conv(x)
        x, gap = self._Segmentation(x, self.K)

        all_x_s = []
        x_s = self.rnn_s[0](x)
        all_x_s.append(self.adder_s(x_s, gap, M, P))
        for i in range(1, self.R):
            x_s = self.rnn_s[i](x_s)
            all_x_s.append(self.adder_s(x_s, gap, M, P))
        return all_x_s

    def _padding(self, input, K):
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

class VisualConv1D(nn.Module):
    def __init__(self, V=256, H=512):
        super(VisualConv1D, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(V)
        conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(H)
        dsconv = nn.Conv1d(H, H, 3, stride=1, padding=1,dilation=1, groups=H, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(H)
        pw_conv = nn.Conv1d(H, V, 1, bias=False)
        self.net = nn.Sequential(relu_0, norm_0, conv1x1, relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x

class RNN(nn.Module):
    def __init__(self, B, H, rnn_type='LSTM', dropout=0, bidirectional=True):
        super(RNN, self).__init__()
        self.intra_rnn = getattr(nn, rnn_type)(B, H, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(B, H, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)        
        self.intra_norm = nn.GroupNorm(1, B, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, B, eps=1e-8)
        self.intra_linear = nn.Linear(H * 2, B)
        self.inter_linear = nn.Linear(H * 2, B)

    def forward(self, x):
        M, D, K, S = x.shape
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(M*S, K, D)
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(M*S*K, -1)).view(M*S, K, -1)
        intra_rnn = intra_rnn.view(M, S, K, D)
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)        
        intra_rnn = intra_rnn + x

        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(M*K, S, D)
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(M*S*K, -1)).view(M*K, S, -1)        
        inter_rnn = inter_rnn.view(M, K, S, D)        
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        out = inter_rnn + intra_rnn
        return out

class Adder(nn.Module):
    def __init__(self, N, B):
        super(Adder, self).__init__()
        self.N, self.B = N, B
        self.prelu        = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)

    def forward(self, x, gap, M, P):
        x = _over_add(x, gap)
        x = self.mask_conv1x1(self.prelu(x))
        x = F.relu(x.view(M, self.N, P))

        return x

'''
Module list: GLN, clone, over_add
'''

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def overlap_and_add(signal, frame_step):
    
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

def _over_add(input, gap):
    B, N, K, S = input.shape
    P = K // 2
    # [B, N, S, K]
    input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

    input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
    input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
    input = input1 + input2
    # [B, N, L]
    if gap > 0:
        input = input[:, :, :-gap]

    return input