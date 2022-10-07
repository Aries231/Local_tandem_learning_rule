import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, layer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc9', 'fc10', 'fc11']

# Hyper parameters for LIF neuron
thresh = 0.6
lens = 0.2
tau = 10  # membrane decaying time constant
decay = torch.exp(torch.tensor(-1. * 1 / tau))


class ReLUX(nn.Module):
    """ Relu function with max value clapped """

    def __init__(self, max_value: float = 1.0, dt=0):
        super(ReLUX, self).__init__()
        self.max_value = float(max_value)
        self.scale = 6.0 / self.max_value
        self.dt = dt

    def forward(self, x):
        out = F.relu6(x * self.scale) / (self.scale)
        out = torch.clamp(out, min=self.dt)
        return out

class ActFun(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # function
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

# membrane potential update
def mem_update(opts, x, mem, spike):
    mem = mem * decay * (1. - spike) + opts(x)
    spike = act_fun(mem)
    return mem, spike

act_fun = ActFun.apply

class boxcar(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # function
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

class CNN_ReluX(nn.Module):

    def __init__(self):
        super(CNN_ReluX, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(64, eps=1e-4, momentum=0.9))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(128, eps=1e-4, momentum=0.9))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(256, eps=1e-4, momentum=0.9))

        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(256, eps=1e-4, momentum=0.9))

        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

        self.conv7 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

        self.conv8 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

        self.fc9 = nn.Sequential(nn.Linear(2 * 2 * 512, 512),
                                 nn.BatchNorm1d(512, eps=1e-4, momentum=0.9))

        self.fc10 = nn.Sequential(nn.Linear(512, 512),
                                  nn.BatchNorm1d(512, eps=1e-4, momentum=0.9))

        self.fc11 = nn.Linear(512, 10)

        self.relux = ReLUX(2)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        # Conv Layer
        x1 = self.relux(self.conv1(x))
        x2 = self.relux(self.conv2(x1))
        x3 = self.relux(self.conv3(x2))
        x4 = self.relux(self.conv4(x3))
        x5 = self.relux(self.conv5(x4))
        x6 = self.relux(self.conv6(x5))
        x7 = self.relux(self.conv7(x6))
        x8 = self.relux(self.conv8(x7))

        # FC Layers
        x8 = x8.view(x8.size(0), -1)
        x9 = self.relux(self.fc9(F.dropout(x8, p=0.2)))
        x10 = self.relux(self.fc10(F.dropout(x9, p=0.2)))
        out = self.fc11(F.dropout(x10, p=0.2))

        hidden_act = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

        return hidden_act, F.log_softmax(out, dim=1)

class SpikingCNN(nn.Module):

    def __init__(self, Tsim):
        super(SpikingCNN, self).__init__()
        self.T = Tsim
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.fc9 = nn.Linear(2 * 2 * 512, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
        c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, 128, 16, 16, device=device)
        c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
        c4_mem = c4_spike = c4_sumspike = torch.zeros(batch_size, 256, 8, 8, device=device)
        c5_mem = c5_spike = c5_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
        c6_mem = c6_spike = c6_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c7_mem = c7_spike = c7_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c8_mem = c8_spike = c8_sumspike = torch.zeros(batch_size, 512, 2, 2, device=device)
        h9_mem = h9_spike = h9_sumspike = torch.zeros(batch_size, 512, device=device)
        h10_mem = h10_spike = h10_sumspike = torch.zeros(batch_size, 512, device=device)
        h11_mem = h11_spike = h11_sumspike = h11_sumMem = torch.zeros(batch_size, 10, device=device)

        for step in range(self.T):
            x = x.view(x.size(0), 3, 32, 32)
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            c1_sumspike += c1_spike
            c2_input = c1_spike.detach()

            c2_mem, c2_spike = mem_update(self.conv2, c2_input, c2_mem, c2_spike)
            c2_sumspike += c2_spike
            c3_input = c2_spike.detach()

            c3_mem, c3_spike = mem_update(self.conv3, c3_input, c3_mem, c3_spike)
            c3_sumspike += c3_spike
            c4_input = c3_spike.detach()

            c4_mem, c4_spike = mem_update(self.conv4, c4_input, c4_mem, c4_spike)
            c4_sumspike += c4_spike
            c5_input = c4_spike.detach()

            c5_mem, c5_spike = mem_update(self.conv5, c5_input, c5_mem, c5_spike)
            c5_sumspike += c5_spike
            c6_input = c5_spike.detach()

            c6_mem, c6_spike = mem_update(self.conv6, c6_input, c6_mem, c6_spike)
            c6_sumspike += c6_spike
            c7_input = c6_spike.detach()

            c7_mem, c7_spike = mem_update(self.conv7, c7_input, c7_mem, c7_spike)
            c7_sumspike += c7_spike
            c8_input = c7_spike.detach()

            c8_mem, c8_spike = mem_update(self.conv8, c8_input, c8_mem, c8_spike)
            c8_sumspike += c8_spike
            h9_input = c8_spike.view(batch_size, -1).detach()

            h9_mem, h9_spike = mem_update(self.fc9, h9_input, h9_mem, h9_spike)
            h9_sumspike += h9_spike
            h10_input = h9_spike.detach()

            h10_mem, h10_spike = mem_update(self.fc10, h10_input, h10_mem, h10_spike)
            h10_sumspike += h10_spike
            h11_input = h10_spike.detach()

            h11_mem, h11_spike = mem_update(self.fc11, h11_input, h11_mem, h11_spike)
            h11_sumspike += h11_spike
            h11_sumMem += self.fc11(h11_input)

        c8_act = c8_sumspike.view(batch_size, -1)
        c1 = c1_sumspike / self.T
        c2 = c2_sumspike / self.T
        c3 = c3_sumspike / self.T
        c4 = c4_sumspike / self.T
        c5 = c5_sumspike / self.T
        c6 = c6_sumspike / self.T
        c7 = c7_sumspike / self.T
        c8 = c8_act / self.T
        h9 = h9_sumspike / self.T
        h10 = h10_sumspike / self.T
        outputs = h11_sumMem / self.T

        return (c1, c2, c3, c4, c5, c6, c7, c8, h9, h10), outputs

class SpikingE2E(nn.Module):
    def __init__(self, Tsim):
        super(SpikingE2E, self).__init__()
        self.T = Tsim
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.fc9 = nn.Linear(2 * 2 * 512, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = torch.zeros(batch_size, 64, 32, 32, device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, 128, 16, 16, device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, 256, 16, 16, device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, 256, 8, 8, device=device)
        c5_mem = c5_spike = torch.zeros(batch_size, 512, 8, 8, device=device)
        c6_mem = c6_spike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c7_mem = c7_spike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c8_mem = c8_spike = torch.zeros(batch_size, 512, 2, 2, device=device)
        h9_mem = h9_spike = torch.zeros(batch_size, 512, device=device)
        h10_mem = h10_spike = torch.zeros(batch_size, 512, device=device)
        h11_mem = h11_spike = h11_sumMem = torch.zeros(batch_size, 10, device=device)

        for step in range(self.T):
            x = x.view(x.size(0), 3, 32, 32)
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            c2_mem, c2_spike = mem_update(self.conv2, c1_spike, c2_mem, c2_spike)
            c3_mem, c3_spike = mem_update(self.conv3, c2_spike, c3_mem, c3_spike)
            c4_mem, c4_spike = mem_update(self.conv4, c3_spike, c4_mem, c4_spike)
            c5_mem, c5_spike = mem_update(self.conv5, c4_spike, c5_mem, c5_spike)
            c6_mem, c6_spike = mem_update(self.conv6, c5_spike, c6_mem, c6_spike)
            c7_mem, c7_spike = mem_update(self.conv7, c6_spike, c7_mem, c7_spike)
            c8_mem, c8_spike = mem_update(self.conv8, c7_spike, c8_mem, c8_spike)
            c9_input = c8_spike.view(batch_size, -1)
            h9_mem, h9_spike = mem_update(self.fc9, c9_input, h9_mem, h9_spike)
            h10_mem, h10_spike = mem_update(self.fc10, h9_spike, h10_mem, h10_spike)
            # h11_mem, h11_spike = mem_update(self.fc11, h10_spike, h11_mem, h11_spike)
            # h11_sumspike += h11_spike
            h11_sumMem += self.fc11(h10_spike)

        outputs = h11_sumMem / self.T

        return None, outputs

class SpikingVGG11OL(nn.Module):
    def __init__(self, Tsim):
        super(SpikingVGG11OL, self).__init__()
        self.T = Tsim
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.fc9 = nn.Linear(2 * 2 * 512, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
        c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, 128, 16, 16, device=device)
        c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
        c4_mem = c4_spike = c4_sumspike = torch.zeros(batch_size, 256, 8, 8, device=device)
        c5_mem = c5_spike = c5_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
        c6_mem = c6_spike = c6_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c7_mem = c7_spike = c7_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c8_mem = c8_spike = c8_sumspike = torch.zeros(batch_size, 512, 2, 2, device=device)
        h9_mem = h9_spike = h9_sumspike = torch.zeros(batch_size, 512, device=device)
        h10_mem = h10_spike = h10_sumspike = torch.zeros(batch_size, 512, device=device)
        h11_mem = h11_spike = h11_sumspike = h11_sumMem = torch.zeros(batch_size, 10, device=device)

        c1_sc_realtime = torch.zeros(self.T, batch_size, 64, 32, 32, device=device)
        c2_sc_realtime = torch.zeros(self.T, batch_size, 128, 16, 16, device=device)
        c3_sc_realtime = torch.zeros(self.T, batch_size, 256, 16, 16, device=device)
        c4_sc_realtime = torch.zeros(self.T, batch_size, 256, 8, 8, device=device)
        c5_sc_realtime = torch.zeros(self.T, batch_size, 512, 8, 8, device=device)
        c6_sc_realtime = torch.zeros(self.T, batch_size, 512, 4, 4, device=device)
        c7_sc_realtime = torch.zeros(self.T, batch_size, 512, 4, 4, device=device)
        c8_sc_realtime = torch.zeros(self.T, batch_size, 512, 2, 2, device=device)
        h9_sc_realtime = torch.zeros(self.T, batch_size, 512, device=device)
        h10_sc_realtime = torch.zeros(self.T, batch_size, 512, device=device)
        h11_sc_realtime = torch.zeros(self.T, batch_size, 10, device=device)

        for step in range(self.T):
            x = x.view(x.size(0), 3, 32, 32)

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem.detach(), c1_spike.detach())
            c1_sumspike = c1_sumspike.detach() + c1_spike
            c1_sc_realtime[step] = c1_sumspike
            c2_input = c1_spike.detach()

            c2_mem, c2_spike = mem_update(self.conv2, c2_input, c2_mem.detach(), c2_spike.detach())
            c2_sumspike = c2_sumspike.detach() + c2_spike
            c2_sc_realtime[step] = c2_sumspike
            c3_input = c2_spike.detach()

            c3_mem, c3_spike = mem_update(self.conv3, c3_input, c3_mem.detach(), c3_spike.detach())
            c3_sumspike = c3_sumspike.detach() + c3_spike
            c3_sc_realtime[step] = c3_sumspike
            c4_input = c3_spike.detach()

            c4_mem, c4_spike = mem_update(self.conv4, c4_input, c4_mem.detach(), c4_spike.detach())
            c4_sumspike = c4_sumspike.detach() + c4_spike
            c4_sc_realtime[step] = c4_sumspike
            c5_input = c4_spike.detach()

            c5_mem, c5_spike = mem_update(self.conv5, c5_input, c5_mem.detach(), c5_spike.detach())
            c5_sumspike = c5_sumspike.detach() + c5_spike
            c5_sc_realtime[step] = c5_sumspike
            c6_input = c5_spike.detach()

            c6_mem, c6_spike = mem_update(self.conv6, c6_input, c6_mem.detach(), c6_spike.detach())
            c6_sumspike = c6_sumspike.detach() + c6_spike
            c6_sc_realtime[step] = c6_sumspike
            c7_input = c6_spike.detach()

            c7_mem, c7_spike = mem_update(self.conv7, c7_input, c7_mem.detach(), c7_spike.detach())
            c7_sumspike = c7_sumspike.detach() + c7_spike
            c7_sc_realtime[step] = c7_sumspike
            c8_input = c7_spike.detach()

            c8_mem, c8_spike = mem_update(self.conv8, c8_input, c8_mem.detach(), c8_spike.detach())
            c8_sumspike = c8_sumspike.detach() + c8_spike
            c8_sc_realtime[step] = c8_sumspike
            h9_input = c8_spike.view(batch_size, -1).detach()

            h9_mem, h9_spike = mem_update(self.fc9, h9_input, h9_mem.detach(), h9_spike.detach())
            h9_sumspike = h9_sumspike.detach() + h9_spike
            h9_sc_realtime[step] = h9_sumspike
            h10_input = h9_spike.detach()

            h10_mem, h10_spike = mem_update(self.fc10, h10_input, h10_mem.detach(), h10_spike.detach())
            h10_sumspike = h10_sumspike.detach() + h10_spike
            h10_sc_realtime[step] = h10_sumspike
            h11_input = h10_spike.detach()

            h11_mem, h11_spike = mem_update(self.fc11, h11_input, h11_mem.detach(), h11_spike.detach())
            # h11_sumspike = h11_sumspike.detach() + h11_spike
            h11_sumMem = h11_sumMem.detach() + self.fc11(h11_input)
            h11_sc_realtime[step] = h11_sumMem

        c8_sc_realtime = c8_sc_realtime.view(self.T, batch_size, -1)
        c1 = c1_sc_realtime
        c2 = c2_sc_realtime
        c3 = c3_sc_realtime
        c4 = c4_sc_realtime
        c5 = c5_sc_realtime
        c6 = c6_sc_realtime
        c7 = c7_sc_realtime
        c8 = c8_sc_realtime
        h9 = h9_sc_realtime
        h10 = h10_sc_realtime
        outputs = h11_sc_realtime

        return (c1, c2, c3, c4, c5, c6, c7, c8, h9, h10), outputs

#############################################################################################
## SpikingJelly version
#############################################################################################
class SJ_SpikingVgg11(nn.Module):

    def __init__(self, Tsim, tau=10.0, decay_input=False, v_threshold=0.6,
                 v_reset=None, surrogate_function= boxcar.apply):
        super(SJ_SpikingVgg11, self).__init__()
        self.T = Tsim
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.fc9 = nn.Sequential(
            nn.Linear(2 * 2 * 512, 512),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.fc10 = nn.Sequential(
            nn.Linear(512, 512),
            neuron.LIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset))#, surrogate_function= boxcar.apply), )
        self.fc11 = nn.Linear(512, 10)

    def reset_model(self):
        functional.reset_net(self.conv1)
        functional.reset_net(self.conv2)
        functional.reset_net(self.conv3)
        functional.reset_net(self.conv4)
        functional.reset_net(self.conv5)
        functional.reset_net(self.conv6)
        functional.reset_net(self.conv7)
        functional.reset_net(self.conv8)
        functional.reset_net(self.fc9)
        functional.reset_net(self.fc10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
        c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, 128, 16, 16, device=device)
        c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
        c4_mem = c4_spike = c4_sumspike = torch.zeros(batch_size, 256, 8, 8, device=device)
        c5_mem = c5_spike = c5_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
        c6_mem = c6_spike = c6_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c7_mem = c7_spike = c7_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
        c8_mem = c8_spike = c8_sumspike = torch.zeros(batch_size, 512, 2, 2, device=device)
        h9_mem = h9_spike = h9_sumspike = torch.zeros(batch_size, 512, device=device)
        h10_mem = h10_spike = h10_sumspike = torch.zeros(batch_size, 512, device=device)
        h11_mem = h11_spike = h11_sumspike = h11_sumMem = torch.zeros(batch_size, 10, device=device)

        for step in range(self.T):
            #x = x.view(x.size(0), 3, 32, 32)
            c1_spike = self.conv1(x.float())
            c1_sumspike += c1_spike
            c2_input = c1_spike.detach()

            c2_spike = self.conv2(c2_input)
            c2_sumspike += c2_spike
            c3_input = c2_spike.detach()

            c3_spike = self.conv3(c3_input)
            c3_sumspike += c3_spike
            c4_input = c3_spike.detach()

            c4_spike = self.conv4(c4_input)
            c4_sumspike += c4_spike
            c5_input = c4_spike.detach()

            c5_spike = self.conv5(c5_input)
            c5_sumspike += c5_spike
            c6_input = c5_spike.detach()

            c6_spike = self.conv6(c6_input)
            c6_sumspike += c6_spike
            c7_input = c6_spike.detach()

            c7_spike = self.conv7(c7_input)
            c7_sumspike += c7_spike
            c8_input = c7_spike.detach()

            c8_spike = self.conv8(c8_input)
            c8_sumspike += c8_spike
            h9_input = c8_spike.view(batch_size, -1).detach()

            h9_spike = self.fc9(h9_input)
            h9_sumspike += h9_spike
            h10_input = h9_spike.detach()

            h10_spike = self.fc10(h10_input)
            h10_sumspike += h10_spike
            h11_input = h10_spike.detach()

            h11_sumMem += self.fc11(h11_input)

        c8_act = c8_sumspike.view(batch_size, -1)
        c1 = c1_sumspike / self.T
        c2 = c2_sumspike / self.T
        c3 = c3_sumspike / self.T
        c4 = c4_sumspike / self.T
        c5 = c5_sumspike / self.T
        c6 = c6_sumspike / self.T
        c7 = c7_sumspike / self.T
        c8 = c8_act / self.T
        h9 = h9_sumspike / self.T
        h10 = h10_sumspike / self.T
        outputs = h11_sumMem / self.T

        return (c1, c2, c3, c4, c5, c6, c7, c8, h9, h10), outputs

class SJ_SpikingVgg11_cupy(nn.Module):

    def __init__(self, Tsim, tau=10.0, decay_input=False, v_threshold=0.6,
                 v_reset=None, surrogate_function= boxcar.apply):
        super(SJ_SpikingVgg11_cupy, self).__init__()
        self.T = Tsim
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, 64, 3, stride=1, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv2 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv3 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv4 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(256, 256, 3, stride=2, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv5 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(256, 512, 3, stride=1, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv6 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=2, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv7 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=1, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.conv8 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=2, padding=1)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.fc9 = nn.Sequential(
            layer.SeqToANNContainer(nn.Linear(2 * 2 * 512, 512)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.fc10 = nn.Sequential(
            layer.SeqToANNContainer(nn.Linear(512, 512)),
            neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset, backend='cupy'))#, surrogate_function= boxcar.apply), )
        self.fc11 = layer.SeqToANNContainer(nn.Linear(512, 10))

    def reset_model(self):
        functional.reset_net(self.conv1)
        functional.reset_net(self.conv2)
        functional.reset_net(self.conv3)
        functional.reset_net(self.conv4)
        functional.reset_net(self.conv5)
        functional.reset_net(self.conv6)
        functional.reset_net(self.conv7)
        functional.reset_net(self.conv8)
        functional.reset_net(self.fc9)
        functional.reset_net(self.fc10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        c1_spike = self.conv1(x_seq.float())
        c1 = c1_spike.mean(0)
        c2_input = c1_spike.detach()

        c2_spike = self.conv2(c2_input)
        c2 = c2_spike.mean(0)
        c3_input = c2_spike.detach()

        c3_spike = self.conv3(c3_input)
        c3 = c3_spike.mean(0)
        c4_input = c3_spike.detach()

        c4_spike = self.conv4(c4_input)
        c4 = c4_spike.mean(0)
        c5_input = c4_spike.detach()

        c5_spike = self.conv5(c5_input)
        c5 = c5_spike.mean(0)
        c6_input = c5_spike.detach()

        c6_spike = self.conv6(c6_input)
        c6 = c6_spike.mean(0)
        c7_input = c6_spike.detach()

        c7_spike = self.conv7(c7_input)
        c7 = c7_spike.mean(0)
        c8_input = c7_spike.detach()

        c8_spike = self.conv8(c8_input)
        c8 = c8_spike.mean(0).view(batch_size, -1)
        h9_input = c8_spike.view(self.T, batch_size, -1).detach()

        h9_spike = self.fc9(h9_input)
        h9 = h9_spike.mean(0)
        h10_input = h9_spike.detach()

        h10_spike = self.fc10(h10_input)
        h10 = h10_spike.mean(0)
        h11_input = h10_spike.detach()

        h11_sumMem = self.fc11(h11_input)
        outputs = h11_sumMem.mean(0)

        return (c1, c2, c3, c4, c5, c6, c7, c8, h9, h10), outputs
