import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven import neuron, functional, layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters for LIF neuron
thresh = 0.6  # 0.3  # neuronal firing threshold
lens = 0.2  # 0.8#0.1  # 0.5  # hyper-parameters of approximation function
# decay = 0.9  # membrane decaying time constant
probs = 0.5  # dropout
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
		# out = F.relu6(x) / (self.scale)
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
		# pydevd.settrace(suspend=False, trace_only_current_thread=True)  # for debuger
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
								   nn.BatchNorm2d(64, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
								   nn.BatchNorm2d(64, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
								   nn.BatchNorm2d(128, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
								   nn.BatchNorm2d(128, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
								   nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1),
								   nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv7 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
								   nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv8 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
								   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv9 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
								   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
								   # nn.ReLU(inplace=True))
								   ReLUX(2))

		self.conv10 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
									nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
									# nn.ReLU(inplace=True))
									ReLUX(2))

		self.conv11 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
									nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
									# nn.ReLU(inplace=True))
									ReLUX(2))

		self.conv12 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
									nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
									# nn.ReLU(inplace=True))
									ReLUX(2))

		self.conv13 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
									nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
									# nn.ReLU(inplace=True))
									ReLUX(2))

		self.fc14 = nn.Sequential(nn.Linear(2 * 2 * 512, 512),
								  nn.BatchNorm1d(512, eps=1e-4, momentum=0.9))

		self.fc15 = nn.Sequential(nn.Linear(512, 512),
								  nn.BatchNorm1d(512, eps=1e-4, momentum=0.9))

		self.fc16 = nn.Linear(512, 10)

		self.relux = ReLUX(2)

	def forward(self, x):
		x = x.view(-1, 3, 32, 32)

		# Conv Layer
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)
		x6 = self.conv6(x5)
		x7 = self.conv7(x6)
		x8 = self.conv8(x7)
		x9 = self.conv9(x8)
		x10 = self.conv10(x9)
		x11 = self.conv11(x10)
		x12 = self.conv12(x11)
		x13 = self.conv13(x12)

		# FC Layers
		x13 = x13.view(x13.size(0), -1)
		x14 = self.relux(self.fc14(F.dropout(x13, p=0.2)))
		x15 = self.relux(self.fc15(F.dropout(x14, p=0.2)))
		out = self.fc16(F.dropout(x15, p=0.2))

		hidden_act = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)

		return hidden_act, F.log_softmax(out, dim=1)

class SpikingCNN(nn.Module):

	def __init__(self, Tsim):
		super(SpikingCNN, self).__init__()
		self.T = Tsim
		self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
		self.conv8 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv13 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		self.fc14 = nn.Linear(2 * 2 * 512, 512)
		self.fc15 = nn.Linear(512, 512)
		self.fc16 = nn.Linear(512, 10)

	def forward(self, x):
		batch_size = x.size(0)
		x = x.view(batch_size, 3, 32, 32)

		c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
		c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
		c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, 128, 32, 32, device=device)
		c4_mem = c4_spike = c4_sumspike = torch.zeros(batch_size, 128, 16, 16, device=device)
		c5_mem = c5_spike = c5_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
		c6_mem = c6_spike = c6_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
		c7_mem = c7_spike = c7_sumspike = torch.zeros(batch_size, 256, 8, 8, device=device)
		c8_mem = c8_spike = c8_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
		c9_mem = c9_spike = c9_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
		c10_mem = c10_spike = c10_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c11_mem = c11_spike = c11_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c12_mem = c12_spike = c12_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c13_mem = c13_spike = c13_sumspike = torch.zeros(batch_size, 512, 2, 2, device=device)
		h14_mem = h14_spike = h14_sumspike = torch.zeros(batch_size, 512, device=device)
		h15_mem = h15_spike = h15_sumspike = torch.zeros(batch_size, 512, device=device)
		h16_mem = h16_spike = h16_sumspike = h16_sumMem = torch.zeros(batch_size, 10, device=device)

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
			c9_input = c8_spike.detach()

			c9_mem, c9_spike = mem_update(self.conv9, c9_input, c9_mem, c9_spike)
			c9_sumspike += c9_spike
			c10_input = c9_spike.detach()

			c10_mem, c10_spike = mem_update(self.conv10, c10_input, c10_mem, c10_spike)
			c10_sumspike += c10_spike
			c11_input = c10_spike.detach()

			c11_mem, c11_spike = mem_update(self.conv11, c11_input, c11_mem, c11_spike)
			c11_sumspike += c11_spike
			c12_input = c11_spike.detach()

			c12_mem, c12_spike = mem_update(self.conv12, c12_input, c12_mem, c12_spike)
			c12_sumspike += c12_spike
			c13_input = c12_spike.detach()

			c13_mem, c13_spike = mem_update(self.conv13, c13_input, c13_mem, c13_spike)
			c13_sumspike += c13_spike
			h14_input = c13_spike.view(batch_size, -1).detach()

			h14_mem, h14_spike = mem_update(self.fc14, h14_input, h14_mem, h14_spike)
			h14_sumspike += h14_spike
			h15_input = h14_spike.detach()

			h15_mem, h15_spike = mem_update(self.fc15, h15_input, h15_mem, h15_spike)
			h15_sumspike += h15_spike
			h16_input = h15_spike.detach()

			h16_mem, h16_spike = mem_update(self.fc16, h16_input, h16_mem, h16_spike)
			h16_sumspike += h16_spike
			h16_sumMem += self.fc16(h16_input)

		c13_act = c13_sumspike.view(batch_size, -1)
		c1 = c1_sumspike / self.T
		c2 = c2_sumspike / self.T
		c3 = c3_sumspike / self.T
		c4 = c4_sumspike / self.T
		c5 = c5_sumspike / self.T
		c6 = c6_sumspike / self.T
		c7 = c7_sumspike / self.T
		c8 = c8_sumspike / self.T
		c9 = c9_sumspike / self.T
		c10 = c10_sumspike / self.T
		c11 = c11_sumspike / self.T
		c12 = c12_sumspike / self.T
		c13 = c13_act / self.T
		h14 = h14_sumspike / self.T
		h15 = h15_sumspike / self.T
		outputs = h16_sumMem / self.T

		return (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, h14, h15), outputs

class SpikingVGG16OL(nn.Module):
	def __init__(self, Tsim):
		super(SpikingVGG16OL, self).__init__()
		self.T = Tsim
		self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
		self.conv8 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv13 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		self.fc14 = nn.Linear(2 * 2 * 512, 512)
		self.fc15 = nn.Linear(512, 512)
		self.fc16 = nn.Linear(512, 10)

	def forward(self, x):
		batch_size = x.size(0)
		x = x.view(batch_size, 3, 32, 32)

		c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
		c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
		c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, 128, 32, 32, device=device)
		c4_mem = c4_spike = c4_sumspike = torch.zeros(batch_size, 128, 16, 16, device=device)
		c5_mem = c5_spike = c5_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
		c6_mem = c6_spike = c6_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
		c7_mem = c7_spike = c7_sumspike = torch.zeros(batch_size, 256, 8, 8, device=device)
		c8_mem = c8_spike = c8_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
		c9_mem = c9_spike = c9_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
		c10_mem = c10_spike = c10_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c11_mem = c11_spike = c11_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c12_mem = c12_spike = c12_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c13_mem = c13_spike = c13_sumspike = torch.zeros(batch_size, 512, 2, 2, device=device)
		h14_mem = h14_spike = h14_sumspike = torch.zeros(batch_size, 512, device=device)
		h15_mem = h15_spike = h15_sumspike = torch.zeros(batch_size, 512, device=device)
		h16_sumMem = torch.zeros(batch_size, 10, device=device)

		c1_sc_realtime = torch.zeros(self.T, batch_size, 64, 32, 32, device=device)
		c2_sc_realtime = torch.zeros(self.T, batch_size, 64, 32, 32, device=device)
		c3_sc_realtime = torch.zeros(self.T, batch_size, 128, 32, 32, device=device)
		c4_sc_realtime = torch.zeros(self.T, batch_size, 128, 16, 16, device=device)
		c5_sc_realtime = torch.zeros(self.T, batch_size, 256, 16, 16, device=device)
		c6_sc_realtime = torch.zeros(self.T, batch_size, 256, 16, 16, device=device)
		c7_sc_realtime = torch.zeros(self.T, batch_size, 256, 8, 8, device=device)
		c8_sc_realtime = torch.zeros(self.T, batch_size, 512, 8, 8, device=device)
		c9_sc_realtime = torch.zeros(self.T, batch_size, 512, 8, 8, device=device)
		c10_sc_realtime = torch.zeros(self.T, batch_size, 512, 4, 4, device=device)
		c11_sc_realtime = torch.zeros(self.T, batch_size, 512, 4, 4, device=device)
		c12_sc_realtime = torch.zeros(self.T, batch_size, 512, 4, 4, device=device)
		c13_sc_realtime = torch.zeros(self.T, batch_size, 512, 2, 2, device=device)
		h14_sc_realtime = torch.zeros(self.T, batch_size, 512, device=device)
		h15_sc_realtime = torch.zeros(self.T, batch_size, 512, device=device)
		h16_sc_realtime = torch.zeros(self.T, batch_size, 10, device=device)

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
			c9_input = c8_spike.detach()

			c9_mem, c9_spike = mem_update(self.conv9, c9_input, c9_mem.detach(), c9_spike.detach())
			c9_sumspike = c9_sumspike.detach() + c9_spike
			c9_sc_realtime[step] = c9_sumspike
			c10_input = c9_spike.detach()

			c10_mem, c10_spike = mem_update(self.conv10, c10_input, c10_mem.detach(), c10_spike.detach())
			c10_sumspike = c10_sumspike.detach() + c10_spike
			c10_sc_realtime[step] = c10_sumspike
			c11_input = c10_spike.detach()

			c11_mem, c11_spike = mem_update(self.conv11, c11_input, c11_mem.detach(), c11_spike.detach())
			c11_sumspike = c11_sumspike.detach() + c11_spike
			c11_sc_realtime[step] = c11_sumspike
			c12_input = c11_spike.detach()

			c12_mem, c12_spike = mem_update(self.conv12, c12_input, c12_mem.detach(), c12_spike.detach())
			c12_sumspike = c12_sumspike.detach() + c12_spike
			c12_sc_realtime[step] = c12_sumspike
			c13_input = c12_spike.detach()

			c13_mem, c13_spike = mem_update(self.conv13, c13_input, c13_mem.detach(), c13_spike.detach())
			c13_sumspike = c13_sumspike.detach() + c13_spike
			c13_sc_realtime[step] = c13_sumspike
			h14_input = c13_spike.view(batch_size, -1).detach()

			h14_mem, h14_spike = mem_update(self.fc14, h14_input, h14_mem.detach(), h14_spike.detach())
			h14_sumspike = h14_sumspike.detach() + h14_spike
			h14_sc_realtime[step] = h14_sumspike
			h15_input = h14_spike.detach()

			h15_mem, h15_spike = mem_update(self.fc15, h15_input, h15_mem.detach(), h15_spike.detach())
			h15_sumspike = h15_sumspike.detach() + h15_spike
			h15_sc_realtime[step] = h15_sumspike
			h16_input = h15_spike.detach()

			h16_sumMem = h16_sumMem.detach() + self.fc16(h16_input)
			h16_sc_realtime[step] = h16_sumMem

		c13_sc_realtime = c13_sc_realtime.view(self.T, batch_size, -1)
		c1 = c1_sc_realtime
		c2 = c2_sc_realtime
		c3 = c3_sc_realtime
		c4 = c4_sc_realtime
		c5 = c5_sc_realtime
		c6 = c6_sc_realtime
		c7 = c7_sc_realtime
		c8 = c8_sc_realtime
		c9 = c9_sc_realtime
		c10 = c10_sc_realtime
		c11 = c11_sc_realtime
		c12 = c12_sc_realtime
		c13 = c13_sc_realtime
		h14 = h14_sc_realtime
		h15 = h15_sc_realtime
		outputs = h16_sc_realtime

		return (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, h14, h15), outputs

class SpikingE2E(nn.Module):
	def __init__(self, Tsim):
		super(SpikingE2E, self).__init__()
		self.T = Tsim
		self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
		self.conv8 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv13 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		self.fc14 = nn.Linear(2 * 2 * 512, 512)
		self.fc15 = nn.Linear(512, 512)
		self.fc16 = nn.Linear(512, 10)

	def forward(self, x):
		batch_size = x.size(0)
		x = x.view(batch_size, 3, 32, 32)

		c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
		c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, 64, 32, 32, device=device)
		c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, 128, 32, 32, device=device)
		c4_mem = c4_spike = c4_sumspike = torch.zeros(batch_size, 128, 16, 16, device=device)
		c5_mem = c5_spike = c5_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
		c6_mem = c6_spike = c6_sumspike = torch.zeros(batch_size, 256, 16, 16, device=device)
		c7_mem = c7_spike = c7_sumspike = torch.zeros(batch_size, 256, 8, 8, device=device)
		c8_mem = c8_spike = c8_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
		c9_mem = c9_spike = c9_sumspike = torch.zeros(batch_size, 512, 8, 8, device=device)
		c10_mem = c10_spike = c10_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c11_mem = c11_spike = c11_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c12_mem = c12_spike = c12_sumspike = torch.zeros(batch_size, 512, 4, 4, device=device)
		c13_mem = c13_spike = c13_sumspike = torch.zeros(batch_size, 512, 2, 2, device=device)
		h14_mem = h14_spike = h14_sumspike = torch.zeros(batch_size, 512, device=device)
		h15_mem = h15_spike = h15_sumspike = torch.zeros(batch_size, 512, device=device)
		h16_mem = h16_spike = h16_sumMem = torch.zeros(batch_size, 10, device=device)

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
			c9_mem, c9_spike = mem_update(self.conv9, c8_spike, c9_mem, c9_spike)
			c10_mem, c10_spike = mem_update(self.conv10, c9_spike, c10_mem, c10_spike)
			c11_mem, c11_spike = mem_update(self.conv11, c10_spike, c11_mem, c11_spike)
			c12_mem, c12_spike = mem_update(self.conv12, c11_spike, c12_mem, c12_spike)
			c13_mem, c13_spike = mem_update(self.conv13, c12_spike, c13_mem, c13_spike)

			c14_input = c13_spike.view(batch_size, -1)
			h14_mem, h14_spike = mem_update(self.fc14, c14_input, h14_mem, h14_spike)
			h15_mem, h15_spike = mem_update(self.fc15, h14_spike, h15_mem, h15_spike)
			# h11_mem, h11_spike = mem_update(self.fc11, h10_spike, h11_mem, h11_spike)
			# h11_sumspike += h11_spike
			h16_sumMem += self.fc16(h15_spike)

		outputs = h16_sumMem / self.T

		return None, outputs


#############################################################################################
## SpikingJelly version
#############################################################################################
class SJ_SpikingVgg16_cupy(nn.Module):
	def __init__(self, Tsim, tau=10.0, decay_input=False, v_threshold=0.6,
				 v_reset=None, surrogate_function=boxcar.apply):
		super(SJ_SpikingVgg16_cupy, self).__init__()
		self.T = Tsim

		self.conv1 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(3, 64, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv2 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv3 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(64, 128, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv4 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(128, 128, 3, stride=2, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv5 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(128, 256, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv6 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(256, 256, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv7 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(256, 256, 3, stride=2, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv8 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(256, 512, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv9 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv10 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=2, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv11 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv12 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=1, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.conv13 = nn.Sequential(
			layer.SeqToANNContainer(nn.Conv2d(512, 512, 3, stride=2, padding=1)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.fc14 = nn.Sequential(
			layer.SeqToANNContainer(nn.Linear(2 * 2 * 512, 512)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.fc15 = nn.Sequential(
			layer.SeqToANNContainer(nn.Linear(512, 512)),
			neuron.MultiStepLIFNode(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
									backend='cupy'))  # , surrogate_function= boxcar.apply), )
		self.fc16 = layer.SeqToANNContainer(nn.Linear(512, 10))

	def reset_model(self):
		functional.reset_net(self.conv1)
		functional.reset_net(self.conv2)
		functional.reset_net(self.conv3)
		functional.reset_net(self.conv4)
		functional.reset_net(self.conv5)
		functional.reset_net(self.conv6)
		functional.reset_net(self.conv7)
		functional.reset_net(self.conv8)
		functional.reset_net(self.conv9)
		functional.reset_net(self.conv10)
		functional.reset_net(self.conv11)
		functional.reset_net(self.conv12)
		functional.reset_net(self.conv13)
		functional.reset_net(self.fc14)
		functional.reset_net(self.fc15)

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
		c8 = c8_spike.mean(0)
		c9_input = c8_spike.detach()

		c9_spike = self.conv9(c9_input)
		c9 = c9_spike.mean(0)
		c10_input = c9_spike.detach()

		c10_spike = self.conv10(c10_input)
		c10 = c10_spike.mean(0)
		c11_input = c10_spike.detach()

		c11_spike = self.conv11(c11_input)
		c11 = c11_spike.mean(0)
		c12_input = c11_spike.detach()

		c12_spike = self.conv12(c12_input)
		c12 = c12_spike.mean(0)
		c13_input = c12_spike.detach()

		c13_spike = self.conv13(c13_input)
		c13 = c13_spike.mean(0).view(batch_size, -1)
		h14_input = c13_spike.view(self.T, batch_size, -1).detach()

		h14_spike = self.fc14(h14_input)
		h14 = h14_spike.mean(0)
		h15_input = h14_spike.detach()

		h15_spike = self.fc15(h15_input)
		h15 = h15_spike.mean(0)
		h16_input = h15_spike.detach()

		h16_sumMem = self.fc16(h16_input)
		outputs = h16_sumMem.mean(0)

		return (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, h14, h15), outputs
