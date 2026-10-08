import random
from code.P1.micrograd.engine import *

class Neuron:
    def __init__(self, nin):  # nin表示接受几个输入，例如Neuron(3)表示接受三个输入
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
        # return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    # nin表示每个神经元的权重，nout表示一层有多少个神经元，nouts表示整个网络每一层的神经元个数
    # 例如，[3, 5, 3]表示第一层有三个神经元，第二层有五个，最后一层也有三个
    def __init__(self, nin, nouts):
        # sz表示尺寸，例如首个输入有2个特征，nouts=[3,4]，那么会得到列表[2,3,4]
        sz = [nin] + nouts
        # len(nouts)表示一共有几层，用i选定某一层。MLP为全连接，第一层有nout个神经元，会产生nout个输出，这些会组成一个列表，
        # 输入下一层的每一个神经元。由于Layer的输入nin必定和上一个的输出相等，所以满足
        # self.layers是一个列表，每一项表示一个layer对象。例如self.layers=[Layer(2,3),layer(3,1)]
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]