import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer import cuda, link, function, initializers

class InstanceNormalization(link.Link):
    def __init__(self, nc, dtype=np.float32):
        super(InstanceNormalization, self).__init__()
        self.nc = nc
        self.dtype = dtype
        self.bn = None
        self.prev_batch = None
        # self._device_id =  cuda.get_device()

        self.add_param('gamma', nc, dtype=dtype)
        initializers.Uniform(self.gamma.data)

        self.add_param('beta', nc, dtype=dtype)
        initializers.Zero(self.beta.data)

    def __call__(self, x, test=True):
        n, c, h, w = x.shape
        assert(c == self.nc)
        if n != self.prev_batch:
            self.bn = L.BatchNormalization(n*c)
            # print(self.bn.device)
            # if self._device_id >= 0:
            self.bn.to_gpu()
            self.bn.gamma = F.tile(self.gamma, n)
            self.bn.beta = F.tile(self.beta, n)
            self.prev_batch = n

        x = F.reshape(x, (1, n*c, h, w))
        return F.reshape(self.bn(x), (n, c, h, w))

class ResidualBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(n_in, n_out, ksize, stride, 1, w),
            c2=L.Convolution2D(n_out, n_out, ksize, 1, 1, w),
            b1=L.BatchNormalization(n_out),
            b2=L.BatchNormalization(n_out)
        )

    def __call__(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        # if x.data.shape != h.data.shape:
        #     xp = chainer.cuda.get_array_module(x.data)
        #     n, c, hh, ww = x.data.shape
        #     pad_c = h.data.shape[1] - c
        #     p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
        #     p = chainer.Variable(p)
        #     x = F.concat((p, x))
        #     if x.data.shape[2:] != h.data.shape[2:]:
        #         x = F.average_pooling_2d(x, 1, 2)
        return h + x

class FastStyleNet(chainer.Chain):
    def __init__(self):
        super(FastStyleNet, self).__init__(
            c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
            r1=ResidualBlock(128, 128),
            r2=ResidualBlock(128, 128),
            r3=ResidualBlock(128, 128),
            r4=ResidualBlock(128, 128),
            r5=ResidualBlock(128, 128),
            d1=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            d2=L.Convolution2D(64, 32, 3, stride=1, pad=1),
            d3=L.Convolution2D(32, 3, 9, stride=1, pad=4),
            b1=L.BatchNormalization(32),
            b2=L.BatchNormalization(64),
            b3=L.BatchNormalization(128),
            b4=L.BatchNormalization(64),
            b5=L.BatchNormalization(32),
        )

    def __call__(self, x, test=False):
        h = F.relu(self.b1(self.c1(x)))
        h = F.relu(self.b2(self.c2(h)))
        h = F.relu(self.b3(self.c3(h)))
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = F.unpooling_2d(h, 2, 2)
        h = F.relu(self.b4(self.d1(h)))
        h = F.unpooling_2d(h, 2, 2)
        h = F.relu(self.b5(self.d2(h)))
        y = self.d3(h)
        # return (F.tanh(y)+1)*127.5
        return F.sigmoid(y) * 255.0

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)
        )
        self.train = False
        self.mean = np.asarray(120, dtype=np.float32)

    def preprocess(self, image):
        return np.rollaxis(image - self.mean, 2)

    def __call__(self, x):
        y1 = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.max_pooling_2d(y1, 2, stride=2)
        y2 = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.max_pooling_2d(y2, 2, stride=2)
        y3 = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        h = F.max_pooling_2d(y3, 2, stride=2)
        y4 = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h))))))
        return [y1, y2, y3, y4]