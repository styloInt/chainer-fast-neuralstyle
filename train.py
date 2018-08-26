import numpy as np
import os
import argparse
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from chainer import cuda, Variable, optimizers, serializers
from net import *
import re

def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

def total_variation_regularization(x, beta=1):
    xp = cuda.get_array_module(x.data)
    wh = Variable(xp.array([[[[1],[-1]],[[1],[-1]],[[1],[-1]]]], dtype=xp.float32))
    ww = Variable(xp.array([[[[1, -1]],[[1, -1]],[[1, -1]]]], dtype=xp.float32))
    tvh = lambda x: F.convolution_2d(x, W=wh, pad=1)
    tvw = lambda x: F.convolution_2d(x, W=ww, pad=1)

    dh = tvh(x)
    dw = tvw(x)
    tv = (F.sum(dh**2) + F.sum(dw**2)) ** (beta / 2.)
    return tv


def save_loss_plot(data_loss, name_file):
    step = 1
    if args.checkpoint:
        step = args.checkpoint

    plt.plot(np.arange(0, len(data_loss)*step, step), data_loss)
    plt.xlabel("Num itteration")
    plt.ylabel("Value loss")
    plt.savefig(name_file)

def save_result_image(output_data, name_file):
    result = cuda.to_cpu(output_data.data)
    result = np.uint8(result[0].transpose((1, 2, 0)))

    Image.fromarray(result).save(name_file)


def create_dir(directory):
    """
        Create a directory if it does not exist
        :type directory: str

    """
    if not os.path.exists(directory):
        os.makedirs(directory)

parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--resume', '-r', default=None, type=str,
                    help='resume the optimization from snapshot')
parser.add_argument('--output', '-o', default=None, type=str,
                    help='output model file path without extension')
parser.add_argument('--image_size', default=512, type=int,
                    help='dimensions to scale style image and dataset')
parser.add_argument('--fullsize', dest='crop', action='store_false',
                    help='do not crop dataset images, but scale only')
parser.add_argument('--lambda_tv', default=10e-6, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', default=1, type=float)
parser.add_argument('--lambda_style', default=1e1, type=float)
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=1000, type=int)
parser.set_defaults(crop=True)
args = parser.parse_args()

chainer.config.train = True
batchsize = args.batchsize

size = args.image_size
n_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style
style_prefix, _ = os.path.splitext(os.path.basename(args.style_image))
parameters_str = "tv{}f{}s{}b{}size{}".format(lambda_tv, lambda_f, lambda_s, batchsize, size)
output = style_prefix if args.output == None else args.output
fs = os.listdir(args.dataset)
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
n_data = len(imagepaths)
print ('num traning images:', n_data)
n_iter = int(n_data / batchsize)
print(n_iter, 'iterations,', n_epoch, 'epochs')

model = FastStyleNet()
vgg = VGG()
serializers.load_npz('vgg16.model', vgg)

start_epoch = 0
start_iter = 0
loss_history = []
if args.initmodel:
    print('load model from', args.initmodel)
    start_iter = int(re.search(r'\d+(?=\D*$)', r"{}".format(args.initmodel)).group(0))
    start_epoch = int(re.search(r'\_(\d+)\_', args.initmodel).group(1))
    serializers.load_npz(args.initmodel, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    vgg.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

O = optimizers.Adam(alpha=args.lr)
O.setup(model)
if args.resume:
    print('load optimizer state from', args.resume)
    start_iter = int(re.search(r'\d+(?=\D*$)', r"{}".format(args.resume)).group(0))
    start_epoch = int(re.search(r'\_(\d+)\_', args.resume).group(1))
    serializers.load_npz(args.resume, O)

style = vgg.preprocess(np.asarray(Image.open(args.style_image).convert('RGB').resize((size,size), 3), dtype=np.float32))
style = xp.asarray(style, dtype=xp.float32)
style_b = xp.zeros((batchsize,) + style.shape, dtype=xp.float32)
for i in range(batchsize):
    style_b[i] = style
feature_s = vgg(Variable(style_b))
gram_s = [gram_matrix(y) for y in feature_s]

create_dir("models/{}".format(output))
lossesBeforeCheckpoint = [];
for epoch in range(start_epoch, n_epoch):
    print ('epoch', epoch)
    for i in range(start_iter, n_iter):
        model.zerograds()
        vgg.zerograds()

        indices = range(i * batchsize, (i+1) * batchsize)
        x = xp.zeros((batchsize, 3, size, size), dtype=xp.float32)
        for j in range(batchsize):
            img = Image.open(imagepaths[i*batchsize + j]).convert('RGB')
            img = ImageOps.fit(img, (size, size), 2) if args.crop else img.resize((size,size), 2)
            x[j] = xp.asarray(img, dtype=np.float32).transpose(2, 0, 1)

        xc = Variable(x.copy())
        x = Variable(x)

        y = model(x)

        xc -= 124
        y -= 124

        feature = vgg(xc)
        feature_hat = vgg(y)

        L_feat = lambda_f * F.mean_squared_error(Variable(feature[2].data), feature_hat[2]) # compute for only the output of layer conv3_3

        L_style = Variable(xp.zeros((), dtype=np.float32))
        for f, f_hat, g_s in zip(feature, feature_hat, gram_s):
            L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), Variable(g_s.data))

        L_tv = lambda_tv * total_variation_regularization(y)
        L = L_feat + L_style + L_tv

        print ('(epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter, L.data))

        L.backward()
        O.update()

        lossesBeforeCheckpoint.append(float(L.data))
        if args.checkpoint > 0 and i % args.checkpoint == 0:
            basname_output = "{}_{}_{}_{}".format(output, parameters_str, epoch, i)
            serializers.save_npz('models/{}.model'.format(basname_output), model)
            serializers.save_npz('models/{}.state'.format(basname_output), O)
            loss_history.append(np.mean(lossesBeforeCheckpoint))
            lossesBeforeCheckpoint = []
            save_loss_plot(loss_history[1:], 'models/{}_{}_{}_{}_loss.png'.format(output, parameters_str, start_epoch, start_iter))
            save_result_image(y+124, 'models/{}_result.png'.format(basname_output))

    start_iter = 0

    print ('save "style.model"')
    serializers.save_npz('models/{}_{}.model'.format(output, epoch), model)
    serializers.save_npz('models/{}_{}.state'.format(output, epoch), O)

serializers.save_npz('models/{}.model'.format(output), model)
serializers.save_npz('models/{}.state'.format(output), O)
