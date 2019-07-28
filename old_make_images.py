import os, time
from pathlib import Path
import shutil
import numpy as np
import argparse
import chainer
from chainer import cuda
from chainer.links import VGG16Layers as VGG
from chainer.training import extensions
import chainermn
from PIL import Image
import yaml
import source.yaml_utils as yaml_utils
from gen_models.ada_generator import AdaBIGGAN, AdaSNGAN
from dis_models.patch_discriminator import Discriminator as PatchDiscriminator
from updater import Updater

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/default.yml")
    parser.add_argument("--gen", "-gen", type=str, default="")
    parser.add_argument("--gen_gen", "-gg", type=str, default="")
    parser.add_argument('--save', type=str, default='images')

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))

    device = args.gpu
    cuda.get_device(device).use()

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu)
        xp = cuda.cupy
    else:
        xp = np

    np.random.seed(1234)

    datasize = config.datasize

    # Model
    if config.gan_type == "BIGGAN":
        gen = AdaBIGGAN(config, datasize, comm=None)
    elif config.gan_type == "SNGAN":
        gen = AdaSNGAN(config, datasize, comm=None)

    if not config.random:  # load pre-trained generator model
        chainer.serializers.load_npz(config.snapshot[config.gan_type], gen.gen)

    chainer.serializers.load_npz(args.gen, gen)
    chainer.serializers.load_npz(args.gen_gen, gen.gen)

    gen.to_gpu(device)
    gen.gen.to_gpu(device)

    with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
        tmp = config.tmp_for_test
        xs = []
        num = 0

        for i in range(200):
            x = gen.random(tmp=tmp, truncate=True)
            xs.append(chainer.cuda.to_cpu(x.data))
            for j in range(len(xs[i])):
                xa = xs[i][j].transpose(2, 1, 0)
                xa = np.asarray(np.clip(xa * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
                pil_img = Image.fromarray(xa)
                pil_img = pil_img.resize((224, 224))#uuuu
                pil_img.save(f"{args.save}/random/{num}.jpg")
                num += 1

        xs = []
        num = 0
        for i in range(5):
            x = gen(perm=np.arange(i * 5, i * 5 + 5))
            xs.append(chainer.cuda.to_cpu(x.data))
            for j in range(len(xs[i])):
                xa = xs[i][j].transpose(2, 1, 0)
                xa = np.asarray(np.clip(xa * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
                pil_img = Image.fromarray(xa)
                pil_img = pil_img.resize((224, 224))
                pil_img.save(f"{args.save}/recon/{num}.jpg")
                num += 1

        xs = []
        num = 0
        for i in range(5):
            x = gen.interpolate(i, i + 5, num=10)
            xs.append(chainer.cuda.to_cpu(x.data))
            for j in range(len(xs[i])):
                xa = xs[i][j].transpose(2, 1, 0)
                xa = np.asarray(np.clip(xa * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
                pil_img = Image.fromarray(xa)
                pil_img = pil_img.resize((224, 224))
                pil_img.save(f"{args.save}/interpolate/{num}.jpg")
                num += 1
