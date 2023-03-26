from utils import load_image, save_img, load_image_PIL
from generate_hybrid_img import gen_hybrid_img_gpu
from tqdm import tqdm
import os
from Normalize import Normalize
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np
import torchvision.models as models
from torchvision.datasets import ImageFolder
from PIL import Image
from pattern import circle, square, prismatic
import argparse
import torch
import timm
import time
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# color_list = ['dodgerblue', 'orange', 'green', 'red', 'mediumslateblue', 'saddlebrown',
#               'violet', 'dimgrey', 'darkkhaki', 'turquoise']

transforms = T.Compose([T.ToTensor()])
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='/mnt/hdd2/zhangqilong/++Image1wRs++', help='The root directory of the clean img')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--batch_size", type=int, default=25, help="How many images process at one time.")
# parser.add_argument('--save_dir', type=str, default='GPU_HIT_output/Tile/Rhom/{}/', help='The root directory of the adversarial img')
parser.add_argument('--cutoff_frequency', type=int, default='4', help='generate a (4*cutoff_frequency+1)*(4*cutoff_frequency+1) gaussian kernel ')
parser.add_argument('--weight_factor', type=float, default='1.0', help='balance the low_frequencies and high_frequencies ')
# parser.add_argument('--max_tile', type=int, default='6', help='balance the low_frequencies and high_frequencies ')
parser.add_argument('--max_epsilon', type=float, default='16', help='balance the low_frequencies and high_frequencies ')
parser.add_argument('--tile_scheme', type=int, default='6', help='tile size')


opt = parser.parse_args()
# dir_name = sorted(os.listdir(opt.input_dir))
res152 = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.resnet152(pretrained=True).eval()).cuda()
inc_v3 = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.inception_v3(pretrained=True).eval()).cuda()
# resnext50_32x4d = torch.nn.Sequential(Normalize(opt.mean, opt.std),models.resnext50_32x4d(pretrained=True).eval()).cuda()
dense121 = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.densenet121(pretrained=True).eval()).cuda()
dense169 = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.densenet169(pretrained=True).eval()).cuda()
shufflenet = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.shufflenet_v2_x1_0(pretrained=True).eval()).cuda()
squee = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.squeezenet1_1(pretrained=True).eval()).cuda()
mobile = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.mobilenet_v2(pretrained=True).eval()).cuda()
wrn50 = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.wide_resnet50_2(pretrained=True).eval()).cuda()
wrn101 = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.wide_resnet101_2(pretrained=True).eval()).cuda()
# se = torch.nn.Sequential(Normalize(opt.mean, opt.std), pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet').eval()).cuda()
vgg = torch.nn.Sequential(Normalize(opt.mean, opt.std), models.vgg19_bn(pretrained=True).eval()).cuda()
se = torch.nn.Sequential(Normalize(opt.mean, opt.std), timm.create_model('legacy_senet154', pretrained=True).eval()).cuda()
pna = torch.nn.Sequential(Normalize(opt.mean, opt.std), timm.create_model('pnasnet5large', pretrained=True).eval()).cuda()



for tile_scheme in range(6, 7, 1):
    img = np.array(Image.open('adversarial_patch/circle.png').convert('RGB'), dtype=np.float64)[149:449, 149:449, :]
    img = np.tile(np.array(Image.fromarray(np.uint8(img)).resize((300 // tile_scheme, 300 // tile_scheme),
                                                                 Image.ANTIALIAS)), (tile_scheme, tile_scheme, 1))
    img = Image.fromarray(img).resize((299, 299))
    img.save('tile_C.png')
    pert = load_image_PIL('tile_C.png').cuda().unsqueeze(0).expand(opt.batch_size, 3, 299, 299)

    # data_loader = DataLoader(ImageNet(opt.input_dir, transforms), batch_size=20, shuffle=False, pin_memory=True, num_workers=8)

    X = ImageFolder(opt.input_dir, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    non_res, non_v3, non_dense121, non_dense169, non_shuff, non_squee, non_mob, non_rext, non_wrn50, non_wrn101, non_se, non_vgg, non_pna = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for images,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        hybrid_image = gen_hybrid_img_gpu(images, pert, opt.cutoff_frequency, opt.weight_factor)
        adv_img = images + torch.clamp(hybrid_image - images, -opt.max_epsilon / 255.0, opt.max_epsilon /255.0)

        # plt.imshow(adv_img[0].permute(1,2,0).detach().cpu().numpy())
        # plt.savefig('1.png')
        # exit()

        with torch.no_grad():
            non_res += (res152(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_v3 += (inc_v3(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_dense121 += (dense121(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_dense169 += (dense169(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_shuff += (shufflenet(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_squee += (squee(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_mob += (mobile(adv_img).max(1)[1] != gt).detach().cpu().sum()
            # non_rext += (resnext50_32x4d(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_wrn50 += (wrn50(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_wrn101 += (wrn101(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_se += (se(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_vgg += (vgg(adv_img).max(1)[1] != gt).detach().cpu().sum()
            non_pna += (pna(adv_img).max(1)[1] != gt).detach().cpu().sum()

    print('vgg = ', non_vgg)
    print('Inc_v3 = ', non_v3)
    print('res152 = ', non_res)
    print('dense121 = ', non_dense121)
    print('dense169 = ', non_dense169)
    print('wrn50 = ', non_wrn50)
    print('wrn101 = ', non_wrn101)
    print('se = ', non_se)
    print('pna = ', non_pna)
    print('shuffle = ', non_shuff)
    print('squeeze = ', non_squee)
    print('mobile = ', non_mob)







