import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn

class Agumenter:
    def __init__(self, device, cutn=10, cut_size=224, cut_pow=1.):
        self.cut_pow = cut_pow
        self.cutn = cutn
        # Note: Sequential is so we can backpropagate through.
        self.cut_size = cut_size
        self.augs = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=30),
            transforms.RandomPerspective(0.2,p=0.4),
            transforms.ColorJitter(hue=0.01, saturation=0.01)
        ).to(device)

        self.norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711)).to(device)
        self.resize = transforms.Resize(cut_size).to(device)

    def __call__(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(self.resize(cutout))
        batch = self.augs(torch.cat(cutouts, dim=0))
        return self.norm(batch)
