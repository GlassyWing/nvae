import torch

from nvae.dataset import ImageFolderDataset
from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_ds = ImageFolderDataset("G:\data\GAN\celeba\img_align_celeba", img_dim=64)

    device = "cpu"
    model = NVAE(z_dim=512, img_dim=(64, 64))
    model.apply(add_sn)
    model.to(device)

    model.load_state_dict(torch.load("checkpoints/ae_ckpt_3_0.795224.pth", map_location=device), strict=False)

    model.eval()

    img = train_ds[54].unsqueeze(0).to(device)
    ori_image = img.permute(0, 2, 3, 1)[0]
    ori_image = ori_image.numpy() * 255
    plt.imshow(ori_image.astype(np.uint8))
    plt.show()

    with torch.no_grad():
        gen_imgs, _ = model(img)
        gen_imgs = gen_imgs.permute(0, 2, 3, 1)
        for gen_img in gen_imgs:
            gen_img = gen_img.cpu().numpy() * 255
            gen_img = gen_img.astype(np.uint8)

            plt.imshow(gen_img)
            # plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, total_loss))
            plt.show()
