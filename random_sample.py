import torch

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = "cpu"
    model = NVAE(z_dim=512, img_dim=(64, 64))
    model.apply(add_sn)
    model.to(device)

    model.load_state_dict(torch.load("checkpoints/ae_ckpt_103_0.074130.pth", map_location=device), strict=False)

    model.eval()

    with torch.no_grad():
        z = torch.randn((25, 512)).to(device)
        gen_imgs, _ = model.decoder(z)
        gen_imgs = gen_imgs.permute(0, 2, 3, 1)
        for gen_img in gen_imgs:
            gen_img = (gen_img.cpu().numpy() + 1) / 2 * 255
            gen_img = gen_img.astype(np.uint8)

            plt.imshow(gen_img)
            # plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, total_loss))
            plt.show()
