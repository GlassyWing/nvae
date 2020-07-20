import torch

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    device = "cpu"
    model = NVAE(z_dim=512, img_dim=(64, 64))
    model.apply(add_sn)
    model.to(device)

    model.load_state_dict(torch.load("checkpoints/ae_ckpt_108_0.077118.pth", map_location=device), strict=False)

    model.eval()

    result = np.zeros((768, 768, 3), dtype=np.uint8)

    with torch.no_grad():
        z = torch.randn((144, 512)).to(device)
        gen_imgs, _ = model.decoder(z)
        gen_imgs = gen_imgs.reshape(12, 12, 3, 64, 64)

        gen_imgs = gen_imgs.permute(0, 1, 3, 4, 2)
        gen_imgs = (gen_imgs.cpu().numpy() + 1) / 2 * 255
        gen_imgs = gen_imgs.astype(np.uint8)

    for i in range(12):
        for j in range(12):
            result[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = gen_imgs[i, j]

    from PIL import Image

    im = Image.fromarray(result)
    im.save("output/demo.jpeg")
