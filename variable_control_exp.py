import torch

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import cv2

def set_bn(model, num_samples=1, iters=100):
    model.train()
    for i in range(iters):
        if i % 10 == 0:
            print('setting BN statistics iter %d out of %d' % (i + 1, iters))
        z = torch.randn((num_samples, z_dim, 2, 2)).to(device)
        model.decoder(z)
    model.eval()

if __name__ == '__main__':
    device = "cuda:0"
    z_dim = 512
    model = NVAE(z_dim=z_dim, img_dim=64)
    model.apply(add_sn)
    model.to(device)

    model.load_state_dict(torch.load("../checkpoints/ae_ckpt_169_0.689621.pth", map_location=device), strict=False)

    z = torch.randn((1, 512, 2, 2)).to(device)
    x = 1
    y = 1
    m = 1
    s = [482, 14, 204, 255, 356, 397, 404, 437]
    alpha = 0.1
    freeze_level = 0

    # m = 14, 204, 255, 356, 397, 404, 437, 482 x = 1, y = 1

    zs = model.decoder.zs

    while True:

        key = cv2.waitKey(200)

        with torch.no_grad():

            gen_imgs, losses = model.decoder(z, mode='fix', freeze_level=freeze_level)

            gen_imgs = gen_imgs.permute(0, 2, 3, 1)
            for gen_img in gen_imgs:
                gen_img = gen_img.cpu().numpy() * 255
                gen_img = gen_img.clip(0, 255).astype(np.uint8)

                # plt.imshow(gen_img)
                # # plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, total_loss))
                # plt.show()
                gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
                gen_img = cv2.resize(gen_img, (int(gen_img.shape[0] * 2), int(gen_img.shape[1] * 2)), cv2.INTER_AREA)
                cv2.putText(gen_img, str(s[m]) + ',' + str(zs[-1][0, s[m], y, x].item()), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=1)
                cv2.imshow("hidden:", gen_img)



        if key == ord('w'):
            zs[-1][:, s[m], y, x] += alpha
            # zs[-1] = torch.randn(1, 64, 16, 16)
        elif key == ord('s'):
            zs[-1][:, s[m], y, x] -= alpha
        elif key == ord('a'):
            m = (m - 1) % len(s)
        elif key == ord('d'):
            m = (m + 1) % len(s)
        elif key == ord('q'):
            exit(0)
