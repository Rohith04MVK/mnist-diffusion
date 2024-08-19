from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from diffusion_model import ContextUnet


def train(ddpm, dataloader, optim, n_epoch, device, save_dir, n_classes, ws_test, save_model=True):
    '''
    trains the ddpm model
    '''
    lrate = optim.param_groups[0]['lr']
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # evaluate and visualize samples
        visualize(ddpm, device, save_dir, n_classes,
                  ws_test, ep, x, c, n_epoch)

        # save model
        if save_model:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


def visualize(ddpm, device, save_dir, n_classes, ws_test, ep, x, c, n_epoch):
    '''
    evaluates the model and visualizes samples
    '''
    ddpm.eval()
    with torch.no_grad():
        n_sample = 4*n_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = ddpm.sample(
                n_sample, (1, 28, 28), device, guide_w=w)

            # append some real images at bottom, order by class also
            x_real = torch.Tensor(x_gen.shape).to(device)
            for k in range(n_classes):
                for j in range(int(n_sample/n_classes)):
                    try:
                        idx = torch.squeeze((c == k).nonzero())[j]
                    except:
                        idx = 0
                    x_real[k+(j*n_classes)] = x[idx]

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all*-1 + 1, nrow=10)
            save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
            print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

            if ep % 5 == 0 or ep == int(n_epoch-1):
                # create gif of images evolving over time, based on x_gen_store
                fig, axs = plt.subplots(nrows=int(
                    n_sample/n_classes), ncols=n_classes, sharex=True, sharey=True, figsize=(8, 3))

                def animate_diff(i, x_gen_store):
                    print(f'gif animating frame {i} of {
                          x_gen_store.shape[0]}', end='\r')
                    plots = []
                    for row in range(int(n_sample/n_classes)):
                        for col in range(n_classes):
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            plots.append(axs[row, col].imshow(-x_gen_store[i, (row*n_classes)+col, 0],
                                         cmap='gray', vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                    return plots
                ani = FuncAnimation(fig, animate_diff, fargs=[
                                    x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])
                ani.save(save_dir + f"gif_ep{ep}_w{w}.gif",
                         dpi=100, writer=PillowWriter(fps=5))
                print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")


def main():
    n_epoch = 20
    batch_size = 256
    n_T = 400  # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 256
    lrate = 1e-4
    save_dir = './data/diffusion/'
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(
        1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # mnist is already normalised 0 to 1
    tf = transforms.Compose([transforms.ToTensor()])

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    train(ddpm, dataloader, optim, n_epoch,
          device, save_dir, n_classes, ws_test)


if __name__ == "__main__":
    main()
