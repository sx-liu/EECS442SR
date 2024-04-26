import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
from torch_utils.resizer import Resizer
import scipy.io
from PIL import Image


def transpose(data, scale_factor=4):
    return F.interpolate(data, scale_factor=scale_factor)


in_shape = (1, 3, 256, 256)
forward = Resizer(in_shape, 1 / 4).to('cuda')


def project(data, measurement):
    return data - transpose(forward(data)) + transpose(measurement)


def grad_and_value(x_prev, x_0_hat, measurement):
    Ax = forward(x_0_hat)
    difference = measurement - Ax
    norm = torch.linalg.norm(difference) / measurement.abs()
    norm = norm.mean()
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

    return norm_grad


#----------------------------------------------------------------------------

def generate_random_mask(shape, mask_prob_range, device):
    prob_mask = torch.rand(shape, device=device)
    mask = (prob_mask > mask_prob_range[0]) & (prob_mask < mask_prob_range[1])
    return mask.float()

def apply_gaussian_noise(x, mask, sigma):
    noise = torch.randn(x.size(), device=x.device) * sigma
    return x * (1 - mask) + noise * mask

def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=1, gridh=1, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, eta_in=1,
):
    # load the image
    image_path = 'input_img/penguin_256.png'
    image = Image.open(image_path)
    np_image = np.array(image.convert('RGB'))
    image_tensor = torch.from_numpy(np_image).permute(2, 0, 1).float().to(device)
    image_tensor=image_tensor.unsqueeze(0)
    print(image_tensor.shape)
    
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    # print(net.img_channels)
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    y = forward(image_tensor)
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        assert not torch.any(torch.isnan(denoised)), f'step {i}'
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # predict x_0
        x_N = denoised

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_N = denoised
            
        # x_N: u; now we want to calculate grad(u) by applying a filter; x_next is torch tensor
        x_N = (x_N * 127.5 + 128).clip(0, 255)

        norm_grad = grad_and_value(x_cur, x_N, y)

        x_next = x_next - eta_in * norm_grad

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    x_final = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = x_final
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(f"div2k/{num_steps}.png")
    print('Done.')

#----------------------------------------------------------------------------

def main():
    for step in [100, 1000]:
        generate_image_grid(f'pretrained-div2k/00000--uncond-ddpmpp-edm-gpus3-batch15-fp32/network-snapshot-000501.pkl', f'results-{step}.png', num_steps=step, eta_in=40.)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
