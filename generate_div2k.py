import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
from PIL import Image


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
    # image_path = 'out/000003.png'
    # image = Image.open(image_path)
    # np_image = np.array(image.convert('RGB'))
    # image_tensor = torch.from_numpy(np_image).permute(2, 0, 1).float().to(device)
    # image_tensor=image_tensor.unsqueeze(0)
    # print(image_tensor.shape)
    
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    print(net.img_channels)
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
    t_N = t_steps[-1]
    
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
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
            
        # # x_N: u; now we want to calculate grad(u) by applying a filter; x_next is torch tensor
        # x_N = (x_N*127.5+128).clip(0, 255)
        # mask = generate_random_mask(x_N.size(), (0.3, 0.7), device)
        # x_noised = apply_gaussian_noise(x_N, mask, 0.05)
        # # print(x_N.shape)
        # total_loss = torch.norm(image_tensor - x_noised, 2)
        # diff_output = total_loss**2
        # eta = eta_in/total_loss
        # norm_grad = torch.autograd.grad(outputs=diff_output, inputs=x_cur)[0]
        # x_next = x_next - eta * norm_grad

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    # Print the PDE loss for x_next
    x_final = (x_next*127.5+128).clip(0, 255).to(torch.uint8)
    # total_loss = torch.norm(image_tensor - x_final, 2)
    # print(f'PDE loss for x_next: {total_loss}')
    # relative_error = torch.norm(x_final - image_tensor, 2)/torch.norm(image_tensor, 2)
    # print(f'Relative error: {relative_error}')
    image = x_final
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(f"div2k/{num_steps}.png")
    print('Done.')

def generate_image_grid_ddim(
    network_pkl, dest_path, randn_like=torch.randn_like,
    seed=0, gridw=1, gridh=1, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    solver='euler', discretization='iddpm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, eta_in=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    
    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # load Darcy/coeff as coefficients
    image_path = 'out/000001.png'
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
    print(net.img_channels)
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h
        
        # predict x_0
        x_N = denoised

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
            x_N = denoised
            
        # x_N: u; now we want to calculate grad(u) by applying a filter; x_next is torch tensor
        x_N = (x_N*127.5+128).clip(0, 255)
        total_loss = torch.norm(image_tensor - x_N)/64**2
        diff_output = total_loss
        print(f'Image loss for x_N: {total_loss}')
        # print(f'f loss: {torch.norm(pde_loss, 2)/100}, b loss: {torch.norm(boundary_loss, 2)/(127*4)}, u loss: {torch.norm(observed_loss, 2)/100}')
        # deep copy x_cur
        eta = eta_in/total_loss
        # grad of total_loss_l2 w.r.t. x_cur
        # total_loss_l2.backward()
        # grad_x_cur = x_cur.grad
        norm_grad = torch.autograd.grad(outputs=diff_output, inputs=x_cur)[0]
        x_next = x_next - eta * norm_grad
        # if i%20==0:
        #     np.save(f'darcy-results-improved/eta-{eta_in}-it-{i}.npy', x_N.to('cpu').detach().numpy())

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    # Print the PDE loss for x_next
    x_final = (x_N*127.5+128).clip(0, 255).to(torch.uint8)
    
    total_loss = torch.norm(image_tensor - x_final)
    print(f'PDE loss for x_next: {total_loss}')
    relative_error = torch.norm(x_final - image_tensor, 2)/torch.norm(image_tensor, 2)
    print(f'Relative error: {relative_error}')
    image = x_final
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(f"ffhq/{num_steps}.png")
    print('Done.')

#----------------------------------------------------------------------------

def main():
    # model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    # generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-vp.pkl',   'cifar10-32x32.png',  num_steps=18) # FID = 1.79, NFE = 35
    # generate_image_grid(f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl',    'ffhq-64x64.png',     num_steps=40) # FID = 2.39, NFE = 79
    # generate_image_grid(f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl',  'afhqv2-64x64.png',   num_steps=40) # FID = 1.96, NFE = 79
    # generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 'imagenet-64x64.png', num_steps=256, S_churn=40, S_min=0.05, S_max=50, S_noise=1.003) # FID = 1.36, NFE = 511
    # for i in [1, 10, 100, 1000, 3000]:
    #     generate_image_grid(f'training-runs/00000--uncond-ddpmpp-edm-gpus2-batch40-fp32/network-snapshot-002500.pkl', f'darcy-results-{i}.png', num_steps=500, eta_in=i)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for step in [20, 100]:
        generate_image_grid(f'/Users/huangjiahe/Downloads/network-snapshot-000400.pkl', f'darcy-results-{step}.png', num_steps=step, eta_in=0.1, device=device)
    #generate_image_grid(f'training-runs/00000--uncond-ddpmpp-edm-gpus2-batch40-fp32/network-snapshot-002500.pkl', f'darcy-results-{step}.png', num_steps=50, eta_in=1000)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
