#每一个epoch都要进行插值以及记录插值loss

import os
import torch
from torch import optim, nn
from model import Siren,MLP,DinerMLP,DinerSiren,DinerMLP_interp,DinerSiren_interp,DinerSiren_mlp
from model import *
from dataio import ImageData, ImageData_interp
import time
import utils
from sklearn.preprocessing import normalize
from tqdm.autonotebook import tqdm
from opt import HyperParameters
from loss import relative_l2_loss
import random
from skimage import io
import rff
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)
from pytorch_msssim import ssim,ms_ssim
import scipy.signal

''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val=1,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

class Logger:
    filename = None
    filename1 = None
    filename2 = None
    filename3 = None
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')
    
    @staticmethod
    def write_file(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')
            
    @staticmethod
    def write1(text):
        with open(Logger.filename1, 'a') as log_file:
            log_file.write(text + '\n')
            
    @staticmethod
    def write2(text):
        with open(Logger.filename2, 'a') as log_file:
            log_file.write(text + '\n')
    @staticmethod
    def write3(text):
        with open(Logger.filename3, 'a') as log_file:
            log_file.write(text + '\n')

def train_img(opt):
    img_path                =       opt.img_path
    steps                   =       opt.steps
    lr                      =       opt.lr
    hidden_layers           =       opt.hidden_layers
    hidden_features         =       opt.hidden_features
    sidelength              =       opt.sidelength
    grayscale               =       opt.grayscale
    first_omega_0           =       opt.w0
    hidden_omega_0          =       opt.w0
    model_type              =       opt.model_type
    steps_til_summary       =       opt.steps_til_summary
    input_dim               =       opt.input_dim
    epochs                  =       opt.epochs
    remain_raw_resolution   =       opt.remain_raw_resolution
    experiment_name         =       opt.experiment_name
    
    name = 'RHINO'
    print(name)
    recon_psnrs = [f'{name}']
    for i in tqdm(range(85,86)):  
        a = opt.img_path.split('/')[-1][-7:]
        img_path = opt.img_path.replace(a, f'{i:003d}.png')
        idx = f'{i:003d}'
        
        # make directory
        log_dir = "/data/liufengyi/MyCode/tidying_up/RHINO/img/log"
        utils.cond_mkdir(os.path.join(log_dir,name))
        utils.cond_mkdir(os.path.join(log_dir,name, "all"))
        utils.cond_mkdir(os.path.join(log_dir,name, "log_psnr"))
        # utils.cond_mkdir(os.path.join(log_dir,name, "log_psnr_interp"))
        utils.cond_mkdir(os.path.join(log_dir,name,"log_time"))
        experiment_name = name
        # check parameters
        
        if steps % steps_til_summary:
            raise ValueError("steps_til_summary could not be devided by steps!")

        Logger.filename = os.path.join(log_dir, name,f'log_time', f'log_{idx}.txt')
        Logger.filename1 = os.path.join(log_dir, name,'log_psnr', f'psnr_log_{idx}.txt')
        Logger.filename2 = os.path.join(log_dir, name,'log_time', f'log_interp_{idx}.txt')
        Logger.filename3 = os.path.join(log_dir, name,'log_time', f'log_interp1_{idx}.txt')

        device = torch.device('cuda')
        criteon = nn.MSELoss()

        out_features = 3

        Dataset = ImageData_interp(image_path = img_path,
                                sidelength = sidelength,
                                grayscale = grayscale,
                                remain_raw_resolution = remain_raw_resolution)
        model_input_all,model_input,model_input_,model_input_interp,\
            gt_all,gt,gt_,gt_interp = Dataset[0]
        
        model_input_all = model_input_all.to(device)  
        model_input_interp = model_input_interp.to(device)
        model_input = model_input.to(device)
        model_input_ = model_input_.to(device)
        gt = gt.to(device)

        hash_table_length = model_input.shape[0]

        if model_type == 'Siren':
            model = Siren(in_features = input_dim,
                            hidden_features = hidden_features,
                            hidden_layers = hidden_layers,
                            out_features = out_features,
                            ).to(device = device)

        
        elif model_type == 'MLP':
            model = MLP(in_features = input_dim,
                        out_features = out_features,
                        hidden_layers = hidden_layers,
                        hidden_features= hidden_features,
                        ).to(device = device)

        elif model_type == 'PeMLP':
            encoding = rff.layers.GaussianEncoding(10.0, 2, 256).to(device) 
            Xp = encoding(model_input)
            Xp_all = encoding(model_input_all)
            Xp_interp = encoding(model_input_interp)
            model = MLP(in_features = 256*2,
                        out_features = out_features,
                        hidden_layers = hidden_layers,
                        hidden_features= hidden_features,
                        ).to(device = device)
            
            
        elif model_type == 'DinerSiren':
            model = DinerSiren(
                        hash_table_length = hash_table_length,
                        in_features = input_dim,
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features,
                        outermost_linear = True,
                        first_omega_0 = first_omega_0,
                        hidden_omega_0 = hidden_omega_0).to(device = device)
        elif model_type == 'DinerSiren_mlp':
            model = DinerSiren_coor1(
                        hash_table_length = hash_table_length,
                        in_features = input_dim,
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features,
                        outermost_linear = True,
                        first_omega_0 = first_omega_0,
                        hidden_omega_0 = hidden_omega_0).to(device = device)        
        elif model_type == 'DinerSiren_interp':
            model = DinerSiren_interp1(
                        #   hash_table_resolution = Dataset.image.shape[:2],
                        hash_table_resolution = torch.tensor([600,600]),
                        in_features = input_dim,
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features,
                        outermost_linear = True,
                        first_omega_0 = first_omega_0,
                        hidden_omega_0 = hidden_omega_0).to(device = device)
            
        elif model_type == 'DinerMLP':
            model = DinerMLP(
                            hash_table_length = hash_table_length, 
                            in_features = input_dim, 
                            hidden_features = hidden_features,
                            hidden_layers = hidden_layers,
                            # mlp_dim=2,
                            out_features = out_features).to(device = device)
        elif model_type == 'RHINO':
            model = RHINO(
                            hash_table_length = hash_table_length, 
                            in_features = input_dim, 
                            hidden_features = hidden_features,
                            hidden_layers = hidden_layers,
                            mlp_dim=2,
                            out_features = out_features).to(device = device)
        else:
            raise NotImplementedError("Model type not supported!")
            
        optimizer = optim.Adam(lr = lr,params = model.parameters())
        
        print(model)

        # training process
        with tqdm(total=epochs) as pbar:
            max_psnr = 0
            time_cost = 0
            for epoch in range(epochs):
                time_start = time.time()

                loss_mse = 0
                if model_type == 'PeMLP':
                    model_output = model(Xp)
                else:
                    model_output = model(model_input)
                loss_mse = criteon(model_output,gt)
                # with torch.no_grad():
                #     model.interp = True
                #     if model_type == 'PeMLP':
                #         model_output_interp = model(Xp_all)
                #     else:
                #         model_output_interp = model(model_input_all)  
                #     loss_mse_interp = criteon(model_output_interp,gt_all.to(model_output_interp))
                #     cur_psnr_interp = utils.loss2psnr(loss_mse_interp)
                    
                #     if model_type == 'PeMLP':
                #         model_output_interp1 = model(Xp_interp)
                #     else:
                #         model_output_interp1 = model(model_input_interp)  
                #     loss_mse_interp1 = criteon(model_output_interp1,gt_interp.to(model_output_interp))
                #     cur_psnr_interp1 = utils.loss2psnr(loss_mse_interp)
                 

                model.interp = False                  
                optimizer.zero_grad()
                loss_mse.backward()
                optimizer.step()

                torch.cuda.synchronize()
                time_cost += time.time() - time_start

                cur_psnr = utils.loss2psnr(loss_mse)
                max_psnr = max(max_psnr,cur_psnr)

                # if (epoch + 1) % steps_til_summary == 0:
                #     # log_str = f"[TRAIN] Epoch: {epoch+1} Loss: {loss_mse.item()} PSNR: {cur_ssim} Time: {round(time_cost, 2)}"
                #     # log_str1 = f"[TRAIN] Epoch: {epoch+1} Loss: {loss_mse_interp.item()} PSNR: {cur_ssim_interp} Time: {round(time_cost, 2)}"
                #     # log_str2 = f"[TRAIN] Epoch: {epoch+1} Loss: {loss_mse_interp1.item()} PSNR: {cur_ssim_interp1} Time: {round(time_cost, 2)}"
                #     log_str = f"[TRAIN] Epoch: {epoch+1} Loss: {loss_mse.item()} PSNR: {cur_psnr} Time: {round(time_cost, 2)}"
                #     log_str1 = f"[TRAIN] Epoch: {epoch+1} Loss: {loss_mse_interp.item()} PSNR: {cur_psnr_interp} Time: {round(time_cost, 2)}"
                #     log_str2 = f"[TRAIN] Epoch: {epoch+1} Loss: {loss_mse_interp1.item()} PSNR: {cur_psnr_interp1} Time: {round(time_cost, 2)}"

                #     Logger.write(log_str)
                #     Logger.write2(log_str1)
                #     Logger.write3(log_str2)
                pbar.update(1)
        print("time_cost: ", time_cost)
        Logger.write1(f"All_time: {time_cost}")
        model.interp = False
        save_path = os.path.join(log_dir,experiment_name)
        if model_type == 'PeMLP':
            utils.render_raw_image_interp(model,encoding(model_input),os.path.join(log_dir,experiment_name,f'{name}_{idx}.png'),[600,600],linear = False)
        else:        
            utils.render_raw_image_interp(model,model_input,os.path.join(log_dir,experiment_name,f'{name}_{idx}.png'),[600,600],linear = False)
        gt_image = (gt.view(600,600,-1) + 1) / 2
        gt_image =  np.round(gt_image.detach().cpu().numpy() * 255).astype(np.uint8)
        io.imsave(f'{save_path}/{name}_gt_{idx}.png',gt_image)
        recon_psnr = utils.calculate_psnr(os.path.join(log_dir,experiment_name,f'{name}_{idx}.png'),f'{save_path}/{name}_gt_{idx}.png')
        print(f"Reconstruction PSNR: {recon_psnr:.2f}")    
        Logger.write1(f"PSNR:{recon_psnr}")
        model.interp = True
        # recon_psnr = utils.calculate_psnr(os.path.join(log_dir,experiment_name,'mlp2.png'),img_path)
        if model_type == 'PeMLP':    
            utils.render_raw_image_interp(model,encoding(model_input_),os.path.join(log_dir,experiment_name,f'{name}_interp_{idx}.png'),[600,600],linear = False)
        else:
            utils.render_raw_image_interp(model,model_input_,os.path.join(log_dir,experiment_name,f'{name}_interp_{idx}.png'),[600,600],linear = False)

        gt_image_ = (gt_.view(600,600,-1) + 1) / 2
        gt_image_ =  np.round(gt_image_.detach().cpu().numpy() * 255).astype(np.uint8)
        io.imsave(f'{save_path}/{name}_interp_gt_{idx}.png',gt_image_)
        recon_psnr = utils.calculate_psnr(os.path.join(log_dir,experiment_name,f'{name}_interp_{idx}.png'),f'{save_path}/{name}_interp_gt_{idx}.png')
        Logger.write1(f"PSNR1:{recon_psnr}")
        print(f"Reconstruction PSNR: {recon_psnr:.2f}")    
        
        if model_type == 'PeMLP':    
            utils.render_raw_image_interp(model,encoding(model_input_all),os.path.join(log_dir,experiment_name,'all',f'{name}_all_{idx}.png'),[600,600],linear = False)
        else:
            utils.render_raw_image_interp(model,model_input_all,os.path.join(log_dir,experiment_name,'all', f'{name}_all_{idx}.png'),[600,600],linear = False)

        gt_image_ = (gt_all.view(1200,1200,-1) + 1) / 2
        gt_image_ =  np.round(gt_image_.detach().cpu().numpy() * 255).astype(np.uint8)
        io.imsave(f'{save_path}/{name}_all_gt_{idx}.png',gt_image_)
        recon_psnr = utils.calculate_psnr(os.path.join(log_dir,experiment_name,'all',f'{name}_all_{idx}.png'),f'{save_path}/{name}_all_gt_{idx}.png')
        Logger.write1(f"PSNR3:{recon_psnr}")
        recon_psnrs += [recon_psnr]
        print(f"Reconstruction PSNR: {recon_psnr:.2f}")  
        
if __name__ == "__main__":

    opt = HyperParameters()
    train_img(opt)
