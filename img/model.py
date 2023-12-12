import torch
from torch import nn
import numpy as np
from opt import HyperParameters
import math
import torch.nn as nn

from typing import Optional
from torch import Tensor
import rff
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
    
class ReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.out_features = out_features
    def forward(self, input):
        return torch.relu(self.linear(input))

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_layers,
                 hidden_features,
                 bias = True):
        super().__init__()
        self.net = []
        N_freqs = 18
        self.embedding_x = Embedding(1, N_freqs) #1+1*2*5
        in_features = 2+2*2*N_freqs
        self.net.append(ReluLayer(in_features = in_features,out_features = hidden_features,bias = bias))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(in_features = hidden_features, out_features = hidden_features,bias = bias))

        self.net.append(nn.Linear(hidden_features, out_features))      

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = self.embedding_x(coords)
        output = self.net(coords)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class DinerMLP(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.hash_mod = True
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []

        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # a1 = self.table[...,0]
        # a2 = self.table[...,1]
        # x_min = a1.min()
        # x_max = a1.max()
        # y_min = a2.min()
        # y_max = a2.max()
        # [x, y] = torch.meshgrid(torch.linspace(x_min, x_max, 1200), torch.linspace(y_min, y_max, 1200))
        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # xy = torch.cat([x, y],dim = -1) # xy shape [H*W,2]
        # output = self.net(xy.to(device='cuda'))
        # output = torch.clamp(output, min = -1.0,max = 1.0)
        # rgb = (output.view(1200,1200,3) + 1) / 2
        # img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # from skimage import io
        # io.imsave('/data/liufengyi/MyCode/DINER/log/test/b1_map.png',img) 
           
        # def find_closest_index(tensor, target):
        #     distances = torch.sum((tensor - target) ** 2, dim=1)
        #     closest_index = torch.argmin(distances)
        #     return closest_index
        # coor = self.table.reshape(1200,1200,2)
        # xy = xy.to(self.table)
        # point1 = coor[591, 356]
        # point2 = coor[604, 373]
        # coor1 = find_closest_index(xy,point1).item()
        # coor2 = find_closest_index(xy,point2).item()
        # coor1 = coor1//1200,coor1%1200
        # coor2 = coor2//1200,coor2%1200     
        # file_path = '/data/liufengyi/MyCode/DINER/map/b1.txt'
        # np.savetxt(file_path, self.table.cpu().numpy())

        # print(f"Tensor saved to {file_path}")

        # if self.hash_mod:
        #     output = self.net(self.table)
        # else:
        #     output = self.net(coords)

        # output = torch.clamp(output, min = -1.0,max = 1.0)

        # return output
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,600,600,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(net_in) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(self.table) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output   
    

class RHINO(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 mlp_dim):

        super().__init__()
        self.interp = False
        self.pe = False
        self.pemlp = False
        self.Gaussian = False
        self.Gaussianmlp = False
        self.flag_direct = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        N_freqs = 11
        Gauss_dim = 256
        if self.pemlp:
            self.embedding_x = Embedding(1, N_freqs) #1+1*2*5
            mlp_feature = 2+2*2*N_freqs
            out_feature = 2
        elif self.Gaussianmlp:
            
            self.encoding = rff.layers.GaussianEncoding(10.0, 2, Gauss_dim)
            mlp_feature = Gauss_dim*2
            out_feature = 2
        else:
            mlp_feature = 2
            out_feature = mlp_dim
            if self.pe:
                self.embedding_x = Embedding(1, N_freqs) #1+1*2*5
                out_feature = 2+2*2*N_freqs
            elif self.Gaussian:
                self.encoding = rff.layers.GaussianEncoding(10.0, 2, Gauss_dim)
                out_feature = Gauss_dim*2
            elif self.flag_direct:
                out_feature = 2

        self.net = []
        self.net.append(ReluLayer(in_features+out_feature, hidden_features))
        
        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features)) 
            
        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)
        
        self.mlpnet = nn.Sequential(
                    nn.Linear(mlp_feature, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(0)
                    ],
                    nn.Linear(hidden_features, mlp_dim),
                )

    
    def forward(self, coords):
        if self.pemlp:
            out_ = self.mlpnet(self.embedding_x(coords))
        elif self.Gaussianmlp:
            out_ = self.mlpnet(self.encoding(coords))
        elif self.pe:
            out_ = self.embedding_x(coords)
        elif self.Gaussian:
            out_ = self.encoding(coords)
        elif self.flag_direct:
            out_ = coords
        else:
            out_ = self.mlpnet(coords)
        # a1 = self.table[...,0]
        # a2 = self.table[...,1]
        # x_min = a1.min()
        # x_max = a1.max()
        # y_min = a2.min()
        # y_max = a2.max()
        # [x, y] = torch.meshgrid(torch.linspace(x_min, x_max, 1200), torch.linspace(y_min, y_max, 1200))
        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # xy = torch.cat([x, y],dim = -1) # xy shape [H*W,2]
        # xy = xy.to(out_)
        # output = self.net(torch.cat([xy.to(out_), out_],dim=-1)) 
        # output = torch.clamp(output, min = -1.0,max = 1.0)
        # rgb = (output.view(1200,1200,3) + 1) / 2
        # img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # from skimage import io
        # io.imsave('/data/liufengyi/MyCode/DINER/log/test/e3_map.png',img)  
        # def find_closest_index(tensor, target):
        #     distances = torch.sum((tensor - target) ** 2, dim=1)
        #     closest_index = torch.argmin(distances)
        #     return closest_index
        # coor = self.table.reshape(1200,1200,2)
        # point1 = coor[591, 356]
        # point2 = coor[604, 373]
        # coor1 = find_closest_index(xy,point1).item()
        # coor2 = find_closest_index(xy,point2).item()
        # coor1 = coor1//1200,coor1%1200
        # coor2 = coor2//1200,coor2%1200
        # file_path = '/data/liufengyi/MyCode/DINER/map/e3.txt'
        # np.savetxt(file_path, self.table.cpu().numpy())
        # a1 = self.table[...,0]
        # a2 = self.table[...,1]
        # a3 = out_[...,0]
        # x_min = a1.min()
        # x_max = a1.max()
        # y_min = a2.min()
        # y_max = a2.max()
        # z_min = a3.min()
        # z_max = a3.max()
        # [x, y, z] = torch.meshgrid(torch.linspace(x_min, x_max, 1200), torch.linspace(y_min, y_max, 1200), torch.linspace(z_min, z_max, 1200))
        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # z = z.contiguous().view(-1, 1)
        # xyz = torch.cat([x, y, z],dim = -1)
        # xyz = xyz.to(out_)
        # xy = torch.cat([x, y],dim = -1)
        # xy = xy.to(out_)
        # output = []
        # split_data = torch.split(xyz, 1200*1200)
        # for batch in split_data:

        #     output_ = self.net(batch)
            
        #     # Transfer the output back to CPU
        #     output_ = output_.to('cpu')
            
        #     # Accumulate the results
        #     output.append(output_)
        # output = torch.cat(output, dim=0)

        # output = torch.clamp(output, min = -1.0,max = 1.0)
        # rgb = (output.view(1200,1200,1200,3) + 1) / 2
        # img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # from skimage import io
        # def find_closest_index(tensor, target):
        #     distances = torch.sum((tensor - target) ** 2, dim=1)
        #     closest_index = torch.argmin(distances)
        #     return closest_index
        # coor = self.table.reshape(1200,1200,2)
        # point1 = coor[591, 356]
        # point2 = coor[604, 373]
        # point1_1 = self.mlpnet(torch.tensor([591/1199, 356/1199]).to('cuda'))
        # point2_1 = self.mlpnet(torch.tensor([604/1199, 373/1199]).to('cuda'))
        # point1_ = torch.cat([point1, point1_1],dim = -1)
        # point2_ = torch.cat([point2, point2_1],dim = -1)
        # coor1 = find_closest_index(xyz.to('cpu'),point1_.to('cpu')).item()
        # coor2 = find_closest_index(xyz.to('cpu'),point2_.to('cpu')).item()  
        # coor1 = coor1//(1200*1200),coor1%(1200*1200)//1200,coor1%(1200*1200)%1200
        # coor2 = coor2//(1200*1200),coor2%(1200*1200)//1200,coor2%(1200*1200)%1200
        # import os
        # def write_ply_ascii(filedir, coords, feats):
        #     if os.path.exists(filedir): os.system('rm '+filedir)
        #     f = open(filedir,'a+')
        #     f.writelines(['ply\n','format ascii 1.0\n'])
        #     f.write('element vertex '+str(coords.shape[0])+'\n')
        #     f.writelines(['property float x\n','property float y\n','property float z\n', 
        #                 'property uchar red\n','property uchar green\n','property uchar blue\n',])
        #     f.write('end_header\n')
        #     coords = coords.astype('int16')
        #     feats = feats.astype('uint8')
        #     for xyz, rgb in zip(coords, feats):
        #         f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
        #                     str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
        #     f.close() 

        #     return
        # [x, y, z] = torch.meshgrid(torch.linspace(0, 1199, 1200), torch.linspace(0, 1199, 1200), torch.linspace(0, 1199, 1200))

        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # z = z.contiguous().view(-1, 1)
        # xyz = torch.cat([x, y, z],dim = -1)       
        # write_ply_ascii('/data/liufengyi/Results/l.ply', xyz.numpy(), img.reshape(-1,3))

        # print(f"Tensor saved to {file_path}")
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,600,600,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(torch.cat([net_in, out_],dim=-1)) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(torch.cat([self.table, out_],dim=-1)) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output
class DinerMLP_mlp1(nn.Module):   # hash 1 mlp 1
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        in_features = 1
        self.in_features = in_features
        
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(ReluLayer(in_features+1, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)
        
        self.mlpnet = nn.Sequential(
                    nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(0)
                    ],
                    nn.Linear(hidden_features, 1),
                )

    
    def forward(self, coords):
        out_ = self.mlpnet(coords)
        # a1 = self.table[...,0]
        # a2 = self.table[...,1]
        # x_min = a1.min()
        # x_max = a1.max()
        # y_min = a2.min()
        # y_max = a2.max()
        # [x, y] = torch.meshgrid(torch.linspace(x_min, x_max, 1200), torch.linspace(y_min, y_max, 1200))
        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # xy = torch.cat([x, y],dim = -1) # xy shape [H*W,2]
        # xy = xy.to(out_)
        # output = self.net(torch.cat([xy.to(out_), out_],dim=-1)) 
        # output = torch.clamp(output, min = -1.0,max = 1.0)
        # rgb = (output.view(1200,1200,3) + 1) / 2
        # img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # from skimage import io
        # io.imsave('/data/liufengyi/MyCode/DINER/log/test/e3_map.png',img)  
        # def find_closest_index(tensor, target):
        #     distances = torch.sum((tensor - target) ** 2, dim=1)
        #     closest_index = torch.argmin(distances)
        #     return closest_index
        # coor = self.table.reshape(1200,1200,2)
        # point1 = coor[591, 356]
        # point2 = coor[604, 373]
        # coor1 = find_closest_index(xy,point1).item()
        # coor2 = find_closest_index(xy,point2).item()
        # coor1 = coor1//1200,coor1%1200
        # coor2 = coor2//1200,coor2%1200
        # file_path = '/data/liufengyi/MyCode/DINER/map/e3.txt'
        # np.savetxt(file_path, self.table.cpu().numpy())
        # a1 = self.table[...,0]
        # a2 = self.table[...,1]
        # a3 = out_[...,0]
        # x_min = a1.min()
        # x_max = a1.max()
        # y_min = a2.min()
        # y_max = a2.max()
        # z_min = a3.min()
        # z_max = a3.max()
        # [x, y, z] = torch.meshgrid(torch.linspace(x_min, x_max, 1200), torch.linspace(y_min, y_max, 1200), torch.linspace(z_min, z_max, 1200))
        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # z = z.contiguous().view(-1, 1)
        # xyz = torch.cat([x, y, z],dim = -1)
        # xyz = xyz.to(out_)
        # xy = torch.cat([x, y],dim = -1)
        # xy = xy.to(out_)
        # output = []
        # split_data = torch.split(xyz, 1200*1200)
        # for batch in split_data:

        #     output_ = self.net(batch)
            
        #     # Transfer the output back to CPU
        #     output_ = output_.to('cpu')
            
        #     # Accumulate the results
        #     output.append(output_)
        # output = torch.cat(output, dim=0)

        # output = torch.clamp(output, min = -1.0,max = 1.0)
        # rgb = (output.view(1200,1200,1200,3) + 1) / 2
        # img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # from skimage import io
        # def find_closest_index(tensor, target):
        #     distances = torch.sum((tensor - target) ** 2, dim=1)
        #     closest_index = torch.argmin(distances)
        #     return closest_index
        # coor = self.table.reshape(1200,1200,2)
        # point1 = coor[591, 356]
        # point2 = coor[604, 373]
        # point1_1 = self.mlpnet(torch.tensor([591/1199, 356/1199]).to('cuda'))
        # point2_1 = self.mlpnet(torch.tensor([604/1199, 373/1199]).to('cuda'))
        # point1_ = torch.cat([point1, point1_1],dim = -1)
        # point2_ = torch.cat([point2, point2_1],dim = -1)
        # coor1 = find_closest_index(xyz.to('cpu'),point1_.to('cpu')).item()
        # coor2 = find_closest_index(xyz.to('cpu'),point2_.to('cpu')).item()  
        # coor1 = coor1//(1200*1200),coor1%(1200*1200)//1200,coor1%(1200*1200)%1200
        # coor2 = coor2//(1200*1200),coor2%(1200*1200)//1200,coor2%(1200*1200)%1200
        # import os
        # def write_ply_ascii(filedir, coords, feats):
        #     if os.path.exists(filedir): os.system('rm '+filedir)
        #     f = open(filedir,'a+')
        #     f.writelines(['ply\n','format ascii 1.0\n'])
        #     f.write('element vertex '+str(coords.shape[0])+'\n')
        #     f.writelines(['property float x\n','property float y\n','property float z\n', 
        #                 'property uchar red\n','property uchar green\n','property uchar blue\n',])
        #     f.write('end_header\n')
        #     coords = coords.astype('int16')
        #     feats = feats.astype('uint8')
        #     for xyz, rgb in zip(coords, feats):
        #         f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
        #                     str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
        #     f.close() 

        #     return
        # [x, y, z] = torch.meshgrid(torch.linspace(0, 1199, 1200), torch.linspace(0, 1199, 1200), torch.linspace(0, 1199, 1200))

        # x = x.contiguous().view(-1, 1)
        # y = y.contiguous().view(-1, 1)
        # z = z.contiguous().view(-1, 1)
        # xyz = torch.cat([x, y, z],dim = -1)       
        # write_ply_ascii('/data/liufengyi/Results/l.ply', xyz.numpy(), img.reshape(-1,3))

        # print(f"Tensor saved to {file_path}")
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,1200,1200,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).reshape(1,-1).permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(torch.cat([net_in, out_],dim=-1)) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(torch.cat([self.table, out_],dim=-1)) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output    
    
class DinerMLP_coor(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.hash_mod = True
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        H,W = int(math.sqrt(hash_table_length)), int(math.sqrt(hash_table_length))

        [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        x = ((x.contiguous().view(-1, 1)) / (W-1) - 0.5) / 0.5
        y = ((y.contiguous().view(-1, 1)) / (H-1) - 0.5) / 0.5
        xy = torch.cat([x, y],dim = -1).to(self.table) # xy shape [H*W,2] 
        self.xy = nn.parameter.Parameter(xy,requires_grad = False)
        self.net = []

        self.net.append(ReluLayer(in_features+2, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        a1 = self.table[...,0]
        a2 = self.table[...,1]
        x_min = a1.min()
        x_max = a1.max()
        y_min = a2.min()
        y_max = a2.max()
        [x, y] = torch.meshgrid(torch.linspace(x_min, x_max, 1200), torch.linspace(y_min, y_max, 1200))
        x = x.contiguous().view(-1, 1)
        y = y.contiguous().view(-1, 1)
        xy = torch.cat([x, y],dim = -1) # xy shape [H*W,2]
        output = self.net(torch.cat([self.xy,xy.to(self.table)],dim=-1)) 
        output = torch.clamp(output, min = -1.0,max = 1.0)
        rgb = (output.view(1200,1200,3) + 1) / 2
        img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        from skimage import io
        io.imsave('/data/liufengyi/MyCode/DINER/log/test/mlp1_map.png',img)    
            
        # if self.hash_mod:
        #     output = self.net(self.table)
        # else:
        #     output = self.net(coords)

        # output = torch.clamp(output, min = -1.0,max = 1.0)

        # return output
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(torch.cat([self.xy,self.table],dim=-1).reshape(1,1200,1200,self.in_features+2).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(net_in) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(torch.cat([self.xy,self.table],dim=-1)) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output   
class DinerMLP_m(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []

        self.net.append(ReluLayer(in_features+1, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):

        output = self.net(self.table)
        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class DinerMLP_idx(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []

        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self,start,end):
        output = self.net(self.table[start:end])
        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class DinerMLP_interp(nn.Module):
    def __init__(self,
                hash_table_resolution, # [H,W]
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True):
        super().__init__()
        self.opt = HyperParameters()
        self.in_features = in_features
        self.hash_table_resolution = hash_table_resolution
    
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_resolution[0]*hash_table_resolution[1],in_features))*2 -1),requires_grad = True)


        self.net = []
        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
                
            self.net.append(final_linear)
        else:
            self.net.append(ReluLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)

    # coords [N,H*W,2]
    def forward(self, coords):
        grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标
        net_in = nn.functional.grid_sample(self.table.reshape(1,self.hash_table_resolution[0],self.hash_table_resolution[1],self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)
        output = self.net(net_in)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output


class DinerMLP_interp1(nn.Module):
    def __init__(self,
                hash_table_resolution, # [H,W]
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True):
        super().__init__()
        self.opt = HyperParameters()
        self.in_features = in_features
        self.hash_table_resolution = hash_table_resolution
    
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_resolution[0]*hash_table_resolution[1],in_features))*2 -1),requires_grad = True)


        self.net = []
        self.net.append(ReluLayer(in_features+2, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
                
            self.net.append(final_linear)
        else:
            self.net.append(ReluLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)

    # coords [N,H*W,2]
    def forward(self, coords):
        grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标
        net_in = nn.functional.grid_sample(self.table.reshape(1,self.hash_table_resolution[0],self.hash_table_resolution[1],self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)
        output = self.net(net_in)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output
    
    
    
class Diner_MLP_CP(nn.Module):
    def __init__(self,
                hash_table_resolution, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,
                n_vectors):
                
        super().__init__()

        self.hash_table_resolution = hash_table_resolution
        self.width = in_features
        self.n = n_vectors
        self.hash_mod = True
        # self.dim = len(hash_table_resolution)

        if len(self.hash_table_resolution) == 2:
            self.x = nn.parameter.Parameter(1e-4 * (torch.rand(self.hash_table_resolution[0],self.width,self.n)*2 -1),requires_grad = True)
            self.y = nn.parameter.Parameter(1e-4 * (torch.rand(self.hash_table_resolution[1],self.width,self.n)*2 -1),requires_grad = True)

        if len(self.hash_table_resolution) == 3:
            self.table = nn.parameter.Parameter(torch.empty((self.hash_table_resolution[0],self.hash_table_resolution[1],self.hash_table_resolution[2],self.width),requires_grad=True))
            self.x = nn.parameter.Parameter(1e-4 * (torch.rand(self.hash_table_resolution[0],self.width,self.n)*2 -1),requires_grad = True)
            self.y = nn.parameter.Parameter(1e-4 * (torch.rand(self.hash_table_resolution[1],self.width,self.n)*2 -1),requires_grad = True)
            self.z = nn.parameter.Parameter(1e-4 * (torch.rand(self.hash_table_resolution[2],self.width,self.n)*2 -1),requires_grad = True)

        self.net = []

        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):

        if self.hash_mod:
            if len(self.hash_table_resolution) == 2:
                table = torch.empty(self.hash_table_resolution[0],self.hash_table_resolution[1],self.width).to(device = 'cuda:0')
                for w in range(self.width):
                    for i in range(self.n):
                        if i == 0:
                            table[...,w] = torch.einsum('i,j -> ij',self.x[:,w,i],self.y[:,w,i])
                        else:
                            # self.table[...,w] += torch.einsum('i,j -> ij',self.x[:,w,i],self.y[:,w,i])
                            table[...,w] = torch.add(table[...,w],torch.einsum('i,j -> ij',self.x[:,w,i],self.y[:,w,i]))


            if len(self.hash_table_resolution) == 3:
                for w in range(self.width):
                    for i in range(self.n):
                        if i == 0:
                            self.table[...,w] = torch.einsum('i,j,k -> ijk',self.x[:,w,i],self.y[:,w,i],self.z[:,w,i])
                        else:
                            self.table[...,w] =  torch.add(self.table[...,w],torch.einsum('i,j,k -> ijk',self.x[:,w,i],self.y[:,w,i],self.z[:,w,i]))

            table = table.view(-1,self.width)
            output = self.net(table)
        else:
            output = self.net(coords)

        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

                # self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) * np.pi / 2 * self.omega_0, 
                #                              np.sqrt(6 / self.in_features) * np.pi / 2 * self.omega_0)
        
    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out

class Siren(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class DinerSiren(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,1200,1200,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(net_in)
            # for i in range(1200):
            #     for j in range(1200):
            #         x, y = self.table.reshape(1200,1200,-1)[i, j]
            #         r = int((x - x_min) / (x_max - x_min) * 255)
            #         g = int((y - y_min) / (y_max - y_min) * 255)
            #         b = 0  # 或者
            #         output_[i,j] = [r,g,b]

            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(self.table)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        return output
class DinerSiren_mlp(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.mlpnet = nn.Sequential(
                    nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(3)
                    ],
                    nn.Linear(hidden_features, 3),
                )
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        rgb_ = self.mlpnet(coords)
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,1200,1200,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(net_in)
            output = output + rgb_
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(self.table)
            output = output + rgb_
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        
        return output


class DinerSiren_mlp1(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features+2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.mlpnet = nn.Sequential(
                    nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(3)
                    ],
                    nn.Linear(hidden_features, 2),
                )
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        out_ = self.mlpnet(coords)
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,1200,1200,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(torch.cat([net_in, out_],dim=-1)) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(torch.cat([self.table, out_],dim=-1)) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output
    

class DinerSiren_mlp2(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.mlpnet = nn.Sequential(
                    nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(3)
                    ],
                    nn.Linear(hidden_features, 2),
                )
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        out_ = self.mlpnet(coords)
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,1200,1200,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(net_in+out_) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(self.table+out_)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output


class DinerSiren_coor(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        H,W = int(math.sqrt(hash_table_length)), int(math.sqrt(hash_table_length))


        [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        x = ((x.contiguous().view(-1, 1)) / (W-1) - 0.5) / 0.5
        y = ((y.contiguous().view(-1, 1)) / (H-1) - 0.5) / 0.5
        xy = torch.cat([x, y],dim = -1).to(self.table) # xy shape [H*W,2] 
        self.xy = nn.parameter.Parameter(xy,requires_grad = False)
        
        
        # self.table[:,:2] =  xy      
        
        self.net = []
        self.net.append(SineLayer(in_features+2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        # self.mlpnet = nn.Sequential(
        #             nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
        #             *[
        #                 nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
        #                 for _ in range(3)
        #             ],
        #             nn.Linear(hidden_features, 2),
        #         )
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # out_ = self.mlpnet(coords)
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(torch.cat([self.xy,self.table],dim=-1).reshape(1,1200,1200,self.in_features+2).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(net_in) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(torch.cat([self.xy,self.table],dim=-1)) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output    
class DinerSiren_coor1(nn.Module):
    def __init__(self, 
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.interp = False
        self.in_features = in_features
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        
        
        # H,W = int(math.sqrt(hash_table_length)), int(math.sqrt(hash_table_length))
        # [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        # x = ((x.contiguous().view(-1, 1)) / (W-1) - 0.5) / 0.5
        # y = ((y.contiguous().view(-1, 1)) / (H-1) - 0.5) / 0.5
        # xy = torch.cat([x, y],dim = -1).to(self.table) # xy shape [H*W,2] 
        # self.xy = nn.parameter.Parameter(xy,requires_grad = False)
        
        
        # self.table[:,:2] =  xy      
        
        self.net = []
        self.net.append(SineLayer(in_features+2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        # self.mlpnet = nn.Sequential(
        #             nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
        #             *[
        #                 nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
        #                 for _ in range(3)
        #             ],
        #             nn.Linear(hidden_features, 2),
        #         )
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # out_ = self.mlpnet(coords)
        if self.interp:
            grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

            # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

            # net_in [640000,4]
            net_in = nn.functional.grid_sample(self.table.reshape(1,1200,1200,self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

            # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
            # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

            output = self.net(torch.cat([coords,net_in],dim=-1)) 
            
            output = torch.clamp(output, min = -1.0,max = 1.0)             
        else:
            output = self.net(torch.cat([coords,self.table],dim=-1)) 
            output = torch.clamp(output, min = -1.0,max = 1.0)
        
        return output    
        
class DinerSiren_idx(nn.Module):
    def __init__(self,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self,start,end):
        output = self.net(self.table[start:end])
        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class DinerSiren_interp(nn.Module):
    def __init__(self,
                hash_table_resolution, # [H,W]
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30.0):


        super().__init__()
        self.opt = HyperParameters()
        self.in_features = in_features
        # self.table_list.append(nn.parameter.Parameter(1e-4 * (torch.rand((opt.input_sidelength[0]*opt.input_sidelength[1]*(4**i),2))*2 -1),requires_grad = True))

        self.hash_table_resolution = hash_table_resolution

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_resolution[0]*hash_table_resolution[1],in_features))*2 -1),requires_grad = True)


        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    
    # coords [N,H*W,2]
    def forward(self, coords):
        # coords.reshape(1,self.opt.sidelength[0],self.opt.sidelength[1],2)
        # coords = coords.permute(0,3,1,2)

        grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

        # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

        # net_in [640000,4]
        net_in = nn.functional.grid_sample(self.table.reshape(1,self.hash_table_resolution[0],self.hash_table_resolution[1],self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

        # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
        # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

        output = self.net(net_in)
        
        output = torch.clamp(output, min = -1.0,max = 1.0) 
        # torch.clamp_(output, min = -1.0,max = 1.0)

        # output = torch.sigmoid(output)
        return output


class DinerSiren_interp1(nn.Module):
    def __init__(self,
                hash_table_resolution, # [H,W]
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30.0):


        super().__init__()
        self.opt = HyperParameters()
        self.in_features = in_features
        # self.table_list.append(nn.parameter.Parameter(1e-4 * (torch.rand((opt.input_sidelength[0]*opt.input_sidelength[1]*(4**i),2))*2 -1),requires_grad = True))

        self.hash_table_resolution = hash_table_resolution

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_resolution[0]*hash_table_resolution[1],in_features))*2 -1),requires_grad = True)
        self.mlpnet = nn.Sequential(
                    nn.Linear(2, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(3)
                    ],
                    nn.Linear(hidden_features, 2),
                )

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    
    # coords [N,H*W,2]
    def forward(self, coords):
        out_ = self.mlpnet(coords)
        # coords.reshape(1,self.opt.sidelength[0],self.opt.sidelength[1],2)
        # coords = coords.permute(0,3,1,2)

        grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

        # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

        # net_in [640000,4]
        net_in = nn.functional.grid_sample(self.table.reshape(1,self.hash_table_resolution[0],self.hash_table_resolution[1],self.in_features).permute(0,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)

        # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
        # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

        output = self.net(net_in+out_)
        
        output = torch.clamp(output, min = -1.0,max = 1.0) 
        # torch.clamp_(output, min = -1.0,max = 1.0)

        # output = torch.sigmoid(output)
        return output


class Sigmoid_layer(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear = nn.Linear(in_features = in_features, out_features = out_features)

    def forward(self, input):
        return torch.sigmoid(self.linear(input))

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) #返回一个一维的tensor（张量），这个张量包含了从start到end（包括端点）的等距的steps个数据点。这里返回的就是 0,1,2,3...,N-1
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self,
                hash_mod,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,
                N_freqs):

        super().__init__()
        self.N_freqs = N_freqs
        self.PEoutput_features = in_features*(2*self.N_freqs+1)
        # self.table = nn.parameter.Parameter(data=data.to("cuda:0"),requires_grad=True)

        self.hash_mod = hash_mod
        if not hash_mod:
            self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(Embedding(in_features,self.N_freqs))
        self.net.append(ReluLayer(self.PEoutput_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        else:
            output = self.net(coords)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class WaveLayer(nn.Module):
    def __init__(self, in_features, out_features,bias=True,
                 is_first=False,omega_0 = 30):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        # 正态分布 N (0,1)
        self.bias = nn.parameter.Parameter(torch.randn(out_features) / 10,requires_grad = True)

        # self.bias = nn.parameter.Parameter(torch.randn(out_features),requires_grad = True)

        # 均匀分布 U (-1,1)
        # self.bias = nn.parameter.Parameter(torch.rand(out_features) * 2 - 1.0,requires_grad = True)

        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        x1 = self.linear(input)
        x2 = self.linear(input) + self.bias
        return torch.sin(self.omega_0 * x1) * torch.exp(- x2*x2 / 2.0)

class WaveNet(nn.Module):
    def __init__(self,hash_mod,hash_table_length, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,first_omega_0=30, hidden_omega_0=30.0):
        super().__init__()


        self.hash_mod = hash_mod
        if hash_mod:
            self.table = nn.parameter.Parameter(torch.randn((hash_table_length,in_features)),requires_grad = True)

        self.net = []
        self.net.append(WaveLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(WaveLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(WaveLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)


    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        else:
            output = self.net(coords)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

# Activation Function
def WaveletActivation(x):
    out = torch.sin(30*x) * torch.exp(- x*x / 2.0)
    return out

def SincActivation(x):
    return torch.sin(x) / torch.abs(x)


class HashSiren_Lessless(nn.Module):
    def __init__(self, hash_table_length, input_dim, hidden_features=64, hidden_layers=2, out_features=1, outermost_linear=True, 
                 first_omega_0=30., hidden_omega_0=30.):
        super(HashSiren_Lessless, self).__init__()
        
        self.model_amp = DinerSiren(
                            hash_table_length = hash_table_length,
                            in_features = input_dim,
                            hidden_features = hidden_features,
                            hidden_layers = hidden_layers,
                            out_features = out_features,
                            outermost_linear = outermost_linear,
                            first_omega_0 = first_omega_0,
                            hidden_omega_0 = hidden_omega_0)
            
        self.model_phs = DinerSiren(
                            hash_table_length = hash_table_length,
                            in_features = input_dim,
                            hidden_features = hidden_features,
                            hidden_layers = hidden_layers,
                            out_features = out_features,
                            outermost_linear = outermost_linear,
                            first_omega_0 = first_omega_0,
                            hidden_omega_0 = hidden_omega_0)
                
    def forward(self):
        amp = (self.model_amp(None) + 1) / 2
        phs = (self.model_phs(None) + 1) / 2
        return amp, phs