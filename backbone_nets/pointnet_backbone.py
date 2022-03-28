import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MLP_for', 'MLP_rot_inv_for','MLP_rev','MLP_rot_inv_rev']
EPS = 1e-6

class MLP_for(nn.Module):
	def __init__(self, num_pts):
		super(MLP_for,self).__init__()
		self.conv1 = torch.nn.Conv1d(3,64,1)
		self.conv2 = torch.nn.Conv1d(64,64,1)
		self.conv3 = torch.nn.Conv1d(64,64,1)
		self.conv4 = torch.nn.Conv1d(64,128,1)
		self.conv5 = torch.nn.Conv1d(128,1024,1)
		self.conv6 = nn.Conv1d(2596, 512, 1) 
		self.conv7 = nn.Conv1d(512, 256, 1)
		self.conv8 = nn.Conv1d(256, 128, 1)
		self.conv9 = nn.Conv1d(128, 3, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6 = nn.BatchNorm1d(512)
		self.bn7 = nn.BatchNorm1d(256)
		self.bn8 = nn.BatchNorm1d(128)
		self.bn9 = nn.BatchNorm1d(3)
		self.num_pts = num_pts
		self.max_pool = nn.MaxPool1d(num_pts)
		
	def forward(self,x, other_input1=None, other_input2=None, other_input3=None):
		x = x.permute(0, 2, 1)
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		point_features = out
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		global_features = self.max_pool(out)
		global_features_repeated = global_features.repeat(1,1, self.num_pts)
		#out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated],1))))
		
		# Avg_pool
		# avgpool = other_input1
		# avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		# out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool],1))))

		#3DMMImg
		avgpool = other_input1
		avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		
		shape_code = other_input2
		shape_code = shape_code.repeat(1,1,self.num_pts)

		expr_code = other_input3
		expr_code = expr_code.repeat(1,1,self.num_pts)

		out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool, shape_code, expr_code],1))))


		out = F.relu(self.bn7(self.conv7(out)))
		out = F.relu(self.bn8(self.conv8(out)))
		out = self.bn9(self.conv9(out))
		return out.permute(0, 2, 1)


class MLP_rot_inv_for(nn.Module):
	def __init__(self, num_pts):
		super(MLP_rot_inv_for,self).__init__()
		self.conv1 = VNLinearLeakyReLU(1,64, negative_slope=0.0)
		#self.conv2 = VNLinearLeakyReLU(64,64, negative_slope=0.0)
		#self.conv3 = VNLinearLeakyReLU(64,64, negative_slope=0.0)
		#self.conv4 = VNLinearLeakyReLU(64,128, negative_slope=0.0)
		#self.conv5 = VNLinearLeakyReLU(128,1024, negative_slope=0.0)
		#self.conv6 = VNLinearLeakyReLU(2596, 512, negative_slope=0.0)
		#self.conv7 = VNLinearLeakyReLU(512, 256, negative_slope=0.0)
		#self.conv8 = VNLinearLeakyReLU(256, 128, negative_slope=0.0)
		#self.conv9 = VNLinearLeakyReLU(128, 1, negative_slope=0.0)
		#self.num_pts = num_pts
		#self.max_pool = VNMaxPool(1024)

		self.conv2 = torch.nn.Conv1d(192,64,1)
		self.conv3 = torch.nn.Conv1d(64,64,1)
		self.conv4 = torch.nn.Conv1d(64,128,1)
		self.conv5 = torch.nn.Conv1d(128,1024,1)
		self.conv6 = nn.Conv1d(2596, 512, 1) 
		self.conv7 = nn.Conv1d(512, 256, 1)
		self.conv8 = nn.Conv1d(256, 128, 1)
		self.conv9 = nn.Conv1d(128, 3, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6 = nn.BatchNorm1d(512)
		self.bn7 = nn.BatchNorm1d(256)
		self.bn8 = nn.BatchNorm1d(128)
		self.bn9 = nn.BatchNorm1d(3)
		self.num_pts = num_pts
		self.max_pool = nn.MaxPool1d(num_pts)


	def forward(self,x, other_input1=None, other_input2=None, other_input3=None):
		x = x.permute(0, 2, 1)
		x = x.unsqueeze(1)
		out = self.conv1(x)
		#out = self.conv2(out)
		#point_features = out
		#out = self.conv3(out)
		#out = self.conv4(out)
		#out = self.conv5(out)
		#global_features = self.max_pool(out)
		#global_features_repeated = global_features.unsqueeze(-1).repeat(1,1,1, self.num_pts)

		#out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated],1))))
		
		# Avg_pool
		# avgpool = other_input1
		# avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		# out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool],1))))

		#3DMMImg
		# avgpool = other_input1
		# avgpool = avgpool.unsqueeze(2).unsqueeze(2).repeat(1,1,3,self.num_pts)
		
		# shape_code = other_input2
		# shape_code = shape_code.unsqueeze(-1).repeat(1,1,3,self.num_pts)

		# expr_code = other_input3
		# expr_code = expr_code.unsqueeze(-1).repeat(1,1,3,self.num_pts)

		# out = self.conv6(torch.cat([point_features, global_features_repeated, avgpool, shape_code, expr_code],1))


		# out = self.conv7(out)
		# out = self.conv8(out)
		# out = self.conv9(out)
		# out = out.squeeze(1)

		#3DMMImg

		out = F.relu(self.bn2(self.conv2(out)))
		point_features = out
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		global_features = self.max_pool(out)
		global_features_repeated = global_features.repeat(1,1, self.num_pts)
		#out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated],1))))
		
		# Avg_pool
		# avgpool = other_input1
		# avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		# out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool],1))))

		#3DMMImg
		avgpool = other_input1
		avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		
		shape_code = other_input2
		shape_code = shape_code.repeat(1,1,self.num_pts)

		expr_code = other_input3
		expr_code = expr_code.repeat(1,1,self.num_pts)

		out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool, shape_code, expr_code],1))))


		out = F.relu(self.bn7(self.conv7(out)))
		out = F.relu(self.bn8(self.conv8(out)))
		out = self.bn9(self.conv9(out))
		return out.permute(0, 2, 1)


class MLP_rev(nn.Module):
	def __init__(self, num_pts):
		super(MLP_rev,self).__init__()
		self.conv1 = torch.nn.Conv1d(3,64,1)
		self.conv2 = torch.nn.Conv1d(64,64,1)
		self.conv3 = torch.nn.Conv1d(64,64,1)
		self.conv4 = torch.nn.Conv1d(64,128,1)
		self.conv5 = torch.nn.Conv1d(128,1024,1)
		self.conv6_1 = nn.Conv1d(1024, 7, 1)
		self.conv6_2 = nn.Conv1d(1024, 199, 1)
		self.conv6_3 = nn.Conv1d(1024, 29, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6_1 = nn.BatchNorm1d(7)
		self.bn6_2 = nn.BatchNorm1d(199)
		self.bn6_3 = nn.BatchNorm1d(29)
		self.num_pts = num_pts
		self.max_pool = nn.MaxPool1d(num_pts)

	def forward(self,x, other_input1=None, other_input2=None, other_input3=None):
		x = x.permute(0, 2, 1)
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		global_features = self.max_pool(out)

		# Global point feature
		out_rot = F.relu(self.bn6_1(self.conv6_1(global_features)))
		out_shape = F.relu(self.bn6_2(self.conv6_2(global_features)))
		out_expr = F.relu(self.bn6_3(self.conv6_3(global_features)))


		out = torch.cat([out_rot, out_shape, out_expr], 1).squeeze(2)

		return out

class MLP_rot_inv_rev(nn.Module):
	def __init__(self, num_pts):
		super(MLP_rot_inv_rev,self).__init__()
		self.conv1 = VNLinearLeakyReLU(1,64, negative_slope=0.0)
		self.conv2 = VNLinearLeakyReLU(64,64, negative_slope=0.0)
		self.conv3 = VNLinearLeakyReLU(64,64, negative_slope=0.0)
		self.conv4 = VNLinearLeakyReLU(64,128, negative_slope=0.0)
		self.conv5 = VNLinearLeakyReLU(128,1024, negative_slope=0.0)
		self.conv6_1 = nn.Conv1d(1024*3, 7, 1)
		self.conv6_2 = nn.Conv1d(1024*3, 199, 1)
		self.conv6_3 = nn.Conv1d(1024*3, 29, 1)

		self.num_pts = num_pts
		self.bn6_1 = nn.BatchNorm1d(7)
		self.bn6_2 = nn.BatchNorm1d(199)
		self.bn6_3 = nn.BatchNorm1d(29)
		self.max_pool = VNMaxPool(1024)

	def forward(self,x, other_input1=None, other_input2=None, other_input3=None):
		x = x.permute(0, 2, 1)
		x = x.unsqueeze(1)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		global_features = self.max_pool(out)
		global_features_flatten = global_features.view(global_features.shape[0], -1).unsqueeze(-1)

		# Global point feature
		out_rot = F.relu(self.bn6_1(self.conv6_1(global_features_flatten)))
		out_shape = F.relu(self.bn6_2(self.conv6_2(global_features_flatten)))
		out_expr = F.relu(self.bn6_3(self.conv6_3(global_features_flatten)))


		out = torch.cat([out_rot, out_shape, out_expr], 1).squeeze(2)

		return out

class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=4, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out

class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim=4):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x

class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max