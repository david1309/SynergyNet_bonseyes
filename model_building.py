import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import scipy.io as sio

# All data parameters import
from utils.params import ParamsPack
# param_pack = ParamsPack()

from backbone_nets import resnet_backbone
from backbone_nets import mobilenetv1_backbone
from backbone_nets import mobilenetv2_backbone
from backbone_nets import ghostnet_backbone
from backbone_nets.pointnet_backbone import MLP_for, MLP_rev
from loss_definition import ParamLoss, WingLoss

from bfm_utils.morphabel_model import MorphabelModel

# Image-to-parameter
class I2P(nn.Module):
	def __init__(self, args):
		super(I2P, self).__init__()
		self.args = args
		# backbone definition
		if 'mobilenet_v2' in self.args.arch:
			self.backbone = getattr(mobilenetv2_backbone, args.arch)(pretrained=False)
		elif 'mobilenet' in self.args.arch:
			self.backbone = getattr(mobilenetv1_backbone, args.arch)()		
		elif 'resnet' in self.args.arch:
			self.backbone = getattr(resnet_backbone, args.arch)(pretrained=False)
		elif 'ghostnet' in self.args.arch:
			self.backbone = getattr(ghostnet_backbone, args.arch)()
		else:
			raise RuntimeError("Please choose [mobilenet_v2, mobilenet_1, resnet50, or ghostnet]")

	def forward(self,input):
		"""Training time forward"""
		_3D_attr, avgpool = self.backbone(input)
		return _3D_attr, avgpool

def get_bfm_params(bfm_path):
	# Load model
	bfm = MorphabelModel(bfm_path)

	# Get additional indices
	kpt_index = list(bfm.kpt_ind)
	# Middle left eye
	kpt_index.append(int((bfm.kpt_ind[43] + bfm.kpt_ind[46])/2))
	# Middle right eye
	kpt_index.append(int((bfm.kpt_ind[37] + bfm.kpt_ind[40])/2)-70)
	# 71 landmark
	kpt_index.append(int((bfm.kpt_ind[25] + bfm.kpt_ind[26])/2))
	# 72 landmark
	kpt_index.append(40424)
	# 73 landmark
	kpt_index.append(int((bfm.kpt_ind[18] + bfm.kpt_ind[19])/2))
	# 74 landmark
	kpt_index.append(int(bfm.kpt_ind[37]) + 40)
	# 75 landmark
	kpt_index.append(int(bfm.kpt_ind[46]) + 20)

	# TODO: Damiano, this is easy but I am tired XD
	# landmarks3d = image_vertices[kpt_index]
	# # Middle of eye
	# middle_eye = (landmarks3d[68] + landmarks3d[69])/2.0
	# # Center head
	# center_head = (image_vertices[20000] + image_vertices[34000])/2.0
	# landmarks3d = np.concatenate([landmarks3d, middle_eye[np.newaxis], center_head[np.newaxis]])

	# Store the base in a tensor type
	shapeMU = torch.tensor(np.reshape(bfm.model['shapeMU'],[int(3), int(len(bfm.model['shapeMU'])/3)],     'F').T[kpt_index]).unsqueeze(1).cuda()
	shapePC = torch.tensor(np.reshape(bfm.model['shapePC'],[int(3), int(len(bfm.model['shapePC'])/3), -1], 'F').transpose(1,2,0)[kpt_index]).cuda()
	expPC   = torch.tensor(np.reshape(bfm.model['expPC'],  [int(3), int(len(bfm.model['expPC'])/3),   -1], 'F').transpose(1,2,0)[kpt_index]).cuda()

	return shapeMU, shapePC, expPC


# Main model SynergyNet definition
class SynergyNet(nn.Module):
	def __init__(self, args):
		super(SynergyNet, self).__init__()
		bfm_path = 'bfm_utils/morphable_models/BFM.mat'
		self.shapeMU, self.shapePC, self.expPC = get_bfm_params(bfm_path)
		self.img_size = args.img_size

		# Image-to-parameter
		self.I2P = I2P(args)

		# Forward
		self.forwardDirection = MLP_for(args.num_lms)

		# Reverse
		self.reverseDirection = MLP_rev(args.num_lms)

		# Losses
		self.LMKLoss_3D = WingLoss()
		self.ParamLoss = ParamLoss()
		self.loss = {
			'loss_lmk_s1': 0.0,
			'loss_lmk_s2': 0.0,
			'loss_param_s1': 0.0,
			'loss_param_s2': 0.0,
			'loss_param_s1s2': 0.0,
			}

	def angle2matrix_3ddfa(self, angles):
		x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
		tensor_0 = torch.zeros_like(x).cuda()
		tensor_1 = tensor_0 + 1
        
		# x
		Rx=torch.stack([
					torch.stack([tensor_1,      tensor_0,      tensor_0]),
                    torch.stack([tensor_0,  torch.cos(x),  torch.sin(x)]),
                    torch.stack([tensor_0, -torch.sin(x),  torch.cos(x)])]).permute(2, 0, 1)
        # y
		Ry=torch.stack([
					torch.stack([torch.cos(y), tensor_0, -torch.sin(y)]),
                    torch.stack([    tensor_0, tensor_1,      tensor_0]),
                    torch.stack([torch.sin(y), tensor_0, torch.cos(y)])]).permute(2, 0, 1)
        # z
		Rz=torch.stack([
					torch.stack([ torch.cos(z),  torch.sin(z), tensor_0]),
                    torch.stack([-torch.sin(z),  torch.cos(z), tensor_0]),
                    torch.stack([     tensor_0,      tensor_0, tensor_1])]).permute(2, 0, 1)
		R = torch.bmm(Rx, Ry)
		R = torch.bmm(R,  Rz)
		return R


	def lm_from_params(self, pose_para, shape_para, exp_para, h):
		# Get parameters
		s = pose_para[:, -1, 0]/100  # Scale
		angles = pose_para[:, :3, 0]  # Rotation angles
		t = pose_para[:, 3:6, 0]*h  # Translation

		# Denormalize values
		shape_para = shape_para*1e7
		exp_para = exp_para*10

		# Generate vertices + apply transforms (rotation, translation, scaling)
		vertices = self.shapeMU.permute(1,0,2) + (shape_para[...,0] @ self.shapePC + exp_para[...,0] @ self.expPC).permute(1,0,2)
		R = self.angle2matrix_3ddfa(angles)

		# Get the  3d landmarks
		landmarks3d = s.unsqueeze(-1).unsqueeze(-1)*torch.bmm(vertices, R.permute(0,2,1)) + t.unsqueeze(1)
		landmarks3d[:, :, 1] = h - landmarks3d[:, :, 1] + 1

		return landmarks3d

	@staticmethod
	def parse_target_params(target):
		pose = target["pose_params"]
		shape = target["shape_params"]
		exp = target["exp_params"]

		return torch.cat((pose, shape, exp), 1).type(torch.cuda.FloatTensor)
	
	@staticmethod
	def parse_pred_params(pred):
		"""
		num_pose= 7,
        num_shape = 199,
        num_exp = 29,
		"""
		pose_para = pred[:, 0:7].reshape(-1, 7, 1)
		shape_para = pred[:, 7: 199+7].reshape(-1, 199, 1)
		exp_para = pred[:, 199+7: 199+7+29].reshape(-1, 29, 1)

		return pose_para, shape_para, exp_para
		
	def forward(self, input, target):
		# Image to 3DMM Parameters
		_3D_attr, avgpool = self.I2P(input)
		_3D_attr_GT = self.parse_target_params(target)
		pose_para, shape_para, exp_para = self.parse_pred_params(_3D_attr)
		vertex_lmk = self.lm_from_params(pose_para, shape_para, exp_para, input.shape[2])  # Coarse landamrks: Lc
		vertex_GT_lmk = target["lm3d"].permute(0, 2, 1)
		# gt = self.lm_from_params(target["pose_params"].unsqueeze(-1), target["shape_params"].unsqueeze(-1), target["exp_params"].unsqueeze(-1), input.shape[2])

		self.loss['loss_lmk_s1'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk)		
		self.loss['loss_param_s1'] = 0.02 * self.ParamLoss(_3D_attr, _3D_attr_GT)
		
		# Coarse landmarks to Refined landmarks
		point_residual = self.forwardDirection(vertex_lmk, avgpool, shape_para, exp_para)
		vertex_lmk = vertex_lmk + point_residual  # Refined landmarks: Lr = Lc + L_residual
		self.loss['loss_lmk_s2'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk)

		# Refined landmarks to 3DMM parameters
		_3D_attr_S2 = self.reverseDirection(vertex_lmk)
		self.loss['loss_param_s2'] = 0.02 * self.ParamLoss(_3D_attr_S2, _3D_attr_GT, mode='only_3dmm')
		self.loss['loss_param_s1s2'] = 0.001 * self.ParamLoss(_3D_attr_S2, _3D_attr, mode='only_3dmm')

		return self.loss

	def forward_test(self, input):
		"""test time forward"""
		with torch.no_grad():
			# Image to 3DMM Parameters
			_3D_attr, avgpool = self.I2P.forward(input)
			pose_para, shape_para, exp_para = self.parse_pred_params(_3D_attr)
			vertex_lmk = self.lm_from_params(pose_para, shape_para, exp_para, input.shape[2])  # Coarse landamrks: Lc

			# Coarse landmarks to Refined landmarks
			point_residual = self.forwardDirection(vertex_lmk, avgpool, shape_para, exp_para)
			vertex_lmk = vertex_lmk + point_residual  # Refined landmarks: Lr = Lc + L_residual

		return vertex_lmk, pose_para, shape_para, exp_para 

	def get_losses(self):
		return self.loss.keys()


if __name__ == '__main__':
	pass