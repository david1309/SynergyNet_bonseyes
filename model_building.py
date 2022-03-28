import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import scipy.io as sio

# All data parameters import
from utils.params import ParamsPack
# param_pack = ParamsPack()

from backbone_nets import resnet_backbone
from backbone_nets import mobilenetv1_backbone
from backbone_nets import mobilenetv2_backbone
from backbone_nets import ghostnet_backbone
from backbone_nets.pointnet_backbone import MLP_for, MLP_rev, MLP_rot_inv_for, MLP_rot_inv_rev
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
	# Landmarks used to compute center of eyes
	kpt_index.append(20000)
	kpt_index.append(34000)

	# Store the base in a tensor type
	shapeMU = torch.tensor(np.reshape(bfm.model['shapeMU'],[int(3), int(len(bfm.model['shapeMU'])/3)],     'F').T[kpt_index]).unsqueeze(1)
	shapePC = torch.tensor(np.reshape(bfm.model['shapePC'],[int(3), int(len(bfm.model['shapePC'])/3), -1], 'F').transpose(1,2,0)[kpt_index])
	expPC   = torch.tensor(np.reshape(bfm.model['expPC'],  [int(3), int(len(bfm.model['expPC'])/3),   -1], 'F').transpose(1,2,0)[kpt_index])

	return shapeMU, shapePC, expPC


# Main model SynergyNet definition
class SynergyNet(nn.Module):
	def __init__(self, args):
		super(SynergyNet, self).__init__()

		# General config
		self.img_size = args.img_size
		self.device = args.device
		self.crop_images = args.crop_images

		# Morphable model parameters
		bfm_path = 'bfm_utils/morphable_models/BFM.mat'
		shapeMU, shapePC, expPC = get_bfm_params(bfm_path)
		self.shapeMU = shapeMU.to(self.device)
		self.shapePC = shapePC.to(self.device)
		self.expPC = expPC.to(self.device)

		# Image-to-parameter
		self.I2P = I2P(args)  # next(self.I2P.parameters()).device  # check if parameters are in device

		# Forward
		if args.use_rot_inv:
			self.forwardDirection = MLP_rot_inv_for(args.num_lms)
		else:
			self.forwardDirection = MLP_for(args.num_lms)

		# Reverse
		# Forward
		if args.use_rot_inv:
			self.reverseDirection = MLP_rev(args.num_lms)
		else:
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

		self.to(self.device)

	def angle2matrix_3ddfa(self, angles):
		x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
		tensor_0 = torch.zeros_like(x).to(self.device)
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
		landmarks3d_with_help_lands = s.unsqueeze(-1).unsqueeze(-1)*torch.bmm(vertices, R.permute(0,2,1)) + t.unsqueeze(1)

		# Compute
		landmarks3d = landmarks3d_with_help_lands[:,:-2]
		# Add center of eyes and center of head
		center_of_eyes = (landmarks3d_with_help_lands[:,68] + landmarks3d_with_help_lands[:,69])/2.0
		center_head = (landmarks3d_with_help_lands[:,-2] + landmarks3d_with_help_lands[:,-1])/2.0
		# Cat these components
		landmarks3d = torch.cat([landmarks3d, center_of_eyes.unsqueeze(1), center_head.unsqueeze(1)], axis=1)
		landmarks3d[:, :, 1] = h - landmarks3d[:, :, 1] + 1

		return landmarks3d

	@staticmethod
	def parse_target_params(target):
		pose = target["pose_params"]
		shape = target["shape_params"]
		exp = target["exp_params"]

		return torch.cat((pose, shape, exp), 1)

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

	def process_input(self, input, bbox=None):
		input = input.clone()

		if self.crop_images and (bbox is not None):
			bbox = bbox.type(torch.int)
			batch_size = input.shape[0]
			valid_samples = []
			
			for i in range(batch_size):
				input_i = input[i,:, bbox[i,1] : bbox[i,1] + bbox[i,3], bbox[i,0] : bbox[i,0] + bbox[i,2]]

				if input_i.shape[1] > 0  and input_i.shape[2] > 0:
					valid_samples.append(i)
					resize = transforms.Resize((self.img_size, self.img_size))
					input[i] = resize(input_i)
			
			input = input[valid_samples]

		return input.to(self.device, non_blocking=True)

	def process_target(self, target):
		for key in target.keys():
			target[key].requires_grad = False
			target[key] = target[key].float().to(self.device, non_blocking=True)
		return target

	def forward(self, input, target):
		# General config
		target = self.process_target(target)
		input = self.process_input(input, bbox=target["bbox"])

		param_loss_factor = 100 * 1
		param_diff_loss_factor = 100 * 1

		# Image to 3DMM Parameters
		_3D_attr, avgpool = self.I2P(input)
		_3D_attr_GT = self.parse_target_params(target)
		pose_para, shape_para, exp_para = self.parse_pred_params(_3D_attr)
		vertex_lmk = self.lm_from_params(pose_para, shape_para, exp_para, input.shape[2])  # Coarse landamrks: Lc
		vertex_GT_lmk = target["lm3d"].permute(0, 2, 1)
		# gt = self.lm_from_params(target["pose_params"].unsqueeze(-1), target["shape_params"].unsqueeze(-1), target["exp_params"].unsqueeze(-1), input.shape[2])

		self.loss['loss_lmk_s1'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk)
		self.loss['loss_param_s1'] = 0.02 * param_loss_factor * self.ParamLoss(_3D_attr, _3D_attr_GT)

		# Coarse landmarks to Refined landmarks
		point_residual = self.forwardDirection(vertex_lmk, avgpool, shape_para, exp_para)
		vertex_lmk = vertex_lmk + point_residual  # Refined landmarks: Lr = Lc + L_residual
		self.loss['loss_lmk_s2'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk)

		# Refined landmarks to 3DMM parameters
		_3D_attr_S2 = self.reverseDirection(vertex_lmk)
		self.loss['loss_param_s2'] = 0.02 * param_loss_factor * self.ParamLoss(_3D_attr_S2, _3D_attr_GT, mode='only_3dmm')
		self.loss['loss_param_s1s2'] = 0.02 * param_diff_loss_factor * self.ParamLoss(_3D_attr_S2, _3D_attr, mode='only_3dmm')  # 0.001

		return self.loss

	def forward_test(self, input, bbox=None):
		"""test time forward"""
		# General config
		input = self.process_input(input, bbox=bbox)

		with torch.no_grad():
			# Image to 3DMM Parameters
			_3D_attr, avgpool = self.I2P.forward(input)
			pose_para, shape_para, exp_para = self.parse_pred_params(_3D_attr)
			vertex_lmk = self.lm_from_params(pose_para, shape_para, exp_para, input.shape[2])  # Coarse landamrks: Lc

			# Coarse landmarks to Refined landmarks
			point_residual = self.forwardDirection(vertex_lmk, avgpool, shape_para, exp_para)
			vertex_lmk = vertex_lmk + point_residual  # Refined landmarks: Lr = Lc + L_residual

		return vertex_lmk, pose_para

	def get_losses(self):
		return self.loss.keys()


if __name__ == '__main__':
	pass