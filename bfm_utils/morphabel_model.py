from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio

class  MorphabelModel(object):
    """MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
    :param shapeMU: [3*nver, 1]. *
    :param shapePC: [3*nver, n_shape_para]. *
    :param shapeEV: [n_shape_para, 1]. ~
    :param expMU: [3*nver, 1]. ~ 
    :param expPC: [3*nver, n_exp_para]. ~
    :param expEV: [n_exp_para, 1]. ~
    :param texMU: [3*nver, 1]. ~
    :param texPC: [3*nver, n_tex_para]. ~
    :param texEV: [n_tex_para, 1]. ~
    :param tri: [ntri, 3] (start from 1, should sub 1 in python and c++). *
    :param tri_mouth: [114, 3] (start from 1, as a supplement to mouth triangles). ~
    :param kpt_ind: [68,] (start from 1). ~
    """
    def __init__(self, model_path, model_type = 'BFM'):
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.model = self.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        # fixed attributes
        self.nver = self.model['shapePC'].shape[0]/3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]
        
        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    @staticmethod
    def load_BFM(model_path):
        ''' load BFM 3DMM model
        Args:
            model_path: path to BFM model. 
        Returns:
            model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
                'shapeMU': [3*nver, 1]
                'shapePC': [3*nver, 199]
                'shapeEV': [199, 1]
                'expMU': [3*nver, 1]
                'expPC': [3*nver, 29]
                'expEV': [29, 1]
                'texMU': [3*nver, 1]
                'texPC': [3*nver, 199]
                'texEV': [199, 1]
                'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
                'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
                'kpt_ind': [68,] (start from 1)
        PS:
            You can change codes according to your own saved data.
            Just make sure the model has corresponding attributes.
        '''
        C = sio.loadmat(model_path)
        model = C['model']
        model = model[0,0]

        # change dtype from double(np.float64) to np.float32, 
        # since big matrix process(espetially matrix dot) is too slow in python.
        model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
        model['shapePC'] = model['shapePC'].astype(np.float32)
        model['shapeEV'] = model['shapeEV'].astype(np.float32)
        model['expEV'] = model['expEV'].astype(np.float32)
        model['expPC'] = model['expPC'].astype(np.float32)

        # matlab start with 1. change to 0 in python.
        model['tri'] = model['tri'].T.copy(order = 'C').astype(np.int32) - 1
        model['tri_mouth'] = model['tri_mouth'].T.copy(order = 'C').astype(np.int32) - 1
        
        # kpt ind
        model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

        return model

    # ------------------------------------- shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, type = 'random'):
        if type == 'zero':
            sp = np.random.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.rand(self.n_shape_para, 1)*1e04
        return sp

    def get_exp_para(self, type = 'random'):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = -1.5 + 3*np.random.random([self.n_exp_para, 1])
            ep[6:, 0] = 0

        return ep 

    def generate_vertices(self, shape_para, exp_para):
        '''
        :param: shape_para: (n_shape_para, 1)
        :param: exp_para: (n_exp_para, 1) 
        :return: vertices: (nver, 3)
        '''
        batch_size = shape_para.shape[0]
        shapeMU = np.repeat(self.model['shapeMU'], batch_size, 1)
        vertices = shapeMU[:, :, np.newaxis] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [batch_size, int(len(vertices)/3, int(3)), 1], 'F')

        return vertices

    # -------------------------------------- texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type = 'random'):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tp = np.random.rand(self.n_tex_para, 1)
        return tp

    def generate_colors(self, tex_para):
        '''
        :param tex_para: (n_tex_para, 1)
        :return: colors: (nver, 3)
        '''
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_para*self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T/255.  
        
        return colors


    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        :param: vertices: [nver, 3]
        :param: angles: [3] x, y, z rotation angle(degree)
        :param: x: pitch. positive for looking down 
        :param: y: yaw. positive for looking left
        :param: z: roll. positive for tilting head right
        :return: vertices: rotated vertices
        '''
        return self.rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = self.angle2matrix(angles)
        return self.similarity_transform(vertices, s, R, t3d)

    def transform_3ddfa(self, vertices, s, angles, t3d): # only used for processing 300W_LP data
        R = self.angle2matrix_3ddfa(angles)
        return self.similarity_transform(vertices, s, R, t3d)

    @staticmethod
    def rotate(vertices, angles):
        ''' rotate vertices. 
        X_new = R.dot(X). X: 3 x 1   
        Args:
            vertices: [nver, 3]. 
            rx, ry, rz: degree angles
            rx: pitch. positive for looking down 
            ry: yaw. positive for looking left
            rz: roll. positive for tilting head right
        Returns:
            rotated vertices: [nver, 3]
        '''
        R = MorphabelModel.angle2matrix(angles)
        rotated_vertices = vertices.dot(R.T)

        return rotated_vertices

    @staticmethod
    def similarity_transform(vertices, s, R, t3d):
        ''' similarity transform. dof = 7.
        3D: s*R.dot(X) + t
        Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
        Args:(float32)
            vertices: [nver, 3]. 
            s: [1,]. scale factor.
            R: [3,3]. rotation matrix.
            t3d: [3,]. 3d translation vector.
        Returns:
            transformed vertices: [nver, 3]
        '''
        t3d = np.squeeze(np.array(t3d, dtype = np.float32))
        transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

        return transformed_vertices

    @staticmethod
    def angle2matrix_3ddfa(angles):
        ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
        Args:
            angles: [3,]. x, y, z angles
            x: pitch.
            y: yaw. 
            z: roll. 
        Returns:
            R: 3x3. rotation matrix.
        '''
        x, y, z = angles[0], angles[1], angles[2]
        
        # x
        Rx=np.array([[1,      0,       0],
                    [0, np.cos(x),  np.sin(x)],
                    [0, -np.sin(x),   np.cos(x)]])
        # y
        Ry=np.array([[ np.cos(y), 0, -np.sin(y)],
                    [      0, 1,      0],
                    [np.sin(y), 0, np.cos(y)]])
        # z
        Rz=np.array([[np.cos(z), np.sin(z), 0],
                    [-np.sin(z),  np.cos(z), 0],
                    [     0,       0, 1]])
        R = Rx.dot(Ry).dot(Rz)
        return R.astype(np.float32)

    @staticmethod
    def angle2matrix(angles):
        ''' get rotation matrix from three rotation angles(degree). right-handed.
        Args:
            angles: [3,]. x, y, z angles
            x: pitch. positive for looking down.
            y: yaw. positive for looking left. 
            z: roll. positive for tilting head right. 
        Returns:
            R: [3, 3]. rotation matrix.
        '''
        x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
        # x
        Rx=np.array([[1,      0,       0],
                    [0, np.cos(x),  -np.sin(x)],
                    [0, np.sin(x),   np.cos(x)]])
        # y
        Ry=np.array([[ np.cos(y), 0, np.sin(y)],
                    [      0, 1,      0],
                    [-np.sin(y), 0, np.cos(y)]])
        # z
        Rz=np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z),  np.cos(z), 0],
                    [     0,       0, 1]])
        
        R=Rz.dot(Ry.dot(Rx))
        return R.astype(np.float32)