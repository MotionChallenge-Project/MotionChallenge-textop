"""TextOp's author hardware deployment: 431D observations and raw ONNX actions."""
import numpy as np
from motion_challenge.adapters.onnx import ONNXController


def multiply(a, b):
    w,x,y,z = np.asarray(a,np.float32).T
    v,i,j,k = np.asarray(b,np.float32).T
    return np.stack([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v],axis=-1)


def inverse(q):
    return q*np.array([1,-1,-1,-1],np.float32)/np.maximum(np.sum(q*q,axis=-1,keepdims=True),np.float32(1e-9))


def heading(q):
    w,x,y,z = q
    yaw = np.arctan2(np.float32(2)*(w*z+x*y),np.float32(1)-np.float32(2)*(y*y+z*z))
    return np.array([np.cos(yaw*np.float32(.5)),0,0,np.sin(yaw*np.float32(.5))],np.float32)


def first_columns(q):
    w,x,y,z = q.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*y+w*z),
                     1-2*(x*x+z*z),2*(x*z-w*y),2*(y*z+w*x)],axis=-1)


def rotate_inverse(q, v):
    w,xyz = q[0],q[1:]
    return v*(2*w*w-1)-np.cross(xyz,v)*w*2+xyz*np.sum(xyz*v)*2


class TextOp(ONNXController):
    action_output = 'actions'
    raw_action_history = True

    def __init__(self, root, runtime):
        super().__init__(root,runtime)
        for name in ('default','scale','kp','kd'):
            setattr(self,name,getattr(self,name).astype(np.float32))
        assert self.session.get_inputs()[0].shape == [1,431]

    def reset(self, state, reference, seed):
        super().reset(state,reference,seed)
        actual = np.asarray(state.root_orientation_xyzw,np.float32)[[3,0,1,2]]
        desired = np.asarray(reference[0].root_orientation_xyzw,np.float32)[[3,0,1,2]]
        self.alignment = multiply(heading(actual),inverse(heading(desired)))
        self.last = (np.asarray(self.cfg['deployment_start_joint_pos'],np.float32)-self.default)/self.scale

    def observations(self, state, reference):
        if len(reference) != 5:
            raise ValueError('TextOp requires current and four future reference frames')
        q = np.asarray(state.joint_position_rad,np.float32)[self.order]
        qd = np.asarray(state.joint_velocity_rad_s,np.float32)[self.order]
        actual = np.asarray(state.root_orientation_xyzw,np.float32)[[3,0,1,2]]
        desired = np.asarray([f.root_orientation_xyzw for f in reference],np.float32)[:,[3,0,1,2]]
        orientation = first_columns(multiply(inverse(actual),multiply(self.alignment,desired)))
        target_q = np.asarray([f.joint_position_rad for f in reference],np.float32)[:,self.order]
        target_v = np.asarray([f.joint_velocity_rad_s for f in reference],np.float32)[:,self.order]
        w,x,y,z = actual
        gravity = np.array([2*(-z*x+w*y),-2*(z*y+w*x),1-2*(w*w+z*z)],np.float32)
        velocity = rotate_inverse(actual,np.asarray(state.root_linear_velocity_world_m_s,np.float32))
        omega = rotate_inverse(actual,np.asarray(state.root_angular_velocity_world_rad_s,np.float32))
        # observation_computer.cpp unconditionally zeros these 15 deployment inputs.
        obs = np.concatenate([target_q.ravel(),target_v.ravel(),np.zeros(15,np.float32),
            orientation.ravel(),gravity,velocity,omega,q-self.default,qd,self.last]).astype(np.float32)
        return {'obs':obs[None]}


def create(root, runtime):
    return TextOp(root,runtime)
