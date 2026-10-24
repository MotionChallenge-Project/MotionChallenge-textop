"""TextOp Tracker's 431D pelvis-relative, five-reference ONNX policy."""
import numpy as np
from motion_challenge.adapters.onnx import ONNXController


def multiply(a, b):
    """Isaac Lab's float32 quaternion product, in wxyz order."""
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    ww = (z1+x1)*(x2+y2)
    yy = (w1-y1)*(w2+z2)
    zz = (w1+y1)*(w2-z2)
    xx = ww+yy+zz
    qq = .5*(xx+(z1-x1)*(x2-y2))
    return np.stack([qq-ww+(z1-y1)*(y2-z2), qq-xx+(x1+w1)*(x2+w2),
                     qq-yy+(w1-x1)*(y2+z2), qq-zz+(z1+y1)*(w2-x2)], axis=-1)


def first_columns(q):
    w, x, y, z = q.T
    s = 2. / np.sum(q*q, axis=-1)
    return np.stack([1-s*(y*y+z*z), s*(x*y-z*w), s*(x*y+z*w),
                     1-s*(x*x+z*z), s*(x*z-y*w), s*(y*z+x*w)], axis=-1)


class TextOp(ONNXController):
    action_output = 'actions'
    raw_action_history = True

    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        for name in ('default', 'scale', 'kp', 'kd'):
            setattr(self, name, getattr(self, name).astype(np.float32))
        assert self.session.get_inputs()[0].shape == [1, 431]

    def observations(self, state, reference):
        if len(reference) != 5:
            raise ValueError('TextOp requires current and four future reference frames')
        q, qd, omega, gravity, rotation = self.proprio(state)
        # The author performs relative future-pose transforms in float32.
        p = np.asarray(state.root_position_world_m, np.float32)
        actual = np.asarray(state.root_orientation_xyzw, np.float32)[[3, 0, 1, 2]]
        inverse = actual.copy()
        inverse[1:] *= -1
        inverse /= np.linalg.norm(inverse)
        inverse = np.repeat(inverse[None], 5, axis=0)
        desired = np.asarray([f.root_orientation_xyzw for f in reference], np.float32)[:, [3, 0, 1, 2]]
        delta = np.asarray([f.root_position_world_m for f in reference], np.float32)-p
        cross = 2*np.cross(inverse[:, 1:], delta)
        position = delta+inverse[:, :1]*cross+np.cross(inverse[:, 1:], cross)
        orientation = first_columns(multiply(inverse, desired))
        target_q = np.asarray([f.joint_position_rad for f in reference])[:, self.order]
        target_v = np.asarray([f.joint_velocity_rad_s for f in reference])[:, self.order]
        velocity = rotation.inv().apply(state.root_linear_velocity_world_m_s)
        obs = np.concatenate([target_q.ravel(), target_v.ravel(), position.ravel(), orientation.ravel(),
                              gravity, velocity, omega, q-self.default, qd, self.last]).astype(np.float32)
        return {'obs': obs[None]}


def create(root, runtime):
    return TextOp(root, runtime)
