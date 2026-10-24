"""Compare TextOp against its original deployment and Isaac Lab 2.1 math.

uv run --with torch validate_inference.py --source /path/to/pinned/source
  --weights /path/to/latest.onnx --output /outside/git/parity.json
The source directory also contains isaaclab-math.py, pinned in native-sources.json.
"""
import argparse
import ast
from dataclasses import replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from adapter import TextOp
from motion_challenge.physics import World, forward_kinematics
from motion_challenge.sdk import ReferenceFrame


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root/'native-sources.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, digest in pins['files'].items():
        assert sha(args.source/path) == digest, path
    math = args.source/'isaaclab-math.py'
    assert sha(math) == pins['isaaclab']['sha256']
    scope = {'np': np, 'torch': torch, 'Enum': Enum}
    functions = {'normalize', 'quat_conjugate', 'quat_inv', 'quat_mul',
                 'quat_apply', 'subtract_frame_transforms', 'matrix_from_quat'}
    nodes = [n for n in ast.parse(math.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    assert {n.name for n in nodes} == functions
    for n in nodes:
        n.decorator_list = []
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+nodes, []), str(math), 'exec'), scope)
    deploy = args.source/'TextOpTracker/scripts/deploy_mujoco.py'
    tree = ast.parse(deploy.read_text())
    observation_functions = {'quat_rotate_inverse_np', 'get_command', 'motion_anchor_pos_b_future',
        'motion_anchor_ori_b_future', 'get_base_lin_vel', 'get_base_ang_vel', 'get_joint_pos_rel',
        'get_joint_vel_rel', 'get_last_action', 'get_projected_gravity', 'compute_observation'}
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == '__fix__add_marker_to_scene':
            break
        if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef)):
            nodes.append(n)
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in observation_functions]
    exec(compile(ast.Module(nodes, []), str(deploy), 'exec'), scope)
    runtime = json.loads((root/'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    policy = TextOp(root, runtime)
    assert policy.names == scope['isaaclab_joint_names']
    assert runtime['reference_offsets_s'] == [0., .02, .04, .06, .08]
    assert runtime['control_period_s'] == .02
    assert policy.cfg['action_clip'] is None
    for key, native in [('default', 'default_angles'), ('scale', 'action_scale'), ('kp', 'kps'), ('kd', 'kds')]:
        np.testing.assert_array_equal(getattr(policy, key), scope[native][scope['mujoco_to_isaaclab_reindex']])
    assert scope['anchor_body'] == scope['AnchorBody'].PELVIS
    target_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'target_dof_pos' for t in n.targets) and
        isinstance(n.value, ast.BinOp))
    target_code = compile(ast.Expression(target_node.value), str(deploy), 'eval')
    world = World(root/'MotionChallenge-Core/robots/g1-29dof-cpu-v0.json')
    assert list(world.joint_names) == scope['mujoco_joint_names']
    rng = np.random.default_rng(241024)
    shuffle = rng.permutation(29)
    names = tuple(np.asarray(world.joint_names)[shuffle])
    canonical = [names.index(n) for n in world.joint_names]
    errors = {k: 0. for k in ('observation', 'network_same_input', 'action', 'motor_target',
        'history', 'reset_history', 'stiffness', 'damping', 'unclipped_target', 'unclipped_history')}
    def compare(k, a, b):
        errors[k] = max(errors[k], float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    for reset in range(4):
        frames = []
        for step in range(40):
            rotation = R.from_euler('xyz', [.1, np.pi/2-1e-6, -.5]) if step % 10 == 0 else R.random(random_state=rng)
            world.data.qpos[:] = np.r_[rng.normal(0, .4, 2), .8, rotation.as_quat()[[3, 0, 1, 2]], rng.normal(0, .3, 29)]
            world.data.qvel[:] = rng.normal(0, .5, 35)
            forward_kinematics(world.model, world.data)
            s = world.state()
            frames.append(ReferenceFrame(step*.02, step*.02, s.joint_position_rad, s.root_position_world_m,
                s.root_orientation_xyzw, s.joint_velocity_rad_s, s.root_linear_velocity_world_m_s,
                s.root_angular_velocity_world_rad_s, s.bodies))
        reorder = scope['mujoco_to_isaaclab_reindex']
        loader = NS(T=40, future_steps=5, anchor_body_name='pelvis', anchor_body_index=0,
            joint_pos=np.asarray([f.joint_position_rad for f in frames])[:, reorder],
            joint_vel=np.asarray([f.joint_velocity_rad_s for f in frames])[:, reorder],
            body_pos=np.asarray([f.root_position_world_m for f in frames])[:, None],
            body_ori=np.asarray([f.root_orientation_xyzw for f in frames])[:, None, [3, 0, 1, 2]])
        for step in range(40):
            world.data.qpos[:] = world._qpos(frames[step])
            world.data.qpos[7:] += rng.normal(0, .2, 29)
            world.data.qvel[:] = rng.normal(0, .7, 35)
            forward_kinematics(world.model, world.data)
            s = world.state()
            s = replace(s, joint_names=names, joint_position_rad=tuple(np.asarray(s.joint_position_rad)[shuffle]),
                        joint_velocity_rad_s=tuple(np.asarray(s.joint_velocity_rad_s)[shuffle]))
            refs = tuple(replace(frames[min(step+i, 39)],
                joint_position_rad=tuple(np.asarray(frames[min(step+i, 39)].joint_position_rad)[shuffle]),
                joint_velocity_rad_s=tuple(np.asarray(frames[min(step+i, 39)].joint_velocity_rad_s)[shuffle])) for i in range(5))
            if step == 0:
                policy.reset(s, refs, reset)
                compare('reset_history', policy.last, 0.)
            obs = scope['compute_observation'](world.data, loader, step, policy.last.copy())[None]
            actual_obs = policy.observations(s, refs)['obs']
            compare('observation', actual_obs, obs)
            expected = policy.session.run(None, {'obs': obs})[0].reshape(29)
            compare('network_same_input', policy.infer({'obs': obs})['actions'].reshape(29), expected)
            target = eval(target_code, dict(scope, action=expected))
            result = policy.act(s, refs)
            compare('action', policy.last, expected)
            compare('motor_target', np.asarray(result.joint_position_target_rad)[canonical], target)
            compare('history', policy.last, policy.infer({'obs': actual_obs})['actions'].reshape(29))
            compare('stiffness', np.asarray(result.kp_nm_rad)[canonical], scope['kps'])
            compare('damping', np.asarray(result.kd_nm_s_rad)[canonical], scope['kds'])
        raw = np.linspace(-25, 25, 29, dtype=np.float32)
        infer = policy.infer
        policy.infer = lambda _: {'actions': raw[None]}
        result = policy.act(s, refs)
        policy.infer = infer
        compare('unclipped_target', np.asarray(result.joint_position_target_rad)[canonical], eval(target_code, dict(scope, action=raw)))
        compare('unclipped_history', policy.last, raw)
    limits = {k: 2e-4 for k in errors}
    for k in ('network_same_input', 'reset_history', 'stiffness', 'damping', 'unclipped_target', 'unclipped_history'):
        limits[k] = 0.
    report = {'passed': all(errors[k] <= limits[k] for k in errors), 'cases': 160, 'resets': 4,
        'max_abs_error': errors, 'limits': limits, 'source_commit': pins['source_commit'],
        'checkpoint_sha256': runtime['weights_sha256'], 'isaaclab_commit': pins['isaaclab']['commit'],
        'scope': 'Original MuJoCo deployment observations, Isaac Lab 2.1 math, raw ONNX, PD targets/gains and action history; shuffled SDK joint order, near-vertical poses and clipped reference endpoints. Canonical benchmark initialization and physics replace the author viewer simulation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
