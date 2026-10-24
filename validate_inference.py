"""Compare the adapter with compiled, hash-pinned author C++ observations.

Needs a C++17 compiler only for validation. ROS transport is stubbed; the author's
observation_computer.cpp and header are compiled unchanged.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from adapter import TextOp
from motion_challenge.sdk import ProprioceptiveStatePacket, ReferenceFrame


def build_native(source, build):
    files = {
        'rclcpp/rclcpp.hpp':'#pragma once\nnamespace rclcpp {class Node{};}\n#define RCLCPP_INFO(...) ((void)0)\n',
        'unitree_hg/msg/low_state.hpp':'''#pragma once
#include <array>
namespace unitree_hg::msg {
struct LowState {struct IMU {std::array<float,4> quaternion; std::array<float,3> gyroscope;} imu_state;
struct Motor {float q,dq;}; std::array<Motor,29> motor_state;};}
''',
        'motion_loader.hpp':'''#pragma once
#include <array>
#include <vector>
struct MotionLoader {int T, future_steps=5, anchor_body_index=0;
std::vector<std::vector<float>> joint_pos,joint_vel;
std::vector<std::vector<std::array<float,3>>> body_pos;
std::vector<std::vector<std::array<float,4>>> body_ori;};
''',
        'wrapper.cpp':'''#include <algorithm>
#include "textop_ctrl/observation_computer.hpp"
extern "C" {
void* make(const float* input,int length) {
 auto m=std::make_shared<MotionLoader>();m->T=length;
 for(int i=0;i<length;++i) {auto x=input+i*65;
  m->joint_pos.emplace_back(x,x+29);m->joint_vel.emplace_back(x+29,x+58);
  m->body_pos.push_back({{x[58],x[59],x[60]}});
  m->body_ori.push_back({{x[61],x[62],x[63],x[64]}});}
 return new ObservationComputer(m,nullptr);}
void step(void* handle,const float* s,const float* last,int t,float* out) {
 unitree_hg::msg::LowState state;
 for(int j=0;j<29;++j) {state.motor_state[j].q=s[j];state.motor_state[j].dq=s[29+j];}
 std::copy(s+58,s+62,state.imu_state.quaternion.begin());
 std::copy(s+62,s+65,state.imu_state.gyroscope.begin());
 auto c=static_cast<ObservationComputer*>(handle);
 c->set_odometry({s[68],s[69],s[70]},{s[65],s[66],s[67]});
 auto result=c->compute_observation(state,t,std::vector<float>(last,last+29));
 std::copy(result.begin(),result.end(),out);}
void destroy(void* handle) {delete static_cast<ObservationComputer*>(handle);}
}
'''}
    for name,content in files.items():
        p=build/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    native=source/'TextOpDeploy/src/textop_ctrl'
    library=build/'observations.so'
    subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off',
        '-include','iostream','-I'+str(build),'-I'+str(native/'include'),
        str(native/'src/observation_computer.cpp'),str(build/'wrapper.cpp'),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.make.argtypes=[array,ctypes.c_int];lib.make.restype=ctypes.c_void_p
    lib.step.argtypes=[ctypes.c_void_p,array,array,ctypes.c_int,array]
    lib.destroy.argtypes=[ctypes.c_void_p]
    return lib


def main():
    parser=argparse.ArgumentParser()
    for key in ('source','weights','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins=json.loads((root/'native-sources.json').read_text())
    for name,digest in pins['files'].items():assert sha(args.source/name)==digest,name
    runtime=json.loads((root/'runtime.json').read_text());runtime['weights']=str(args.weights.resolve())
    assert sha(args.weights)==runtime['weights_sha256']
    policy=TextOp(root,runtime)
    native=args.source/'TextOpDeploy/src/textop_ctrl'
    yaml=(native/'config/g1_29dof.yaml').read_text()
    extract=lambda name:json.loads(re.search(r'^'+name+r':\s*(\[[\s\S]*?\])',yaml,re.M).group(1))
    names=extract('joint_names');order=[names.index(n) for n in policy.names]
    for field,key in [('default','default_angles'),('scale','action_scale'),('kp','kps'),('kd','kds')]:
        np.testing.assert_array_equal(getattr(policy,field),np.asarray(extract(key),np.float32)[order])
    cpp=(native/'src/textop_onnx_controller.cpp').read_text()
    start=re.search(r'std::vector<float> start_pose = \{(.*?)\};',cpp,re.S).group(1)
    start=np.array([float(x.strip().removesuffix('f')) for x in start.split(',') if x.strip()],np.float32)[order]
    rng=np.random.default_rng(241025);shuffled=tuple(np.asarray(names)[rng.permutation(29)])
    canonical=[shuffled.index(n) for n in names]
    errors={k:0. for k in ('observation','action','motor_target','reset_history','history','position_independence')}
    def compare(key,a,b):errors[key]=max(errors[key],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    with tempfile.TemporaryDirectory() as directory:
        lib=build_native(args.source,Path(directory))
        for reset in range(4):
            frames=[]
            for t in range(40):
                rotation=Rotation.random(random_state=rng).as_quat()
                if t%9==0:rotation=-rotation
                frames.append(ReferenceFrame(t*.02,t*.02,tuple(rng.normal(0,.3,29)),
                    tuple(rng.normal(0,1,3)),tuple(rotation),tuple(rng.normal(0,.5,29))))
            policyorder=[shuffled.index(n) for n in policy.names]
            data=np.ascontiguousarray([(*np.asarray(f.joint_position_rad)[policyorder],
                *np.asarray(f.joint_velocity_rad_s)[policyorder],*f.root_position_world_m,
                *np.asarray(f.root_orientation_xyzw)[[3,0,1,2]]) for f in frames],dtype=np.float32)
            handle=lib.make(data,40)
            try:
                last=(start-policy.default)/policy.scale
                for t in range(40):
                    rotation=Rotation.random(random_state=rng)
                    q=rotation.as_quat()
                    if t%7==0:q=-q
                    state=ProprioceptiveStatePacket(t*.02,shuffled,tuple(rng.normal(0,.3,29)),
                        tuple(rng.normal(0,.7,29)),tuple(q),tuple(rng.normal(0,.5,3)),tuple(rng.normal(0,.5,3)))
                    refs=tuple(frames[min(t+i,39)] for i in range(5))
                    if t==0:
                        policy.reset(state,refs,reset);compare('reset_history',policy.last,last)
                    s=np.array([*np.asarray(state.joint_position_rad)[canonical],
                        *np.asarray(state.joint_velocity_rad_s)[canonical],*q[[3,0,1,2]],
                        *rotation.inv().apply(state.root_angular_velocity_world_rad_s),
                        *state.root_linear_velocity_world_m_s,0,0,0],np.float32)
                    obs=np.empty(431,np.float32);lib.step(handle,s,last,t,obs)
                    other=np.empty(431,np.float32);s[68:]=rng.normal(0,100,3);lib.step(handle,s,last,t,other)
                    compare('position_independence',obs,other)
                    actual=policy.observations(state,refs)['obs'][0];compare('observation',actual,obs)
                    expected=policy.infer({'obs':obs[None]})['actions'].reshape(29)
                    result=policy.act(state,refs);compare('action',policy.last,expected)
                    compare('motor_target',np.asarray(result.joint_position_target_rad)[policy.order],policy.default+policy.scale*expected)
                    compare('history',policy.last,policy.infer({'obs':actual[None]})['actions'].reshape(29))
                    last=expected.copy()
            finally:lib.destroy(handle)
    limits={k:2e-4 for k in errors}
    for key in ('reset_history','history','position_independence'):limits[key]=0.
    report=dict(passed=all(errors[k]<=limits[k] for k in errors),cases=160,resets=4,
        source_commit=pins['source_commit'],checkpoint_sha256=runtime['weights_sha256'],
        max_abs_error=errors,limits=limits,compiled_source_sha256=pins['files'],
        scope='Unmodified author C++ observations with ROS transport stubs, exact hardware policy, initial heading, author-zeroed15 position inputs, start-pose action history, raw action and PD targets. Shuffled joints, signed quaternions, endpoints, repeated reset; position perturbation cannot change input. Benchmark physical initialization stays canonical.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
