#!/usr/bin/env node
/**
 * RFM Simulation on AWS — CDK app entry point
 *
 * Usage: deploy.py calls this automatically (examples for direct use):
 *   npx cdk deploy GR00T-Demo                  -c region=us-east-1 -c vla=gr00t
 *   npx cdk deploy GR00T-Bridge-Demo           -c region=us-east-1 -c vla=gr00t-bridge
 *   npx cdk deploy GR00T-GR1-Demo              -c region=us-east-1 -c vla=gr00t-gr1
 *   npx cdk deploy Pi-Demo                     -c region=us-east-1 -c vla=pi
 *   npx cdk deploy OpenVLA-OFT-Demo            -c region=us-east-1 -c vla=openvla-oft
 *   npx cdk deploy OpenVLA-OFT-Spatial-Demo    -c region=us-east-1 -c vla=openvla-oft -c libero_suite=spatial
 *   npx cdk deploy LAP-Demo                    -c region=us-east-1 -c vla=lap
 *   npx cdk deploy RLDX-Demo                   -c region=us-east-1 -c vla=rldx
 *   npx cdk deploy RLDX-Simpler-Demo           -c region=us-east-1 -c vla=rldx-simpler
 *   npx cdk deploy RLDX-GR1-Demo               -c region=us-east-1 -c vla=rldx-gr1
 *   npx cdk deploy RLDX-Kitchen-Demo  -c region=us-east-1 -c vla=rldx-kitchen
 *   npx cdk deploy OpenArm-Isaac-Demo          -c region=us-east-1 -c vla=openarm-isaac
 *   npx cdk deploy OpenArm-Lift-ACT-Demo       -c region=us-east-1 -c vla=openarm-lift-act
 *   npx cdk deploy MolmoAct2-Demo              -c region=us-west-2 -c vla=molmoact2
 *   npx cdk deploy XVLA-Demo                   -c region=us-east-1 -c vla=xvla
 *   npx cdk deploy Real2Sim-Demo               -c region=ap-northeast-2 -c vla=real2sim
 */

import * as cdk from 'aws-cdk-lib';
import { Validations } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

const app = new cdk.App();

const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';
const vla: string = app.node.tryGetContext('vla');

if (!vla || !['gr00t', 'gr00t-bridge', 'gr00t-gr1', 'gr00t-g1', 'pi', 'openvla-oft', 'lap', 'rldx', 'rldx-simpler', 'rldx-gr1', 'rldx-kitchen', 'openarm-isaac', 'openarm-lift-act', 'molmoact2', 'xvla', 'smolvla', 'lingbot-va', 'vla-adapter', 'fastwam', 'cosmos-policy', 'aspire', 'real2sim'].includes(vla)) {
  throw new Error(
    'CDK context "vla" is required. Pass -c vla=gr00t, -c vla=gr00t-bridge, -c vla=gr00t-gr1, -c vla=gr00t-g1, -c vla=pi, -c vla=openvla-oft, -c vla=lap, -c vla=rldx, -c vla=rldx-simpler, -c vla=rldx-gr1, -c vla=rldx-kitchen, -c vla=openarm-isaac, -c vla=openarm-lift-act, -c vla=molmoact2, -c vla=xvla, -c vla=smolvla, -c vla=lingbot-va, -c vla=vla-adapter, -c vla=fastwam, -c vla=cosmos-policy, -c vla=aspire, or -c vla=real2sim.\n' +
    'Use deploy.py which sets this automatically.',
  );
}

// OpenVLA-OFT only: LIBERO suite ("10" = default, legacy stack name preserved)
const OFT_SUITES = ['spatial', 'object', 'goal', '10', 'long'] as const;
const normaliseSuite = (s: string): string => (s === 'long' ? '10' : s);
const rawSuite: string = app.node.tryGetContext('libero_suite') ?? '10';
if (vla === 'openvla-oft' && !OFT_SUITES.includes(rawSuite as typeof OFT_SUITES[number])) {
  throw new Error(
    `Invalid libero_suite "${rawSuite}". Supported: ${OFT_SUITES.join(', ')}.`,
  );
}
const liberoSuite = vla === 'openvla-oft' ? normaliseSuite(rawSuite) : '';

const suiteCap = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);
const oftStackName = (suite: string): string =>
  suite === '10' ? 'OpenVLA-OFT-Demo' : `OpenVLA-OFT-${suiteCap(suite)}-Demo`;

const stackNameMap: Record<string, string> = {
  'gr00t':       'GR00T-Demo',
  'gr00t-bridge': 'GR00T-Bridge-Demo',
  'gr00t-gr1':   'GR00T-GR1-Demo',
  'gr00t-g1':    'GR00T-G1-Demo',
  'pi':          'Pi-Demo',
  'openvla-oft': oftStackName(liberoSuite),
  'lap':         'LAP-Demo',
  'rldx':        'RLDX-Demo',
  'rldx-simpler': 'RLDX-Simpler-Demo',
  'rldx-gr1':    'RLDX-GR1-Demo',
  'rldx-kitchen': 'RLDX-Kitchen-Demo',
  'openarm-isaac': 'OpenArm-Isaac-Demo',
  'openarm-lift-act': 'OpenArm-Lift-ACT-Demo',
  'molmoact2':   'MolmoAct2-Demo',
  'xvla':        'XVLA-Demo',
  'smolvla':     'SmolVLA-Demo',
  'lingbot-va':  'LingBot-VA-Demo',
  'vla-adapter': 'VLA-Adapter-Demo',
  'fastwam':     'FastWAM-Demo',
  'cosmos-policy': 'Cosmos-Policy-Demo',
  'aspire':      'ASPIRE-Demo',
  'real2sim':    'Real2Sim-Demo',
};
const stackName = stackNameMap[vla];

const oftDescription = (suite: string): string => {
  if (suite === '10') return 'OpenVLA-OFT + LIBERO-10 (Franka Panda, long-horizon)';
  return `OpenVLA-OFT + LIBERO-${suiteCap(suite)} (Franka Panda)`;
};

const descriptionMap: Record<string, string> = {
  'gr00t':       'GR00T N1.7 + LIBERO (Franka Panda)',
  'gr00t-bridge': 'GR00T N1.7 + SimplerEnv Bridge (WidowX 250, SIMPLER_ENV_WIDOWX, ZeroMQ policy server + SAPIEN sim client)',
  'gr00t-gr1':   'GR00T N1.6 + RoboCasa (Fourier GR1 humanoid)',
  'gr00t-g1':    'GR00T N1.6 + WBC loco-manipulation (Unitree G1 humanoid)',
  'pi':          'π0.5',
  'openvla-oft': oftDescription(liberoSuite),
  'lap':         'LAP-3B + LIBERO-Spatial (Franka Panda, JAX policy server + sim client)',
  'rldx':        'RLDX-1 (RLWRLD MSAT/Qwen3-VL-8B) + LIBERO (Franka Panda, ZeroMQ policy server + sim client, eager)',
  'rldx-simpler': 'RLDX-1 (RLWRLD MSAT/Qwen3-VL-8B) + SimplerEnv (Google Robot, OXE_FRACTAL real-robot embodiment, ZeroMQ policy server + sim client, eager)',
  'rldx-gr1':    'RLDX-1 (RLWRLD MSAT/Qwen3-VL-8B) + RoboCasa GR-1 Tabletop (GR-1 bimanual humanoid + waist, ZeroMQ policy server + MuJoCo sim client, eager)',
  'rldx-kitchen': 'RLDX-1 (RLWRLD MSAT/Qwen3-VL-8B) + RoboCasa Kitchen 24-task (PandaOmron single-arm mobile manipulator, ZeroMQ policy server + MuJoCo sim client, eager)',
  'openarm-isaac': 'π0.5 (LeRobot pi05 folding_latest, 16-DOF) + Isaac Lab bimanual OpenArm (pipe-proof)',
  'openarm-lift-act': 'OpenArm unimanual Lift-Cube × ACT — scripted teleop-free demo collection (HDF5)',
  'molmoact2': 'MolmoAct2 (Ai2 5B Action Reasoning Model) + LIBERO (Franka Panda, allenai/lerobot fork, fp32 in-process eval)',
  'xvla': 'X-VLA-0.9B (2toINF/Tsinghua AIR) + LIBERO (Franka Panda, upstream LeRobot lerobot-eval, in-process)',
  'smolvla': 'SmolVLA-450M (Hugging Face) + LIBERO (Franka Panda, upstream LeRobot lerobot-eval, in-process)',
  'lingbot-va': 'LingBot-VA world action model (Robbyant) + LIBERO-Long (Franka Panda, upstream LeRobot lerobot-eval, in-process)',
  'vla-adapter': 'VLA-Adapter-Pro (OpenHelix, 0.5B) + LIBERO (Franka Panda, OpenVLA-OFT-style in-process eval)',
  'aspire': 'ASPIRE agentic skill discovery (Claude on Amazon Bedrock writes cap-harness programs) + LIBERO-Pro fork (Franka Panda)',
  'fastwam': 'FastWAM world action model (Wan2.2 video DiT + action expert) + LIBERO (Franka Panda, upstream LeRobot lerobot-eval, in-process)',
  'cosmos-policy': 'Cosmos Policy (NVIDIA world action model, research-only weights) + LIBERO (Franka Panda, in its own Docker image)',
  'real2sim': 'Real2Sim: capture to 3D Gaussian Splatting (nerfstudio splatfacto) to per-object collision meshes to scripted MuJoCo grasp (UR5e + Robotiq 2F-85)',
};

const stack = new VlaSimulatorStack(app, stackName, {
  vla,
  liberoSuite,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `vla-simulator ${descriptionMap[vla]} fire-and-forget simulation pipeline`,
});

Validations.of(app).addPlugins(new AwsSolutionsChecks(app, { verbose: true }));
