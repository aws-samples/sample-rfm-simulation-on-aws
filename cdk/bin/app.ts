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
 *   npx cdk deploy RLDX-Bridge-Demo            -c region=us-east-1 -c vla=rldx-bridge
 *   npx cdk deploy RLDX-GR1-Demo               -c region=us-east-1 -c vla=rldx-gr1
 *   npx cdk deploy RLDX-Kitchen-Demo  -c region=us-east-1 -c vla=rldx-kitchen
 *   npx cdk deploy OpenArm-Isaac-Demo          -c region=us-east-1 -c vla=openarm-isaac
 *   npx cdk deploy OpenArm-Lift-ACT-Demo       -c region=us-east-1 -c vla=openarm-lift-act
 *   npx cdk deploy MolmoAct2-Demo              -c region=us-west-2 -c vla=molmoact2
 *   npx cdk deploy XVLA-Demo                   -c region=us-east-1 -c vla=xvla
 *   npx cdk deploy Real2Sim-Demo               -c region=ap-northeast-2 -c vla=real2sim
 *   npx cdk deploy SmolVLA-Demo                -c region=us-east-1 -c vla=smolvla -c libero_variant=plus
 *   npx cdk deploy Pi-Demo                     -c region=us-east-1 -c vla=pi -c libero_objects=real2sim -c real2sim_run=s3://<bucket>/<RUN_ID>/real2sim
 *   npx cdk deploy RoboLab-Cosmos3-Demo        -c region=us-east-1 -c vla=robolab-cosmos3
 *   npx cdk deploy SmolVLA-RoboTwin-Demo       -c region=us-east-1 -c vla=smolvla-robotwin [-c robotwin_protocol=clean]
 *   npx cdk deploy LingBot-VLA2-Demo           -c region=us-east-1 -c vla=lingbot-vla2
 *   npx cdk deploy RLinf-OFT-LIBERO-Demo       -c region=us-east-1 -c vla=rlinf-oft-libero [-c rl_profile=full]
 */

import * as cdk from 'aws-cdk-lib';
import { Validations } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

const app = new cdk.App();

const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';
const vla: string = app.node.tryGetContext('vla');

if (!vla || !['gr00t', 'gr00t-bridge', 'gr00t-gr1', 'gr00t-g1', 'pi', 'openvla-oft', 'lap', 'rldx', 'rldx-simpler', 'rldx-bridge', 'rldx-gr1', 'rldx-kitchen', 'openarm-isaac', 'openarm-lift-act', 'molmoact2', 'xvla', 'smolvla', 'lingbot-va', 'vla-adapter', 'fastwam', 'cosmos-policy', 'aspire', 'real2sim', 'robolab-cosmos3', 'robolab-pi05', 'smolvla-robotwin', 'lingbot-vla2', 'rlinf-oft-libero'].includes(vla)) {
  throw new Error(
    'CDK context "vla" is required. Pass -c vla=gr00t, -c vla=gr00t-bridge, -c vla=gr00t-gr1, -c vla=gr00t-g1, -c vla=pi, -c vla=openvla-oft, -c vla=lap, -c vla=rldx, -c vla=rldx-simpler, -c vla=rldx-bridge, -c vla=rldx-gr1, -c vla=rldx-kitchen, -c vla=openarm-isaac, -c vla=openarm-lift-act, -c vla=molmoact2, -c vla=xvla, -c vla=smolvla, -c vla=lingbot-va, -c vla=vla-adapter, -c vla=fastwam, -c vla=cosmos-policy, -c vla=aspire, -c vla=real2sim, -c vla=robolab-cosmos3, -c vla=robolab-pi05, -c vla=smolvla-robotwin, -c vla=lingbot-vla2, or -c vla=rlinf-oft-libero.\n' +
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

// LIBERO-Plus robustness sweep (deploy.py --libero-variant plus): same stack and results bucket as the LIBERO
// run; the stack only takes the timeout / EBS size from models/<vla>.yaml libero_plus. Absent = LIBERO.
const liberoVariant: string = app.node.tryGetContext('libero_variant') ?? 'libero';
if (!['libero', 'plus'].includes(liberoVariant)) {
  throw new Error(`Invalid libero_variant "${liberoVariant}". Supported: libero, plus.`);
}

// LIBERO-R2S real2sim object swap (deploy.py --libero-objects real2sim): same stack and results bucket; the stack
// grants s3:GetObject on exactly the stage-1 outputs the UserData downloads and takes the timeout / EBS size from
// models/<vla>.yaml libero_r2s. Absent = LIBERO.
const liberoObjects: string = app.node.tryGetContext('libero_objects') ?? 'libero';
if (!['libero', 'real2sim'].includes(liberoObjects)) {
  throw new Error(`Invalid libero_objects "${liberoObjects}". Supported: libero, real2sim.`);
}
const real2simRun: string = String(app.node.tryGetContext('real2sim_run') ?? '').trim();

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
  'rldx-bridge': 'RLDX-Bridge-Demo',
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
  'robolab-cosmos3': 'RoboLab-Cosmos3-Demo',
  'robolab-pi05': 'RoboLab-Pi05-Demo',
  'smolvla-robotwin': 'SmolVLA-RoboTwin-Demo',
  'lingbot-vla2': 'LingBot-VLA2-Demo',
  'rlinf-oft-libero': 'RLinf-OFT-LIBERO-Demo',
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
  'rldx-bridge': 'RLDX-1 (RLWRLD MSAT/Qwen3-VL-8B) + SimplerEnv Bridge (WidowX 250, OXE_BRIDGE_ORIG real-robot embodiment, ZeroMQ policy server + sim client, eager)',
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
  'robolab-cosmos3': 'Cosmos3-Policy-DROID (NVIDIA world action model, policy server) + NVIDIA RoboLab (Isaac Lab, DROID Franka, non-commercial assets)',
  'robolab-pi05': 'pi0.5-DROID (openpi policy server) + NVIDIA RoboLab (Isaac Lab, DROID Franka, non-commercial assets)',
  'smolvla-robotwin': 'SmolVLA-450M (Hugging Face) + RoboTwin 2.0 bimanual (Aloha-AgileX, SAPIEN + CuRobo, upstream LeRobot lerobot-eval, in-process)',
  'lingbot-vla2': 'LingBot-VLA 2.0 (Robbyant) + RoboTwin 2.0 bimanual (Aloha-AgileX, SAPIEN + CuRobo, upstream LeRobot lerobot-eval, in-process)',
  'rlinf-oft-libero': 'RLinf GRPO RL post-training of OpenVLA-OFT + LIBERO (baseline eval, GRPO, re-eval; RLinf Docker image)',
};

const stack = new VlaSimulatorStack(app, stackName, {
  vla,
  liberoSuite,
  liberoVariant,
  liberoObjects,
  real2simRun,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `vla-simulator ${descriptionMap[vla]} fire-and-forget simulation pipeline`,
});

Validations.of(app).addPlugins(new AwsSolutionsChecks(app, { verbose: true }));
