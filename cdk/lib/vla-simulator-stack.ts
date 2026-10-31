/**
 * RFM Simulation on AWS — unified CDK stack (file/class names keep the original vla-simulator name)
 *
 * Supports GR00T and π0.5 via a single stack parameterised by `vla` context.
 *
 * Resources: VPC + SG + S3 + SNS + IAM + AzSelector + EC2
 *            (+ GalleryPublisher Lambda when deploy.py passes the gallery_* context, see gallery-stack.ts)
 * Deploy:    python deploy.py --vla gr00t [--bridge]
 *            python deploy.py --vla pi    [--bridge]
 * Teardown:  python destroy.py --vla {gr00t|pi}
 *
 * Config entry point: ../simulator-config.yaml + ../models/{vla}.yaml
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';
import { acknowledge, acknowledgeByPath } from './nag';

import { AzSelectorConstruct } from './az-selector';

export interface VlaSimulatorStackProps extends cdk.StackProps {
  vla: string;
  /** OpenVLA-OFT only: selected LIBERO suite (spatial | object | goal | 10). */
  liberoSuite?: string;
  /** LIBERO targets: "plus" = LIBERO-Plus sweep (models/<vla>.yaml libero_plus); default "libero". */
  liberoVariant?: string;
  /** openvla-oft / pi: "real2sim" = LIBERO-R2S object swap (models/<vla>.yaml libero_r2s); default "libero". */
  liberoObjects?: string;
  /** --libero-objects real2sim: s3://<bucket>/<RUN_ID>/real2sim of the stage-1 run whose objects are read. */
  real2simRun?: string;
}

/**
 * DLAMI AMI mapping (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1)
 * Pre-installed: NVIDIA driver 550 + CUDA + Docker + Python
 * ap-southeast-1 (Singapore) excluded — only T4/V100 (pre-Ampere, FlashAttention incompatible).
 *
 * To add a new region: EC2 console → AMIs → Public images →
 *   search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * Ubuntu 22.04"
 */
const DLAMI_MAPPING: Record<string, string> = {
  'ap-northeast-2': 'ami-0e8f4fdb799f677ca', // Seoul     (PyTorch 2.4.1, 2025-06-24)
  'ap-northeast-1': 'ami-01e1a167769a23572', // Tokyo     (PyTorch 2.4.1)
  'us-east-1':      'ami-0aee7b90d684e107d', // Virginia  (PyTorch 2.4.1)
  'us-west-2':      'ami-0ed3bd866951103a1', // Oregon    (PyTorch 2.4.1)
  'eu-central-1':   'ami-03aa80bc63bbd3638', // Frankfurt (PyTorch 2.4.1)
};

const INSTANCE_TYPES: Record<string, string[]> = {
  gr00t:         ['g6.12xlarge', 'g5.12xlarge', 'g6.xlarge',  'g5.xlarge'],
  'gr00t-gr1':   ['g6.12xlarge', 'g5.12xlarge', 'g6.xlarge',  'g5.xlarge'],
  // G1 WBC rollout = 5 parallel MuJoCo envs (CPU/EGL render) + 1 CUDA policy server.
  // 12xlarge tier mirrors gr00t-gr1; xlarge single-GPU kept last as a capacity fallback.
  'gr00t-g1':    ['g6.12xlarge', 'g5.12xlarge', 'g6.xlarge',  'g5.xlarge'],
  pi:            ['g5.xlarge',   'g5.2xlarge',  'g6.xlarge',  'g6.2xlarge'],
  'openvla-oft': ['g6.xlarge',   'g5.xlarge',   'g6.2xlarge', 'g5.2xlarge'],
  lap:           ['g6.xlarge',   'g6.2xlarge',  'g5.xlarge',  'g5.2xlarge'],
  // Phase 0 GATE PASS (2026-06-19): eager rollout peak VRAM ~17GB on L40S(sm_89) → L4 24GB FITS
  // (~6.6GB headroom). Single-GPU xlarge sufficient; .2xlarge offers more system RAM for n_envs>1.
  rldx:          ['g6.xlarge',   'g6.2xlarge',  'g5.xlarge',  'g5.2xlarge'],
  // SIMPLER = same ~17GB MSAT/Qwen3-VL-8B policy server as rldx, + SAPIEN/ManiSkill renderer.
  // [TO VERIFY] VRAM/RAM unmeasured (Gate 1) — lead with .2xlarge for SAPIEN + 5-env-spawn RAM
  // headroom, single-GPU xlarge kept as fallback. MUST have an explicit entry: the
  // `?? INSTANCE_TYPES['gr00t']` fallback below would otherwise pick gr00t's 12xlarge (wrong+costly).
  'rldx-simpler': ['g6.2xlarge', 'g5.2xlarge',  'g6.xlarge',  'g5.xlarge'],
  // GR-1 Tabletop = same ~17GB MSAT/Qwen3-VL-8B policy server as rldx, + MuJoCo/robosuite
  // renderer (LIGHTER than SIMPLER's SAPIEN — no Vulkan device alloc). n_envs=1 (sequential).
  // [TO VERIFY] VRAM/RAM unmeasured (Gate 1) — lead with .2xlarge for robosuite + tabletop-asset
  // RAM headroom, single-GPU xlarge as fallback. MUST have an explicit entry: the
  // `?? INSTANCE_TYPES['gr00t']` fallback below would otherwise pick gr00t's 12xlarge (wrong+costly).
  'rldx-gr1':    ['g6.2xlarge', 'g5.2xlarge',  'g6.xlarge',  'g5.xlarge'],
  // RoboCasa Kitchen = same ~17GB MSAT/Qwen3-VL-8B policy server as rldx, + MuJoCo/robosuite
  // renderer (like GR-1 — no Vulkan device alloc). n_envs=1 (sequential). Loads full kitchen
  // scenes (heavier ~5GB asset set than GR-1 tabletop) → lead with .2xlarge for system RAM,
  // single-GPU xlarge as fallback. [TO VERIFY] VRAM/RAM unmeasured (Gate 3 pending). MUST have an
  // explicit entry: the `?? INSTANCE_TYPES['gr00t']` fallback below would otherwise pick gr00t's
  // 12xlarge (wrong+costly).
  'rldx-kitchen': ['g6.2xlarge', 'g5.2xlarge',  'g6.xlarge',  'g5.xlarge'],
  // Isaac Sim 5.1 + --enable_cameras render is heavy → 12xlarge tier (mirrors gr00t-gr1).
  'openarm-isaac': ['g6.12xlarge', 'g5.12xlarge', 'g6.2xlarge', 'g5.2xlarge'],
  // 16 collection envs × 2 TiledCameras → ResourceLoader needs >24 GB device mem, so a single-GPU
  // L4/A10G (24 GB) OOMs at first render (run 0418: ERROR_OUT_OF_DEVICE_MEMORY before SM loop).
  // MULTI-GPU ONLY (≥4 GPU): never let AzSelector fall back to a single-GPU type. 24xlarge is the
  // capacity fallback (also 4 GPU, more vCPU). g6/g5 .16xlarge is a single-GPU trap — excluded.
  // CAPACITY (2026-06-11, run9): all 16 g5/g6 .12xl/.24xl × AZ combos returned InsufficientInstance-
  // Capacity in us-east-1 → 0-cost rollback (AzSelector probes via run_instances BEFORE launch).
  // Widened with g6e (4× L40S 48 GB — 4-GPU, MORE VRAM than L4, kept LAST so cheaper g5/g6 win first).
  'openarm-lift-act': ['g6.12xlarge', 'g5.12xlarge', 'g6.24xlarge', 'g5.24xlarge', 'g6e.12xlarge', 'g6e.24xlarge'],
  // MolmoAct2 5B eval runs fp32 (official LIBERO recipe, use_amp=false) → ~20GB weights + acts/KV
  // peaks well over 24GB. Needs a SINGLE GPU >=40GB → g6e.* (1x L40S 48GB). The 4-GPU 12xlarge
  // types (g5/g6) are 4x24GB but inference uses ONE GPU → 24GB → OOM, so they are EXCLUDED (the
  // userdata also enforces a >=40GB VRAM-floor preflight).
  // ⚠️ HOST RAM is also binding: g6e.xlarge = only 32 GiB RAM → fp32 from_pretrained (~20GB resident
  // + 22GB safetensors page-cache, no swap) THRASHES, GPU stuck 0%. g6e.2xlarge = 64 GiB RAM (same
  // L40S) is the floor; the userdata enforces a host-RAM preflight too. xlarge DROPPED.
  // MUST have an explicit entry: the `?? INSTANCE_TYPES['gr00t']` fallback would pick 12xlarge (24GB OOM).
  'molmoact2':   ['g6e.2xlarge', 'g6e.4xlarge'],
};

// Targets without an INSTANCE_TYPES entry (e.g. the LeRobot × LIBERO family) take the ordered
// `instance.preferred` list from their models/<vla>.yaml. Never fall back to another target's list:
// gr00t's 12xlarge default was the wrong (and costly) choice for every target that hit it.
function preferredFromYaml(vla: string, instanceCfg: any): string[] {
  const pref = instanceCfg?.preferred;
  if (!Array.isArray(pref) || pref.length === 0 || !pref.every((t: unknown) => typeof t === 'string')) {
    throw new Error(`models/${vla}.yaml: instance.preferred (list of instance types) is required for ${vla}`);
  }
  return pref;
}

// Targets with a `variants:` block (robolab-*) select one with model.variant; that variant's `instance:`
// keys (preferred list, EBS, creation timeout) override the top-level `instance:`. Same rule as
// generate.resolve_variant(). Models without `variants:` are returned unchanged.
export function resolveInstanceCfg(vla: string, modelConfig: any): any {
  const base = modelConfig?.instance ?? {};
  const variants = modelConfig?.variants;
  if (!variants || typeof variants !== 'object') return base;
  const name = String(modelConfig?.model?.variant ?? '').trim();
  if (!Object.prototype.hasOwnProperty.call(variants, name)) {
    throw new Error(`models/${vla}.yaml: model.variant "${name}" is not one of ${Object.keys(variants).join(', ')}`);
  }
  return { ...base, ...(variants[name]?.instance ?? {}) };
}

// LIBERO-R2S: s3://bucket/<RUN_ID>[/real2sim][/] -> {bucket, prefix "<RUN_ID>/real2sim"} (the stage-1 outputs the
// instance may read); throws on anything else. Mirrors libero_r2s.parse_run.
export function real2simRunPrefix(uri: string): { bucket: string; prefix: string } {
  const m = /^s3:\/\/([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])\/((?:[A-Za-z0-9_.=+-]+\/)*?[A-Za-z0-9_.=+-]+?)(?:\/real2sim)?\/?$/.exec(
    String(uri ?? '').trim());
  if (!m || m[2].includes('..') || m[2].includes('*')) {
    throw new Error(`real2sim_run must be s3://<bucket>/<RUN_ID> of a real2sim stage-1 run, got "${uri}"`);
  }
  return { bucket: m[1], prefix: `${m[2]}/real2sim` };
}

// Real2Sim user capture: models/<vla>.yaml input = {source: capture, capture_s3_uri: s3://bucket/key.zip}
// -> "bucket/key.zip" (the one object the instance may read); undefined for any other input.
export function captureObjectPath(vla: string, input: any): string | undefined {
  if (input?.source !== 'capture') return undefined;
  const uri = String(input?.capture_s3_uri ?? '').trim();
  const m = /^s3:\/\/([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])\/([A-Za-z0-9_./=+-]+\.zip)$/.exec(uri);
  if (!m) {
    throw new Error(`models/${vla}.yaml: input.capture_s3_uri must be s3://<bucket>/<key>.zip, got "${uri}"`);
  }
  return `${m[1]}/${m[2]}`;
}

export class VlaSimulatorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: VlaSimulatorStackProps) {
    super(scope, id, props);

    const { vla, liberoSuite, liberoVariant, liberoObjects, real2simRun } = props;

    // ── Load config ──────────────────────────────────────────────
    const simConfigPath = path.resolve(__dirname, '../../simulator-config.yaml');
    const modelConfigPath = path.resolve(__dirname, `../../models/${vla}.yaml`);
    const simConfig = yaml.load(fs.readFileSync(simConfigPath, 'utf8')) as any;
    const modelConfig = yaml.load(fs.readFileSync(modelConfigPath, 'utf8')) as any;
    const deployment = { ...simConfig.deployment, ...(modelConfig.deployment ?? {}) };
    // --libero-variant plus: libero_plus.creationpolicy_timeout / ebs_gb override instance.* (the fork's
    // 6.4 GB asset pack and the sweep need more disk and time). Absent → the LIBERO stack, unchanged.
    const plusCfg = liberoVariant === 'plus' ? modelConfig.libero_plus : undefined;
    if (liberoVariant === 'plus' && !plusCfg) {
      throw new Error(`libero_variant=plus needs a libero_plus block in models/${vla}.yaml`);
    }
    // --libero-objects real2sim: libero_r2s.creationpolicy_timeout / ebs_gb likewise, plus a read grant on the
    // stage-1 run (below). Absent → the LIBERO stack, unchanged.
    const r2sCfg = liberoObjects === 'real2sim' ? modelConfig.libero_r2s : undefined;
    if (liberoObjects === 'real2sim' && !r2sCfg) {
      throw new Error(`libero_objects=real2sim needs a libero_r2s block in models/${vla}.yaml`);
    }
    if (r2sCfg && liberoVariant === 'plus') {
      throw new Error('libero_objects=real2sim cannot be combined with libero_variant=plus');
    }
    const r2sRun = r2sCfg ? real2simRunPrefix(real2simRun ?? '') : undefined;
    const knobCfg = plusCfg ?? r2sCfg;
    const instanceCfg = {
      ...(resolveInstanceCfg(vla, modelConfig) ?? {}),
      ...(knobCfg?.creationpolicy_timeout ? { creationpolicy_timeout: knobCfg.creationpolicy_timeout } : {}),
      ...(knobCfg?.ebs_gb ? { ebs_gb: knobCfg.ebs_gb } : {}),
    };

    // CDK context takes precedence (injected by deploy.py)
    const notifyEmail: string = this.node.tryGetContext('notify_email') ?? deployment.notify_email;
    const s3ResultsPrefix: string = deployment.s3_results_prefix ?? 'vla-sim-results';
    const autoTerminate: boolean = deployment.auto_terminate ?? true;
    const stackName = id;

    // Video history gallery (deploy.py, see gallery-stack.ts): all three set → publish videos to
    // the shared RFM-Gallery bucket. Absent/empty → no gallery resources (stack as before).
    const galleryBucketName: string = ((this.node.tryGetContext('gallery_bucket') as string | undefined) ?? '').trim();
    const galleryHash: string = ((this.node.tryGetContext('gallery_hash') as string | undefined) ?? '').trim();
    const galleryUrl: string = ((this.node.tryGetContext('gallery_url') as string | undefined) ?? '').trim();
    const galleryOn = !!(galleryBucketName || galleryHash || galleryUrl);
    if (galleryOn) {
      if (!/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(galleryBucketName) || !/^[0-9a-f]{32}$/.test(galleryHash)
          || !/^https:\/\/[A-Za-z0-9.-]+\/g\/[0-9a-f]{32}\/$/.test(galleryUrl)) {
        throw new Error(
          'Gallery context needs all of gallery_bucket (S3 bucket name), gallery_hash (32 hex chars) and ' +
          'gallery_url (https://<domain>/g/<hash>/). deploy.py sets them; pass none of them to deploy without the gallery.',
        );
      }
    }
    // Gallery folder of this stack: one per target (per LIBERO suite for OpenVLA-OFT).
    const galleryTarget = vla === 'openvla-oft' && liberoSuite && liberoSuite !== '10' ? `${vla}-${liberoSuite}` : vla;

    // Bridge mode: vpc_id context set → import existing VPC
    const vpcIdCtx: string | undefined =
      (this.node.tryGetContext('vpc_id') as string | undefined) || undefined;

    // π0.5 bridge additionally needs nlb_endpoint
    const nlbEndpoint: string =
      (this.node.tryGetContext('nlb_endpoint') as string | undefined) ?? '';
    const isGr00tVla = vla === 'gr00t' || vla === 'gr00t-gr1' || vla === 'gr00t-g1';
    const bridgeMode = !!(vpcIdCtx && (isGr00tVla || nlbEndpoint));

    // EBS size: bridge mode may use smaller disk (π0.5 skips checkpoint download)
    const ebsGb: number = bridgeMode
      ? (instanceCfg.ebs_gb_bridge ?? instanceCfg.ebs_gb ?? 200)
      : (instanceCfg.ebs_gb ?? 200);

    // CreationPolicy timeout from model config. For openvla-oft this is a suite-keyed
    // map (PT120M for short-horizon, PT240M for LIBERO-10); the RoboTwin targets key it by
    // --robotwin-protocol (context robotwin_protocol, default subset); for others it's a scalar.
    const rawTimeout: string | Record<string, string> = instanceCfg.creationpolicy_timeout ?? 'PT180M';
    let creationTimeout: string;
    if (typeof rawTimeout === 'string') {
      creationTimeout = rawTimeout;
    } else if (rawTimeout['subset'] !== undefined) {
      const protocol: string = (this.node.tryGetContext('robotwin_protocol') as string | undefined) || 'subset';
      if (rawTimeout[protocol] === undefined) {
        throw new Error(`models/${vla}.yaml: instance.creationpolicy_timeout has no "${protocol}" entry (robotwin_protocol)`);
      }
      creationTimeout = rawTimeout[protocol];
    } else {
      const suiteKey = liberoSuite || '10';
      creationTimeout = rawTimeout[suiteKey] ?? rawTimeout['10'] ?? 'PT180M';
    }

    // ── AMI ───────────────────────────────────────────────────────
    const region = this.region;
    const amiId = DLAMI_MAPPING[region];
    if (!amiId) {
      throw new Error(
        `Unsupported region: ${region}\n` +
        `Supported regions: ${Object.keys(DLAMI_MAPPING).join(', ')}\n` +
        `To add a new region, update DLAMI_MAPPING in cdk/lib/vla-simulator-stack.ts.`,
      );
    }

    // ── VPC ───────────────────────────────────────────────────────
    const vpcLogicalId = `${vla.toUpperCase()}VPC`;
    let vpc: ec2.IVpc;
    if (vpcIdCtx) {
      vpc = ec2.Vpc.fromLookup(this, vpcLogicalId, { vpcId: vpcIdCtx });
    } else {
      vpc = new ec2.Vpc(this, vpcLogicalId, {
        maxAzs: 4,
        natGateways: 1,
        subnetConfiguration: [
          { name: 'public',  subnetType: ec2.SubnetType.PUBLIC,               cidrMask: 24 },
          { name: 'private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,  cidrMask: 24 },
        ],
      });
    }

    // ── Security Group ────────────────────────────────────────────
    const sg = new ec2.SecurityGroup(this, `${vla.toUpperCase()}SG`, {
      vpc,
      description: `${vla} simulator instance - outbound only`,
      allowAllOutbound: true,
    });

    // ── S3 bucket ─────────────────────────────────────────────────
    // S3 bucket names are capped at 63 chars. The longer stack names (OpenArm-Lift-ACT-Demo,
    // OpenVLA-OFT-<Suite>-Demo) overflow it in ap-northeast-* / eu-central-1, so drop the
    // "-demo" part only for those. Names that already fit are unchanged, so existing (retained)
    // buckets keep their names. deploy.py mirrors this rule for the orphan-bucket import.
    let bucketName = `${s3ResultsPrefix}-${stackName.toLowerCase()}-${this.region}-${this.account}`;
    if (!cdk.Token.isUnresolved(bucketName) && bucketName.length > 63) {
      bucketName = `${s3ResultsPrefix}-${stackName.toLowerCase().replace(/-demo$/, '')}-${this.region}-${this.account}`;
      if (bucketName.length > 63) {
        throw new Error(
          `Results bucket name is longer than 63 chars: ${bucketName}\n` +
          'Use a shorter deployment.s3_results_prefix in simulator-config.yaml.',
        );
      }
    }
    const bucket = new s3.Bucket(this, 'ResultsBucket', {
      bucketName,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      enforceSSL: true,
    });

    // ── SNS ───────────────────────────────────────────────────────
    const topic = new sns.Topic(this, 'NotificationTopic', {
      topicName: `${stackName}-${vla}-notify`,
    });
    new sns.Subscription(this, 'EmailSubscription', {
      topic,
      protocol: sns.SubscriptionProtocol.EMAIL,
      endpoint: notifyEmail,
    });

    // N1.7 rollout server (gr00t-g1 Step 2): if models/gr00t-g1.yaml has server.s3_ckpt_uri, the
    // instance must READ the fine-tuned ckpt from that (external) artifacts bucket. Parse the bucket
    // name so the IAM role can be scoped to it (GetObject/ListBucket). Empty → N1.6 demo path (no extra grant).
    const serverS3CkptUri: string = (modelConfig.server?.s3_ckpt_uri ?? '').trim();
    const serverCkptBucket: string = serverS3CkptUri.startsWith('s3://')
      ? serverS3CkptUri.slice('s3://'.length).split('/')[0]
      : '';

    // ── IAM Role ──────────────────────────────────────────────────
    const logGroupPrefix = `/${vla}/*`;
    const role = new iam.Role(this, `${vla.charAt(0).toUpperCase() + vla.slice(1)}InstanceRole`, {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
      inlinePolicies: {
        VlaSimPolicy: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: ['s3:PutObject'],
              resources: [`${bucket.bucketArn}/*`],
            }),
            new iam.PolicyStatement({
              actions: ['sns:Publish'],
              resources: [topic.topicArn],
            }),
            // Self-terminate at the end of a run (AutoTerminate); only instances of this stack.
            // No cloudformation:SignalResource: the run reports to the WaitCondition presigned URL.
            new iam.PolicyStatement({
              actions: ['ec2:TerminateInstances'],
              resources: [`arn:${cdk.Aws.PARTITION}:ec2:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:instance/*`],
              conditions: { StringEquals: { 'ec2:ResourceTag/StackName': stackName } },
            }),
            new iam.PolicyStatement({
              actions: [
                'logs:CreateLogGroup',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
                'logs:DescribeLogStreams',
              ],
              resources: [
                `arn:${cdk.Aws.PARTITION}:logs:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:log-group:${logGroupPrefix}`,
              ],
            }),
            new iam.PolicyStatement({
              actions: ['ssm:GetParameter'],
              resources: [
                `arn:${cdk.Aws.PARTITION}:ssm:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:parameter/vla-simulator/*`,
              ],
            }),
            new iam.PolicyStatement({
              actions: ['kms:Decrypt'],
              resources: [`arn:${cdk.Aws.PARTITION}:kms:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:key/*`],
              conditions: { StringEquals: { 'kms:ViaService': `ssm.${cdk.Aws.REGION}.amazonaws.com` } },
            }),
          ],
        }),
      },
    });

    // ASPIRE (instance.bedrock: true): the discovery loop runs Claude Code on Amazon Bedrock from the
    // instance. Invocation only (no model management); cross-region inference profiles resolve to
    // foundation models in other regions, hence the region wildcard on foundation-model.
    if (instanceCfg.bedrock === true) {
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
        resources: [
          `arn:${cdk.Aws.PARTITION}:bedrock:*:${cdk.Aws.ACCOUNT_ID}:inference-profile/*`,
          `arn:${cdk.Aws.PARTITION}:bedrock:*::foundation-model/anthropic.*`,
        ],
      }));
    }

    // N1.7 rollout server (gr00t-g1 Step 2): allow reading the fine-tuned ckpt from the external
    // artifacts bucket. Scoped to that bucket only; added only when server.s3_ckpt_uri is set.
    if (serverCkptBucket) {
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [`arn:${cdk.Aws.PARTITION}:s3:::${serverCkptBucket}/*`],
      }));
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:ListBucket'],
        resources: [`arn:${cdk.Aws.PARTITION}:s3:::${serverCkptBucket}`],
      }));
    }

    // Real2Sim with your own capture (models/real2sim.yaml input.source: capture): read exactly that
    // object. The HOPE-Video default downloads from the public internet and needs no grant.
    const captureObject = captureObjectPath(vla, modelConfig.input);
    if (captureObject) {
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [`arn:${cdk.Aws.PARTITION}:s3:::${captureObject}`],
      }));
    }

    // LIBERO-R2S: read exactly the stage-1 outputs the UserData downloads (<run>/segment/* and splat_world.ply);
    // ListBucket only under <run>/segment/ (aws s3 cp --recursive lists before it copies).
    if (r2sRun) {
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [
          `arn:${cdk.Aws.PARTITION}:s3:::${r2sRun.bucket}/${r2sRun.prefix}/segment/*`,
          `arn:${cdk.Aws.PARTITION}:s3:::${r2sRun.bucket}/${r2sRun.prefix}/splat_world.ply`,
        ],
      }));
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:ListBucket'],
        resources: [`arn:${cdk.Aws.PARTITION}:s3:::${r2sRun.bucket}`],
        conditions: { StringLike: { 's3:prefix': [`${r2sRun.prefix}/segment/`, `${r2sRun.prefix}/segment/*`] } },
      }));
    }

    const instanceProfile = new iam.CfnInstanceProfile(this, `${vla.toUpperCase()}InstanceProfile`, {
      roles: [role.roleName],
    });

    // ── WaitCondition (replaces CfnInstance CreationPolicy) ───────
    // UserData sends SUCCESS/FAILURE to the presigned URL instead of cfn signal-resource.
    const waitHandle = new cdk.CfnWaitConditionHandle(this, `${vla.toUpperCase()}WaitHandle`);
    const waitCondition = new cdk.CfnWaitCondition(this, `${vla.toUpperCase()}WaitCondition`, {
      count: 1,
      timeout: String(
        parseInt(creationTimeout.replace(/\D/g, '')) * 60  // PT180M → 10800 seconds
      ),
      handle: waitHandle.ref,
    });

    // ── UserData (S3 Asset bootstrap) ─────────────────────────────
    const userDataPath = path.resolve(__dirname, `../../assets/userdata/${vla}.sh`);
    if (!fs.existsSync(userDataPath)) {
      throw new Error(
        `UserData script not found: ${userDataPath}\n` +
        `Use deploy.py instead of cdk deploy directly — it generates ${vla}.sh automatically.\n` +
        `Or run manually: python generate.py --vla ${vla}`,
      );
    }
    const userDataAsset = new s3assets.Asset(this, `${vla.toUpperCase()}UserDataAsset`, {
      path: userDataPath,
    });
    userDataAsset.grantRead(role);

    // Bootstrap script: inject WaitCondition presigned URL instead of CFN signal resource
    const baseBootstrapLines = [
      '#!/bin/bash',
      `# vla-simulator ${vla} — S3 bootstrap`,
      'export S3BucketName="${S3BucketName}"',
      'export SnsTopicArn="${SnsTopicArn}"',
      'export StackName="${StackName}"',
      'export WaitConditionUrl="${WaitConditionUrl}"',
      'export Region="${Region}"',
      'export AutoTerminate="${AutoTerminate}"',
    ];

    const bootstrapSubVars: Record<string, string> = {
      S3BucketName:     bucket.bucketName,
      SnsTopicArn:      topic.topicArn,
      StackName:        stackName,
      WaitConditionUrl: waitHandle.ref,
      Region:           this.region,
      AutoTerminate:    autoTerminate ? 'true' : 'false',
    };

    if (galleryOn) {
      // Read by the templates' SNS message ("Video history" link); absent → no link.
      baseBootstrapLines.push('export GalleryUrl="${GalleryUrl}"');
      bootstrapSubVars['GalleryUrl'] = galleryUrl;
    }

    if (vla === 'pi' || vla === 'lap') {
      // pi & lap both run the openpi WebSocket↔gRPC bridge in bridge mode.
      baseBootstrapLines.push(
        'export BridgeMode="${BridgeMode}"',
        'export NlbEndpoint="${NlbEndpoint}"',
      );
      bootstrapSubVars['BridgeMode'] = bridgeMode ? 'true' : 'false';
      bootstrapSubVars['NlbEndpoint'] = nlbEndpoint;
    }
    // gr00t-gr1 uses the same bootstrap as gr00t (no extra variables needed)

    baseBootstrapLines.push(
      `aws s3 cp ${userDataAsset.s3ObjectUrl} /tmp/${vla}.sh`,
      `chmod +x /tmp/${vla}.sh`,
      `exec /tmp/${vla}.sh`,
    );

    const userDataRendered = cdk.Fn.base64(
      cdk.Fn.sub(baseBootstrapLines.join('\n'), bootstrapSubVars),
    );

    // ── AzSelector: launch real instance (no probe+terminate race) ─
    const azSelector = new AzSelectorConstruct(this, 'AzSelector', {
      instanceTypes: INSTANCE_TYPES[vla] ?? preferredFromYaml(vla, instanceCfg),
      amiId,
      subnetIds: vpc.privateSubnets.map((s) => s.subnetId),
      userDataBase64: userDataRendered,
      iamInstanceProfileArn: instanceProfile.attrArn,
      instanceRoleArn: role.roleArn,
      securityGroupIds: [sg.securityGroupId],
      ebsGb,
      tags: [
        { key: 'Name',      value: `${stackName}-${vla}` },
        { key: 'StackName', value: stackName },
      ],
      scopeTag: { key: 'StackName', value: stackName },
    });
    // WaitCondition timeout starts when CFN processes it. The instance is launched by AzSelector
    // Lambda. No explicit DependsOn needed — WaitConditionHandle URL is embedded in UserData,
    // so the instance will signal back once the script completes.

    // ── Video history gallery publisher (optional) ────────────────
    if (galleryOn) {
      const galleryPrefix = `g/${galleryHash}/`;
      const galleryBucketArn = `arn:${cdk.Aws.PARTITION}:s3:::${galleryBucketName}`;
      const publisherRole = new iam.Role(this, 'GalleryPublisherRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        ],
        inlinePolicies: {
          GalleryPublisherPolicy: new iam.PolicyDocument({
            statements: [
              // Read the run videos/captions (HeadObject is covered by s3:GetObject).
              new iam.PolicyStatement({
                actions: ['s3:GetObject'],
                resources: [`${bucket.bucketArn}/*`],
              }),
              // Write copies + index under this e-mail's gallery folder only; read back manifest/captions.
              new iam.PolicyStatement({
                actions: ['s3:PutObject', 's3:GetObject'],
                resources: [`${galleryBucketArn}/${galleryPrefix}*`],
              }),
              new iam.PolicyStatement({
                actions: ['s3:ListBucket'],
                resources: [galleryBucketArn],
                conditions: { StringLike: { 's3:prefix': [`${galleryPrefix}*`] } },
              }),
            ],
          }),
        },
      });
      const publisher = new lambda.Function(this, 'GalleryPublisher', {
        runtime: lambda.Runtime.PYTHON_3_13,
        handler: 'index.handler',
        role: publisherRole,
        timeout: cdk.Duration.minutes(5),
        memorySize: 256,
        code: lambda.Code.fromAsset(path.resolve(__dirname, '../lambda/gallery_publisher')),
        description: `Copies ${vla} run videos into the video history gallery and rebuilds its index`,
        environment: {
          GALLERY_BUCKET: galleryBucketName,
          GALLERY_PREFIX: galleryPrefix,
          GALLERY_TARGET: galleryTarget,
        },
      });
      bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.LambdaDestination(publisher), { suffix: '.mp4' });
      bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.LambdaDestination(publisher), { suffix: '.txt' });

      new cdk.CfnOutput(this, 'GalleryUrl', {
        value: galleryUrl,
        description: 'Video history gallery (password: python vlasim.py gallery)',
      });
      new cdk.CfnOutput(this, 'GalleryBucket', {
        value: galleryBucketName,
        description: 'Video history gallery bucket',
      });
      new cdk.CfnOutput(this, 'GalleryPrefix', {
        value: `${galleryPrefix}${galleryTarget}/`,
        description: 'Gallery folder of this stack (removed by destroy.py --delete-all)',
      });

      acknowledge(publisherRole, [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'AWSLambdaBasicExecutionRole is the minimal managed policy for Lambda CloudWatch Logs access.',
          appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'],
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Video and caption keys are created per run at run time; read-only on this stack\'s results bucket objects.',
          appliesTo: ['Resource::<ResultsBucketA95A2103.Arn>/*'],
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Gallery copies and the index are written under this e-mail\'s gallery folder g/<hash>/ only.',
          appliesTo: [`Resource::arn:<AWS::Partition>:s3:::${galleryBucketName}/${galleryPrefix}*`],
        },
      ]);
      acknowledge(publisher, [
        { id: 'AwsSolutions-L1', reason: 'Python 3.13 is the latest stable production runtime for AWS Lambda.' },
      ]);
      // CDK's singleton handler that applies the bucket notification configuration.
      const notificationsHandler = this.node.tryFindChild('BucketNotificationsHandler050a0587b7544547bf325f094a3db834');
      if (notificationsHandler) {
        acknowledge(notificationsHandler, [
          {
            id: 'AwsSolutions-IAM4',
            reason: 'CDK-managed S3 bucket notification handler; AWSLambdaBasicExecutionRole is the minimal managed policy for its CloudWatch Logs access.',
            appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'],
          },
          {
            id: 'AwsSolutions-IAM5',
            reason: 'CDK-managed S3 bucket notification handler: s3:PutBucketNotification is granted on * by aws-cdk-lib and only applied to this stack\'s results bucket.',
            appliesTo: ['Resource::*'],
          },
        ]);
      }
    }

    // ── Outputs ───────────────────────────────────────────────────
    new cdk.CfnOutput(this, 'S3BucketName', {
      value: bucket.bucketName,
      description: `Results S3 bucket (download: aws s3 sync s3://BUCKET/RUN_ID/ ./${vla}-results/RUN_ID/)`,
    });
    new cdk.CfnOutput(this, 'SNSTopicArn', {
      value: topic.topicArn,
      description: 'Completion notification SNS topic ARN',
    });
    new cdk.CfnOutput(this, 'InstanceId', {
      value: azSelector.instanceId,
      description: 'EC2 instance ID',
    });
    new cdk.CfnOutput(this, 'SelectedInstanceType', {
      value: azSelector.resolvedInstanceType,
      description: 'GPU instance type selected by AzSelector',
    });
    new cdk.CfnOutput(this, 'SelectedAZ', {
      value: azSelector.availabilityZone,
      description: 'Availability zone selected by AzSelector',
    });
    new cdk.CfnOutput(this, 'NotifyEmail', {
      value: notifyEmail,
      description: 'Completion notification email address',
    });
    if (vla === 'pi' || vla === 'lap') {
      new cdk.CfnOutput(this, 'Mode', {
        value: bridgeMode ? `bridge (NLB: ${nlbEndpoint})` : 'local',
        description: 'Deployment mode: local (Docker Compose) or bridge (gRPC NLB)',
      });
    }

    // ── cdk-nag suppressions ──────────────────────────────────────
    // EC2 instance is launched by AzSelector Lambda (not a CfnInstance resource)
    // EC28/EC29 nag rules do not apply to Lambda-launched instances.

    if (!vpcIdCtx) {
      acknowledgeByPath(this, `/${id}/${vpcLogicalId}/Resource`, [
        { id: 'AwsSolutions-VPC7', reason: 'VPC Flow Logs omitted for this sample to reduce cost. Short-lived stack with no inbound traffic allowed.' },
      ]);
    }

    acknowledge(bucket, [
      { id: 'AwsSolutions-S1', reason: 'Server access logging omitted for this sample results bucket. Bucket is private (BlockPublicAccess.BLOCK_ALL).' },
    ]);

    acknowledge(topic, [
      { id: 'AwsSolutions-SNS3', reason: 'SNS topic used only for email notifications to end users. Enforcing SSL on publish not required for this notification-only use case.' },
    ]);

    const roleLogicalId = `${vla.charAt(0).toUpperCase() + vla.slice(1)}InstanceRole`;
    acknowledge(role, [
      { id: 'AwsSolutions-IAM4', reason: 'AmazonSSMManagedInstanceCore enables SSM Session Manager for browser-based terminal access.', appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore'] },
      { id: 'AwsSolutions-IAM5', reason: 'Wildcard on results bucket objects required for dynamic output file names.', appliesTo: ['Resource::<ResultsBucketA95A2103.Arn>/*'] },
      { id: 'AwsSolutions-IAM5', reason: 'Self-termination: the instance ID is unknown at synth time; ec2:TerminateInstances is constrained by StringEquals on the StackName tag.', appliesTo: ['Resource::arn:<AWS::Partition>:ec2:<AWS::Region>:<AWS::AccountId>:instance/*'] },
      {
        id: 'AwsSolutions-IAM5',
        reason: `CloudWatch Logs wildcard limited to /${vla}/* log group prefix.`,
        appliesTo: [`Resource::arn:<AWS::Partition>:logs:<AWS::Region>:<AWS::AccountId>:log-group:/${vla}/*`],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'KMS key ARN cannot be determined at synth time; scoped to SSM service via kms:ViaService condition.',
        appliesTo: [`Resource::arn:<AWS::Partition>:kms:<AWS::Region>:<AWS::AccountId>:key/*`],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'SSM parameter path wildcard scoped to /vla-simulator/* prefix for HF token and other simulator secrets.',
        appliesTo: [`Resource::arn:<AWS::Partition>:ssm:<AWS::Region>:<AWS::AccountId>:parameter/vla-simulator/*`],
      },
    ]);

    // Finding ids carry the policy resource as cdk-nag renders it: unresolved parts stay as
    // <AWS::...> placeholders (env-agnostic synth, e.g. CI), resolved ones are literal.
    const assetRegion = cdk.Token.isUnresolved(this.region) ? '<AWS::Region>' : this.region;
    const assetAccount = cdk.Token.isUnresolved(this.account) ? '<AWS::AccountId>' : this.account;
    const assetPartition = cdk.Token.isUnresolved(this.region) ? '<AWS::Partition>' : (this.region.startsWith('cn-') ? 'aws-cn' : 'aws');
    acknowledgeByPath(this, `/${id}/${roleLogicalId}/DefaultPolicy/Resource`, [
      {
        id: 'AwsSolutions-IAM5',
        reason: 'CDK auto-generates this policy for S3 Asset download from CDK bootstrap bucket. Required by s3assets.Asset.grantRead().',
        appliesTo: [
          'Action::s3:GetBucket*',
          'Action::s3:GetObject*',
          'Action::s3:List*',
          // The CDK bootstrap asset bucket ARN: concrete when the stack has an account/region,
          // placeholder form when it synthesizes environment-agnostic (CI). v3 has no regex match.
          `Resource::arn:${assetPartition}:s3:::cdk-hnb659fds-assets-${assetAccount}-${assetRegion}/*`,
        ],
      },
      ...(serverCkptBucket ? [{
        id: 'AwsSolutions-IAM5',
        reason: `N1.7 rollout reads the fine-tuned ckpt objects from ${serverCkptBucket}; object key wildcard required for the ckpt directory contents (read-only).`,
        appliesTo: [`Resource::arn:<AWS::Partition>:s3:::${serverCkptBucket}/*`],
      }] : []),
      ...(r2sRun ? [{
        id: 'AwsSolutions-IAM5',
        reason: 'LIBERO-R2S reads the per-cluster hull files of one real2sim stage-1 run (read-only); the key wildcard is limited to that run\'s segment/ directory.',
        appliesTo: [`Resource::arn:<AWS::Partition>:s3:::${r2sRun.bucket}/${r2sRun.prefix}/segment/*`],
      }] : []),
      ...(instanceCfg.bedrock === true ? [{
        id: 'AwsSolutions-IAM5',
        reason: 'ASPIRE discovery loop invokes Claude through Bedrock cross-region (global.*) inference profiles, which route to Anthropic foundation models in any region; invocation-only actions.',
        appliesTo: [
          'Resource::arn:<AWS::Partition>:bedrock:*:<AWS::AccountId>:inference-profile/*',
          'Resource::arn:<AWS::Partition>:bedrock:*::foundation-model/anthropic.*',
        ],
      }] : []),
    ]);
  }
}
