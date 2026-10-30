import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { captureObjectPath, resolveInstanceCfg, VlaSimulatorStack } from '../lib/vla-simulator-stack';

// The stack reads the rendered UserData from assets/userdata/<vla>.sh (gitignored, written by
// generate.py / deploy.py). Seed a placeholder for any target that has not been rendered yet so
// the tests run on a fresh checkout, and remove only the placeholders we created.
const USERDATA_DIR = path.resolve(__dirname, '../../assets/userdata');
const VLAS = ['gr00t', 'pi'];
const SEED_VLAS = [...VLAS, 'openarm-lift-act', 'xvla', 'real2sim', 'gr00t-bridge', 'rldx-bridge', 'smolvla', 'openvla-oft', 'robolab-cosmos3'];
const seeded: string[] = [];

beforeAll(() => {
  fs.mkdirSync(USERDATA_DIR, { recursive: true });
  for (const vla of SEED_VLAS) {
    const p = path.join(USERDATA_DIR, `${vla}.sh`);
    if (!fs.existsSync(p)) {
      fs.writeFileSync(p, '#!/bin/bash\n# test placeholder\n');
      seeded.push(p);
    }
  }
});

afterAll(() => {
  for (const p of seeded) fs.rmSync(p, { force: true });
});

function makeApp(vla: string, extraCtx: Record<string, string> = {}) {
  const app = new cdk.App({
    context: { region: 'us-east-1', vla, notify_email: 'test@example.com', ...extraCtx },
  });
  const stackName = vla === 'gr00t' ? 'GR00T-Demo' : 'Pi-Demo';
  return new VlaSimulatorStack(app, stackName, {
    vla,
    env: { account: '123456789012', region: 'us-east-1' },
  });
}

// The GPU instance is launched by the AzSelector custom resource (capacity fallback across
// AZs/instance types), not by an AWS::EC2::Instance, and the run is gated by a WaitCondition.
test.each(VLAS)('%s stack launches the instance through AzSelector', (vla) => {
  const template = Template.fromStack(makeApp(vla));
  template.resourceCountIs('AWS::EC2::Instance', 0);
  template.resourceCountIs('AWS::CloudFormation::CustomResource', 1);
  template.resourceCountIs('AWS::CloudFormation::WaitCondition', 1);
});

test.each(VLAS)('%s stack has S3 results bucket', (vla) => {
  const template = Template.fromStack(makeApp(vla));
  template.resourceCountIs('AWS::S3::Bucket', 1);
});

test.each(VLAS)('%s stack has SNS topic', (vla) => {
  const template = Template.fromStack(makeApp(vla));
  template.resourceCountIs('AWS::SNS::Topic', 1);
});

// S3 bucket names are capped at 63 chars; OpenArm-Lift-ACT-Demo overflows it in ap-northeast-2.
test('long stack names get a results bucket name within 63 chars', () => {
  const app = new cdk.App({
    context: { region: 'ap-northeast-2', vla: 'openarm-lift-act', notify_email: 'test@example.com' },
  });
  const stack = new VlaSimulatorStack(app, 'OpenArm-Lift-ACT-Demo', {
    vla: 'openarm-lift-act',
    env: { account: '123456789012', region: 'ap-northeast-2' },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
    BucketName: 'vla-sim-results-openarm-lift-act-ap-northeast-2-123456789012',
  });
});

test('bucket names that already fit are unchanged', () => {
  Template.fromStack(makeApp('gr00t')).hasResourceProperties('AWS::S3::Bucket', {
    BucketName: 'vla-sim-results-gr00t-demo-us-east-1-123456789012',
  });
});

// Targets without an INSTANCE_TYPES entry take instance.preferred from models/<vla>.yaml.
test('yaml-driven targets launch their own preferred instance list', () => {
  const app = new cdk.App({ context: { region: 'us-east-1', vla: 'xvla', notify_email: 'test@example.com' } });
  const stack = new VlaSimulatorStack(app, 'XVLA-Demo', {
    vla: 'xvla',
    env: { account: '123456789012', region: 'us-east-1' },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFormation::CustomResource', {
    InstanceTypes: 'g6.2xlarge,g5.2xlarge,g6.xlarge,g5.xlarge',
  });
});

// robolab-*: model.variant selects variants.<name>.instance (instance list, EBS, timeout) over instance.
test('robolab-cosmos3 launches the instance list of its default variant', () => {
  const app = new cdk.App({ context: { region: 'us-east-1', vla: 'robolab-cosmos3', notify_email: 'test@example.com' } });
  const stack = new VlaSimulatorStack(app, 'RoboLab-Cosmos3-Demo', {
    vla: 'robolab-cosmos3',
    env: { account: '123456789012', region: 'us-east-1' },
  });
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::CloudFormation::CustomResource', { InstanceTypes: 'g6e.2xlarge,g6e.4xlarge' });
  template.hasResourceProperties('AWS::CloudFormation::WaitCondition', { Timeout: String(540 * 60) });
  template.hasResourceProperties('AWS::S3::Bucket', { BucketName: 'vla-sim-results-robolab-cosmos3-demo-us-east-1-123456789012' });
});

test('resolveInstanceCfg merges the selected variant and rejects unknown ones', () => {
  const cfg = {
    model: { variant: 'nano' },
    instance: { preferred: ['g6e.2xlarge'], ebs_gb: 600, creationpolicy_timeout: 'PT540M' },
    variants: { edge: { instance: { preferred: ['g6e.2xlarge'] } }, nano: { instance: { preferred: ['g6e.12xlarge'], ebs_gb: 700 } } },
  };
  expect(resolveInstanceCfg('robolab-cosmos3', cfg)).toEqual({
    preferred: ['g6e.12xlarge'], ebs_gb: 700, creationpolicy_timeout: 'PT540M',
  });
  expect(() => resolveInstanceCfg('robolab-cosmos3', { ...cfg, model: { variant: 'huge' } })).toThrow(/not one of edge, nano/);
  const plain = { instance: { preferred: ['g6.xlarge'] } };
  expect(resolveInstanceCfg('xvla', plain)).toBe(plain.instance);
});

// SimplerEnv WidowX/Bridge targets: single-GPU list from their yaml (not gr00t's 12xlarge), and the
// CloudWatch log grant follows the target id, matching the /<vla>/userdata group the template writes.
test.each([
  ['gr00t-bridge', 'GR00T-Bridge-Demo'],
  ['rldx-bridge', 'RLDX-Bridge-Demo'],
])('%s launches a single-GPU instance list and grants its own log group', (vla, stackName) => {
  const app = new cdk.App({ context: { region: 'us-east-1', vla, notify_email: 'test@example.com' } });
  const stack = new VlaSimulatorStack(app, stackName, { vla, env: { account: '123456789012', region: 'us-east-1' } });
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::CloudFormation::CustomResource', {
    InstanceTypes: 'g6.2xlarge,g5.2xlarge,g6.xlarge,g5.xlarge',
  });
  expect(JSON.stringify(template.toJSON())).toContain(`log-group:/${vla}/*`);
  template.hasResourceProperties('AWS::S3::Bucket', {
    BucketName: `vla-sim-results-${stackName.toLowerCase()}-us-east-1-123456789012`,
  });
});

// IAM hardening (audit A20/A21): PassRole is pinned to the instance role and to EC2, nothing passes
// any role, EC2 mutations are scoped, and the launched instance requires IMDSv2.
type Stmt = { Action: string | string[]; Resource: unknown; Condition?: Record<string, Record<string, unknown>> };
function statements(template: Template): Stmt[] {
  const out: Stmt[] = [];
  const collect = (doc: { Statement?: Stmt[] } | undefined) => out.push(...(doc?.Statement ?? []));
  for (const r of Object.values(template.findResources('AWS::IAM::Role'))) {
    for (const p of (r.Properties.Policies ?? []) as Array<{ PolicyDocument: { Statement: Stmt[] } }>) collect(p.PolicyDocument);
  }
  for (const r of Object.values(template.findResources('AWS::IAM::Policy'))) collect(r.Properties.PolicyDocument);
  return out;
}
const actionsOf = (s: Stmt) => (Array.isArray(s.Action) ? s.Action : [s.Action]);
const resourcesOf = (s: Stmt) => (Array.isArray(s.Resource) ? s.Resource : [s.Resource]);

// Real2Sim: instance list from models/real2sim.yaml; the HOPE-Video default needs no extra S3 read grant,
// a user capture gets s3:GetObject on exactly that object.
test('real2sim launches its yaml instance list and grants no capture read by default', () => {
  const app = new cdk.App({ context: { region: 'ap-northeast-2', vla: 'real2sim', notify_email: 'test@example.com' } });
  const stack = new VlaSimulatorStack(app, 'Real2Sim-Demo', {
    vla: 'real2sim',
    env: { account: '123456789012', region: 'ap-northeast-2' },
  });
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::CloudFormation::CustomResource', {
    InstanceTypes: 'g5.2xlarge,g6.2xlarge,g5.4xlarge,g6.4xlarge',
  });
  const reads = statements(template).filter((s) => actionsOf(s).includes('s3:GetObject'));
  for (const s of reads) expect(JSON.stringify(s.Resource)).not.toContain('.zip');
});

test('captureObjectPath only accepts s3://bucket/key.zip for input.source capture', () => {
  expect(captureObjectPath('real2sim', { source: 'hope', capture_s3_uri: 's3://b-1/x.zip' })).toBeUndefined();
  expect(captureObjectPath('real2sim', undefined)).toBeUndefined();
  expect(captureObjectPath('real2sim', { source: 'capture', capture_s3_uri: 's3://my-bucket/caps/desk.zip' }))
    .toBe('my-bucket/caps/desk.zip');
  expect(() => captureObjectPath('real2sim', { source: 'capture', capture_s3_uri: 's3://my-bucket/*' })).toThrow();
  expect(() => captureObjectPath('real2sim', { source: 'capture', capture_s3_uri: '' })).toThrow();
});


test.each(VLAS)('%s: iam:PassRole is never on Resource "*" and is limited to EC2', (vla) => {
  const stmts = statements(Template.fromStack(makeApp(vla)));
  const passRole = stmts.filter((s) => actionsOf(s).some((a) => a === 'iam:PassRole' || a === 'iam:*' || a === '*'));
  expect(passRole.length).toBeGreaterThan(0);
  for (const s of passRole) {
    expect(resourcesOf(s)).not.toContain('*');
    expect(s.Condition?.StringEquals?.['iam:PassedToService']).toBe('ec2.amazonaws.com');
  }
});

test.each(VLAS)('%s: EC2 mutating actions are not granted on Resource "*"', (vla) => {
  const stmts = statements(Template.fromStack(makeApp(vla)));
  for (const s of stmts) {
    if (actionsOf(s).some((a) => ['ec2:RunInstances', 'ec2:CreateTags', 'ec2:TerminateInstances'].includes(a))) {
      expect(resourcesOf(s)).not.toContain('*');
    }
    // Neither of these is used by any template (WaitCondition URL, private bucket without ACLs).
    expect(actionsOf(s)).not.toContain('s3:PutObjectAcl');
    expect(actionsOf(s)).not.toContain('cloudformation:SignalResource');
  }
  const create = stmts.filter((s) => actionsOf(s).includes('ec2:CreateTags'));
  expect(create.length).toBeGreaterThan(0);
  for (const s of create) expect(s.Condition?.StringEquals?.['ec2:CreateAction']).toBe('RunInstances');
  const terminate = stmts.filter((s) => actionsOf(s).includes('ec2:TerminateInstances'));
  expect(terminate.length).toBe(2); // AzSelector Lambda (stack delete) + instance (self-terminate)
  for (const s of terminate) expect(s.Condition?.StringEquals?.['ec2:ResourceTag/StackName']).toBeDefined();
});

test.each(VLAS)('%s: AzSelector launches with IMDSv2 required (hop limit 2 for Docker)', (vla) => {
  const template = Template.fromStack(makeApp(vla));
  const fns = Object.values(template.findResources('AWS::Lambda::Function'))
    .map((r) => String(r.Properties.Code?.ZipFile ?? ''))
    .filter((code) => code.includes('run_instances'));
  expect(fns).toHaveLength(1);
  expect(fns[0]).toMatch(/'HttpTokens': 'required'/);
  expect(fns[0]).toMatch(/'HttpPutResponseHopLimit': 2/);
  const run = statements(template).filter((s) => actionsOf(s).includes('ec2:RunInstances') && s.Condition);
  expect(run.some((s) => s.Condition?.StringEquals?.['ec2:MetadataHttpTokens'] === 'required')).toBe(true);
});

// LIBERO-Plus sweep (deploy.py --libero-variant plus): same stack name, timeout / EBS from models/<vla>.yaml
// libero_plus; without the context the stack is the LIBERO one. A target without the block is rejected.
function plusStack(vla: string, stackName: string, liberoVariant?: string, liberoSuite = '') {
  const app = new cdk.App({ context: { region: 'us-east-1', vla, notify_email: 'test@example.com' } });
  return new VlaSimulatorStack(app, stackName, {
    vla, liberoSuite, liberoVariant, env: { account: '123456789012', region: 'us-east-1' },
  });
}

test('libero_variant=plus takes the timeout and EBS size from the libero_plus block', () => {
  const plus = Template.fromStack(plusStack('smolvla', 'SmolVLA-Demo', 'plus'));
  plus.hasResourceProperties('AWS::CloudFormation::WaitCondition', { Timeout: String(180 * 60) });
  plus.hasResourceProperties('AWS::CloudFormation::CustomResource', { EbsGb: '150' });
  const base = Template.fromStack(plusStack('smolvla', 'SmolVLA-Demo'));
  base.hasResourceProperties('AWS::CloudFormation::WaitCondition', { Timeout: String(150 * 60) });
  base.hasResourceProperties('AWS::CloudFormation::CustomResource', { EbsGb: '120' });
  // OpenVLA-OFT: the suite-keyed timeout map is replaced by the scalar libero_plus value.
  const oft = Template.fromStack(plusStack('openvla-oft', 'OpenVLA-OFT-Spatial-Demo', 'plus', 'spatial'));
  oft.hasResourceProperties('AWS::CloudFormation::WaitCondition', { Timeout: String(240 * 60) });
  oft.hasResourceProperties('AWS::CloudFormation::CustomResource', { EbsGb: '270' });
});

test('libero_variant=plus is rejected for a target without a libero_plus block', () => {
  expect(() => plusStack('xvla', 'XVLA-Demo', 'plus')).toThrow(/libero_plus block/);
});
