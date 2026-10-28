import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { JudgeStack, judgeStackName, JUDGE_AMI_MAPPING } from '../lib/judge-stack';

// The stack reads the rendered UserData (assets/userdata/flora-judge.sh, gitignored); seed a
// placeholder on a fresh checkout and remove only what we created.
const USERDATA = path.resolve(__dirname, '../../assets/userdata/flora-judge.sh');
let seeded = false;
beforeAll(() => {
  fs.mkdirSync(path.dirname(USERDATA), { recursive: true });
  if (!fs.existsSync(USERDATA)) {
    fs.writeFileSync(USERDATA, '#!/bin/bash\n# test placeholder\n');
    seeded = true;
  }
});
afterAll(() => {
  if (seeded) fs.rmSync(USERDATA, { force: true });
});

const BUCKET = 'vla-sim-results-smolvla-demo-ap-northeast-2-123456789012';
const RUN = '2026-10-20-0130';

function makeStack(overrides: Partial<{ runId: string; sourceBucket: string; region: string }> = {}) {
  const app = new cdk.App({ context: { notify_email: 'test@example.com' } });
  return new JudgeStack(app, judgeStackName('flora-judge', 'SmolVLA-Demo'), {
    judge: 'flora-judge',
    target: 'smolvla',
    targetStack: 'SmolVLA-Demo',
    sourceBucket: overrides.sourceBucket ?? BUCKET,
    runId: overrides.runId ?? RUN,
    env: { account: '123456789012', region: overrides.region ?? 'ap-northeast-2' },
  });
}

test('stack name is derived from the judged stack', () => {
  expect(makeStack().stackName).toBe('FloraJudge-SmolVLA-Demo');
});

test('one AzSelector-launched instance, a WaitCondition, an SNS topic, no bucket of its own', () => {
  const t = Template.fromStack(makeStack());
  t.resourceCountIs('AWS::EC2::Instance', 0);
  t.resourceCountIs('AWS::CloudFormation::CustomResource', 1);
  t.resourceCountIs('AWS::CloudFormation::WaitCondition', 1);
  t.resourceCountIs('AWS::SNS::Topic', 1);
  t.resourceCountIs('AWS::S3::Bucket', 0);
});

test('instance types come from models/flora-judge.yaml with the Blackwell-capable AMI', () => {
  const t = Template.fromStack(makeStack());
  t.hasResourceProperties('AWS::CloudFormation::CustomResource', {
    InstanceTypes: 'g7e.2xlarge,g7e.4xlarge,g6e.12xlarge,p5.4xlarge',
    AmiId: JUDGE_AMI_MAPPING['ap-northeast-2'],
    EbsGb: '250',
  });
});

test('instance role is scoped to the judged run', () => {
  const t = Template.fromStack(makeStack());
  const roles = t.findResources('AWS::IAM::Role');
  const json = JSON.stringify(Object.entries(roles).filter(([id]) => id.startsWith('JudgeInstanceRole')));
  expect(json).toContain(`:s3:::${BUCKET}/${RUN}/*`);
  expect(json).toContain(`:s3:::${BUCKET}/${RUN}/judge*`);
  expect(json).toContain(`:s3:::${BUCKET}/${RUN}/task-*/videos/*.txt`);
  expect(json).toContain(`"s3:prefix":["${RUN}/*"]`);
  expect(json).not.toMatch(/"s3:\*"|"Resource":"\*"/);
  t.hasResourceProperties('AWS::IAM::Role', {
    Policies: Match.arrayWith([Match.objectLike({ PolicyName: 'JudgePolicy' })]),
  });
});

test('the bootstrap exports the run to the UserData', () => {
  const t = Template.fromStack(makeStack());
  const json = JSON.stringify(t.findResources('AWS::CloudFormation::CustomResource'));
  for (const v of ['JudgeRunId', 'JudgeTarget', 'JudgeTargetStack', 'S3BucketName']) {
    expect(json).toContain(`export ${v}=`);
  }
  expect(json).toContain(RUN);
});

test('rejects run ids and buckets that would widen the IAM grant', () => {
  expect(() => makeStack({ runId: '*' })).toThrow(/run id/);
  expect(() => makeStack({ runId: '2026-10-20-0130/../x' })).toThrow(/run id/);
  expect(() => makeStack({ sourceBucket: 'Bad_Bucket' })).toThrow(/bucket/);
});

test('unsupported region fails at synth', () => {
  expect(() => makeStack({ region: 'sa-east-1' })).toThrow(/Unsupported region/);
});
