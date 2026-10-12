import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

// The stack reads the rendered UserData from assets/userdata/<vla>.sh (gitignored, written by
// generate.py / deploy.py). Seed a placeholder for any target that has not been rendered yet so
// the tests run on a fresh checkout, and remove only the placeholders we created.
const USERDATA_DIR = path.resolve(__dirname, '../../assets/userdata');
const VLAS = ['gr00t', 'pi'];
const seeded: string[] = [];

beforeAll(() => {
  fs.mkdirSync(USERDATA_DIR, { recursive: true });
  for (const vla of VLAS) {
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
