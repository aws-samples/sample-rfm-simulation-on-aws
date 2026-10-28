/**
 * RFM Simulation on AWS — post-run judge stack (Flora-Rewarder)
 *
 * Scores the rollout videos of a FINISHED run of another target: one GPU instance reads
 * s3://<results bucket of that target>/<run_id>/task-N/videos/*, runs the judge model, writes
 * <run_id>/judge.json + judge.txt next to the run, notifies by e-mail (SNS) and terminates itself.
 *
 * Resources: VPC + SG + SNS + IAM + AzSelector + WaitCondition. No bucket of its own: the results
 * stay in the evaluated target's bucket (which outlives that target's stack, RemovalPolicy.RETAIN).
 * Deploy:    python deploy.py --judge --vla smolvla [--run RUN_ID]   (= python vlasim.py judge ...)
 * Teardown:  python destroy.py --judge --vla smolvla
 * Config:    ../models/<judge>.yaml (instance.preferred, ebs_gb, creationpolicy_timeout)
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';
import { acknowledge, acknowledgeByPath } from './nag';
import { AzSelectorConstruct } from './az-selector';

/** Post-run judges (targets.py JUDGE_TARGETS). */
export const JUDGE_TARGETS = ['flora-judge'] as const;

/**
 * Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04), 20261027. Unlike the PyTorch 2.4.1
 * DLAMI the policy targets use (driver 550), its driver supports Blackwell (G7e: RTX PRO 6000) and
 * the CUDA 12.9/13 wheels vLLM installs. Refresh per region with the public SSM parameter
 *   /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id
 */
export const JUDGE_AMI_MAPPING: Record<string, string> = {
  'ap-northeast-2': 'ami-0233dba07f826c984', // Seoul
  'ap-northeast-1': 'ami-04afffefcaa016081', // Tokyo
  'us-east-1':      'ami-05414055cdd6e338f', // Virginia
  'us-west-2':      'ami-0fcc8fa7ee8d2dc4c', // Oregon
  'eu-central-1':   'ami-0b9e63bb13a37ba97', // Frankfurt
};

export const JUDGE_STACK_PREFIX: Record<string, string> = {
  'flora-judge': 'FloraJudge',
};

/** Stack id of the judge of one evaluated stack, e.g. FloraJudge-SmolVLA-Demo. */
export function judgeStackName(judge: string, targetStack: string): string {
  return `${JUDGE_STACK_PREFIX[judge]}-${targetStack}`;
}

export interface JudgeStackProps extends cdk.StackProps {
  judge: string;
  /** Evaluated --vla target and its stack name (messages, tags). */
  target: string;
  targetStack: string;
  /** Results bucket of the evaluated target and the run id inside it. */
  sourceBucket: string;
  runId: string;
}

const BUCKET_RE = /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/;
// Run ids are YYYY-MM-DD-HHMM; allow any plain id but never IAM wildcard characters.
const RUN_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const STACK_RE = /^[A-Za-z][A-Za-z0-9-]{0,100}$/;
const TARGET_RE = /^[a-z0-9][a-z0-9-]{0,40}$/;

export class JudgeStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: JudgeStackProps) {
    super(scope, id, props);
    const { judge, target, targetStack, sourceBucket, runId } = props;

    if (!(JUDGE_TARGETS as readonly string[]).includes(judge)) {
      throw new Error(`Unknown judge "${judge}". Supported: ${JUDGE_TARGETS.join(', ')}.`);
    }
    if (!BUCKET_RE.test(sourceBucket)) throw new Error(`judge_bucket is not a bucket name: "${sourceBucket}"`);
    if (!RUN_ID_RE.test(runId)) throw new Error(`judge_run is not a run id: "${runId}"`);
    if (!STACK_RE.test(targetStack)) throw new Error(`judge_target_stack is not a stack name: "${targetStack}"`);
    if (!TARGET_RE.test(target)) throw new Error(`judge_target is not a target id: "${target}"`);

    const simConfig = yaml.load(fs.readFileSync(path.resolve(__dirname, '../../simulator-config.yaml'), 'utf8')) as any;
    const modelConfig = yaml.load(fs.readFileSync(path.resolve(__dirname, `../../models/${judge}.yaml`), 'utf8')) as any;
    const deployment = { ...simConfig.deployment, ...(modelConfig.deployment ?? {}) };
    const instanceCfg = modelConfig.instance ?? {};
    const notifyEmail: string = this.node.tryGetContext('notify_email') ?? deployment.notify_email;
    const autoTerminate: boolean = deployment.auto_terminate ?? true;
    const preferred = instanceCfg.preferred;
    if (!Array.isArray(preferred) || preferred.length === 0 || !preferred.every((t: unknown) => typeof t === 'string')) {
      throw new Error(`models/${judge}.yaml: instance.preferred (list of instance types) is required`);
    }
    const ebsGb: number = instanceCfg.ebs_gb ?? 250;
    const creationTimeout: string = instanceCfg.creationpolicy_timeout ?? 'PT180M';
    const stackName = id;

    const amiId = JUDGE_AMI_MAPPING[this.region];
    if (!amiId) {
      throw new Error(
        `Unsupported region for ${judge}: ${this.region}\n` +
        `Supported regions: ${Object.keys(JUDGE_AMI_MAPPING).join(', ')} (JUDGE_AMI_MAPPING in cdk/lib/judge-stack.ts).`,
      );
    }

    const vpc = new ec2.Vpc(this, 'JudgeVPC', {
      maxAzs: 4,
      natGateways: 1,
      subnetConfiguration: [
        { name: 'public',  subnetType: ec2.SubnetType.PUBLIC,              cidrMask: 24 },
        { name: 'private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
      ],
    });
    const sg = new ec2.SecurityGroup(this, 'JudgeSG', {
      vpc,
      description: `${judge} instance - outbound only`,
      allowAllOutbound: true,
    });

    const topic = new sns.Topic(this, 'NotificationTopic', { topicName: `${stackName}-notify` });
    new sns.Subscription(this, 'EmailSubscription', {
      topic,
      protocol: sns.SubscriptionProtocol.EMAIL,
      endpoint: notifyEmail,
    });

    // The evaluated run only: list + read under <run>/, write judge.* and the per-video captions.
    const bucketArn = `arn:${cdk.Aws.PARTITION}:s3:::${sourceBucket}`;
    const runPrefix = `${runId}/`;
    const role = new iam.Role(this, 'JudgeInstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore')],
      inlinePolicies: {
        JudgePolicy: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: ['s3:ListBucket'],
              resources: [bucketArn],
              conditions: { StringLike: { 's3:prefix': [`${runPrefix}*`] } },
            }),
            new iam.PolicyStatement({
              actions: ['s3:GetObject'],
              resources: [`${bucketArn}/${runPrefix}*`],
            }),
            new iam.PolicyStatement({
              actions: ['s3:PutObject'],
              resources: [`${bucketArn}/${runPrefix}judge*`, `${bucketArn}/${runPrefix}task-*/videos/*.txt`],
            }),
            new iam.PolicyStatement({ actions: ['sns:Publish'], resources: [topic.topicArn] }),
            new iam.PolicyStatement({
              actions: ['ec2:TerminateInstances'],
              resources: [`arn:${cdk.Aws.PARTITION}:ec2:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:instance/*`],
              conditions: { StringEquals: { 'ec2:ResourceTag/StackName': stackName } },
            }),
            new iam.PolicyStatement({
              actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents', 'logs:DescribeLogStreams'],
              resources: [`arn:${cdk.Aws.PARTITION}:logs:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:log-group:/${judge}/*`],
            }),
          ],
        }),
      },
    });
    const instanceProfile = new iam.CfnInstanceProfile(this, 'JudgeInstanceProfile', { roles: [role.roleName] });

    const waitHandle = new cdk.CfnWaitConditionHandle(this, 'JudgeWaitHandle');
    new cdk.CfnWaitCondition(this, 'JudgeWaitCondition', {
      count: 1,
      timeout: String(parseInt(creationTimeout.replace(/\D/g, ''), 10) * 60),
      handle: waitHandle.ref,
    });

    const userDataPath = path.resolve(__dirname, `../../assets/userdata/${judge}.sh`);
    if (!fs.existsSync(userDataPath)) {
      throw new Error(
        `UserData script not found: ${userDataPath}\n` +
        `Use deploy.py --judge (it renders it), or run: python generate.py --judge-target ${judge}`,
      );
    }
    const userDataAsset = new s3assets.Asset(this, 'JudgeUserDataAsset', { path: userDataPath });
    userDataAsset.grantRead(role);

    const bootstrap = [
      '#!/bin/bash',
      `# vla-simulator ${judge} — S3 bootstrap`,
      'export S3BucketName="${S3BucketName}"',
      'export JudgeRunId="${JudgeRunId}"',
      'export JudgeTarget="${JudgeTarget}"',
      'export JudgeTargetStack="${JudgeTargetStack}"',
      'export SnsTopicArn="${SnsTopicArn}"',
      'export StackName="${StackName}"',
      'export WaitConditionUrl="${WaitConditionUrl}"',
      'export Region="${Region}"',
      'export AutoTerminate="${AutoTerminate}"',
      `aws s3 cp ${userDataAsset.s3ObjectUrl} /tmp/${judge}.sh`,
      `chmod +x /tmp/${judge}.sh`,
      `exec /tmp/${judge}.sh`,
    ];
    const userData = cdk.Fn.base64(cdk.Fn.sub(bootstrap.join('\n'), {
      S3BucketName: sourceBucket,
      JudgeRunId: runId,
      JudgeTarget: target,
      JudgeTargetStack: targetStack,
      SnsTopicArn: topic.topicArn,
      StackName: stackName,
      WaitConditionUrl: waitHandle.ref,
      Region: this.region,
      AutoTerminate: autoTerminate ? 'true' : 'false',
    }));

    const azSelector = new AzSelectorConstruct(this, 'AzSelector', {
      instanceTypes: preferred,
      amiId,
      subnetIds: vpc.privateSubnets.map((s) => s.subnetId),
      userDataBase64: userData,
      iamInstanceProfileArn: instanceProfile.attrArn,
      instanceRoleArn: role.roleArn,
      securityGroupIds: [sg.securityGroupId],
      ebsGb,
      tags: [
        { key: 'Name', value: `${stackName}` },
        { key: 'StackName', value: stackName },
        { key: 'JudgedStack', value: targetStack },
      ],
      scopeTag: { key: 'StackName', value: stackName },
    });

    new cdk.CfnOutput(this, 'JudgeJsonUri', {
      value: `s3://${sourceBucket}/${runPrefix}judge.json`,
      description: 'judge.json of the evaluated run (judge.txt next to it)',
    });
    new cdk.CfnOutput(this, 'JudgedRun', { value: `${targetStack} ${runId}`, description: 'Evaluated stack and run id' });
    new cdk.CfnOutput(this, 'LogGroup', { value: `/${judge}/userdata`, description: 'CloudWatch log group of the judge instance' });
    new cdk.CfnOutput(this, 'SNSTopicArn', { value: topic.topicArn, description: 'Completion notification SNS topic ARN' });
    new cdk.CfnOutput(this, 'InstanceId', { value: azSelector.instanceId, description: 'EC2 instance ID' });
    new cdk.CfnOutput(this, 'SelectedInstanceType', { value: azSelector.resolvedInstanceType, description: 'GPU instance type selected by AzSelector' });
    new cdk.CfnOutput(this, 'SelectedAZ', { value: azSelector.availabilityZone, description: 'Availability zone selected by AzSelector' });

    // ── cdk-nag acknowledgements ──────────────────────────────────
    acknowledgeByPath(this, `/${id}/JudgeVPC/Resource`, [
      { id: 'AwsSolutions-VPC7', reason: 'VPC Flow Logs omitted for this sample to reduce cost. Short-lived stack with no inbound traffic allowed.' },
    ]);
    acknowledge(topic, [
      { id: 'AwsSolutions-SNS3', reason: 'SNS topic used only for email notifications to end users. Enforcing SSL on publish not required for this notification-only use case.' },
    ]);
    acknowledge(role, [
      { id: 'AwsSolutions-IAM4', reason: 'AmazonSSMManagedInstanceCore enables SSM Session Manager for browser-based terminal access.', appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore'] },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'Read-only on the objects of the one evaluated run (<run_id>/*); video names are created per run.',
        appliesTo: [`Resource::arn:<AWS::Partition>:s3:::${sourceBucket}/${runPrefix}*`],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'Writes only judge.json/judge.txt/judge-* and the per-video .txt captions of the evaluated run.',
        appliesTo: [
          `Resource::arn:<AWS::Partition>:s3:::${sourceBucket}/${runPrefix}judge*`,
          `Resource::arn:<AWS::Partition>:s3:::${sourceBucket}/${runPrefix}task-*/videos/*.txt`,
        ],
      },
      { id: 'AwsSolutions-IAM5', reason: 'Self-termination: the instance ID is unknown at synth time; ec2:TerminateInstances is constrained by StringEquals on the StackName tag.', appliesTo: ['Resource::arn:<AWS::Partition>:ec2:<AWS::Region>:<AWS::AccountId>:instance/*'] },
      {
        id: 'AwsSolutions-IAM5',
        reason: `CloudWatch Logs wildcard limited to /${judge}/* log group prefix.`,
        appliesTo: [`Resource::arn:<AWS::Partition>:logs:<AWS::Region>:<AWS::AccountId>:log-group:/${judge}/*`],
      },
    ]);
    const assetRegion = cdk.Token.isUnresolved(this.region) ? '<AWS::Region>' : this.region;
    const assetAccount = cdk.Token.isUnresolved(this.account) ? '<AWS::AccountId>' : this.account;
    const assetPartition = cdk.Token.isUnresolved(this.region) ? '<AWS::Partition>' : (this.region.startsWith('cn-') ? 'aws-cn' : 'aws');
    acknowledgeByPath(this, `/${id}/JudgeInstanceRole/DefaultPolicy/Resource`, [
      {
        id: 'AwsSolutions-IAM5',
        reason: 'CDK auto-generates this policy for S3 Asset download from CDK bootstrap bucket. Required by s3assets.Asset.grantRead().',
        appliesTo: [
          'Action::s3:GetBucket*',
          'Action::s3:GetObject*',
          'Action::s3:List*',
          `Resource::arn:${assetPartition}:s3:::cdk-hnb659fds-assets-${assetAccount}-${assetRegion}/*`,
        ],
      },
    ]);
  }
}
