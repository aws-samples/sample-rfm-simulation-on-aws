/**
 * AzSelectorConstruct
 *
 * Uses a Custom Resource Lambda to launch the real instance directly instead of
 * probing first, which removes the race between probe + terminate and launch.
 *
 * How it works:
 * 1. Try the instance-type fallback list in order
 * 2. For each instance type, list supported AZs with describe-instance-type-offerings
 * 3. Shuffle the AZ list so launches do not pile onto one AZ
 * 4. Call RunInstances (with UserData + IAM + SG) directly in each AZ
 * 5. On success, return InstanceId + AZ + InstanceType (no probe needed)
 * 6. On InsufficientInstanceCapacity, try the next AZ
 * 7. If every AZ fails for a type, fall back to the next instance type
 * 8. Delete: terminate by InstanceId
 *
 * IAM: RunInstances is limited to the AMI, subnets and security groups passed in, and new instances
 * must carry the scope tag and require IMDSv2. CreateTags only works as part of RunInstances,
 * TerminateInstances only on instances with the scope tag, and PassRole only for the instance role
 * and only to EC2.
 */
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';
import { acknowledge, acknowledgeByPath } from './nag';

export interface AzSelectorProps {
  instanceTypes: string[];
  amiId: string;
  subnetIds: string[];
  // Real instance launch params (replaces probe+terminate)
  userDataBase64: string;
  iamInstanceProfileArn: string;
  // Role inside the instance profile; the only role the Lambda may pass (to EC2).
  instanceRoleArn: string;
  securityGroupIds: string[];
  ebsGb: number;
  tags: Array<{ key: string; value: string }>;
  // Tag that RunInstances must set and TerminateInstances requires (e.g. StackName=<stack>).
  // It must also be one of `tags`.
  scopeTag: { key: string; value: string };
  // After every On-Demand type/AZ has no capacity, try the same list again as one-time Spot
  // (interruption = terminate). Off by default; models/<vla>.yaml instance.spot_fallback: true.
  spotFallback?: boolean;
}

export class AzSelectorConstruct extends Construct {
  public readonly instanceId: string;
  public readonly availabilityZone: string;
  public readonly resolvedInstanceType: string;
  public readonly subnetId: string;
  public readonly market: string;

  constructor(scope: Construct, id: string, props: AzSelectorProps) {
    super(scope, id);

    if (!props.tags.some((t) => t.key === props.scopeTag.key && t.value === props.scopeTag.value)) {
      throw new Error(`AzSelector: scopeTag ${props.scopeTag.key} must be one of the instance tags`);
    }
    const ec2Arn = (resource: string, account: string = cdk.Aws.ACCOUNT_ID) =>
      `arn:${cdk.Aws.PARTITION}:ec2:${cdk.Aws.REGION}:${account}:${resource}`;
    const instanceArns = ec2Arn('instance/*');

    const lambdaRole = new iam.Role(this, 'AzSelectorRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
      inlinePolicies: {
        AzSelectorPolicy: new iam.PolicyDocument({
          statements: [
            // Describe* actions do not support resource-level permissions.
            new iam.PolicyStatement({
              actions: ['ec2:DescribeInstanceTypeOfferings', 'ec2:DescribeSubnets'],
              resources: ['*'],
            }),
            // RunInstances: the instance itself must be tagged with the scope tag and use IMDSv2.
            new iam.PolicyStatement({
              actions: ['ec2:RunInstances'],
              resources: [instanceArns],
              conditions: {
                StringEquals: {
                  [`aws:RequestTag/${props.scopeTag.key}`]: props.scopeTag.value,
                  'ec2:MetadataHttpTokens': 'required',
                },
              },
            }),
            // RunInstances: only the given AMI, subnets and security groups; the root volume and
            // the primary ENI are created by the call (their IDs are unknown until then).
            new iam.PolicyStatement({
              actions: ['ec2:RunInstances'],
              resources: [
                ec2Arn(`image/${props.amiId}`, ''),
                ...props.subnetIds.map((id) => ec2Arn(`subnet/${id}`)),
                ...props.securityGroupIds.map((id) => ec2Arn(`security-group/${id}`)),
                ec2Arn('volume/*'),
                ec2Arn('network-interface/*'),
                // A one-time Spot launch also creates a spot-instances-request (only with spotFallback).
                ...(props.spotFallback ? [ec2Arn('spot-instances-request/*')] : []),
              ],
            }),
            // Tags only as part of the launch (TagSpecifications), never on existing resources.
            new iam.PolicyStatement({
              actions: ['ec2:CreateTags'],
              resources: [instanceArns],
              conditions: { StringEquals: { 'ec2:CreateAction': 'RunInstances' } },
            }),
            new iam.PolicyStatement({
              actions: ['ec2:TerminateInstances'],
              resources: [instanceArns],
              conditions: {
                StringEquals: { [`ec2:ResourceTag/${props.scopeTag.key}`]: props.scopeTag.value },
              },
            }),
            new iam.PolicyStatement({
              actions: ['iam:PassRole'],
              resources: [props.instanceRoleArn],
              conditions: { StringEquals: { 'iam:PassedToService': 'ec2.amazonaws.com' } },
            }),
          ],
        }),
      },
    });

    const azSelectorFn = new lambda.Function(this, 'AzSelectorFunction', {
      runtime: lambda.Runtime.PYTHON_3_13,
      handler: 'index.handler',
      role: lambdaRole,
      timeout: cdk.Duration.minutes(10),
      code: lambda.Code.fromInline(AZ_SELECTOR_LAMBDA_CODE),
      description: 'Launches GPU instance with capacity fallback across AZs/types',
    });

    const customResource = new cdk.CustomResource(this, 'AzSelectorResource', {
      serviceToken: azSelectorFn.functionArn,
      properties: {
        InstanceTypes: props.instanceTypes.join(','),
        AmiId: props.amiId,
        SubnetIds: props.subnetIds.join(','),
        UserDataBase64: props.userDataBase64,
        IamInstanceProfileArn: props.iamInstanceProfileArn,
        SecurityGroupIds: props.securityGroupIds.join(','),
        EbsGb: props.ebsGb.toString(),
        Tags: JSON.stringify(props.tags),
        SpotFallback: props.spotFallback ? 'true' : 'false',
        Timestamp: Date.now().toString(),
      },
    });

    azSelectorFn.addPermission('CfnInvoke', {
      principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
    });

    this.instanceId = customResource.getAttString('InstanceId');
    this.availabilityZone = customResource.getAttString('AvailabilityZone');
    this.resolvedInstanceType = customResource.getAttString('InstanceType');
    this.subnetId = customResource.getAttString('SubnetId');
    this.market = customResource.getAttString('Market');

    acknowledge(lambdaRole, [
      {
        id: 'AwsSolutions-IAM4',
        reason: 'AWSLambdaBasicExecutionRole is the minimal managed policy for Lambda CloudWatch Logs access.',
        appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'ec2:DescribeInstanceTypeOfferings and ec2:DescribeSubnets do not support resource-level permissions.',
        appliesTo: ['Resource::*'],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'Instance IDs are unknown until RunInstances returns. RunInstances requires the scope tag and IMDSv2, CreateTags is limited to ec2:CreateAction=RunInstances, and TerminateInstances requires the scope tag.',
        appliesTo: ['Resource::arn:<AWS::Partition>:ec2:<AWS::Region>:<AWS::AccountId>:instance/*'],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'The root EBS volume and primary network interface are created by RunInstances, so their IDs are unknown at synth time. The AMI, subnets and security groups are pinned.',
        appliesTo: [
          'Resource::arn:<AWS::Partition>:ec2:<AWS::Region>:<AWS::AccountId>:volume/*',
          'Resource::arn:<AWS::Partition>:ec2:<AWS::Region>:<AWS::AccountId>:network-interface/*',
        ],
      },
    ]);

    acknowledge(azSelectorFn, [
      {
        id: 'AwsSolutions-L1',
        reason: 'Python 3.13 is the latest stable production runtime for AWS Lambda.',
      },
    ]);
  }
}

const AZ_SELECTOR_LAMBDA_CODE = `
import json
import boto3
import random
import cfnresponse

def handler(event, context):
    print(json.dumps({k: v for k, v in event.items() if k != 'ResourceProperties'} | {'ResourceProperties': {k: (v[:80]+'...' if isinstance(v,str) and len(v)>80 else v) for k,v in event.get('ResourceProperties',{}).items()}}))

    region = context.invoked_function_arn.split(':')[3]
    ec2 = boto3.client('ec2', region_name=region)

    if event['RequestType'] == 'Delete':
        instance_id = event.get('PhysicalResourceId', '')
        if instance_id and instance_id.startswith('i-'):
            try:
                ec2.terminate_instances(InstanceIds=[instance_id])
                print(f'Terminated instance {instance_id}')
            except Exception as e:
                print(f'Terminate failed (ignoring): {e}')
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {},
            physicalResourceId=instance_id or 'az-selector-deleted')
        return

    try:
        props = event['ResourceProperties']
        instance_types = props['InstanceTypes'].split(',')
        ami_id = props['AmiId']
        subnet_ids = props['SubnetIds'].split(',')
        user_data_b64 = props['UserDataBase64']
        iam_profile_arn = props['IamInstanceProfileArn']
        sg_ids = props['SecurityGroupIds'].split(',')
        ebs_gb = int(props['EbsGb'])
        tags = json.loads(props['Tags'])
        markets = ['on-demand'] + (['spot'] if props.get('SpotFallback') == 'true' else [])

        subnets_resp = ec2.describe_subnets(SubnetIds=subnet_ids)
        subnet_az_map = {s['SubnetId']: s['AvailabilityZone'] for s in subnets_resp['Subnets']}
        print(f'Subnet AZ map: {subnet_az_map}')

        all_tried = []

        for market, instance_type in [(m, t) for m in markets for t in instance_types]:
            print(f'--- Trying instance type: {instance_type} ({market}) ---')
            label = instance_type if market == 'on-demand' else f'{instance_type}[spot]'

            resp = ec2.describe_instance_type_offerings(
                LocationType='availability-zone',
                Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
            )
            supported_azs = {o['Location'] for o in resp['InstanceTypeOfferings']}
            print(f'Supported AZs for {instance_type}: {supported_azs}')

            candidates = [(sid, az) for sid, az in subnet_az_map.items() if az in supported_azs]
            if not candidates:
                print(f'{instance_type} not available in any VPC subnet AZ, skipping...')
                all_tried.append(f'{label}(no subnet AZ match)')
                continue

            random.shuffle(candidates)

            for subnet_id, az in candidates:
                print(f'Launching {instance_type} ({market}) in {az} (subnet: {subnet_id})')
                try:
                    tag_specs = [{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': t['key'], 'Value': t['value']} for t in tags]
                    }]
                    market_opts = {} if market == 'on-demand' else {'InstanceMarketOptions': {
                        'MarketType': 'spot',
                        'SpotOptions': {'SpotInstanceType': 'one-time', 'InstanceInterruptionBehavior': 'terminate'},
                    }}
                    run_resp = ec2.run_instances(
                        **market_opts,
                        InstanceType=instance_type,
                        ImageId=ami_id,
                        MinCount=1,
                        MaxCount=1,
                        SubnetId=subnet_id,
                        SecurityGroupIds=sg_ids,
                        IamInstanceProfile={'Arn': iam_profile_arn},
                        UserData=user_data_b64,
                        BlockDeviceMappings=[{
                            'DeviceName': '/dev/sda1',
                            'Ebs': {
                                'VolumeSize': ebs_gb,
                                'VolumeType': 'gp3',
                                'DeleteOnTermination': True,
                                'Encrypted': True,
                            }
                        }],
                        TagSpecifications=tag_specs,
                        # IMDSv2 only. Hop limit 2 so containers on the default Docker bridge
                        # network can still reach IMDS (one extra hop).
                        MetadataOptions={
                            'HttpTokens': 'required',
                            'HttpPutResponseHopLimit': 2,
                            'HttpEndpoint': 'enabled',
                        },
                    )
                    instance_id = run_resp['Instances'][0]['InstanceId']
                    print(f'SUCCESS: {instance_type} ({market}) in {az} (instance: {instance_id})')

                    cfnresponse.send(event, context, cfnresponse.SUCCESS,
                        {'InstanceId': instance_id, 'AvailabilityZone': az,
                         'InstanceType': instance_type, 'SubnetId': subnet_id, 'Market': market},
                        physicalResourceId=instance_id)
                    return

                except Exception as e:
                    error_msg = str(e)
                    if any(c in error_msg for c in ('InsufficientInstanceCapacity', 'SpotMaxPriceTooLow',
                                                    'MaxSpotInstanceCountExceeded')):
                        print(f'InsufficientCapacity: {label} in {az}')
                        all_tried.append(f'{label}/{az}')
                        continue
                    elif 'Unsupported' in error_msg:
                        print(f'Unsupported: {instance_type} in {az}')
                        all_tried.append(f'{label}/{az}(unsupported)')
                        continue
                    else:
                        print(f'Unexpected error: {e}')
                        raise

            print(f'All AZs exhausted for {label}, falling back...')

        cfnresponse.send(event, context, cfnresponse.FAILED, {},
            reason=f'No capacity available for any instance type: {all_tried}',
            physicalResourceId='az-selector-failed')

    except Exception as e:
        print(f'Error: {e}')
        cfnresponse.send(event, context, cfnresponse.FAILED, {},
            reason=str(e),
            physicalResourceId='az-selector-error')
`;
