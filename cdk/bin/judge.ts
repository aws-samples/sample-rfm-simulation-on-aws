#!/usr/bin/env node
/**
 * RFM Simulation on AWS — CDK app for the post-run judges (Flora-Rewarder)
 *
 * A separate app (like bin/gallery.ts) because a judge is not a `vla` target: it scores a finished run
 * of one. deploy.py --judge calls it:
 *   npx cdk deploy FloraJudge-SmolVLA-Demo --app "npx ts-node --prefer-ts-exts bin/judge.ts" \
 *     -c region=ap-northeast-2 -c notify_email=you@example.com -c judge=flora-judge -c judge_target=smolvla \
 *     -c judge_target_stack=SmolVLA-Demo -c judge_bucket=vla-sim-results-smolvla-demo-... -c judge_run=2026-10-20-0130
 */

import * as cdk from 'aws-cdk-lib';
import { Validations } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { JUDGE_TARGETS, JudgeStack, judgeStackName } from '../lib/judge-stack';

const app = new cdk.App();
const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';
const ctx = (k: string): string => ((app.node.tryGetContext(k) as string | undefined) ?? '').trim();

const judge = ctx('judge') || 'flora-judge';
if (!(JUDGE_TARGETS as readonly string[]).includes(judge)) {
  throw new Error(`CDK context "judge" must be one of: ${JUDGE_TARGETS.join(', ')}. Use deploy.py --judge.`);
}
const missing = ['judge_target', 'judge_target_stack', 'judge_bucket', 'judge_run'].filter((k) => !ctx(k));
if (missing.length) {
  throw new Error(`CDK context ${missing.join(', ')} required. Use deploy.py --judge --vla <target> [--run RUN_ID].`);
}

new JudgeStack(app, judgeStackName(judge, ctx('judge_target_stack')), {
  judge,
  target: ctx('judge_target'),
  targetStack: ctx('judge_target_stack'),
  sourceBucket: ctx('judge_bucket'),
  runId: ctx('judge_run'),
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `vla-simulator ${judge} post-run judge of ${ctx('judge_target_stack')} run ${ctx('judge_run')}`,
});

Validations.of(app).addPlugins(new AwsSolutionsChecks(app, { verbose: true }));
