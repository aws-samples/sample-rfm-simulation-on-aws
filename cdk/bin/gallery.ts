#!/usr/bin/env node
/**
 * RFM Simulation on AWS — CDK app for the shared video history gallery (RFM-Gallery)
 *
 * A separate app so the gallery deploys and destroys without a `vla` context. deploy.py calls it:
 *   npx cdk deploy RFM-Gallery --app "npx ts-node --prefer-ts-exts bin/gallery.ts" -c region=us-east-1
 */

import * as cdk from 'aws-cdk-lib';
import { Validations } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { GALLERY_STACK_NAME, GalleryStack } from '../lib/gallery-stack';

const app = new cdk.App();
const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';

new GalleryStack(app, GALLERY_STACK_NAME, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: 'vla-simulator video history gallery (S3 + CloudFront, password per notification e-mail)',
});

Validations.of(app).addPlugins(new AwsSolutionsChecks(app, { verbose: true }));
