/**
 * RFM-Gallery — shared video history gallery (one per account: its CloudFront OAC and KeyValueStore
 * names are account-global, so a second region fails with AlreadyExists; deploy.py stops before that,
 * deploy other regions with --no-gallery)
 *
 * Resources: private S3 bucket + CloudFront distribution (OAC) + CloudFront KeyValueStore +
 *            CloudFront Function (viewer-request, HTTP Basic auth against the KVS)
 *
 * Every target stack deployed with the gallery on gets a GalleryPublisher Lambda that copies the
 * run videos into g/<hash>/<target>/<run_id>/ of this bucket and rebuilds g/<hash>/index.html.
 * <hash> is derived from the notification e-mail, and the KVS holds sha256(<hash> + password) for
 * it. deploy.py deploys this stack on demand and writes the KVS entries (see gallery.py).
 *
 * Deploy:   python deploy.py --vla <target>             (creates it when missing)
 * Direct:   npx cdk deploy RFM-Gallery --app "npx ts-node --prefer-ts-exts bin/gallery.ts" -c region=<r>
 * Teardown: python destroy.py --vla <target> --delete-all (when it is the last target stack)
 */

import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';
import { acknowledge } from './nag';

export const GALLERY_STACK_NAME = 'RFM-Gallery';

export class GalleryStack extends cdk.Stack {
  public readonly bucket: s3.Bucket;
  public readonly distribution: cloudfront.Distribution;
  public readonly keyValueStore: cloudfront.KeyValueStore;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // RETAIN like the results buckets: the history outlives the stack unless
    // `destroy.py --delete-all` empties and deletes it explicitly.
    this.bucket = new s3.Bucket(this, 'GalleryBucket', {
      bucketName: `rfm-gallery-${this.account}-${this.region}`,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: false,
    });

    this.keyValueStore = new cloudfront.KeyValueStore(this, 'GalleryPasswords', {
      comment: 'RFM video history: gallery id -> sha256(id + password)',
    });

    const authFunction = new cloudfront.Function(this, 'GalleryAuth', {
      code: cloudfront.FunctionCode.fromFile({ filePath: path.resolve(__dirname, '../functions/gallery-auth.js') }),
      runtime: cloudfront.FunctionRuntime.JS_2_0,
      keyValueStore: this.keyValueStore,
      comment: 'RFM video history: Basic auth for /g/<id>/ against the KeyValueStore',
    });

    this.distribution = new cloudfront.Distribution(this, 'GalleryDistribution', {
      comment: 'RFM video history gallery',
      priceClass: cloudfront.PriceClass.PRICE_CLASS_200,
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(this.bucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
        functionAssociations: [{ function: authFunction, eventType: cloudfront.FunctionEventType.VIEWER_REQUEST }],
      },
    });

    new cdk.CfnOutput(this, 'GalleryBucketName', {
      value: this.bucket.bucketName,
      description: 'Video history gallery bucket (objects under g/<id>/)',
    });
    new cdk.CfnOutput(this, 'DistributionDomainName', {
      value: this.distribution.distributionDomainName,
      description: 'Video history gallery CloudFront domain (https://DOMAIN/g/<id>/)',
    });
    new cdk.CfnOutput(this, 'DistributionId', {
      value: this.distribution.distributionId,
      description: 'Video history gallery CloudFront distribution ID',
    });
    new cdk.CfnOutput(this, 'KvsArn', {
      value: this.keyValueStore.keyValueStoreArn,
      description: 'CloudFront KeyValueStore with the gallery password digests',
    });

    // ── cdk-nag acknowledgements ──────────────────────────────────
    acknowledge(this.bucket, [
      { id: 'AwsSolutions-S1', reason: 'Server access logging omitted for this sample gallery bucket. Bucket is private (BlockPublicAccess.BLOCK_ALL) and read only through CloudFront OAC.' },
    ]);
    acknowledge(this.distribution, [
      { id: 'AwsSolutions-CFR1', reason: 'The gallery is a personal video history; viewers are not restricted by country. Every request needs the per-e-mail password (CloudFront Function + KeyValueStore).' },
      { id: 'AwsSolutions-CFR2', reason: 'Static, password-protected video files only (GET/HEAD); AWS WAF would add a fixed monthly cost to a sample with near-zero traffic.' },
      { id: 'AwsSolutions-CFR3', reason: 'Access logging omitted for this sample gallery to avoid an extra log bucket; viewers are authenticated by the viewer-request function.' },
      { id: 'AwsSolutions-CFR4', reason: 'Uses the default *.cloudfront.net certificate (no custom domain), for which the viewer security policy cannot be changed. Add a custom domain + ACM certificate to raise the minimum TLS version.' },
    ]);
  }
}
