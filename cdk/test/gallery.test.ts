import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { GalleryStack } from '../lib/gallery-stack';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

const USERDATA_DIR = path.resolve(__dirname, '../../assets/userdata');
const seeded: string[] = [];
beforeAll(() => {
  fs.mkdirSync(USERDATA_DIR, { recursive: true });
  for (const vla of ['smolvla', 'openvla-oft']) {
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

const GID = '0123456789abcdef0123456789abcdef';
const GALLERY_CTX = {
  gallery_bucket: 'rfm-gallery-123456789012-us-east-1',
  gallery_hash: GID,
  gallery_url: `https://d111.cloudfront.net/g/${GID}/`,
};
const ENV = { account: '123456789012', region: 'us-east-1' };

// ── RFM-Gallery stack ───────────────────────────────────────────────

describe('GalleryStack', () => {
  const template = Template.fromStack(new GalleryStack(new cdk.App(), 'RFM-Gallery', { env: ENV }));

  test('private, SSL-only, retained gallery bucket', () => {
    template.resourceCountIs('AWS::S3::Bucket', 1);
    template.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Retain',
      Properties: {
        BucketName: 'rfm-gallery-123456789012-us-east-1',
        PublicAccessBlockConfiguration: {
          BlockPublicAcls: true, BlockPublicPolicy: true, IgnorePublicAcls: true, RestrictPublicBuckets: true,
        },
      },
    });
    template.hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({ Effect: 'Deny', Condition: { Bool: { 'aws:SecureTransport': 'false' } } }),
          Match.objectLike({ Principal: { Service: 'cloudfront.amazonaws.com' }, Action: 's3:GetObject' }),
        ]),
      },
    });
  });

  test('CloudFront with OAC, HTTPS and the Basic-auth function on viewer-request', () => {
    template.resourceCountIs('AWS::CloudFront::OriginAccessControl', 1);
    template.resourceCountIs('AWS::CloudFront::KeyValueStore', 1);
    template.hasResourceProperties('AWS::CloudFront::Function', {
      FunctionConfig: Match.objectLike({
        Runtime: 'cloudfront-js-2.0',
        KeyValueStoreAssociations: [{ KeyValueStoreARN: Match.anyValue() }],
      }),
    });
    template.hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: Match.objectLike({
        DefaultCacheBehavior: Match.objectLike({
          ViewerProtocolPolicy: 'redirect-to-https',
          AllowedMethods: ['GET', 'HEAD'],
          FunctionAssociations: [{ EventType: 'viewer-request', FunctionARN: Match.anyValue() }],
        }),
      }),
    });
  });

  test('outputs used by deploy.py / destroy.py / vlasim gallery', () => {
    for (const o of ['GalleryBucketName', 'DistributionDomainName', 'DistributionId', 'KvsArn']) {
      template.hasOutput(o, {});
    }
  });
});

// ── CloudFront Function logic (run in Node with the cloudfront module stubbed) ──

type CfResult = { statusCode?: number; headers?: Record<string, { value: string }>; uri?: string };
function loadAuthHandler(store: Record<string, string>): (event: unknown) => Promise<CfResult> {
  const src = fs.readFileSync(path.resolve(__dirname, '../functions/gallery-auth.js'), 'utf8')
    .replace(/^import .*$/gm, '');
  const cf = {
    kvs: () => ({
      get: async (k: string) => {
        if (!(k in store)) throw new Error(`key not found: ${k}`);
        return store[k];
      },
    }),
  };
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  return new Function('crypto', 'cf', `${src}\nreturn handler;`)(crypto, cf);
}

function req(uri: string, password?: string) {
  const headers: Record<string, { value: string }> = {};
  if (password !== undefined) {
    headers.authorization = { value: 'Basic ' + Buffer.from(`anyone:${password}`).toString('base64') };
  }
  return { request: { uri, headers } };
}

describe('gallery-auth CloudFront Function', () => {
  // Same vector as tests/test_gallery.py (gallery.password_digest).
  const digest = '7e51c6871c3dd77b9379e976ba2c4205ef41cdd638e18ac635fad0c4627d21bb';
  const handler = loadAuthHandler({ [GID]: digest });

  test('digest matches gallery.py', () => {
    expect(crypto.createHash('sha256').update(GID + 's3cret-Pass_1').digest('hex')).toBe(digest);
  });

  test('right password: index rewrite and pass-through', async () => {
    expect((await handler(req(`/g/${GID}/`, 's3cret-Pass_1'))).uri).toBe(`/g/${GID}/index.html`);
    const r = await handler(req(`/g/${GID}/smolvla/run/task-0/ep0.mp4`, 's3cret-Pass_1'));
    expect(r.statusCode).toBeUndefined();
    expect(r.uri).toBe(`/g/${GID}/smolvla/run/task-0/ep0.mp4`);
  });

  test.each([
    ['no credentials', `/g/${GID}/`, undefined],
    ['wrong password', `/g/${GID}/index.html`, 'nope'],
    ['unknown id', '/g/ffffffffffffffffffffffffffffffff/', 's3cret-Pass_1'],
  ])('%s -> 401 with a Basic challenge', async (_name, uri, pw) => {
    const r = await handler(req(uri, pw));
    expect(r.statusCode).toBe(401);
    expect(r.headers?.['www-authenticate'].value).toMatch(/^Basic realm=/);
  });

  test('malformed Authorization -> 401', async () => {
    const r = await handler({ request: { uri: `/g/${GID}/`, headers: { authorization: { value: 'Basic %%%' } } } });
    expect(r.statusCode).toBe(401);
  });

  test('/g/<id> redirects to /g/<id>/, other paths 404', async () => {
    const r = await handler(req(`/g/${GID}`));
    expect(r.statusCode).toBe(302);
    expect(r.headers?.location.value).toBe(`/g/${GID}/`);
    for (const uri of ['/', '/index.html', '/g/', '/g/ABC/', `/x/g/${GID}/`]) {
      expect((await handler(req(uri, 's3cret-Pass_1'))).statusCode).toBe(404);
    }
  });
});

// ── Target stack with / without the gallery context ─────────────────

function targetStack(vla: string, ctx: Record<string, string> = {}, stackName = 'SmolVLA-Demo', liberoSuite?: string) {
  const app = new cdk.App({ context: { region: 'us-east-1', vla, notify_email: 'test@example.com', ...ctx } });
  return new VlaSimulatorStack(app, stackName, { vla, liberoSuite, env: ENV });
}

describe('VlaSimulatorStack gallery publisher', () => {
  test('without gallery context: no publisher, no notification, no GalleryUrl', () => {
    const t = Template.fromStack(targetStack('smolvla'));
    t.resourceCountIs('Custom::S3BucketNotifications', 0);
    const fns = Object.keys(t.findResources('AWS::Lambda::Function'));
    expect(fns.some((id) => id.startsWith('GalleryPublisher'))).toBe(false);
    expect(Object.keys(t.findOutputs('*'))).not.toContain('GalleryUrl');
    expect(JSON.stringify(t.toJSON())).not.toContain('GalleryUrl');
  });

  test('with gallery context: Lambda on .mp4/.txt, scoped IAM, outputs, bootstrap export', () => {
    const t = Template.fromStack(targetStack('smolvla', GALLERY_CTX));
    t.hasResourceProperties('AWS::Lambda::Function', {
      Handler: 'index.handler',
      Environment: {
        Variables: { GALLERY_BUCKET: GALLERY_CTX.gallery_bucket, GALLERY_PREFIX: `g/${GID}/`, GALLERY_TARGET: 'smolvla' },
      },
    });
    t.hasResourceProperties('Custom::S3BucketNotifications', {
      BucketName: { Ref: 'ResultsBucketA95A2103' },
      NotificationConfiguration: {
        LambdaFunctionConfigurations: [
          Match.objectLike({ Events: ['s3:ObjectCreated:*'], Filter: { Key: { FilterRules: [{ Name: 'suffix', Value: '.mp4' }] } } }),
          Match.objectLike({ Events: ['s3:ObjectCreated:*'], Filter: { Key: { FilterRules: [{ Name: 'suffix', Value: '.txt' }] } } }),
        ],
      },
    });
    const role = Object.entries(t.findResources('AWS::IAM::Role')).find(([id]) => id.startsWith('GalleryPublisherRole'));
    expect(role).toBeDefined();
    const stmts = JSON.stringify(role![1].Properties.Policies);
    expect(stmts).toContain(`:s3:::${GALLERY_CTX.gallery_bucket}/g/${GID}/*`);
    expect(stmts).toContain(`"s3:prefix":["g/${GID}/*"]`);
    expect(stmts).not.toContain('"Resource":"*"');
    expect(stmts).not.toContain('s3:DeleteObject');
    t.hasOutput('GalleryUrl', { Value: GALLERY_CTX.gallery_url });
    t.hasOutput('GalleryPrefix', { Value: `g/${GID}/smolvla/` });
    expect(JSON.stringify(t.toJSON())).toContain('export GalleryUrl=\\"${GalleryUrl}\\"');
  });

  test('OpenVLA-OFT suites publish to their own gallery folder', () => {
    const t = Template.fromStack(targetStack('openvla-oft', GALLERY_CTX, 'OpenVLA-OFT-Spatial-Demo', 'spatial'));
    t.hasOutput('GalleryPrefix', { Value: `g/${GID}/openvla-oft-spatial/` });
  });

  test('partial or malformed gallery context is rejected', () => {
    expect(() => targetStack('smolvla', { gallery_hash: GID })).toThrow(/Gallery context/);
    expect(() => targetStack('smolvla', { ...GALLERY_CTX, gallery_hash: 'xyz' })).toThrow(/Gallery context/);
    expect(() => targetStack('smolvla', { ...GALLERY_CTX, gallery_url: 'http://evil/' })).toThrow(/Gallery context/);
  });
});
