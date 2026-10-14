/**
 * cdk-nag v3 acknowledgement helpers.
 *
 * cdk-nag v3 dropped `NagSuppressions` for CDK's native `Validations.of(construct).acknowledge()`.
 * These helpers keep the v2 call shape used across this app ({ id, reason, appliesTo? }) and map it
 * onto v3: each `appliesTo` entry becomes its own finding id `<rule>[<entry>]` (v3 acknowledges
 * findings individually), and a path lookup replaces `addResourceSuppressionsByPath`.
 */
import { Stack, Validations } from 'aws-cdk-lib';
import { IConstruct } from 'constructs';

export interface NagAck {
  readonly id: string;
  readonly reason: string;
  readonly appliesTo?: string[];
}

export function acknowledge(construct: IConstruct, acks: NagAck[]): void {
  for (const a of acks) {
    if (a.appliesTo && a.appliesTo.length > 0) {
      for (const entry of a.appliesTo) {
        Validations.of(construct).acknowledge({ id: `${a.id}[${entry}]`, reason: a.reason });
      }
    } else {
      Validations.of(construct).acknowledge({ id: a.id, reason: a.reason });
    }
  }
}

/** Acknowledge on the construct at `path` (`/<StackId>/<child>/...`, as in cdk-nag v2's ByPath). */
export function acknowledgeByPath(stack: Stack, path: string, acks: NagAck[]): void {
  const parts = path.split('/').filter((p) => p.length > 0);
  if (parts[0] === stack.node.id) parts.shift();
  let node: IConstruct = stack;
  for (const p of parts) {
    const child = node.node.tryFindChild(p);
    if (!child) {
      throw new Error(`cdk-nag acknowledgement: no construct at ${path} (missing '${p}')`);
    }
    node = child;
  }
  acknowledge(node, acks);
}
