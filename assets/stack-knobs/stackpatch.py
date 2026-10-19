# Stack-knob patcher for upstream LIBERO eval scripts (AMLC multi-model study, 2026-10-15).
# usage: python3 stackpatch.py <eval.py> <step_scale> <random_init:true|false> [replan_head:true|false]
# replan_head: OpenVLA-OFT upstream fills deque(maxlen=num_open_loop_steps) with the full 8-action chunk,
# so num_open_loop_steps<8 silently executes the chunk TAIL. replan_head=true executes the first k instead.
# Restores <eval.py>.stackorig before patching so each task starts from upstream bytes.
import os, re, shutil, sys
p, scale, rnd = sys.argv[1], float(sys.argv[2]), sys.argv[3].lower() == 'true'
head = len(sys.argv) > 4 and sys.argv[4].lower() == 'true'
orig = p + '.stackorig'
if not os.path.exists(orig):
    shutil.copy2(p, orig)
shutil.copy2(orig, p)
s = open(p).read()
if scale != 1.0:
    f = lambda m: m.group(1) + str(int(round(int(m.group(2)) * scale)))
    pats = [r'((?:"libero_(?:spatial|object|goal|10|90)"|TaskSuite\.LIBERO_\w+)\s*:\s*)(\d+)',
            r'(max_steps\s*=\s*)(\d+)\b']
    for pat in pats:
        s2, k = re.subn(pat, f, s)
        if k:
            s = s2
            print(f'STACKPATCH step cap x{scale}: {k} sites ({pat[:30]})')
            break
    else:
        sys.exit('STACKPATCH FAIL: no step-cap site found')
if rnd:
    s2, k = re.subn(r'env\.set_init_state\([^)]*\)', 'env.reset()', s)
    if not k:
        sys.exit('STACKPATCH FAIL: no set_init_state site found')
    s = s2
    print(f'STACKPATCH random init: {k} sites')
if head:
    s2, k = re.subn(r'action_queue\.extend\(actions\)', 'action_queue.extend(actions[:cfg.num_open_loop_steps])', s)
    if not k:
        sys.exit('STACKPATCH FAIL: no action_queue.extend site found')
    s = s2
    print(f'STACKPATCH replan head: {k} sites')
open(p, 'w').write(s)
