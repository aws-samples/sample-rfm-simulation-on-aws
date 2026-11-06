# Self-run benchmark advisor (`vlasim bench`)

Back to the [README](../README.md) · [Documentation index](README.md)

> **Self-run benchmark in your account; results depend on the stack and settings; AWS does not certify model performance.**
> The advisor picks candidates and runs them for you on the same settings. It does not publish or guarantee a ranking:
> the numbers come from your own runs, with the interval that goes with their episode count.

Comparing every robot foundation model (RFM) in this repository is a lot of deploys. `vlasim bench` narrows it
down from a few plain questions, runs the 2-4 candidates on the **same actions, episodes, seed and condition
label** with the existing per-target CDK stacks, and writes **one HTML report** (success rate with 95% interval,
per-action table, rollouts side by side, wall time and cost, a ranked recommendation).

```bash
python vlasim.py bench plan                                    # 5 questions -> candidates -> pick 1-3 actions
python vlasim.py bench run --plan bench-plan.yaml --dry-run    # exact deploy commands + offline render check
python vlasim.py bench run --plan bench-plan.yaml              # deploy every candidate in parallel
python vlasim.py bench report --plan bench-plan.yaml           # one HTML report (published to the video gallery)
python vlasim.py bench destroy --plan bench-plan.yaml          # destroy.py --keep-data for every candidate
```

## 1. Plan: questions, candidates, actions

`bench plan` asks five numbered questions (or reads them from `--answers answers.yaml`):

| Key | Question | Answers |
|---|---|---|
| `robot` | What does your robot look like? | `single-arm` (LIBERO, Franka Panda) · `bimanual` (RoboTwin 2.0) · `humanoid` |
| `action` | What kind of action matters most? | `pick-place` · `articulated` (drawer, door, knob) · `long-horizon` · `spatial` · `press` · `tool-use` · `bimanual` · `any` |
| `commercial` | Do you need a license that allows commercial use? | `yes` · `no` |
| `gpu` | Which GPU size can the model use? | `24gb` (L4 / A10G) · `48gb` (L40S) · `any` |
| `edge` | Should the model be small enough for an edge device? | `yes` (about 1B parameters or less) · `no` |

The candidates come from [`bench/catalog.yaml`](../bench/catalog.yaml): one entry per target that can run a
shared benchmark, with embodiment, suites the checkpoint covers, license (commercial use OK / not / to review),
parameters, GPU tier and instance, the default run's time from [cost.md](cost.md) and seconds per episode where
the repo measured one. Values come from `models/*.yaml` and the docs; unknown values are `null` and are never
guessed (an unknown license fails `commercial: yes`, an unknown size fails `edge: yes`).

Selection, in order: robot form, license, GPU tier, edge size; then the benchmark suite most of the remaining
targets cover for the chosen action type; then the **4 cheapest** by estimated cost. Every target that was left
out is listed with the reason (for example `needs a 48gb GPU`, `checkpoint does not cover libero_spatial`,
`not on a shared bench yet: runs whole suites`). `--targets smolvla,xvla,pi` compares a list you choose
(each must still fit the answers).

The actions are the tasks valid for **all** candidates, grouped (pick and place, articulated, long-horizon,
spatial; RoboTwin: tool use, press, two-handed, pick and place), with LIBERO's natural-language instruction.
Pick 1-3, the episodes per action (default 10) and a seed (default: drawn once and written to the plan).
The plan shows the estimate per target and asks before it writes `bench-plan.yaml`:

```text
Estimate (EC2 On-Demand, catalog figures):
  smolvla            g6.xlarge       ~35 min     $0.58   (default run 10 ep 15-30 min + 30 s/episode)
  xvla               g6.2xlarge      ~32 min     $0.64   (default run 10 ep 15-30 min + 13 s/episode)
  pi                 g5.xlarge       ~57 min     $1.18   (default run 120 ep 57 min, scaled by episodes (upper bound))
  total $2.40, wall time ~57 min (parallel)
```

Non-interactive (`answers.yaml` may also hold `actions`, `episodes`, `seed`, `targets`):

```bash
cat > answers.yaml <<'EOF'
robot: single-arm
action: spatial
commercial: "no"
gpu: 24gb
edge: "no"
actions: [libero_spatial:0, libero_spatial:1]
episodes: 10
EOF
python vlasim.py bench plan --answers answers.yaml --targets smolvla,xvla,pi --region ap-northeast-2 --yes
```

Targets on a bench today: LIBERO `smolvla`, `xvla`, `fastwam`, `lingbot-va` (LIBERO-10 only), `multitask-dit`
(LIBERO-10 only), `vla-jepa`, `pi`; RoboTwin 2.0 `smolvla-robotwin`, `lingbot-vla2` (both research / evaluation
only: RoboTwin needs NVIDIA CuRobo). Humanoid targets have no shared task list yet. The other LIBERO targets
are listed under `not_benchable` in the catalog with the reason (whole-suite runs, no condition label, no seed
control).

## 2. Run: one tasks file per target

`bench run` writes `bench-plan.d/tasks-<target>.yaml` next to the plan, in each target's own `tasks:` format
(LeRobot: `task_ids: [id]` + `env_name` + `n_episodes`; π0.5: `task_ids` + `num_episodes`; RoboTwin:
`robotwin_tasks` + `n_episodes`), every entry with the plan's `seed` and `condition: bench-<id>`, and calls the
normal deploy path for each target in parallel:

```text
python deploy.py --vla smolvla --tasks-file .../bench-plan.d/tasks-smolvla.yaml --seed 20261106 --region ap-northeast-2
python deploy.py --vla xvla --tasks-file .../bench-plan.d/tasks-xvla.yaml --seed 20261106 --region ap-northeast-2
python deploy.py --vla pi --tasks-file .../bench-plan.d/tasks-pi.yaml --seed 20261106 --region ap-northeast-2
```

`deploy.py --tasks-file PATH` (and `generate.py --tasks-file`) replaces the `tasks:` list of `models/<vla>.yaml`
for that deploy only, exactly like a [task set](targets.md#task-sets-and-action-types) but from a separate
file, so nothing in `models/` changes; same stack, results bucket and checkpoint revision. `--seed N` is the
shared seed of [results.md](results.md#shared-seed---seed) and is recorded in every `protocol.yaml`.
The condition label is carried into every `summary.txt`, so the bench runs stay out of the default
leaderboard (`--matrix` shows baseline conditions only) and `bench report` can find exactly them.

`--dry-run` stops before AWS: it prints the commands above, renders each target's user data with its tasks
file (`generate.py`, no AWS call) and runs `bash -n` on it. Without it, `bench run` creates the shared video
gallery once (each deploy would otherwise race to create it), starts the deploys in parallel with one log per
target (`bench-plan.d/deploy-<target>.log`) and returns when every `cdk deploy` has returned; the evaluations
continue on the instances and e-mail when they finish. `--no-gallery` and `--email` are passed through.

The plan warns when the estimate gets close to a target's CloudFormation wait limit
(`instance.creationpolicy_timeout`); use fewer episodes or actions then.

## 3. Report

`bench report` reads the candidates' results buckets in the plan's region, keeps only tasks whose
`summary.txt` has the plan's condition (runs from the plan's date on), and writes one self-contained HTML file
(inline CSS and SVG, no JavaScript):

- the disclaimer above, your answers, the chosen set with the reasons and the excluded targets;
- a bar chart of the success rate with the 95% Wilson interval per model;
- a per-action table (k/n, rate, interval) and the first two rollout videos per action and model side by side;
- measured wall time (RUN_ID to the last upload) and EC2 cost at the pinned On-Demand price
  ([results.md](results.md#cross-model-matrix-leaderboard)), next to the plan's estimate;
- a ranked recommendation with the **tie rule**: when the 95% intervals of the leader and another model overlap,
  the runs do not separate them; the report says "tie" and picks by license (commercial use OK first), then cost.
  More episodes narrow the intervals.

With the gallery (default) the page is uploaded to `g/<id>/bench/<plan id>.html` of the `RFM-Gallery` bucket and
opened with the gallery password (`python vlasim.py gallery`); the videos are the gallery copies, so they play
behind the same password. `--no-gallery` writes the file locally with presigned video links (valid 7 days);
`--from-dir DIR` reads an `aws s3 sync` copy (`DIR/<results bucket>/<RUN_ID>/...`) with no AWS call.

## 4. Clean up

`bench destroy --plan bench-plan.yaml` runs `destroy.py --vla <target> --keep-data --yes` for every candidate
(`--dry-run` prints the commands): the compute stacks go, the results buckets and gallery videos stay, so the
report can be regenerated later. Use `destroy.py --delete-all` per target to remove the data too.

## Limits

- Estimates come from the catalog figures (EC2 On-Demand only); stack create/delete, NAT, EBS and S3 are not included.
  A scaled estimate (`upper bound`) also scales the setup time.
- One run per model. LIBERO and RoboTwin repeat under the same seed ([results.md](results.md#re-run-null-per-simulator)),
  but 10 episodes per action give wide intervals: a tie is the common outcome, and it is reported as one.
- π0.5 runs a task-id subset through the same main.py patch as LIBERO-Plus; its `summary.txt` then carries
  `task_ids`, `n_episodes` and `n_success` from the client's own counters.
