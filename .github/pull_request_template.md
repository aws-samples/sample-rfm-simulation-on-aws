## Summary

<!-- What does this change and why? Link the related issue, e.g. "Closes #123". -->

## Checklist

- [ ] Userdata renders for the affected targets and passes `bash -n` (`python generate.py --vla <target> --output-dir assets/userdata && bash -n assets/userdata/<target>.sh`)
- [ ] `python -m pytest -q tests` passes
- [ ] `npx cdk synth -c vla=<target> ...` succeeds for the affected targets (run in `cdk/`)
- [ ] Documentation updated (`README.md` / `docs/`) where behavior, targets or options changed
- [ ] Third-party code, models or datasets added: license noted in `docs/licenses.md` (and `NOTICE` if required)

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
