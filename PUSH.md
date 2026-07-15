# Pushing the pipeline subtree to fredsanto/GENOVA2I

Run these yourself (`!`-prefixed in chat, or directly in a terminal) — do not
paste your PAT into any command I run, only into git's own interactive
username/password prompt.

**First, rotate the PAT you pasted earlier in this session** — it's been
exposed in the transcript and must be treated as compromised. Generate a new
one at GitHub → Settings → Developer settings → Personal access tokens, then
use the new one below.

```bash
cd /work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI

# 1. Split out just the pipeline subtree (not the whole monorepo) as its own
#    branch with linear history rewritten to that subdirectory's root.
git subtree split --prefix=ServerQwen/Qwen_Engine_GENOVA2I/genova_vllm_556_0610 -b genova2i-export

# 2. Push that branch to GENOVA2I's main. Git will prompt for a username
#    and password — enter "fredsanto" and your NEW PAT as the password.
git push genova2i genova2i-export:main

# 3. Clean up the local export branch (optional).
git branch -D genova2i-export
```

## If step 2 is rejected (non-fast-forward)

That means `fredsanto/GENOVA2I` already has commits that don't share history
with this subtree split (e.g. it was seeded independently). Two options:

- **Merge instead of overwrite** (safer, keeps remote history):
  ```bash
  git fetch genova2i main
  git checkout genova2i-export
  git merge genova2i/main --allow-unrelated-histories
  # resolve any conflicts, then:
  git push genova2i genova2i-export:main
  ```
- **Force-overwrite remote main** (only if you're sure nothing on the remote
  needs keeping — this discards whatever is there now):
  ```bash
  git push genova2i genova2i-export:main --force-with-lease
  ```

## Context

- `genova2i` remote → `https://github.com/fredsanto/GENOVA2I.git` (private,
  owned by fredsanto).
- Only `ServerQwen/Qwen_Engine_GENOVA2I/genova_vllm_556_0610` is being pushed
  — not the rest of the GenMasterAI monorepo (unrelated LoRA scripts, other
  server dirs, etc. stay out of it).
- The pipeline's proprietary `LICENSE` file (copyright Eric Ducret) has been
  removed from this copy — confirmed authorized before doing so.
