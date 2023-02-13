# Randomizer Keywords
This extension for stable-diffusion-webui adds some keywords using the extra networks syntax to allow randomization of parameters when combined with dynamic_prompts.

## Usage
The following keywords were added:

- `<cfg_scale:7>` - CFG scale
- `<checkpoint:animefull-latest.ckpt>` - SD Checkpoint

**NOTE**: These keywords will be applied *per-batch*, not per-prompt. This is because you can't change things like checkpoints or sampler parameters for a single batch of images.

## Example
This prompt will pick from a random CFG scale each batch:

## Extension
Other extensions can add their own keywords.
