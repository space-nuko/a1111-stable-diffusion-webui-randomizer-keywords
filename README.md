# Randomizer Keywords
This extension for stable-diffusion-webui adds some keywords using the extra networks syntax to allow randomization of parameters when combined with dynamic_prompts.

## Usage
The following keywords were added:

- `<cfg_scale:7>` - CFG scale
- `<checkpoint:animefull-latest.ckpt>` - SD Checkpoint

**NOTE**: These keywords will be applied *per-batch*, not per-prompt. This is because you can't change things like checkpoints or sampler parameters for a single batch of images.

## Example
When used with dynamic_prompts, this prompt will pick from a random checkpoint each batch:

```
{<checkpoint:animefull-latest>|<checkpoint:wd15-beta1-fp32>}
```
