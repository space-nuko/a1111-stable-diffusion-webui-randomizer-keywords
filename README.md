# Randomizer Keywords
This extension for stable-diffusion-webui adds some keywords using the extra networks syntax to allow randomization of parameters when combined with the [Dynamic Prompts](https://github.com/adieyal/sd-dynamic-prompts/tree/main/sd_dynamic_prompts) extension.

## Example
When used with Dynamic Prompts, this prompt will pick from a random checkpoint each batch:

```
{<checkpoint:animefull-latest>|<checkpoint:wd15-beta1-fp32>}
```

You can also assemble a wildcard list containing text like model names to choose from, and deploy them inside the keywords:

```
<addnet_model_1:__artist_loras__>
```

## List of Keywords
This extension adds the following special keywords to be used in prompts:

- `<checkpoint:animefull-latest.ckpt>` - SD Checkpoint
- `<vae:animefull-latest.vae.pt>` - SD VAE
- `<cfg_scale:7>` - CFG Scale
- `<seed:1>` - Seed
- `<subseed:1>` - Subseed
- `<subseed_strength:1>` - Subseed Strength
- `<sampler_name:Euler a>` - Sampler Name
- `<steps:20>` - Sampling Steps
- `<width:512>` - Width
- `<height:512>` - Height
- `<tiling:true>` - Tiling
- `<restore_faces:true>` - Restore Faces
- `<s_churn:true>` - Sigma Churn
- `<s_tmin:true>` - Sigma Min
- `<s_tmax:true>` - Sigma Max
- `<s_noise:true>` - Sigma Noise
- `<eta:512>` - Eta
- `<clip_skip:1>` - CLIP Skip
- `<ddim_discretize:quad>` - DDIM Discretize
- `<denoising_strength:0.7>` - Denoising Strength
- `<hr_upscaler:Latent>` - Hires. Fix Upscaler (txt2img)
- `<hr_second_pass_steps:10>` - Hires. Fix Steps (txt2img)
- `<mask_blur:2>` - Mask Blur (img2img)
- `<inpainting_mask_weight:2>` - Inpainting Mask Weight (img2img)
- `<addnet_model_1:rembrandt>` - [Additional Networks](https://github.com/kohya-ss/sd-webui-additional-networks) Model (up to 5)
- `<addnet_weight_1:1.0>` - Additional Networks Weight (up to 5)
- `<addnet_unet_weight_1:1.0>` - Additional Networks UNet Weight (up to 5)
- `<addnet_tenc_weight_1:1.0>` - Additional Networks TEnc Weight (up to 5)

**NOTE**: These keywords will be applied *per-batch*, not per-prompt. This is because you can't change things like checkpoints or sampler parameters for individual images in a batch.
