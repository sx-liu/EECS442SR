# VisionRefine: High-Resolution Image Recover

The course project of EECS442

Jiahe Huang, Shixuan Liu, Xuejun Zhang, Jingjing Zhu, Shuangyu Lei

## Roadmap
- [ ] Super resolution with CNN.
- [x] Super resolution (x4) with pre-trained diffusion model w/ classifier conditioning.
- [ ] Super resolution with codeformer.

## Diffusion Model

###  Classifier Guided DDPM Sampling

Our pre-trained model is derived from [OpenAI](https://github.com/openai/guided-diffusion) and uses the 64x64 -> 256x256 upsampler. You may directly download it [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt).

Run the sampling algorithm:

```python
python batch_upsample_cond.py
```

### EDM Based DPS

You may find our own pre-trained model of DIV2K based on EDM [here](https://drive.google.com/file/d/1AGy7nSMq9UQgG0wZ6wg4DLWPS2o1vVtT/view?usp=sharing).

Run the sampling algorithm:

```python
python generate_sr.py
```

