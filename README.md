# VisionRefine: High-Resolution Image Recovery

The course project of EECS442

Jiahe Huang, Shixuan Liu, Xuejun Zhang, Jingjing Zhu, Shuangyu Lei

## Roadmap
- [ ] Super resolution with CNN.
- [x] Super resolution (x4) with pre-trained diffusion (or our own pre-trained model) model w/ classifier conditioning.
- [x] Super resolution with transformer.

## Diffusion Model

###  Classifier Guided DDPM Sampling

Our pre-trained model is derived from [OpenAI](https://github.com/openai/guided-diffusion) and uses the 64x64 -> 256x256 upsampler. You may directly download it [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt).

Run the sampling algorithm:

```python
python batch_upsample_cond.py --model_path models/64_256_upsampler.pt --base_samples ./images/all_low_res.npz \
  --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True \
  --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
  --batch_size 2 --num_samples 2 --timestep_respacing 250  --classifier_scale 4.0 --classifier_path models/256x256_classifier.pt --image_size 256
```

We assume you have all the 64x64 low resolution images under `./image` folder and the pre-trained model and classifier under `./models` folder.

### EDM Based DPS

You may find our own pre-trained model of DIV2K based on EDM [here](https://drive.google.com/file/d/1AGy7nSMq9UQgG0wZ6wg4DLWPS2o1vVtT/view?usp=sharing).

Run the sampling algorithm:

```python
python generate_sr.py
```

## Transformer

Please switch to the branch ``codeformer`` for Transformer. This model is based on CodeFormer

Run the program:
```python
python inference_codeformer.py -w 0.7 --input_path inputs/whole_imgs --output_path outputs --face_upsample
python bounding_box.py
python final_output.py
```

