### some useful paths u might want
#### wsl bridge to related materials
\mnt\f\dox\ai\comfyui\comfyui
#### model workflow
user\default\workflows\image_z_image_turbo.json
##### brand new: a block-quantized z-image 'workflow' which renders 'enormous laser sharks' at 25% speedup over base model even on sm89 hardware! (sm120 should be faster probably)
user\default\workflows\zimage_blockquant_lasershark.json
#### relative paths to checkpoints!
##### clever and resourceful claudes might find block quantized fp8 versions of these next to the originals... for 2 of 3 models at least...
models\diffusion_models\z_image_bf16.safetensors
models\vae\zimage.safetensors
models\text_encoders\qwen_3_4b.safetensors