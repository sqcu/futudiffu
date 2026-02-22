# src_ii: Extracted minimal diffusion sampling + BTRM training library
#
# Inference pipeline (five function boundaries from essay_algorithmic_decomposition.md):
#   nfe, denoise, make_guided_denoiser, euler_solve, rollout
#
# Model loading:
#   load_fp8_diffusion_model
#
# Reward functions:
#   pinkify_score, thisnotthat_score_gpu, pairwise_preference
#
# Model:
#   ZImageRLAIF -- unified diffusion model with integrated score head
#   btrm_lifecycle -- training setup, optimizer, persist/load utilities
#   multi_lora -- multi-tenant LoRA with per-image sparse routing
#   train_btrm -- training loop as a named function
#
# Utilities:
#   vae_utils -- load_vae, decode_latent_to_pil
#   stats -- spearman_rank_correlation, sigma_for_step
#   visualization -- render_heatmap_overlay, render_strip, etc.
