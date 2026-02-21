# src_ii: Extracted minimal diffusion sampling + BTRM training library
#
# Inference pipeline (five function boundaries from essay_algorithmic_decomposition.md):
#   nfe, denoise, make_guided_denoiser, euler_solve, rollout
#
# Model loading:
#   load_fp8_diffusion_model, configure_sage_attention
#
# Reward functions:
#   pinkify_score, thisnotthat_score_gpu, pairwise_preference
#
# BTRM compound model (prevents defect 24):
#   BTRMCompoundModel -- enforces backbone + adapter + head coupling
#   train_btrm -- training loop as a named function
#
# Utilities:
#   vae_utils -- load_vae, decode_latent_to_pil
#   attention_capture -- AttentionCapture for mechanistic interpretability
#   stats -- spearman_rank_correlation, sigma_for_step
#   visualization -- render_heatmap_overlay, render_strip, etc.
