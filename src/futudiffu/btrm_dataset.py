"""BTRM training dataset generation for diffusion QAT.

Generates paired trajectory data across prompts, seeds, step counts, and
attention backends. Each trajectory records per-step latents (decodeable by VAE)
for BTRM head training on step-count and attention quantization discrimination.

All trajectories use the FP8 blockwise diffusion model (~5.8GB VRAM). The
"precision" axis compares attention backends:
  - "sdpa": PyTorch SDPA (BF16, full precision) — gold standard
  - "sage": SageAttention INT8 QK + BF16 PV — introduces quantization artifacts

Two trajectory families:
  - t2i: text-to-image from gaussian noise (24 prompts x seeds x steps x attn_backend)
  - i2i: image-to-image from off-policy reference images forward-noised to
    configurable levels, denoised with object+transformative text queries.
    These cover latent trajectories unreachable from gaussian init.

Dataset shape:
    - 24 prompt templates (covering diverse content)
    - 11 off-policy reference images with object labels + style transforms
    - 100 PRNG seeds
    - 10 step count schedules (4, 6, 8, 10, 12, 15, 18, 20, 25, 30)
    - 3 i2i denoise strengths (0.75, 0.85, 0.95)
    - Each trajectory saves latent at every step + final

Pairing strategy (constructed post-hoc from the trajectory pool):
    scrongle pairs: same (seed, prompt), different step count
    scrimble pairs: same (seed, prompt, steps), different attention backend
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Prompt template library (24 templates)
# ---------------------------------------------------------------------------

# Mad-libs slots: {animal}, {object}, {environment}, {style}, {text}, {color}
# Each template exercises different model capabilities:
#   - scene composition, text rendering, fine detail, abstract concepts,
#     character design, lighting, texture, spatial relationships

PROMPT_TEMPLATES: list[str] = [
    # --- Core project prompts (laser shark family) ---

    # 0: Golden reference
    'ahem.\n*ting ting ting ting ting*\nthe query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME when they are participating in dialogue. however, in this situation, they are being used as a hidden state generator to steer an *image generation model*, z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".',

    # 1-5: Laser shark variations
    'qwen-3-4b, draw me a "gigantic laser shark breaching out of the ocean at sunset".',
    'qwen-3-4b, draw me a "laser shark swimming through a neon cyberpunk cityscape at night".',
    'qwen-3-4b, draw me "three laser sharks circling a coral reef, underwater photography".',
    'qwen-3-4b, draw me a "tiny laser shark in a fishbowl on a desk, office background".',
    'qwen-3-4b, draw me a "laser shark made of chrome and glass, studio lighting, product photography".',

    # --- Text rendering (known model weakness) ---

    # 6-9: Text in various contexts
    'A neon sign reading "OPEN 24 HOURS" above a rain-soaked Tokyo alleyway at night.',
    'A handwritten letter on aged parchment that reads "Dear Future Self" in elegant cursive.',
    'A chalkboard in a classroom with the equation "E = mc²" written in white chalk.',
    'A storefront window with painted gold lettering reading "ANTIQUES & CURIOSITIES" in a foggy English village.',

    # --- Scene composition and spatial relationships ---

    # 10-13: Complex scenes
    'A cat sitting on top of a stack of books next to a window with rain outside, warm interior lighting.',
    'An astronaut riding a horse across a desert under a starfield, photorealistic.',
    'A robot watering potted plants on a balcony overlooking a futuristic city skyline at dawn.',
    'A child building a sandcastle on a beach while enormous waves curl in the background.',

    # --- Fine detail and texture ---

    # 14-17: Texture-heavy subjects
    'Macro photography of a mechanical pocket watch with visible gears, golden light.',
    'A weathered wooden door with peeling blue paint and a rusty iron knocker, Mediterranean village.',
    'Close-up of a dragonfly perched on a dewdrop-covered blade of grass, morning light.',
    'A slice of layered cake showing distinct chocolate, cream, and raspberry layers, food photography.',

    # --- Abstract and artistic styles ---

    # 18-21: Style diversity
    'An oil painting of a mountain lake at twilight in the style of the Hudson River School.',
    'A vaporwave aesthetic rendering of ancient Greek ruins with pink and cyan gradients.',
    'Bauhaus-inspired geometric composition of primary colored shapes on a white background.',
    'A double exposure photograph combining a wolf portrait with a pine forest landscape.',

    # --- Edge cases and challenging content ---

    # 22-23: Unusual compositions
    'A mirror reflecting a room that does not match the room it is placed in, surrealist.',
    'An isometric pixel art scene of a busy medieval marketplace with at least 20 distinct characters.',
]

assert len(PROMPT_TEMPLATES) == 24, f"Expected 24 templates, got {len(PROMPT_TEMPLATES)}"


# ---------------------------------------------------------------------------
# Step count schedules
# ---------------------------------------------------------------------------

# Focused on the 8-22 range where scrongle artifacts are visible but
# not total garbage. Very low counts (4-6) produce obvious noise; very
# high (25-30) are near-indistinguishable from gold.
STEP_SCHEDULES: list[int] = [8, 10, 12, 14, 16, 18, 20, 22]

assert len(STEP_SCHEDULES) == 8, f"Expected 8 schedules, got {len(STEP_SCHEDULES)}"


# ---------------------------------------------------------------------------
# Image-to-image off-policy reference images
# ---------------------------------------------------------------------------

# Each entry: filename, object_label (describes structure to preserve),
# and per-image transform hints (optional overrides to the shared pool).
#
# The i2i query is: object_label + ", " + transformative_label
# The object_label anchors the denoiser to the spatial layout and subject
# matter of the source image. The transformative_label applies a style or
# medium change that doesn't ablate/replace the composition.

I2I_IMAGES: list[dict] = [
    {
        "file": "00500-3023556536_re_nightmode2.png",
        "object_label": (
            "anime character with long rabbit ears holding a polearm weapon "
            "against a dark background with magenta and teal lighting"
        ),
    },
    {
        "file": "1bit redraw.png",
        "object_label": (
            "cartoon face with large round eyes and small features "
            "in stark black and white 1-bit dithered pixel art"
        ),
    },
    {
        "file": "bubblegum-zinesona-4.png",
        "object_label": (
            "standing figure with voluminous curly hair and crossed arms "
            "in minimal zine-style line drawing"
        ),
    },
    {
        "file": "clear-sky-thick-mkii.png",
        "object_label": (
            "rough pen sketch of a face with heavy brushwork "
            "and handwritten text reading CLEAR SKY"
        ),
    },
    {
        "file": "deviantart-is-my-spine-moe-is-my-face.png",
        "object_label": (
            "two side-by-side figures, left rendered in chaotic abstract "
            "scribbles and right as a clean anime-style character"
        ),
    },
    {
        "file": "mspaint-enso-i-couldnt-forget-ii.png",
        "object_label": (
            "single minimalist brush circle enso on white background "
            "with visible brush texture"
        ),
    },
    {
        "file": "offhand_pleometric.png",
        "object_label": (
            "purple cartoon bird creature wearing a red santa hat "
            "with yellow beak and feet and a white belly, bold outlines"
        ),
    },
    {
        "file": "pizza-ratto.png",
        "object_label": (
            "pen sketch of a buck-toothed rat creature behind a brick "
            "counter with game UI elements labeled SMOOSH and SAVE"
        ),
    },
    {
        "file": "red-tonegraph.png",
        "object_label": (
            "abstract flowchart diagram of red and pink rectangular "
            "nodes connected by directional arrows on a light background"
        ),
    },
    {
        "file": "snek-heavy.png",
        "object_label": (
            "clean lineart of a witch character wearing a large pointed "
            "hat sitting atop a coiled serpent body in anime style"
        ),
    },
    {
        "file": "widemeister.png",
        "object_label": (
            "crude wireframe sketch of an extremely wide-bodied humanoid "
            "figure with thin angular limbs and blocky feet"
        ),
    },
]

assert len(I2I_IMAGES) == 11, f"Expected 11 i2i images, got {len(I2I_IMAGES)}"


# Transformative labels: style/medium transforms that preserve spatial structure.
# Sampled and appended to object_label to form the i2i text query.

TRANSFORMATIVE_LABELS: list[str] = [
    "rendered as a detailed oil painting with rich impasto texture",
    "reimagined in watercolor with soft bleeding edges",
    "in a neon-lit cyberpunk aesthetic with glowing accents",
    "as a vintage sepia-toned photograph",
    "in the style of Japanese woodblock print ukiyo-e",
    "rendered in cel-shaded animation with clean flat colors",
    "with dramatic chiaroscuro lighting in the style of Caravaggio",
    "in soft pastel vaporwave colors with a dreamy atmosphere",
    "as a stained glass window with bold lead lines",
    "rendered in photorealistic 3D with studio lighting",
    "in the style of a classic art nouveau poster",
    "as a bold pop art piece with halftone dots and primary colors",
    "rendered in delicate pencil crosshatching",
    "in the palette and style of a faded retro VHS screenshot",
    "as a mosaic made of small colored tiles",
    "reimagined in the dense linework of a Moebius comic panel",
    "rendered in luminous gouache with opaque layered washes",
    "as a risograph print with misregistered cyan and magenta layers",
    "in the style of a Soviet-era propaganda poster with bold typography",
    "reimagined as an illuminated manuscript page with gold leaf borders",
    "rendered in scratchboard with fine white lines on black",
    "as a thermal infrared photograph with false-color heat mapping",
    "in the style of a Meiji-era Japanese lithograph with mineral pigments",
    "reimagined as a cyanotype blueprint with Prussian blue tones",
    "rendered in thick palette knife oil impasto under raking gallery light",
    "as a linocut print with rough carved texture and uneven ink coverage",
]


# Denoise strengths for i2i: how much of the original structure survives.
# Lower = more structure preserved, higher = more creative freedom.
I2I_DENOISE_STRENGTHS: list[float] = [0.75, 0.85, 0.95]


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class BTRMDatasetConfig:
    """Configuration for BTRM training dataset generation."""

    # Sampling budget
    n_seeds: int = 100
    prompts_per_seed: int = 4     # sample from 24
    steps_per_seed: int = 3       # sample from 10

    # Attention backends for variations (gold is always sdpa-30)
    # Including "sdpa" gives clean scrongle pairs (sdpa-30 vs sdpa-N).
    # Including "sage" gives clean scrimble pairs (sdpa-30 vs sage-30)
    # and mixed pairs (sdpa-30 vs sage-N).
    precision_levels: list[str] = field(
        default_factory=lambda: ["sdpa", "sage"])

    # Generation params (match reference workflow)
    cfg: float = 4.0
    width: int = 1280
    height: int = 832
    sampling_shift: float = 1.0
    multiplier: float = 1.0
    negative_prompt: str = ""

    # Which steps to save latents at (None = all steps)
    # Sparse saves reduce storage: 7 checkpoints per trajectory
    save_steps: Optional[list[int]] = field(
        default_factory=lambda: [0, 4, 9, 14, 19, 24, 29])

    # Seed offset (add to trajectory index for PRNG)
    seed_base: int = 1000

    # i2i off-policy settings
    i2i_seeds_per_image: int = 3
    i2i_transforms_per_combo: int = 2  # transforms sampled per (image, noise_level)
    i2i_denoise_strengths: list[float] = field(
        default_factory=lambda: [0.75, 0.85, 0.95])
    i2i_n_steps: int = 30  # step count for denoising (independent of denoise strength)
    i2i_dir: str = ""  # path to i2i_off_policies/ directory

    # Latent rendering: how many trajectories to VAE-decode for visual QA
    render_count: int = 10   # render this many trajectories
    render_steps: list[int] = field(
        default_factory=lambda: [0, 14, 29])  # early/mid/late noise levels

    # Progress reporting: print milestone at every N fraction of total (0.01 = 1%)
    report_interval: float = 0.01

    @property
    def n_trajectories(self) -> int:
        n_per_seed = self.prompts_per_seed * self.steps_per_seed
        n_precision = len(self.precision_levels)
        # +1 for the sdpa 30-step gold standard per (seed, prompt)
        t2i = self.n_seeds * (n_per_seed * n_precision
                              + self.prompts_per_seed)  # gold standards
        return t2i + self.n_i2i_trajectories

    @property
    def n_i2i_trajectories(self) -> int:
        n_images = len(I2I_IMAGES)
        n_strengths = len(self.i2i_denoise_strengths)
        n_transforms = self.i2i_transforms_per_combo
        n_seeds = self.i2i_seeds_per_image
        # Per (image, strength): sample n_transforms labels, each with n_seeds
        # x2 for sdpa gold + sage variant
        return n_images * n_strengths * n_transforms * n_seeds * 2

    @property
    def approx_trajectories(self) -> int:
        """Rough count including gold standards and i2i."""
        t2i = self.n_seeds * self.prompts_per_seed * (
            self.steps_per_seed * len(self.precision_levels) + 1)  # +1 for sdpa gold
        return t2i + self.n_i2i_trajectories


# ---------------------------------------------------------------------------
# Trajectory record
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryRecord:
    """One complete diffusion trajectory with metadata."""
    seed: int
    prompt_idx: int          # t2i: index into PROMPT_TEMPLATES; i2i: -1
    prompt: str
    n_steps: int
    precision: str           # "sdpa" or "sage" (attention backend)
    is_gold: bool            # True = sdpa 30-step reference
    latents: dict            # step_idx (int) -> latent tensor path (str)
    final_latent_path: str = ""
    traj_type: str = "t2i"   # "t2i" or "i2i"
    i2i_image_idx: int = -1  # index into I2I_IMAGES (-1 for t2i)
    i2i_denoise: float = 0.0 # denoise strength (0 for t2i)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sampling plan: which (seed, prompt, steps, precision) combos to generate
# ---------------------------------------------------------------------------

def build_generation_plan(
    config: BTRMDatasetConfig,
    rng_seed: int = 42,
) -> list[dict]:
    """Build the list of trajectories to generate.

    For each of n_seeds PRNG seeds:
        1. Sample prompts_per_seed prompts (without replacement) from 24
        2. Sample steps_per_seed step counts (without replacement) from 10
        3. For each prompt:
           a. sdpa 30-step gold standard (full-precision attention)
           b. sage 30-step scrimble reference (INT8 QK attention)
           c. Each sampled step count × each precision level (mixed variations)

    All trajectories use FP8 diffusion model. The precision field controls
    the attention backend ("sdpa" or "sage").

    Returns list of dicts with keys: seed, prompt_idx, prompt, n_steps, precision, is_gold
    """
    rng = random.Random(rng_seed)
    plan = []

    for seed_offset in range(config.n_seeds):
        seed = config.seed_base + seed_offset

        # Sample prompts for this seed
        prompt_indices = rng.sample(range(len(PROMPT_TEMPLATES)),
                                    config.prompts_per_seed)

        # Sample step counts for this seed
        step_counts = rng.sample(STEP_SCHEDULES, config.steps_per_seed)

        for pidx in prompt_indices:
            prompt = PROMPT_TEMPLATES[pidx]

            # Gold standard: sdpa, 30 steps (full-precision attention)
            plan.append({
                "seed": seed,
                "prompt_idx": pidx,
                "prompt": prompt,
                "n_steps": 30,
                "precision": "sdpa",
                "is_gold": True,
                "type": "t2i",
            })

            # Scrimble reference: sage, 30 steps (INT8 QK attention)
            plan.append({
                "seed": seed,
                "prompt_idx": pidx,
                "prompt": prompt,
                "n_steps": 30,
                "precision": "sage",
                "is_gold": False,
                "type": "t2i",
            })

            # Variations: each sampled step count x each precision level
            for n_steps in step_counts:
                for precision in config.precision_levels:
                    # Skip duplicates (sdpa-30 is gold, sage-30 is scrimble ref)
                    if n_steps == 30:
                        continue
                    plan.append({
                        "seed": seed,
                        "prompt_idx": pidx,
                        "prompt": prompt,
                        "n_steps": n_steps,
                        "precision": precision,
                        "is_gold": False,
                        "type": "t2i",
                    })

    # --- i2i off-policy trajectories ---
    if config.i2i_dir:
        i2i_plan = build_i2i_plan(config, rng)
        plan.extend(i2i_plan)

    # Interleave t2i and i2i entries so both types get generated even with
    # --max-trajectories caps. Deterministic shuffle for reproducibility.
    rng_shuffle = random.Random(rng_seed + 1)
    rng_shuffle.shuffle(plan)

    return plan


def build_i2i_plan(
    config: BTRMDatasetConfig,
    rng: random.Random,
) -> list[dict]:
    """Build i2i trajectory entries for the generation plan.

    For each of 11 reference images:
        For each denoise strength (0.3, 0.5, 0.7):
            Sample i2i_transforms_per_combo transformative labels
            For each transform:
                For each of i2i_seeds_per_image seeds:
                    - sdpa gold reference (full-precision attention)
                    - sage variant (INT8 QK attention, scrimble comparison)
    """
    plan = []
    n_strengths = len(config.i2i_denoise_strengths)

    for img_idx, img_info in enumerate(I2I_IMAGES):
        object_label = img_info["object_label"]

        for denoise in config.i2i_denoise_strengths:
            # Sample transforms for this (image, denoise) combo
            n_t = min(config.i2i_transforms_per_combo, len(TRANSFORMATIVE_LABELS))
            transforms = rng.sample(TRANSFORMATIVE_LABELS, n_t)

            for transform in transforms:
                prompt = f"{object_label}, {transform}"

                for seed_offset in range(config.i2i_seeds_per_image):
                    # Use a distinct seed range for i2i (offset by 10000)
                    seed = config.seed_base + 10000 + (
                        img_idx * 1000 + seed_offset)

                    # Step count is independent of denoise — denoise only
                    # controls the starting sigma, not iteration count.
                    n_steps = config.i2i_n_steps

                    # sdpa gold (full-precision attention)
                    plan.append({
                        "seed": seed,
                        "prompt_idx": -1,
                        "prompt": prompt,
                        "n_steps": n_steps,
                        "precision": "sdpa",
                        "is_gold": True,
                        "type": "i2i",
                        "image_idx": img_idx,
                        "image_file": img_info["file"],
                        "denoise": denoise,
                    })

                    # sage variant (INT8 QK attention)
                    plan.append({
                        "seed": seed,
                        "prompt_idx": -1,
                        "prompt": prompt,
                        "n_steps": n_steps,
                        "precision": "sage",
                        "is_gold": False,
                        "type": "i2i",
                        "image_idx": img_idx,
                        "image_file": img_info["file"],
                        "denoise": denoise,
                    })

    return plan


def plan_summary(plan: list[dict]) -> str:
    """Print human-readable summary of a generation plan."""
    from collections import Counter

    t2i_plan = [p for p in plan if p.get("type", "t2i") == "t2i"]
    i2i_plan = [p for p in plan if p.get("type") == "i2i"]

    lines = []
    lines.append(f"Total trajectories: {len(plan)}")
    lines.append(f"  t2i: {len(t2i_plan)}")
    lines.append(f"  i2i: {len(i2i_plan)}")

    gold = sum(1 for p in plan if p["is_gold"])
    lines.append(f"  Gold standards: {gold}")
    lines.append(f"  Variations: {len(plan) - gold}")

    step_counts = Counter(p["n_steps"] for p in plan)
    lines.append(f"  Step count distribution: {dict(sorted(step_counts.items()))}")

    precision_counts = Counter(p["precision"] for p in plan)
    lines.append(f"  Precision distribution: {dict(precision_counts)}")

    if t2i_plan:
        prompt_counts = Counter(p["prompt_idx"] for p in t2i_plan)
        lines.append(f"  Unique t2i prompts used: {len(prompt_counts)}")

    if i2i_plan:
        image_counts = Counter(p["image_idx"] for p in i2i_plan)
        lines.append(f"  i2i images used: {len(image_counts)}")
        denoise_counts = Counter(p["denoise"] for p in i2i_plan)
        lines.append(f"  i2i denoise strengths: {dict(sorted(denoise_counts.items()))}")

    seed_counts = Counter(p["seed"] for p in plan)
    lines.append(f"  Seeds: {len(seed_counts)} "
                 f"(range {min(seed_counts)}..{max(seed_counts)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pair construction (post-hoc from trajectory pool)
# ---------------------------------------------------------------------------

def build_training_pairs(
    records: list[TrajectoryRecord],
    gold_step_count: int = 30,
) -> dict[str, list[tuple[int, int]]]:
    """Construct scrongle and scrimble training pairs from trajectory records.

    Returns dict with keys:
        "scrongle": list of (gold_idx, variant_idx) pairs — same seed+prompt,
                    different step count (gold=30 steps vs variant=fewer steps)
        "scrimble": list of (gold_idx, variant_idx) pairs — same seed+prompt+steps,
                    different attention backend (gold=sdpa vs variant=sage)

    Indices refer to positions in the records list.
    """
    # Index records by (seed, prompt_idx)
    by_key: dict[tuple[int, int], list[int]] = {}
    for i, rec in enumerate(records):
        key = (rec.seed, rec.prompt_idx)
        by_key.setdefault(key, []).append(i)

    scrongle_pairs = []
    scrimble_pairs = []

    for key, indices in by_key.items():
        # Find gold (sdpa 30-step)
        gold_indices = [i for i in indices
                        if records[i].is_gold]
        if not gold_indices:
            continue
        gold_idx = gold_indices[0]

        for i in indices:
            if i == gold_idx:
                continue
            rec = records[i]

            # Scrongle: same attention as gold (sdpa), fewer steps
            if rec.precision == "sdpa" and rec.n_steps < gold_step_count:
                scrongle_pairs.append((gold_idx, i))

            # Scrimble: same step count as gold (30), different attention
            elif rec.n_steps == gold_step_count and rec.precision != "sdpa":
                scrimble_pairs.append((gold_idx, i))

            # Mixed: both fewer steps AND different attention — contributes
            # to both heads (the scrongle head sees step-count diff, the
            # scrimble head sees attention quant diff)
            elif rec.precision != "sdpa" and rec.n_steps < gold_step_count:
                scrongle_pairs.append((gold_idx, i))
                scrimble_pairs.append((gold_idx, i))

    return {
        "scrongle": scrongle_pairs,
        "scrimble": scrimble_pairs,
    }


# ---------------------------------------------------------------------------
# Dataset manifest (JSON serializable)
# ---------------------------------------------------------------------------

def save_manifest(
    plan: list[dict],
    records: list[TrajectoryRecord],
    config: BTRMDatasetConfig,
    output_dir: Path,
) -> Path:
    """Save dataset manifest as JSON for reproducibility."""
    manifest = {
        "config": asdict(config),
        "plan": plan,
        "records": [
            {
                "seed": r.seed,
                "prompt_idx": r.prompt_idx,
                "n_steps": r.n_steps,
                "precision": r.precision,
                "is_gold": r.is_gold,
                "latents": r.latents,
                "final_latent_path": r.final_latent_path,
                "traj_type": r.traj_type,
                "i2i_image_idx": r.i2i_image_idx,
                "i2i_denoise": r.i2i_denoise,
            }
            for r in records
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
