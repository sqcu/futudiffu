"""Test pinkify_score() with synthetic and challenge set images.

Creates programmatic test images and scores them, plus scores the PINKIFY
challenge set (PINKER_A through PINKER_F). Verifies that the ranking function
produces the required orderings. Outputs a table and saves scores to disk.
"""

import sys
import os
import json
import time

# Ensure src_ii is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_ii'))

from PIL import Image, ImageDraw
import numpy as np

from reward_functions import (
    pinkify_score,
    _rgb_to_hsv_array,
    _is_pink_mask,
    _continuous_pinkness,
    _hue_pink_weight,
    _coverage_contrast_noscipy,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'pinkify_test_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 256


def make_pink_circle_on_white() -> Image.Image:
    """Pink circle (#FF69B4) centered on white background."""
    img = Image.new('RGB', (SIZE, SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy, r = SIZE // 2, SIZE // 2, SIZE // 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 105, 180))
    return img


def make_monochrome_red() -> Image.Image:
    """Solid red (#FF0000)."""
    return Image.new('RGB', (SIZE, SIZE), (255, 0, 0))


def make_monochrome_white() -> Image.Image:
    """Solid white (#FFFFFF)."""
    return Image.new('RGB', (SIZE, SIZE), (255, 255, 255))


def make_monochrome_pink() -> Image.Image:
    """Solid pink (#FF69B4)."""
    return Image.new('RGB', (SIZE, SIZE), (255, 105, 180))


def make_pink_blue_stripes() -> Image.Image:
    """Alternating 8-pixel-wide pink and blue vertical stripes."""
    img = Image.new('RGB', (SIZE, SIZE))
    arr = np.array(img)
    stripe_width = 8
    for x in range(SIZE):
        if (x // stripe_width) % 2 == 0:
            arr[:, x, :] = [255, 105, 180]  # pink
        else:
            arr[:, x, :] = [0, 0, 255]      # blue
    return Image.fromarray(arr)


def make_red_pink_white_gradient() -> Image.Image:
    """Horizontal gradient: red (left) -> pink (middle) -> white (right)."""
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for x in range(SIZE):
        t = x / (SIZE - 1)  # 0 to 1
        if t < 0.5:
            # Red to pink: interpolate from (255,0,0) to (255,105,180)
            s = t * 2  # 0 to 1
            r = 255
            g = int(105 * s)
            b = int(180 * s)
        else:
            # Pink to white: interpolate from (255,105,180) to (255,255,255)
            s = (t - 0.5) * 2  # 0 to 1
            r = 255
            g = int(105 + (255 - 105) * s)
            b = int(180 + (255 - 180) * s)
        arr[:, x, :] = [r, g, b]
    return Image.fromarray(arr)


def make_black() -> Image.Image:
    """Solid black (#000000)."""
    return Image.new('RGB', (SIZE, SIZE), (0, 0, 0))


def run_synthetic_tests():
    """Run 7 synthetic test cases and return results + pass/fail."""
    test_cases = [
        ("1. Pink circle on white", make_pink_circle_on_white()),
        ("2. Monochrome red", make_monochrome_red()),
        ("3. Monochrome white", make_monochrome_white()),
        ("4. Monochrome pink", make_monochrome_pink()),
        ("5. Pink-blue stripes", make_pink_blue_stripes()),
        ("6. Red-pink-white gradient", make_red_pink_white_gradient()),
        ("7. Black", make_black()),
    ]

    results = []
    print(f"\n{'='*70}")
    print(f"SYNTHETIC PINKIFY SCORING TEST")
    print(f"{'='*70}")
    print(f"\n{'Test Case':<35} {'Score':>10}")
    print(f"{'-'*35} {'-'*10}")

    for label, img in test_cases:
        score = pinkify_score(img)

        # Save the image
        safe_name = label.split('. ')[1].replace(' ', '_').replace('-', '_').lower()
        img.save(os.path.join(OUTPUT_DIR, f'{safe_name}.png'))

        result = {'label': label, 'score': score}
        results.append(result)

        print(f"{label:<35} {score:>10.6f}")

    print(f"\n{'='*70}")
    print(f"RANKING (highest to lowest score):")
    print(f"{'='*70}")
    sorted_results = sorted(results, key=lambda r: r['score'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['label']:<35} score={r['score']:.6f}")

    # Expected ranking analysis
    print(f"\n{'='*70}")
    print(f"SYNTHETIC RANKING CHECKS:")
    print(f"{'='*70}")

    scores_by_name = {r['label']: r['score'] for r in results}

    checks = [
        ("Pink-blue stripes > Monochrome pink (contrast matters)",
         scores_by_name["5. Pink-blue stripes"] > scores_by_name["4. Monochrome pink"]),
        ("Pink-blue stripes > Red-pink-white gradient",
         scores_by_name["5. Pink-blue stripes"] > scores_by_name["6. Red-pink-white gradient"]),
        ("White score ~= 0",
         scores_by_name["3. Monochrome white"] < 0.001),
        ("Black score ~= 0",
         scores_by_name["7. Black"] < 0.001),
        ("Pink circle > White",
         scores_by_name["1. Pink circle on white"] > scores_by_name["3. Monochrome white"]),
        ("Pink circle > Black",
         scores_by_name["1. Pink circle on white"] > scores_by_name["7. Black"]),
    ]

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")

    return results, all_pass


def run_challenge_set_tests():
    """Run the PINKIFY challenge set (PINKER_A through PINKER_F)."""
    cases_dir = os.path.join(os.path.dirname(__file__), '..', 'i2i_off_policies', 'PINKIFY_cases')

    if not os.path.isdir(cases_dir):
        print(f"\nChallenge set directory not found: {cases_dir}")
        print("Skipping challenge set tests.")
        return [], True

    print(f"\n{'='*70}")
    print(f"PINKIFY CHALLENGE SET TEST")
    print(f"{'='*70}")

    names = ['PINKER_A.png', 'PINKER_B.png', 'PINKER_C.png',
             'PINKER_D.png', 'PINKER_E.png', 'PINKER_F.png']

    scores = {}
    results = []
    print(f"\n{'Image':<20} {'Score':>12}")
    print(f"{'-'*20} {'-'*12}")

    for name in names:
        path = os.path.join(cases_dir, name)
        img = Image.open(path)
        score = pinkify_score(img)
        letter = name.split('_')[1].split('.')[0]
        scores[letter] = score
        results.append({'label': name, 'score': score})
        print(f"{name:<20} {score:>12.8f}")

    print(f"\n{'='*70}")
    print(f"CHALLENGE SET RANKING CHECKS:")
    print(f"{'='*70}")

    checks = [
        ("A < B (no pink < faint pink)",
         scores['A'] < scores['B']),
        ("B < C (faint pink < vivid pink)",
         scores['B'] < scores['C']),
        ("C < D (vivid lines < vivid accents + lavender lines)",
         scores['C'] < scores['D']),
        ("D ~= E (same accents, different neutral tint)",
         abs(scores['D'] - scores['E']) < 0.001),
        ("D < F (accents only < accents + pink-washed background)",
         scores['D'] < scores['F']),
        ("E < F (accents only < accents + pink-washed background)",
         scores['E'] < scores['F']),
        ("Clear tier separation: F >> D",
         scores['F'] > scores['D'] * 2.0),
    ]

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")

    return results, all_pass


def run_diagnostics():
    """Print diagnostic information about the scoring internals."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTICS: CONTINUOUS PINKNESS")
    print(f"{'='*70}")

    key_colors = {
        'Hot pink (#FF69B4)': (255, 105, 180),
        'Pure red (#FF0000)': (255, 0, 0),
        'Pure white (#FFFFFF)': (255, 255, 255),
        'Pure black (#000000)': (0, 0, 0),
        'Magenta (#FF00FF)': (255, 0, 255),
        'Light pink (#FFB6C1)': (255, 182, 193),
        'Deep pink (#FF1493)': (255, 20, 147),
        'Blue (#0000FF)': (0, 0, 255),
        'Pure green (#00FF00)': (0, 255, 0),
        'Lavender (#AA8CA8)': (170, 140, 168),
    }

    print(f"\n{'Color':<25} {'H':>7} {'S':>7} {'V':>7} {'Hue_W':>7} {'Pinkness':>10}")
    print(f"{'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")
    for name, (r, g, b) in key_colors.items():
        pixel = np.array([[[r, g, b]]], dtype=np.uint8)
        hsv = _rgb_to_hsv_array(pixel)
        h_val, s_val, v_val = hsv[0, 0, 0], hsv[0, 0, 1], hsv[0, 0, 2]
        hue_w = _hue_pink_weight(hsv[..., 0])[0, 0]
        pinkness = _continuous_pinkness(hsv)[0, 0]
        print(f"{name:<25} {h_val:>7.1f} {s_val:>7.3f} {v_val:>7.3f} {hue_w:>7.3f} {pinkness:>10.4f}")

    # Edge artifact check for monochrome pink
    print(f"\n{'='*70}")
    print(f"EDGE ARTIFACT CHECK (monochrome pink, reflect padding):")
    print(f"{'='*70}")

    mono_pink = make_monochrome_pink()
    rgb = np.array(mono_pink)
    hsv = _rgb_to_hsv_array(rgb)
    pinkness_map = _continuous_pinkness(hsv)
    presence = (pinkness_map > 0.01).astype(np.float32)
    contrast = _coverage_contrast_noscipy(presence)

    nonzero_contrast = contrast[contrast > 0]
    print(f"  Pink pixel count: {(pinkness_map > 0.01).sum()}")
    print(f"  Pixels with nonzero coverage contrast: {len(nonzero_contrast)}")
    print(f"  Coverage contrast sum: {contrast.sum():.6f}")
    print(f"  (Should be 0.0 with reflect padding for uniform image)")


def main():
    syn_results, syn_pass = run_synthetic_tests()
    ch_results, ch_pass = run_challenge_set_tests()
    run_diagnostics()

    all_pass = syn_pass and ch_pass

    # Save results to JSON
    output_data = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'synthetic_results': syn_results,
        'challenge_results': ch_results,
        'synthetic_pass': syn_pass,
        'challenge_pass': ch_pass,
        'all_pass': all_pass,
        'scoring_config': {
            'hue_core_range': '[300, 360] or [0, 30)',
            'hue_falloff_degrees': 20,
            'value_floor': 0.1,
            'compression_power': 0.5,
            'kernel_size': 7,
            'intensity_weight': 0.2,
            'presence_threshold': 0.01,
        }
    }

    json_path = os.path.join(OUTPUT_DIR, 'pinkify_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"OVERALL RESULT: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'='*70}")
    print(f"  Synthetic tests: {'PASS' if syn_pass else 'FAIL'}")
    print(f"  Challenge tests: {'PASS' if ch_pass else 'FAIL'}")
    print(f"\nResults saved to: {json_path}")

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
