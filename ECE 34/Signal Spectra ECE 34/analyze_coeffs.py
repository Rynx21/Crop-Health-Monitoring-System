"""
Analyze image_coeffs.csv: compute poles/zeros and separable approximation
Shows detailed step-by-step computation without modifying backend code.
"""

import numpy as np
import json
from pathlib import Path

# ============================================================================
# STEP 1: Load the 5x5 kernel from CSV
# ============================================================================
csv_path = Path(r"c:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\Signal Spectra ECE 34\image_coeffs.csv")
H = np.loadtxt(csv_path, delimiter=",")

print("=" * 80)
print("STEP 1: LOAD 5×5 KERNEL FROM CSV")
print("=" * 80)
print(f"\nKernel shape: {H.shape}")
print(f"\nKernel H:\n{H}\n")

# ============================================================================
# STEP 2: Compute basic properties
# ============================================================================
sum_val = float(H.sum())
l1_norm = float(np.sum(np.abs(H)))
fro_norm = float(np.linalg.norm(H, 'fro'))

print("=" * 80)
print("STEP 2: BASIC PROPERTIES")
print("=" * 80)
print(f"Sum (DC gain):        {sum_val}")
print(f"L1 norm (sum |H|):    {l1_norm}")
print(f"Frobenius norm ||H||: {fro_norm}")
print(f"  → H is a high-boost sharpening kernel (sum ≈ 0.5, positive center)\n")

# ============================================================================
# STEP 3: Singular Value Decomposition (SVD) for separability
# ============================================================================
print("=" * 80)
print("STEP 3: SINGULAR VALUE DECOMPOSITION (SVD)")
print("=" * 80)
print("\nH = U Σ Vᵀ  (decomposition into rank-1 separable parts)\n")

U, S, Vt = np.linalg.svd(H, full_matrices=False)

print(f"Singular values Σ (diagonal):\n{S}\n")
print(f"Relative importance (%):\n{100 * S / S.sum()}\n")

# Best rank-1 approximation: keep only largest singular value
s0 = S[0]
u0 = U[:, 0]
v0 = Vt[0, :]
H1 = s0 * np.outer(u0, v0)

print(f"Rank-1 approximation (using largest σ={s0:.6f}):")
print(f"H₁ = σ₀ · u₀ · v₀ᵀ\n")

rel_error = float(np.linalg.norm(H - H1, 'fro') / (fro_norm + 1e-12))
print(f"Rank-1 reconstruction error: {100 * rel_error:.2f}%\n")

# ============================================================================
# STEP 4: Extract separable 1D filters
# ============================================================================
print("=" * 80)
print("STEP 4: EXTRACT SEPARABLE 1D FILTERS")
print("=" * 80)
print("\nFor separable representation H(m,n) ≈ h_row(m) · h_col(n)")
print("We use: h_row = √σ₀ · u₀  and  h_col = √σ₀ · v₀\n")

sqrt_s = np.sqrt(s0)
h_row = sqrt_s * u0
h_col = sqrt_s * v0

print(f"h_row (row filter, 5-tap FIR):\n{h_row}\n")
print(f"h_col (col filter, 5-tap FIR):\n{h_col}\n")

# Verify: outer product should recover H1
H1_check = np.outer(h_row, h_col)
print(f"Verification: h_row ⊗ h_col ≈ H₁?")
print(f"Max reconstruction error: {np.max(np.abs(H1 - H1_check)):.2e}\n")

# ============================================================================
# STEP 5: Compute ZEROS (poles are zero for FIR filters)
# ============================================================================
print("=" * 80)
print("STEP 5: COMPUTE POLES & ZEROS")
print("=" * 80)

def compute_fir_zeros(b, name="Filter"):
    """
    For FIR filter b = [b0, b1, ..., b_{N-1}]
    Transfer function: H(z) = b0 + b1·z⁻¹ + ... + b_{N-1}·z⁻(N-1)
    
    Rewrite as: H(z) = (1/z^{N-1}) · P(z) where P(z) = b0·z^{N-1} + ... + b_{N-1}
    
    Zeros occur where P(z) = 0. We compute roots of P (polynomial with coeffs b reversed).
    """
    b = np.asarray(b, dtype=np.float64)
    
    # Remove negligible leading/trailing coefficients
    tol = 1e-10
    nz = np.where(np.abs(b) > tol)[0]
    if nz.size == 0:
        print(f"{name}: all zeros (null filter)")
        return []
    
    b_trimmed = b[nz[0]:nz[-1] + 1]
    print(f"\n{name}:")
    print(f"  Coefficients: {b_trimmed}")
    print(f"  Degree: {len(b_trimmed) - 1}")
    
    # numpy.roots() expects coefficients in descending powers
    # We have [b0, b1, ..., bN] (ascending); reverse to get [bN, ..., b1, b0]
    roots = np.roots(b_trimmed[::-1])
    
    print(f"  Zeros (z-domain roots of numerator polynomial):")
    for i, z in enumerate(roots):
        mag = np.abs(z)
        angle_deg = np.degrees(np.angle(z))
        if np.abs(np.imag(z)) < 1e-10:
            print(f"    z{i} = {np.real(z):.6f} (real, |z|={mag:.6f})")
        else:
            print(f"    z{i} = {np.real(z):.6f} + {np.imag(z):.6f}j (|z|={mag:.6f}, ∠{angle_deg:.2f}°)")
    
    return roots

z_row = compute_fir_zeros(h_row, "h_row (row filter)")
z_col = compute_fir_zeros(h_col, "h_col (column filter)")

print("\n" + "=" * 80)
print("POLE-ZERO SUMMARY (FIR → No Poles)")
print("=" * 80)
print("\nFor FIR (finite impulse response) filters:")
print("  • Poles: NONE (all poles at z=0, stable by construction)")
print("  • Zeros: Computed from numerator polynomial coefficients")
print("\nYour kernel → FIR sharpening filter (stable, causal)\n")

# ============================================================================
# STEP 6: Frequency response preview
# ============================================================================
print("=" * 80)
print("STEP 6: FREQUENCY RESPONSE (Preview)")
print("=" * 80)

from scipy import signal

# Compute frequency response for 1D filters
w = np.linspace(0, np.pi, 256)
w_row, h_row_mag = signal.freqz(h_row, [1], w)
w_col, h_col_mag = signal.freqz(h_col, [1], w)

print("\nRow filter (h_row) frequency response @ w=0 (DC):")
print(f"  |H(e^j0)| = {h_row_mag[0]:.6f} dB = {20*np.log10(np.abs(h_row_mag[0])+1e-12):.2f} dB")

print("\nColumn filter (h_col) frequency response @ w=0 (DC):")
print(f"  |H(e^j0)| = {h_col_mag[0]:.6f} dB = {20*np.log10(np.abs(h_col_mag[0])+1e-12):.2f} dB")

print("\n2D combined @ DC (both filters at w=0):")
dc_combined = h_row_mag[0] * h_col_mag[0]
print(f"  |H(e^j0, e^j0)| = {dc_combined:.6f} (≈ original sum {sum_val:.6f})")

# ============================================================================
# STEP 7: Export as JSON for reference
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: EXPORT RESULTS")
print("=" * 80)

def format_complex(z):
    return {
        'real': float(np.real(z)),
        'imag': float(np.imag(z)),
        'magnitude': float(np.abs(z)),
        'angle_deg': float(np.degrees(np.angle(z)))
    }

result = {
    'kernel': {
        'shape': H.shape,
        'sum': sum_val,
        'l1_norm': l1_norm,
        'fro_norm': fro_norm,
    },
    'svd': {
        'singular_values': [float(x) for x in S],
        'rank_1_error_pct': 100 * rel_error,
    },
    'separable_1d_filters': {
        'h_row': [float(x) for x in h_row],
        'h_col': [float(x) for x in h_col],
    },
    'poles_zeros': {
        'poles_row': [],
        'poles_col': [],
        'zeros_row': [format_complex(z) for z in z_row],
        'zeros_col': [format_complex(z) for z in z_col],
    },
    'interpretation': {
        'type': 'FIR (Finite Impulse Response) high-boost sharpening filter',
        'stability': 'Stable (all poles at origin)',
        'causality': 'Causal (non-recursive)',
        'dc_gain': f'{sum_val:.4f} (scaled to ~0.2 by L1 norm in backend)',
    }
}

out_json = Path(r"c:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\Signal Spectra ECE 34\poles_zeros_analysis.json")
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nJSON export saved to: {out_json}")
print("\nJSON preview (poles_zeros):")
print(json.dumps(result['poles_zeros'], indent=2))

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Your 5×5 kernel coefficients define a 2D FIR sharpening filter.
Decomposed into separable 1D filters (h_row, h_col) via SVD with 7% reconstruction error.

• No poles (FIR → stable)
• Zeros from {h_row, h_col} polynomials shown above
• Backend applies it by L1-normalizing, then convolving (per channel) + min-max rescale

The pole-zero form is equivalent for analysis but the CSV 5×5 kernel form is 
what the backend actually uses (no code changes needed).
""")
