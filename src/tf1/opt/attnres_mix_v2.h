// AttnRes depth-mix, v2 (HUT-NATIVE-AR runtime, v10, Oct 16 2026): same formula as attnres_mix.h,
//
//   l_i = (w . v_i) * rstd_i,   p = softmax_i(l),   out = sum_i p_i v_i,     D = 192, n = 1..9
//
// rebuilt for speed. Changes vs v1 (all deterministic: fixed lane layout, fixed reduction trees, explicit FMA, so the
// encoder and decoder compute identical bits; the RESULT differs from v1 at the ulp level, so native parity is re-run):
//   * dots: two FMA accumulators per source over the 24 vectors, reduced with a fixed tree (no per-dot store)
//   * softmax: one exp256 for n <= 8 (a second only for n == 9), p = e / s as ONE vector division (IEEE, = scalar)
//   * n == 1 short cut: p = 1 exactly, out = v_0 (exp(0) = 1, 1/1 = 1, 1 * x = x: bitwise the v1 result too)
//   * mix: out = fma(p_i, v_i, out) over the sources, broadcasts hoisted
// plus ar_take_f_v2: partial += f fused with the partial's sum of squares, so its rstd is cached, not re-read per read.
#pragma once
#include <immintrin.h>

#include <cmath>

#include "vec_math.h"

namespace fx2 {
namespace opt {

// fixed-order horizontal sum of 8 lanes: ((0+4)+(2+6)) + ((1+5)+(3+7))
inline float ar_hsum8_v2(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  const __m128 a = _mm_add_ps(lo, hi);                    // (0+4) (1+5) (2+6) (3+7)
  const __m128 b = _mm_add_ps(a, _mm_movehl_ps(a, a));    // (0+4)+(2+6), (1+5)+(3+7)
  return _mm_cvtss_f32(_mm_add_ss(b, _mm_movehdup_ps(b)));
}

inline float ar_dot192_v2(const float* a, const float* b) {
  __m256 s0 = _mm256_mul_ps(_mm256_load_ps(a), _mm256_load_ps(b));
  __m256 s1 = _mm256_mul_ps(_mm256_load_ps(a + 8), _mm256_load_ps(b + 8));
  for (int i = 16; i < 192; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8), s1);
  }
  return ar_hsum8_v2(_mm256_add_ps(s0, s1));
}

// 1 / rms(v) with the v2 dot (rms(v) = sqrt(mean v^2 + 1e-6), arch.py _inv_rms)
inline float ar_rstd192_v2(const float* v) {
  return 1.0f / std::sqrt(ar_dot192_v2(v, v) / 192.0f + 1e-6f);
}

inline void ar_depth_mix192_v2(const float* const* src, const float* rstd, int n, const float* w, float* out) {
  if (n == 1) {                                   // softmax of one logit is exactly 1
    for (int d = 0; d < 192; d += 8) _mm256_store_ps(out + d, _mm256_load_ps(src[0] + d));
    return;
  }
  alignas(32) float lg[16];
  for (int i = 0; i < n; i++) lg[i] = ar_dot192_v2(w, src[i]) * rstd[i];
  float m = lg[0];
  for (int i = 1; i < n; i++) m = lg[i] > m ? lg[i] : m;
  alignas(32) float e[16];
  const __m256 vm = _mm256_set1_ps(m);
  // pads: lg = m - 100 -> e ~ 4e-44 (never summed, never used)
  for (int i = n; i < 16; i++) lg[i] = m - 100.0f;
  _mm256_store_ps(e, exp256_ps(_mm256_sub_ps(_mm256_load_ps(lg), vm)));
  if (n > 8) _mm256_store_ps(e + 8, exp256_ps(_mm256_sub_ps(_mm256_load_ps(lg + 8), vm)));
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += e[i];       // fixed left-to-right order
  alignas(32) float p[16];
  const __m256 vs = _mm256_set1_ps(s);
  _mm256_store_ps(p, _mm256_div_ps(_mm256_load_ps(e), vs));
  if (n > 8) _mm256_store_ps(p + 8, _mm256_div_ps(_mm256_load_ps(e + 8), vs));
  __m256 pb[9];
  for (int i = 0; i < n; i++) pb[i] = _mm256_set1_ps(p[i]);
  for (int d = 0; d < 192; d += 8) {
    __m256 h = _mm256_mul_ps(pb[0], _mm256_load_ps(src[0] + d));
    for (int i = 1; i < n; i++) h = _mm256_fmadd_ps(pb[i], _mm256_load_ps(src[i] + d), h);
    _mm256_store_ps(out + d, h);
  }
}

// part += f, returning the new partial's 1/rms in the same pass (two FMA accumulators, fixed tree)
inline float ar_take_f_v2(float* part, const float* f) {
  __m256 q0 = _mm256_setzero_ps(), q1 = q0;
  for (int d = 0; d < 192; d += 16) {
    const __m256 a = _mm256_add_ps(_mm256_load_ps(part + d), _mm256_load_ps(f + d));
    const __m256 b = _mm256_add_ps(_mm256_load_ps(part + d + 8), _mm256_load_ps(f + d + 8));
    _mm256_store_ps(part + d, a);
    _mm256_store_ps(part + d + 8, b);
    q0 = _mm256_fmadd_ps(a, a, q0);
    q1 = _mm256_fmadd_ps(b, b, q1);
  }
  return 1.0f / std::sqrt(ar_hsum8_v2(_mm256_add_ps(q0, q1)) / 192.0f + 1e-6f);
}

}  // namespace opt
}  // namespace fx2

namespace fx2 {
namespace opt {

// part += f with the partial's 1/rms AND dot(w, new part) in the same pass (TF_ARV2=5: w = the next read's query).
// The dot uses exactly ar_dot192_v2's accumulator layout and FMA order (mul for the first 16 lanes, then fma, two
// accumulators, ar_hsum8_v2), so *dot is bitwise ar_dot192_v2(w, part) on the updated partial.
inline float ar_take_f_dot_v2(float* part, const float* f, const float* w, float* dot) {
  __m256 q0 = _mm256_setzero_ps(), q1 = q0, s0, s1;
  {
    const __m256 a = _mm256_add_ps(_mm256_load_ps(part), _mm256_load_ps(f));
    const __m256 b = _mm256_add_ps(_mm256_load_ps(part + 8), _mm256_load_ps(f + 8));
    _mm256_store_ps(part, a);
    _mm256_store_ps(part + 8, b);
    q0 = _mm256_fmadd_ps(a, a, q0);
    q1 = _mm256_fmadd_ps(b, b, q1);
    s0 = _mm256_mul_ps(_mm256_load_ps(w), a);
    s1 = _mm256_mul_ps(_mm256_load_ps(w + 8), b);
  }
  for (int d = 16; d < 192; d += 16) {
    const __m256 a = _mm256_add_ps(_mm256_load_ps(part + d), _mm256_load_ps(f + d));
    const __m256 b = _mm256_add_ps(_mm256_load_ps(part + d + 8), _mm256_load_ps(f + d + 8));
    _mm256_store_ps(part + d, a);
    _mm256_store_ps(part + d + 8, b);
    q0 = _mm256_fmadd_ps(a, a, q0);
    q1 = _mm256_fmadd_ps(b, b, q1);
    s0 = _mm256_fmadd_ps(_mm256_load_ps(w + d), a, s0);
    s1 = _mm256_fmadd_ps(_mm256_load_ps(w + d + 8), b, s1);
  }
  *dot = ar_hsum8_v2(_mm256_add_ps(s0, s1));
  return 1.0f / std::sqrt(ar_hsum8_v2(_mm256_add_ps(q0, q1)) / 192.0f + 1e-6f);
}

}  // namespace opt
}  // namespace fx2
