// AttnRes depth-mix for the native engine (v7 arch.py replace mode; HUT-NATIVE-AR, v10).
//
//   l_i = (w . v_i) / rms(v_i),  rms(v) = sqrt(mean_d v_d^2 + 1e-6)     (arch.py _inv_rms, EPS 1e-6)
//   p   = softmax_i(l)                                                 (engine exp: exp256_ps)
//   out = sum_i p_i v_i                                                (sources in the given order)
//
// D = 192.  Every reduction has a fixed order (4 x 8-lane partial sums, then a fixed lane tree), so with the
// engine's flags (-ffp-model=precise -ffp-contract=off -mrecip=none) encoder and decoder compute identical bits.
// Shared by model_opt.cpp and the unit test (test_attnres_mix.cpp).
#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstring>

#include "vec_math.h"

namespace fx2 {
namespace opt {

// a.b over 192 floats, fixed order (a, b 32-byte aligned)
inline float ar_dot192(const float* a, const float* b) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  for (int i = 0; i < 192; i += 32) {
    s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
    s2 = _mm256_add_ps(s2, _mm256_mul_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
    s3 = _mm256_add_ps(s3, _mm256_mul_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24)));
  }
  const __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  alignas(32) float l[8];
  _mm256_store_ps(l, s);
  return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

// 1 / rms(v)
inline float ar_rstd192(const float* v) {
  return 1.0f / std::sqrt(ar_dot192(v, v) / 192.0f + 1e-6f);
}

// out = depth_mix(src[0..n-1], w); rstd[i] = ar_rstd192(src[i]) (cached by the caller for fixed sources).
// n in 1..16.  out may not alias a source.
inline void ar_depth_mix192(const float* const* src, const float* rstd, int n, const float* w, float* out) {
  float lg[16];
  for (int i = 0; i < n; i++) lg[i] = ar_dot192(w, src[i]) * rstd[i];
  float m = lg[0];
  for (int i = 1; i < n; i++) m = lg[i] > m ? lg[i] : m;
  alignas(32) float e[16];
  for (int i = 0; i < 16; i++) e[i] = (i < n ? lg[i] : m - 100.0f) - m;   // pads are never summed
  _mm256_store_ps(e, exp256_ps(_mm256_load_ps(e)));
  _mm256_store_ps(e + 8, exp256_ps(_mm256_load_ps(e + 8)));
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += e[i];
  float p[16];
  for (int i = 0; i < n; i++) p[i] = e[i] / s;
  for (int d = 0; d < 192; d += 8) {
    __m256 h = _mm256_mul_ps(_mm256_set1_ps(p[0]), _mm256_load_ps(src[0] + d));
    for (int i = 1; i < n; i++)
      h = _mm256_add_ps(h, _mm256_mul_ps(_mm256_set1_ps(p[i]), _mm256_load_ps(src[i] + d)));
    _mm256_store_ps(out + d, h);
  }
}

}  // namespace opt
}  // namespace fx2
