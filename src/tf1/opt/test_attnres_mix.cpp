// Unit test for attnres_mix.h (HUT-NATIVE-AR): engine depth-mix vs an fp64 reference of the same formula,
// over source counts 1..9 (the replace-B2 range), normal and peaked logits, small/large source scales.
// Also checks bit-identical repeat calls (determinism).  Build with the engine's TF1 flags:
//   clang++-18 -O3 -std=c++17 -march=x86-64-v3 -mrecip=none -ffp-model=precise -ffp-contract=off \
//              test_attnres_mix.cpp -o test_attnres_mix
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "attnres_mix.h"

using namespace fx2::opt;

static void ref_mix(const float* const* src, int n, const float* w, double* out) {
  double lg[16];
  for (int i = 0; i < n; i++) {
    double dw = 0, ss = 0;
    for (int d = 0; d < 192; d++) {
      dw += double(w[d]) * src[i][d];
      ss += double(src[i][d]) * src[i][d];
    }
    lg[i] = dw / std::sqrt(ss / 192.0 + 1e-6);
  }
  double m = lg[0];
  for (int i = 1; i < n; i++) m = std::max(m, lg[i]);
  double s = 0, p[16];
  for (int i = 0; i < n; i++) { p[i] = std::exp(lg[i] - m); s += p[i]; }
  for (int d = 0; d < 192; d++) {
    double h = 0;
    for (int i = 0; i < n; i++) h += p[i] / s * src[i][d];
    out[d] = h;
  }
}

int main() {
  std::mt19937 rng(1);
  std::normal_distribution<float> N(0.f, 1.f);
  alignas(32) static float S[10][192];
  alignas(32) float w[192], out[192], out2[192];
  double ref[192];
  double worst = 0;
  int fails = 0, cases = 0;
  const float scales[] = {1e-3f, 1.f, 1e2f};
  const float wscales[] = {0.1f, 1.f, 8.f};   // 8: very peaked softmax
  for (int rep = 0; rep < 200; rep++)
    for (int n = 1; n <= 9; n++)
      for (float sc : scales)
        for (float ws : wscales) {
          for (int i = 0; i < n; i++)
            for (int d = 0; d < 192; d++) S[i][d] = N(rng) * sc * (1.f + 0.5f * i);
          for (int d = 0; d < 192; d++) w[d] = N(rng) * ws / std::sqrt(192.f);
          const float* src[10];
          float rs[10];
          for (int i = 0; i < n; i++) { src[i] = S[i]; rs[i] = ar_rstd192(S[i]); }
          ar_depth_mix192(src, rs, n, w, out);
          ar_depth_mix192(src, rs, n, w, out2);
          if (std::memcmp(out, out2, sizeof(out)) != 0) { std::printf("NONDETERMINISTIC n=%d\n", n); fails++; }
          ref_mix(src, n, w, ref);
          double num = 0, den = 0;
          for (int d = 0; d < 192; d++) {
            num += (out[d] - ref[d]) * (out[d] - ref[d]);
            den += ref[d] * ref[d];
            if (!std::isfinite(out[d])) { std::printf("NONFINITE n=%d\n", n); fails++; break; }
          }
          const double rel = std::sqrt(num / (den + 1e-300));
          if (rel > worst) worst = rel;
          if (rel > 1e-5) { std::printf("n=%d sc=%g ws=%g rel %.3e\n", n, sc, ws, rel); fails++; }
          cases++;
        }
  std::printf("cases %d, worst rel L2 vs fp64 %.3e, fails %d -> %s\n", cases, worst, fails, fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
