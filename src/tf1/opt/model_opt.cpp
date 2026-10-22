// Changes in this file: van_layer_kv(), the `gelesen` ("read") flag with its
// early exit in step() and set_gelesen(), for rows whose output is not read;
// and the mlp.down density threshold, which can follow the layer's neuron
// count instead of DMLP; the side-channel token row (proj_add,
// paket_token_row) and last_hidden().  All additive only.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/model_opt.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md,
// among them a K/V-only last layer for rows whose output is not read.
// TransformerOpt implementation. Wiring order matches SPEC section 3.3 and
// the arena stream order of arena_build.cpp exactly.
//
// Profiling (-DFX2_PROF): raw rdtsc (no lfence/rdtscp - the ~67-cycle
// serialized stamp would distort short sections; raw back-to-back stamps
// measure ~10-20 core cycles) section accumulators, one per call site,
// aggregated into the four SPEC section-9 groups at print time and
// overhead-corrected by the measured empty-section delta.

#include "model_opt.h"
#include "../../cold_size.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "arena_build.h"
#if TF_AR
#include "attnres_mix.h"
#if TF_ARV2 || TF_AR_COSTDIAG >= 3
#include "attnres_mix_v2.h"   // v10 runtime rework (FMA dots/mix, one exp, cached partial rstd)
#endif
#endif
#include "attn.h"
#include "glue.h"
#include "kda.h"
#include "kda_math.h"
#include "qmat_dense.h"
#include "qmat_sparse.h"

// Debug-only component isolation for the naive-vs-opt cross-check (never
// defined in the production build targets): swap the fused KDA step and/or
// the attention kernel for the naive path (bit-exact vs src/model.cpp) to
// attribute divergence per component.
#if defined(FX2_XCHECK_NAIVE_KDA) || defined(FX2_XCHECK_NAIVE_ATTN)
#include "../kernels.h"
#endif

namespace fx2 {
namespace opt {

namespace {
constexpr int V = 208, D = 192, NL = 16, DH = 64, NH = 3,
              DMLP = TF_DMLP;
constexpr int NKIMI = 12, NVAN = 4,
              NSKIP = 8;

constexpr int WIN = ATTN_WIN, ROPE_LEN = 131072;
static_assert(WIN <= ATTN_WIN,
              "the ring cannot be wider than the scan.");
static_assert(WIN % ATTN_BLK == 0 && WIN >= 256,
              "the ring follows the block width of the scan.");
// density thresholds for the sparse column kernels (qmat.h)
constexpr int PRIOR_DENSE_NNZ =
    static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * V);  // 194

// beta projection: 3x192 fp32 matvec on xn. Bit-identical to the naive
// fx2::dot_f32(xn, w, 192) (same 8-wide fmadd chain + hsum tree).
inline float dot192(const float* a, const float* b) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 192; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  return hsum256v(acc);
}
}  // namespace

// ---------------------------------------------------------------------------
// profiling sites
// ---------------------------------------------------------------------------
namespace {

enum PSite : int {
  // group 0: int4 x int8 matmuls (ALL arenas)
  PS_MM_PRIOR = 0,
  PS_MM_KQKV,
  PS_MM_GUP,
  PS_MM_GDN,
  PS_MM_KOP,
  PS_MM_VQKV,
  PS_MM_VOP,
  PS_MM_MLP_UP,
  PS_MM_MLP_DOWN,
  PS_MM_UNEMB,
  // group 1: vanilla attention mechanism
  PS_AT_INSERT,
  PS_AT_STEP,
  // group 2: kimi linear attention mechanism
  PS_KD_BETA,
  PS_KD_STEP,
  // group 3: everything else
  PS_EL_PRIORQ,
  PS_EL_EMBED,
  PS_EL_RESID,
  PS_EL_NQ_KIMI,
  PS_EL_NQ_VAN,
  PS_EL_NQ_MLP,
  PS_EL_NQ_FIN,
  PS_EL_HEADPREP,
  PS_EL_OQ,
  PS_EL_HEAD,
  PS_N
};

#ifdef FX2_PROF
struct PSiteInfo {
  const char* name;
  int group;
};
const PSiteInfo kPSites[PS_N] = {
    {"prior spmv (int4 cols)", 0},
    {"kimi q/k/v proj (27)", 0},
    {"gate up (fg+og, 18)", 0},
    {"gate down (fg+og, 18)", 0},
    {"kimi out proj (9)", 0},
    {"van q/k/v proj (9)", 0},
    {"van out proj (3)", 0},
    {"mlp up + relu2q (12)", 0},
    {"mlp down spmv (12)", 0},
    {"unembed (1)", 0},
    {"kv insert (3)", 1},
    {"attn step qk+sm+pv (3)", 1},
    {"beta matvec (9)", 2},
    {"kda fused step (9)", 2},
    {"prior quant+idx", 3},
    {"embed combine", 3},
    {"resid/skip/tok ops", 3},
    {"norm+quant kimi (9)", 3},
    {"norm+quant van (3)", 3},
    {"norm+quant mlp (12)", 3},
    {"final norm+quant", 3},
    {"head-prep norm/rope/q (3)", 3},
    {"attnout quant (12)", 3},
    {"softcap+softmax head", 3},
};
const char* kGroups[4] = {"int4xint8 matmuls", "vanilla attention mechanism",
                          "kimi (KDA) mechanism", "everything else"};

inline uint64_t prof_rdtsc() {
  unsigned lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi)::"memory");
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
uint64_t g_pacc[PS_N] = {};
uint64_t g_pcalls[PS_N] = {};
struct PScope {
  int s;
  uint64_t t0;
  explicit PScope(int s_) : s(s_), t0(prof_rdtsc()) {}
  ~PScope() {
    g_pacc[s] += prof_rdtsc() - t0;
    g_pcalls[s]++;
  }
};
#define PSCOPE(site) PScope pscope_(site)
#else
#define PSCOPE(site) \
  do {               \
  } while (0)
#endif

}  // namespace

void prof_reset() {
#ifdef FX2_PROF
  std::memset(g_pacc, 0, sizeof(g_pacc));
  std::memset(g_pcalls, 0, sizeof(g_pcalls));
#endif
}

bool prof_enabled() {
#ifdef FX2_PROF
  return true;
#else
  return false;
#endif
}

void prof_print(double tsc_to_core, long tokens) {
#ifndef FX2_PROF
  (void)tsc_to_core;
  (void)tokens;
  std::printf("prof: not compiled in (build with -DFX2_PROF)\n");
#else
  // empty-section overhead: median back-to-back raw-rdtsc delta
  const int NS = 4001;
  std::vector<uint64_t> st(NS);
  for (int i = 0; i < NS; i++) st[i] = prof_rdtsc();
  std::vector<double> dd(NS - 1);
  for (int i = 0; i + 1 < NS; i++) dd[i] = double(st[i + 1] - st[i]);
  std::nth_element(dd.begin(), dd.begin() + (NS - 1) / 2, dd.end());
  const double stamp_ticks = dd[(NS - 1) / 2];
  std::printf("\n== per-part profile (raw rdtsc, overhead-corrected by %.1f "
              "ticks/section = %.1f core cyc) ==\n",
              stamp_ticks, stamp_ticks * tsc_to_core);
  double gcyc[4] = {}, gcalls[4] = {};
  double site_cyc[PS_N];
  double total = 0;
  for (int s = 0; s < PS_N; s++) {
    double ticks = double(g_pacc[s]) - double(g_pcalls[s]) * stamp_ticks;
    if (ticks < 0) ticks = 0;
    site_cyc[s] = ticks * tsc_to_core / double(tokens > 0 ? tokens : 1);
    gcyc[kPSites[s].group] += site_cyc[s];
    gcalls[kPSites[s].group] += double(g_pcalls[s]);
    total += site_cyc[s];
  }
  std::printf("%-34s %12s %6s %9s\n", "group", "cyc/token", "%", "calls/tok");
  for (int g = 0; g < 4; g++)
    std::printf("%-34s %12.0f %5.1f%% %9.1f\n", kGroups[g], gcyc[g],
                100.0 * gcyc[g] / total,
                gcalls[g] / double(tokens > 0 ? tokens : 1));
  std::printf("%-34s %12.0f %5.1f%%\n", "TOTAL (instrumented sections)", total,
              100.0);
  std::printf("\n%-34s %12s %6s %9s\n", "  site", "cyc/token", "%",
              "calls/tok");
  for (int g = 0; g < 4; g++) {
    std::printf("  -- %s --\n", kGroups[g]);
    for (int s = 0; s < PS_N; s++) {
      if (kPSites[s].group != g) continue;
      std::printf("  %-32s %12.0f %5.1f%% %9.1f\n", kPSites[s].name,
                  site_cyc[s], 100.0 * site_cyc[s] / total,
                  double(g_pcalls[s]) / double(tokens > 0 ? tokens : 1));
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------
#if TF_AR
#define TF_ACC acc
#else
#define TF_ACC x
#endif
struct TransformerOptImpl {
  OptModel M;
  AttnKind kind;

  // streaming state (hugepage pool: KDA states, then KV caches)
  HugeBuf spool;
  KdaState* kst = nullptr;   // [9]
  AttnKVF32* kvf = nullptr;  // [3] iff kind == KVF32
  AttnKV* kvi = nullptr;     // [3] iff kind == KVI8
  int64_t t = 0;
  int64_t rope_off = 0;

#ifdef FX2_XCHECK_NAIVE_KDA
  struct NaiveKst {
    alignas(64) float S[NH][DH * DH];
    alignas(64) float hist[3][3][D];
  };
  std::vector<NaiveKst> nkst = std::vector<NaiveKst>(NKIMI);
#endif
#ifdef FX2_XCHECK_NAIVE_ATTN
  struct NaiveVst {
    alignas(64) int8_t kring[WIN][D];
    alignas(64) int8_t vring[WIN][D];
  };
  std::vector<NaiveVst> nvst = std::vector<NaiveVst>(3);
#endif

  // per-token skip stack (written by layers 0..5, read by 6..11 of the SAME
  // token) - transient within step
  alignas(64) float skip_store[NSKIP][D] = {};

#if TF_AR
  // Sub-layer output target (TF_ACC): ar_f in AttnRes replace mode, where the
  // sub-layer reads its input from x (= the depth-mixed h) but its output must
  // stay separate (partial sums); the residual stream x itself in the shipped
  // form (TF_ACC = x, so that build is byte-identical to the original).
  float* acc = x;
  // ---- AttnRes replace mode (v7 arch.py replace-B2, v7_AR_FORWARD_SPEC.md) ----
  // sources of a read: x0, the completed block sums (fixed once complete, their
  // 1/rms cached), and the running partial sum of the current block.
  static constexpr int ARB = TF_AR;
  static constexpr int AR_MAXC = NL / ARB + 1;    // x0 + completed block sums
  alignas(64) float ar_src[AR_MAXC][D] = {};
  float ar_rstd[AR_MAXC] = {};
  int ar_n = 0;
  alignas(64) float ar_part[D] = {};
#if TF_ARV2
  float ar_part_rstd = 0.0f;   // TF_ARV2: 1/rms of ar_part, kept current by ar_take_f
#endif
#if TF_ARV2 >= 5
  // TF_ARV2=5: ONLINE softmax per future read q over the COMPLETED sources (x0 + frozen block sums), maintained at push
  // time: M_q = max logit, Z_q = sum exp(l - M_q), A_q = sum exp(l - M_q) v. A read then only adds the partial:
  // m = max(M_q, l_p); a = exp(M_q - m), b = exp(l_p - m); out = (a A_q + b part) / (a Z_q + b).
  // Same function as the direct softmax (reassociated, deterministic); the read's chain is one dot + one exp + a mix.
  alignas(64) float ar_qA[2 * NL + 1][D] = {};
  float ar_qM[2 * NL + 1] = {}, ar_qZ[2 * NL + 1] = {};
#endif
#if TF_ARV2 >= 5
  // read index q: 2l = mixer read of layer l, 2l+1 = its MLP read, 2*NL = the final read
  static constexpr int AR_NQ = 2 * NL + 1;
  const float* ar_qw(int q) const {
    return q == 2 * NL ? M.ar_q_final : ((q & 1) ? M.ar_q_mlp[q >> 1] : M.ar_q_mix[q >> 1]);
  }
  int ar_qnext = 0;            // first read index that has not happened yet
  float ar_part_dot = 0.0f;    // dot(w_q, ar_part) for read q == ar_part_dot_q, fused into ar_take_f
  int ar_part_dot_q = -1;
#endif
  bool ar_has_part = false;
  alignas(64) float ar_f[D] = {};
#if TF_AR_COSTDIAG >= 2
  alignas(64) float ar_diag_scratch[D] = {};
  float ar_diag_sum = 0.0f;
  ~TransformerOptImpl() { if (ar_diag_sum == 1.2345f) std::printf("."); }   // keeps the scratch work observable
#endif

  void ar_push(const float* v) {
    std::memcpy(ar_src[ar_n], v, sizeof(float) * D);
#if TF_ARV2
    // the partial's rstd is already current; any other source (x0) gets it computed once here
    ar_rstd[ar_n] = (v == ar_part && ar_has_part) ? ar_part_rstd : ar_rstd192_v2(v);
#if TF_ARV2 >= 5
    {
      const float* sv = ar_src[ar_n];
      const float rsj = ar_rstd[ar_n];
      if (ar_n == 0) {
        // first source: M = l, Z = 1, A = x0 kept LAZY (A_q == ar_src[0] until the first block sum arrives)
        for (int q = ar_qnext; q < AR_NQ; q++) {
          ar_qM[q] = ar_dot192_v2(ar_qw(q), sv) * rsj;
          ar_qZ[q] = 1.0f;
        }
      } else {
        // all remaining reads: logits first, then the two exps 8 reads at a time, then the A updates
        alignas(32) float ea[2 * NL + 8], eb[2 * NL + 8];
        const int q0 = ar_qnext, nq = AR_NQ - q0;
        for (int k = 0; k < nq; k++) {
          const int q = q0 + k;
          const float l = ar_dot192_v2(ar_qw(q), sv) * rsj;
          const float m2 = l > ar_qM[q] ? l : ar_qM[q];
          ea[k] = ar_qM[q] - m2;                              // old-sum scale exponent (<= 0)
          eb[k] = l - m2;                                     // new-source exponent (<= 0)
          ar_qM[q] = m2;
        }
        for (int k = nq; k < ((nq + 7) & ~7); k++) ea[k] = eb[k] = 0.0f;
        for (int k = 0; k < nq; k += 8) {
          _mm256_store_ps(ea + k, exp256_ps(_mm256_load_ps(ea + k)));
          _mm256_store_ps(eb + k, exp256_ps(_mm256_load_ps(eb + k)));
        }
        for (int k = 0; k < nq; k++) {
          const int q = q0 + k;
          const float* Aold = ar_n == 1 ? ar_src[0] : ar_qA[q];   // lazy x0
          float* Aq = ar_qA[q];
          const __m256 va = _mm256_set1_ps(ea[k]), vb = _mm256_set1_ps(eb[k]);
          for (int d = 0; d < D; d += 8)
            _mm256_store_ps(Aq + d, _mm256_fmadd_ps(vb, _mm256_load_ps(sv + d), _mm256_mul_ps(va, _mm256_load_ps(Aold + d))));
          ar_qZ[q] = ea[k] * ar_qZ[q] + eb[k];
        }
      }
    }
#endif
#else
    ar_rstd[ar_n] = ar_rstd192(v);
#endif
    ar_n++;
  }
  // x = depth_mix(x0, completed block sums, partial; w)   (attnres_mix.h)
  void ar_read(const float* w) {
#if TF_AR_COSTDIAG
    // DIAGNOSTIC ONLY (never a real build): the read degenerates to a copy of the newest source. COSTDIAG=2 also
    // computes the real v1 depth mix into a scratch row (kept alive by a checksum) before copying, so 2-vs-1 have
    // IDENTICAL activations and downstream work (v5 1631: MLP-down sparsity depends on activations) and their paired
    // wall difference is the depth-mix compute alone.
#if TF_AR_COSTDIAG >= 2
    {
      const float* src2[AR_MAXC + 1];
      float rs2[AR_MAXC + 1];
      int n2 = 0;
      for (int i = 0; i < ar_n; i++) { src2[n2] = ar_src[i]; rs2[n2++] = ar_rstd[i]; }
#if TF_AR_COSTDIAG >= 3
      // COSTDIAG=3: the v2 mix (FMA dots/mix, one exp) in place of v1's; rstd of the partial computed here (v2 proper
      // caches it at take_f, so this slightly overstates v2's cost)
      if (ar_has_part) { src2[n2] = ar_part; rs2[n2++] = ar_rstd192_v2(ar_part); }
      ar_depth_mix192_v2(src2, rs2, n2, w, ar_diag_scratch);
#else
      if (ar_has_part) { src2[n2] = ar_part; rs2[n2++] = ar_rstd192(ar_part); }
      ar_depth_mix192(src2, rs2, n2, w, ar_diag_scratch);
#endif
      ar_diag_sum += ar_diag_scratch[ar_n & 127];
    }
#endif
    (void)w;
    std::memcpy(x, ar_has_part ? ar_part : ar_src[ar_n - 1], sizeof(float) * D);
    return;
#endif
    const float* src[AR_MAXC + 1];
    float rs[AR_MAXC + 1];
    int n = 0;
    for (int i = 0; i < ar_n; i++) { src[n] = ar_src[i]; rs[n++] = ar_rstd[i]; }
#if TF_ARV2 >= 5
    (void)rs;
    (void)src;
    (void)w;
    const int q = ar_qnext++;
    const float* Aq = ar_n == 1 ? ar_src[0] : ar_qA[q];   // lazy x0 (see ar_push)
    if (!ar_has_part) {                                    // completed sources only: out = A / Z
      if (ar_n == 1) {                                     // x0 alone: exactly x0
        for (int d = 0; d < D; d += 8) _mm256_store_ps(x + d, _mm256_load_ps(Aq + d));
      } else {
        const __m256 r = _mm256_set1_ps(1.0f / ar_qZ[q]);
        for (int d = 0; d < D; d += 8) _mm256_store_ps(x + d, _mm256_mul_ps(_mm256_load_ps(Aq + d), r));
      }
    } else {
      const float lp = (ar_part_dot_q == q ? ar_part_dot : ar_dot192_v2(ar_qw(q), ar_part)) * ar_part_rstd;
      const float m = lp > ar_qM[q] ? lp : ar_qM[q];
      alignas(32) float ex[8] = {ar_qM[q] - m, lp - m, 0, 0, 0, 0, 0, 0};
      _mm256_store_ps(ex, exp256_ps(_mm256_load_ps(ex)));
      const float den = ex[0] * ar_qZ[q] + ex[1];
      const __m256 ra = _mm256_set1_ps(ex[0] / den), rb = _mm256_set1_ps(ex[1] / den);
      for (int d = 0; d < D; d += 8)
        _mm256_store_ps(x + d, _mm256_fmadd_ps(rb, _mm256_load_ps(ar_part + d), _mm256_mul_ps(ra, _mm256_load_ps(Aq + d))));
    }
#elif TF_ARV2
    if (ar_has_part) { src[n] = ar_part; rs[n++] = ar_part_rstd; }
    ar_depth_mix192_v2(src, rs, n, w, x);
#else
    if (ar_has_part) { src[n] = ar_part; rs[n++] = ar_rstd192(ar_part); }
    ar_depth_mix192(src, rs, n, w, x);
#endif
  }
#if TF_ARV2 >= 5
  // next_q: the read that will use this partial next (-1: the partial is pushed before any read sees it)
  void ar_take_f(int next_q) {
    if (ar_has_part && next_q >= 0) {
      ar_part_rstd = ar_take_f_dot_v2(ar_part, ar_f, ar_qw(next_q), &ar_part_dot);   // add + rms + next dot, one pass
      ar_part_dot_q = next_q;
      return;
    }
    ar_part_dot_q = -1;
    if (ar_has_part) {
      ar_part_rstd = ar_take_f_v2(ar_part, ar_f);
    } else {
      std::memcpy(ar_part, ar_f, sizeof(float) * D);
      ar_part_rstd = ar_rstd192_v2(ar_part);
      ar_has_part = true;
    }
  }
#else
  void ar_take_f() {   // partial += f (the block's running sum), f = sub-layer output
#if TF_ARV2
    if (ar_has_part) {
      ar_part_rstd = ar_take_f_v2(ar_part, ar_f);       // add + sum of squares in one pass
    } else {
      std::memcpy(ar_part, ar_f, sizeof(float) * D);
      ar_part_rstd = ar_rstd192_v2(ar_part);
      ar_has_part = true;
    }
#else
    if (ar_has_part) {
      for (int d = 0; d < D; d++) ar_part[d] += ar_f[d];
    } else {
      std::memcpy(ar_part, ar_f, sizeof(float) * D);
      ar_has_part = true;
    }
#endif
  }
#endif

#endif

  // scratch
  alignas(64) float x[D] = {}, xn[D] = {}, yb[D] = {};
  alignas(64) uint8_t q5[5 * D] = {};  // up to 5 biased-u8 copies of xn
  alignas(64) uint8_t q64b[DH] = {};
  alignas(64) float bq[D] = {}, bk[D] = {}, bv[D] = {};
  alignas(64) float graw[D] = {}, og192[D] = {}, gn[D] = {};
  alignas(32) float braw[8] = {};
  alignas(64) float qf[D] = {}, kf[D] = {}, vf[D] = {}, pre[D] = {};
  alignas(64) int8_t qq[D] = {}, kk[D] = {}, vv[D] = {};
  alignas(64) uint8_t q768[DMLP] = {};
  alignas(64) uint16_t idx768[DMLP + 16] = {};
  alignas(64) float prior_f32[208] = {};
  alignas(64) uint8_t q8p[224] = {};  // V rounded to 32
  alignas(64) uint16_t idxp[224 + 16] = {};
  alignas(64) float logits[208] = {};
  alignas(64) float probs208[208] = {};
  alignas(32) float sfall[32] = {}, cfall[32] = {};  // rope fallback row
  // The assembled token row with side channels.  `tok_paket` is the normed
  // sum, `pk` the channels of the row that step() is about to consume.
  alignas(64) float tok_paket[D] = {};
  PaketChannels pk;

  COLD_SIZE_ATTR explicit TransformerOptImpl(const char* path, AttnKind k) : kind(k) {
    M.load(path);
    const size_t kda_sz = sizeof(KdaState) * NKIMI;
    const size_t kv_sz =
        (kind == AttnKind::KVF32 ? sizeof(AttnKVF32) : sizeof(AttnKV)) * NVAN;
    spool.alloc(kda_sz + kv_sz);
    spool.trim(kda_sz + kv_sz);
    kst = reinterpret_cast<KdaState*>(spool.p);
    if (kind == AttnKind::KVF32)
      kvf = reinterpret_cast<AttnKVF32*>(spool.p + kda_sz);
    else
      kvi = reinterpret_cast<AttnKV*>(spool.p + kda_sz);
    for (int i = 0; i < NKIMI; i++) kda_layer_reset(kst[i]);
  }

  void begin(int64_t rope_position_offset) {
    for (int i = 0; i < NKIMI; i++) kda_layer_reset(kst[i]);
#ifdef FX2_XCHECK_NAIVE_KDA
    std::memset(nkst.data(), 0, sizeof(NaiveKst) * NKIMI);
#endif
    t = 0;  // invalidates the KV rings (validity is [0, t] within an article)
    rope_off = rope_position_offset;
  }

  void kimi_layer(int ki) {
    KimiArenas& L = M.kimi[ki];
    {
      PSCOPE(PS_EL_NQ_KIMI);
      const float s5[5] = {L.qp.s_act, L.kp.s_act, L.vp.s_act, L.fg_up.s_act,
                           L.og_up.s_act};
      rms_norm_quant192_multi(x, xn, s5, 5, q5);
    }
    {
      PSCOPE(PS_MM_KQKV);
      qgemv_f32(L.qp.m, q5, bq);
      qgemv_f32(L.kp.m, q5 + D, bk);
      qgemv_f32(L.vp.m, q5 + 2 * D, bv);
    }
    {
      PSCOPE(PS_MM_GUP);
      qgemv_quant_bias(L.fg_up.m, q5 + 3 * D, L.fg_down.s_act, q64b);
    }
    {
      PSCOPE(PS_MM_GDN);
      qgemv_f32(L.fg_down.m, q64b, graw);
    }
    {
      PSCOPE(PS_MM_GUP);
      qgemv_quant_bias(L.og_up.m, q5 + 4 * D, L.og_down.s_act, q64b);
    }
    {
      PSCOPE(PS_MM_GDN);
      qgemv_f32(L.og_down.m, q64b, og192);
    }
    {
      PSCOPE(PS_KD_BETA);
      for (int h = 0; h < NH; h++)
        braw[h] = dot192(xn, L.beta_w + size_t(h) * D);
    }
    {
      PSCOPE(PS_KD_STEP);
#ifdef FX2_XCHECK_NAIVE_KDA
      // naive scalar kimi path (bit-exact vs src/model.cpp kimi_attention)
      NaiveKst& st = nkst[ki];
      alignas(32) float cq[D], ck[D], cv[D], o192[D];
      conv4_silu(L.kw.conv_w[0], st.hist[0][0], st.hist[0][1], st.hist[0][2],
                 bq, cq, D);
      conv4_silu(L.kw.conv_w[1], st.hist[1][0], st.hist[1][1], st.hist[1][2],
                 bk, ck, D);
      conv4_silu(L.kw.conv_w[2], st.hist[2][0], st.hist[2][1], st.hist[2][2],
                 bv, cv, D);
      const float* newest[3] = {bq, bk, bv};
      for (int c = 0; c < 3; c++) {
        std::memmove(st.hist[c][0], st.hist[c][1], sizeof(float) * 2 * D);
        std::memcpy(st.hist[c][2], newest[c], sizeof(float) * D);
      }
      for (int h = 0; h < NH; h++)
        kda_head_step(st.S[h], cq + h * DH, ck + h * DH, cv + h * DH,
                      graw + h * DH, L.kw.dt_bias + h * DH, L.kw.a_neg[h],
                      sigmoid1f(braw[h]), o192 + h * DH);
      for (int h = 0; h < NH; h++)
        gated_rms_norm64(o192 + h * DH, og192 + h * DH, L.kw.gn_w,
                         gn + h * DH);
#else
      kda_layer_step(L.kw, kst[ki], bq, bk, bv, graw, og192, braw, gn);
#endif
    }
    {
      PSCOPE(PS_EL_OQ);
      quant192_u8(gn, L.op.s_act, q5);
    }
    {
      PSCOPE(PS_MM_KOP);
      qgemv_add(L.op.m, q5, TF_ACC);
    }
  }

  // GQA / value embeddings (build constants TF_KVH, TF_VE): k and v come out
  // of the projections with TF_KVH heads; v gets c0*v + c1*VE[token] (token of
  // the row being fed; sentinel rows >= 208 per TF_VE_SENT), then the K/V heads
  // are repeated for the NH query heads, so the attention kernels and the KV
  // ring stay as they are.
  int cur_tok = 0;
  void kv_expand(const VanArenas& L) {
#if TF_VE
    {
      constexpr int W = TF_KVH * DH;
      int row = cur_tok;
#ifndef TF_VE_SENT
#define TF_VE_SENT 0
#endif
      if (row >= V) row = (TF_VE_SENT == 1) ? V - 1 : (TF_VE_SENT == 2 ? row - V : -1);
      if (row >= 0) {
        const float* e = L.ve + size_t(row) * W;
        for (int i = 0; i < W; i++) vf[i] = L.ve_c0 * vf[i] + L.ve_c1 * e[i];
      } else {
        for (int i = 0; i < W; i++) vf[i] = L.ve_c0 * vf[i];
      }
    }
#else
    (void)L;
#endif
#if TF_KVH != 3
    for (int h = TF_KVH; h < NH; h++) {
      std::memcpy(kf + h * DH, kf + (h % TF_KVH) * DH, sizeof(float) * DH);
      std::memcpy(vf + h * DH, vf + (h % TF_KVH) * DH, sizeof(float) * DH);
    }
#endif
  }

  void van_layer(int vi) {
    VanArenas& L = M.van[vi];
    {
      PSCOPE(PS_EL_NQ_VAN);
      const float s3[3] = {L.qp.s_act, L.kp.s_act, L.vp.s_act};
      rms_norm_quant192_x3(x, xn, s3, q5, q5 + D, q5 + 2 * D);
    }
    {
      PSCOPE(PS_MM_VQKV);
      qgemv_f32(L.qp.m, q5, qf);
      qgemv_f32(L.kp.m, q5 + D, kf);
      qgemv_f32(L.vp.m, q5 + 2 * D, vf);
    }
    kv_expand(L);
    {
      PSCOPE(PS_EL_HEADPREP);
      for (int h = 0; h < NH; h++) rms_norm64(qf + h * DH, qf + h * DH);
      for (int h = 0; h < NH; h++) rms_norm64(kf + h * DH, kf + h * DH);
      const int64_t pos = rope_off + t;
      const float *sp, *cp;
      if (pos < ROPE_LEN) {
        sp = M.rope_sin.data() + size_t(pos) * 32;
        cp = M.rope_cos.data() + size_t(pos) * 32;
      } else {  // beyond the shipped table: libm fallback (same as naive)
        const float fpos = static_cast<float>(pos);
        for (int i = 0; i < 32; i++) {
          const float ang = fpos * M.inv_freq[i];
          sfall[i] = std::sin(ang);
          cfall[i] = std::cos(ang);
        }
        sp = sfall;
        cp = cfall;
      }
      rope_apply_192(qf, sp, cp);
      rope_apply_192(kf, sp, cp);
      for (int h = 0; h < NH; h++) {
        quant64_i8(qf + h * DH, L.sq[h], qq + h * DH);
        quant64_i8(kf + h * DH, L.sk[h], kk + h * DH);
        quant64_i8(vf + h * DH, L.sv[h], vv + h * DH);
      }
    }
    const int slot = static_cast<int>(t % WIN);
    const int64_t n = t + 1;
#ifdef FX2_EXP_KPF
    {  // experiment: warm the first 2KB of each head's K stream (the scan's
       // software prefetch runs +2KB ahead, so the first 2KB are cold-start
       // demand misses paced by the QK compute)
      const char* kb = kind == AttnKind::KVF32
                           ? reinterpret_cast<const char*>(kvf[vi].k)
                           : reinterpret_cast<const char*>(kvi[vi].k);
      for (int h = 0; h < 3; h++)
        for (int off = 0; off < 2048; off += 64)
          _mm_prefetch(kb + h * (ATTN_NBLK * 32 * ATTN_DH) + off, _MM_HINT_T0);
    }
#endif
#ifdef FX2_XCHECK_NAIVE_ATTN
    {
      NaiveVst& st = nvst[vi];
      std::memcpy(st.kring[slot], kk, D);
      std::memcpy(st.vring[slot], vv, D);
      if (n < WIN)
        fx2::attention_step_var(qq, st.kring[0], st.vring[0], L.coef, L.sv,
                                static_cast<int>(n), pre, 0.0f);
      else
        fx2::attention_step_fixed(qq, st.kring[0], st.vring[0], L.coef, L.sv,
                                  pre, 0.0f);
    }
#else
    if (!((3 & 2) && 1) && kind == AttnKind::KVF32) {
      {
        PSCOPE(PS_AT_INSERT);
        attn_kv_insert(kvf[vi], slot, kk, vv);
      }
      {
        PSCOPE(PS_AT_STEP);
        if (n < WIN)
          attn_step_var(kvf[vi], qq, L.coef, L.sv, static_cast<int>(n), pre,
                        0.0f);
        else
          attn_step_fixed(kvf[vi], qq, L.coef, L.sv, pre, 0.0f);
      }
    } else {
      {
        PSCOPE(PS_AT_INSERT);
        attn_kv_insert(kvi[vi], slot, kk, vv);
      }
      {
        PSCOPE(PS_AT_STEP);
        if (n < WIN)
          attn_step_var(kvi[vi], qq, L.coef, L.sv, static_cast<int>(n), pre,
                        0.0f);
        else
          attn_step_fixed(kvi[vi], qq, L.coef, L.sv, pre, 0.0f);
      }
    }
#endif
    {
      PSCOPE(PS_EL_OQ);
      quant192_u8(pre, L.op.s_act, q5);
    }
    {
      PSCOPE(PS_MM_VOP);
      qgemv_add(L.op.m, q5, TF_ACC);
    }
  }

#ifdef FX2_XCHECK_NAIVE_ATTN
#error "the K/V-only last layer writes only the opt KV ring; the naive cross-check needs the full path"
#endif

  // The last layer for a row whose distribution nobody reads.  Line for line
  // the same as van_layer() up to and including attn_kv_insert, and nothing
  // after it: the KV ring of this layer must hold k and v of this position,
  // everything else only feeds `pre` -> `x` -> the output head, and `x` is
  // rewritten by embed_combine192() at the next step anyway.  `xn` is
  // overwritten here; nothing may call last_hidden() between an unread row and
  // the next read one (in this coder no bit is coded inside a prompt block).
  void van_layer_kv(int vi) {
    VanArenas& L = M.van[vi];
    {
      PSCOPE(PS_EL_NQ_VAN);
      const float s3[3] = {L.qp.s_act, L.kp.s_act, L.vp.s_act};
      rms_norm_quant192_x3(x, xn, s3, q5, q5 + D, q5 + 2 * D);
    }
    {
      PSCOPE(PS_MM_VQKV);
      qgemv_f32(L.kp.m, q5 + D, kf);
      qgemv_f32(L.vp.m, q5 + 2 * D, vf);
    }
    kv_expand(L);
    {
      PSCOPE(PS_EL_HEADPREP);
      for (int h = 0; h < NH; h++) rms_norm64(kf + h * DH, kf + h * DH);
      const int64_t pos = rope_off + t;
      const float *sp, *cp;
      if (pos < ROPE_LEN) {
        sp = M.rope_sin.data() + size_t(pos) * 32;
        cp = M.rope_cos.data() + size_t(pos) * 32;
      } else {  // beyond the shipped table: libm fallback (same as naive)
        const float fpos = static_cast<float>(pos);
        for (int i = 0; i < 32; i++) {
          const float ang = fpos * M.inv_freq[i];
          sfall[i] = std::sin(ang);
          cfall[i] = std::cos(ang);
        }
        sp = sfall;
        cp = cfall;
      }
      rope_apply_192(kf, sp, cp);
      for (int h = 0; h < NH; h++) {
        quant64_i8(kf + h * DH, L.sk[h], kk + h * DH);
        quant64_i8(vf + h * DH, L.sv[h], vv + h * DH);
      }
    }
    const int slot = static_cast<int>(t % WIN);
    {
      PSCOPE(PS_AT_INSERT);
      // the same branch choice as in van_layer()
      if (!((3 & 2) && 1) &&
          kind == AttnKind::KVF32)
        attn_kv_insert(kvf[vi], slot, kk, vv);
      else
        attn_kv_insert(kvi[vi], slot, kk, vv);
    }
  }

  void mlp_block(int l) {
    MlpArenas& mm = M.mlp[l];
    {
      PSCOPE(PS_EL_NQ_MLP);
      rms_norm_quant192(x, xn, mm.up.s_act, q5);
    }
    int nnz;
    {
      PSCOPE(PS_MM_MLP_UP);
      nnz = qgemv_relu2q(mm.up.m, q5, mm.down_s_act, q768, idx768);
    }
    {
      PSCOPE(PS_MM_MLP_DOWN);
#ifdef TF_W5
      if (mm.down_wide) {
        if (nnz > static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * mm.down8.d_in))
          qsparse_dense_add(mm.down8, q768, TF_ACC);
        else
          qsparse_add(mm.down8, q768, idx768, nnz, TF_ACC);
      } else
#endif
      if (nnz > static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * mm.down.d_in))
        qsparse4_dense_add(mm.down, q768, TF_ACC);
      else
        qsparse4_add(mm.down, q768, idx768, nnz, TF_ACC);
    }
  }

  static void proj_add(const float* f, int d_in, const int8_t* q,
                       const float* fold, float s_act, float* acc_out) {
    int32_t qa[PaketArenas::DIN_LAST > PaketArenas::DIN_MATCH
                   ? PaketArenas::DIN_LAST
                   : PaketArenas::DIN_MATCH];
    for (int i = 0; i < d_in; i++) {
      // True division and clamp-before-round, exactly as glue.cpp
      // quant16_i8 / kernels.cpp quantize_i8 do it (a reciprocal multiply is
      // not bit-identical, and clamp/round commute here).
      float v = f[i] / s_act;
      if (v < -128.0f) v = -128.0f;
      if (v > 127.0f) v = 127.0f;
      qa[i] = static_cast<int32_t>(std::nearbyintf(v));  // round half to even
    }
    for (int o = 0; o < D; o++) {
      int32_t s = 0;
      const int8_t* w = q + size_t(o) * d_in;
      for (int i = 0; i < d_in; i++) s += qa[i] * static_cast<int32_t>(w[i]);
      acc_out[o] += fold[o] * static_cast<float>(s);
    }
  }

  // The token row with side channels: raw embedding row + the two projection
  // terms, then the embedding norm (the order of the training model).
  const float* paket_token_row(int token) {
    const PaketArenas& P = M.paket;
    if (token >= P.emb_rows ||
        pk.prev1 >= P.emb_rows + 1 ||
        pk.prev2 >= P.emb_rows + 1 ||
        pk.match_byte >= PaketArenas::NMATCH ||
        (pk.match_class & ~15) != 0) {
      // A row index out of range would read past a table and produce numbers
      // that look plausible.  The engine never generates one; a caller that
      // does gets a message instead of a silent wrong distribution.
      std::fprintf(stderr, "model_opt: row out of range (token %d, prev %d/%d, match %d/%d, embedding rows %d)\n",
          token, pk.prev1, pk.prev2, pk.match_byte, pk.match_class, P.emb_rows);
      std::exit(1);
    }
    alignas(64) float t[D];
    std::memcpy(t, P.emb_raw + size_t(token) * D, sizeof(float) * D);
    if (P.has_last) {
      alignas(64) float f[PaketArenas::DIN_LAST];
      static_assert(PaketArenas::X == 2,
          "the predecessor list below is written out for two "
          "slots.  With a different slot count the remaining slots would "
          "silently read row 0 -- widen PaketChannels and this initializer "
          "together, and re-export the last_table (its row count is X*rows).");
      const int prev[PaketArenas::X] = {pk.prev1, pk.prev2};
      for (int k = 0; k < PaketArenas::X; k++) {
        const size_t row = size_t(k) * (P.emb_rows + 1)
                         + size_t(prev[k]);
        std::memcpy(f + k * PaketArenas::DSLOT,
                    P.last_rows + row * PaketArenas::DSLOT,
                    sizeof(float) * PaketArenas::DSLOT);
      }
      proj_add(f, PaketArenas::DIN_LAST, P.last_q, P.last_fold, P.last_s_act,
               t);
    }
    if (P.has_match) {
      alignas(64) float f[PaketArenas::DIN_MATCH];
      // as zctx/zpmodel.py match_features: rows [match_byte] and [NMATCH + class]
      std::memcpy(f, P.match_rows + size_t(pk.match_byte) * PaketArenas::DSLOT,
                  sizeof(float) * PaketArenas::DSLOT);
      std::memcpy(f + PaketArenas::DSLOT,
                  P.match_rows +
                      size_t(PaketArenas::NMATCH + pk.match_class) *
                          PaketArenas::DSLOT,
                  sizeof(float) * PaketArenas::DSLOT);
      proj_add(f, PaketArenas::DIN_MATCH, P.match_q, P.match_fold,
               P.match_s_act, t);
    }
    rms_norm192(t, tok_paket);
    return tok_paket;
  }

  // gelesen ("read"): the output of the next step() is used.  Applies to
  // that step only.
  bool gelesen = true;
  static_assert((16 % 4) == 0,
      "the K/V-only last layer needs (NL-1) % 4 == 3 (a window layer)");

  // prior must be the cap-208 internal buffer with pads [205..208) zeroed
  void step(uint8_t token, float* probs_out) {
    // Fail-safe: the flag holds for exactly one step.
    const bool gel = gelesen;
    gelesen = true;
    cur_tok = token;
    const float* tok = M.tok_table + size_t(token) * D;
    if (M.paket.present) tok = paket_token_row(token);
    int nnzp;
    {
      PSCOPE(PS_EL_PRIORQ);
      prior_quant_raw(prior_f32, M.prior_s_act, q8p);  // raw u8; q8p[208..224)=0
      nnzp = qsparse_make_idx(q8p, V, idxp);
      const uint16_t fillv = nnzp ? idxp[nnzp - 1] : 0;
      for (int k = nnzp; k < nnzp + 8; k++) idxp[k] = fillv;
    }
    {
      PSCOPE(PS_MM_PRIOR);
#ifdef TF_W5
      if (M.prior_wide) {
        if (nnzp > PRIOR_DENSE_NNZ)
          qsparse_dense_f32(M.prior8, q8p, yb);
        else
          qsparse_f32(M.prior8, q8p, idxp, nnzp, yb);
      } else
#endif
      if (nnzp > PRIOR_DENSE_NNZ)
        qsparse4_dense_f32(M.prior, q8p, yb);
      else
        qsparse4_f32(M.prior, q8p, idxp, nnzp, yb);
    }
    {
      PSCOPE(PS_EL_EMBED);
      embed_combine192(tok, yb, x);  // x0 = tok_row + rms_norm(prior_y)
    }

#if TF_AR
    // AttnRes replace mode: no U-Net skips, no residual/token coefficients.
    // Every sub-layer reads h = depth_mix(x0, completed block sums, partial)
    // and its output f goes into the partial sum of the current block.
    ar_n = 0;
    ar_has_part = false;
#if TF_ARV2 >= 5
    ar_qnext = 0;
    ar_part_dot_q = -1;
#endif
    ar_push(x);                                   // source 0 = x0
    for (int l = 0; l < NL; l++) {
      {
        PSCOPE(PS_EL_RESID);
        if (l % ARB == 0 && ar_has_part) {        // block boundary: freeze its sum
          ar_push(ar_part);
          ar_has_part = false;
        }
        ar_read(M.ar_q_mix[l]);                   // x = h for the mixer
      }
      if (!gel && l == NL - 1) {                  // unread row: K/V of the last layer only
        van_layer_kv(M.layer2van[l]);
        t++;
        return;
      }
      std::memset(ar_f, 0, sizeof(ar_f));
      acc = ar_f;
      if (OptModel::KIMI_L[l])
        kimi_layer(M.layer2kimi[l]);
      else
        van_layer(M.layer2van[l]);
      {
        PSCOPE(PS_EL_RESID);
#if TF_ARV2 >= 5
        ar_take_f(2 * l + 1);
#else
        ar_take_f();
#endif
        ar_read(M.ar_q_mlp[l]);                   // x = h for the MLP
      }
      std::memset(ar_f, 0, sizeof(ar_f));
      mlp_block(l);
      {
        PSCOPE(PS_EL_RESID);
#if TF_ARV2 >= 5
        // next read: the final read (2*NL) after the last layer; else layer l+1's mixer read, unless a block
        // boundary pushes the partial first
        ar_take_f(l == NL - 1 ? 2 * NL : ((l + 1) % ARB != 0 ? 2 * l + 2 : -1));
#else
        ar_take_f();
#endif
      }
      acc = x;
    }
    {
      PSCOPE(PS_EL_RESID);
      ar_read(M.ar_q_final);                      // x = final read; rms_norm + unembedding below
    }
    (void)tok;
#else
    for (int l = 0; l < NL; l++) {
      {
        PSCOPE(PS_EL_RESID);
        // skip-add BEFORE the token connection (SPEC 3.3); pairing 11-l
        if (l >= NSKIP)
          add_scaled192(x, M.skip_w[l - NSKIP], skip_store[NL - 1 - l]);
        axpby_tok192(M.rsc[l], x, M.tec[l], tok);
      }
      // The last layer of an unread row: K and V for the ring only, then the
      // step is over.  probs_out and `logits` keep the values of the last step
      // with output, `xn` does not (see van_layer_kv); `t` advances as always.
      if (!gel && l == NL - 1) {
        van_layer_kv(M.layer2van[l]);
        t++;
        return;
      }
      if (OptModel::KIMI_L[l])
        kimi_layer(M.layer2kimi[l]);
      else
        van_layer(M.layer2van[l]);
      mlp_block(l);
      if (l < NSKIP) {
        PSCOPE(PS_EL_RESID);
        std::memcpy(skip_store[l], x, sizeof(float) * D);  // push AFTER block
      }
    }
#endif

    {
      PSCOPE(PS_EL_NQ_FIN);
      rms_norm_quant192(x, xn, M.unembed.s_act, q5);
    }
    {
      PSCOPE(PS_MM_UNEMB);
      qgemv_f32(M.unembed.m, q5, logits);  // rows_padded 208
    }
    {
      PSCOPE(PS_EL_HEAD);
#if TF_PLM
      // xn = the unembedding input; prior_f32 = this step's prior row
      head_plm_softcap_softmax(logits, prior_f32, dot192(M.plm_w, xn), probs208);
#else
      head_softcap_softmax(logits, probs208);
#endif
      std::memcpy(probs_out, probs208, sizeof(float) * V);
    }
    t++;
  }

};

COLD_SIZE_ATTR TransformerOpt::TransformerOpt(const char* weights_path, AttnKind attn)
    : impl(new TransformerOptImpl(weights_path, attn)) {}

COLD_SIZE_ATTR TransformerOpt::~TransformerOpt() = default;

void TransformerOpt::begin_article(int64_t rope_position_offset) {
  impl->begin(rope_position_offset);
}

void TransformerOpt::step(uint8_t token, const uint16_t* prior_f16,
                          float* probs_out) {
  prior_f16_to_f32(prior_f16, impl->prior_f32);  // exact; pads zeroed
  impl->step(token, probs_out);
}

void TransformerOpt::step(uint8_t token, const float* prior205,
                          float* probs_out) {
  std::memcpy(impl->prior_f32, prior205, sizeof(float) * V);
  for (int i = 208; i < 208; ++i) impl->prior_f32[i] = 0.0f;
  impl->step(token, probs_out);
}

void TransformerOpt::step(uint8_t token, const uint16_t* prior_f16,
                          float* probs_out, const PaketChannels& channels) {
  impl->pk = channels;
  prior_f16_to_f32(prior_f16, impl->prior_f32);  // exact; pads zeroed
  impl->step(token, probs_out);
}

bool TransformerOpt::paket_head() const { return impl->M.paket.present; }

const float* TransformerOpt::paket_row(uint8_t token,
                                       const PaketChannels& channels) {
  impl->pk = channels;
  return impl->paket_token_row(token);
}

int TransformerOpt::embedding_rows() const { return impl->M.paket.emb_rows; }

void TransformerOpt::set_gelesen(bool gelesen) { impl->gelesen = gelesen; }

const float* TransformerOpt::last_logits() const { return impl->logits; }

// `xn` is x after the final rms_norm, i.e. exactly the vector fed to
// the unembedding matmul one line later in step().  Nothing is written here.
const float* TransformerOpt::last_hidden() const { return impl->xn; }
int TransformerOpt::hidden_size() { return D; }

AttnKind TransformerOpt::attn_kind() const { return impl->kind; }

}  // namespace opt
}  // namespace fx2
