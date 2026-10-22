// Research instrumentation and switches (environment variables, read once).
// With no variable set, the program behaves exactly as the original.
#ifndef CMX_EXP_H
#define CMX_EXP_H

#include <cstdlib>

namespace cmx {

inline long long EnvLL(const char* name, long long dflt) {
  const char* v = std::getenv(name);
  return v ? std::atoll(v) : dflt;
}

struct Opts {
  long long log_every;   // CMX_LOG_EVERY: print a CMX line every N coder bytes (0 = off)
  long long stop_at;     // CMX_STOP_AT: stop coding at this coder byte (0 = never)
  int nolstm;            // CMX_NOLSTM: byte LSTM bypassed (PPMd distribution passed through)
  int bitl_off;          // CMX_BITL_OFF: BITLSTM correction off
  int tfmix;             // CMX_TFMIX: extra layer-0 mixer with transformer-confidence context
  int tffinal;           // CMX_TFFINAL: final mixer context includes transformer confidence
  int tau;               // CMX_TAU: adaptive-depth threshold x1000 (0 = compiled 8000)
  int lr0, lr1;          // CMX_LR0 / CMX_LR1: layer-0 / final mixer learning-rate multiplier x1000 (0 = 1000)
  int bytecost;          // CMX_BYTECOST: write the per-coder-byte code length (bits, float32) to bytecost.f32 at CMX_STOP_AT
  int metamix;          // CMX_METAMIX: inside the metadata block the last layer-0 mixer selects by (part, field, byte in field, partial byte)
  int metafin;          // CMX_METAFIN: inside the metadata block the final mixer selects by (part, field, byte in field)
  int aliasfix;          // CMX_ALIASFIX: after transformer-only alias rows, use the row of the last alias as the transformer slot
  Opts() {
    log_every = EnvLL("CMX_LOG_EVERY", 0);
    stop_at = EnvLL("CMX_STOP_AT", 0);
    nolstm = (int)EnvLL("CMX_NOLSTM", 0);
    bitl_off = (int)EnvLL("CMX_BITL_OFF", 0);
    tfmix = (int)EnvLL("CMX_TFMIX", 0);
    tffinal = (int)EnvLL("CMX_TFFINAL", 0);
    tau = (int)EnvLL("CMX_TAU", 0);
    lr0 = (int)EnvLL("CMX_LR0", 0);
    lr1 = (int)EnvLL("CMX_LR1", 0);
    aliasfix = (int)EnvLL("CMX_ALIASFIX", 0);
    bytecost = (int)EnvLL("CMX_BYTECOST", 0);
    metamix = (int)EnvLL("CMX_METAMIX", 0);
    metafin = (int)EnvLL("CMX_METAFIN", 0);
  }
};

// Switches act only inside the main coding run (set by RunCompression right
// before the Predictor is built); helper runs (dictionary derivation, block
// decoding in child processes) see the defaults.
inline bool& Main() {
  static bool m = false;
  return m;
}
// only the top-level "-e" process may switch them on (child -C/-D runs never)
inline bool& Allowed() {
  static bool a = false;
  return a;
}
#ifdef CMX_FINAL
// Final entry build: the chosen modelling switches are compile-time constants that act in EVERY
// process (compression and decompression alike); environment values are ignored and all
// instrumentation is off (except the CMX_TRACE_DIR trace writer in bit_head.cpp).  -DCMX_FINAL -DCMX_F_NOLSTM=1 -DCMX_F_LR0=2000 -DCMX_F_TFMIX=1 ...
#ifndef CMX_F_NOLSTM
#define CMX_F_NOLSTM 0
#endif
#ifndef CMX_F_TFMIX
#define CMX_F_TFMIX 0
#endif
#ifndef CMX_F_LR0
#define CMX_F_LR0 0
#endif
#ifndef CMX_F_LR1
#define CMX_F_LR1 0
#endif
inline bool& Main();
inline const Opts& O() {
  static const Opts zf = [] { Opts z; z.log_every = 0; z.stop_at = 0; z.nolstm = 0; z.bitl_off = 0; z.tfmix = 0; z.tffinal = 0;
                              z.tau = 0; z.lr0 = 0; z.lr1 = 0; z.aliasfix = 0; z.bytecost = 0; z.metamix = 0; z.metafin = 0; return z; }();
  static const Opts f = [] { Opts z; z.log_every = 0; z.stop_at = 0; z.nolstm = CMX_F_NOLSTM; z.bitl_off = 0;
                             z.tfmix = CMX_F_TFMIX; z.tffinal = 0; z.tau = 0; z.lr0 = CMX_F_LR0; z.lr1 = CMX_F_LR1;
                             z.aliasfix = 0; z.bytecost = 0; z.metamix = 0; z.metafin = 0; return z; }();
  // only the main text stream (set before its Predictor in compression AND decompression);
  // blob-form side blocks (-C/-D children) keep the original model, the byte LSTM matters there
  return Main() ? f : zf;
}
#else
inline const Opts& O() {
  static Opts o;
  static const Opts zero = [] { Opts z; z.log_every = 0; z.stop_at = 0; z.nolstm = 0; z.bitl_off = 0; z.tfmix = 0; z.tffinal = 0; z.tau = 0; z.lr0 = 0; z.lr1 = 0; z.aliasfix = 0; z.bytecost = 0; z.metamix = 0; z.metafin = 0; return z; }();
  return Main() ? o : zero;
}
#endif

// code-length accounting (bits), filled by the encoder
struct Acct {
  double final_bits = 0, tf_bits = 0, pre_bits = 0;
  unsigned long long nbits = 0;
};
inline Acct& A() {
  static Acct a;
  return a;
}

// transformer confidence of the current byte (set when the transformer row arrives)
struct TfConf {
  int ex = 0;          // most probable byte
  float best = 0.0f;   // its probability
  float second = 0.0f; // second best
  int valid = 0;
};
inline TfConf& T() {
  static TfConf t;
  return t;
}

// field state of the metadata block, advanced by every CODED byte (runner)
struct MetaState {
  long long line = 0;     // line index from the head of the stream
  int field = 0, fpos = 0;
  int in = 1;             // 1 while inside the block
  unsigned int Ctx() const {
    if (!in) return 0;
    const long long N = 243425;
    int part = line < 3 ? 0 : line < 3 + N ? 1 : (line - 3 - N) < 87813 ? 2 : 3;
    const int f = field > 3 ? 3 : field, b = fpos > 15 ? 15 : fpos;
    return 1u + (unsigned int)((part * 4 + f) * 16 + b);
  }
  void Byte(unsigned char c) {
    if (!in) return;
    if (c == 0x0a) { ++line; field = 0; fpos = 0; if (line >= 3 + 2 * 243425LL) in = 0; }
    else if (c == 0x09) { ++field; fpos = 0; }
    else ++fpos;
  }
};
inline MetaState& M() {
  static MetaState m;
  return m;
}

}  // namespace cmx

#endif
