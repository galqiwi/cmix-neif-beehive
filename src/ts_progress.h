#ifndef TS_PROGRESS_H_
#define TS_PROGRESS_H_

#include "metablock.h"
#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include "timestamp_algebra.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace ts_progress {

using ts_algebra::Fdiv;
using ts_algebra::DaysFromCivil;
using ts_algebra::CivilFromDays;
using ts_algebra::kEpocheTage;
using ts_algebra::TsZuSek;
using ts_algebra::SekZuTs;
using ts_algebra::Schaetzer;
using ts_algebra::RohFail;
using ts_algebra::TsRohZuSek;
using ts_algebra::SekZuTsRoh;
using ts_algebra::DezimalRoh;
using ts_algebra::ZahlRoh;

struct Stats {
  unsigned long long block_bytes = 0;
  unsigned long long a_zeilen = 0;      // part-A records (= revids)
  unsigned long long b_zeilen = 0;      // part-B records (= timestamps)
  unsigned long long stellen = 0;       // feed points
  unsigned long long bytes = 0;         // fed bytes
  unsigned long long hits = 0;
  unsigned long long ausgelassen = 0;
  long long abs_error = 0;
};

static const unsigned char kNL = 0x0a;
static const unsigned char kTab = 0x09;

class Aut {
 public:
  void Observe(unsigned char b) {
    have_block_ = false;
#if AUX_BUILD
    (void)b; return;
#else
    if (phase_ == kFertig) return;
    if (!block_opened_ &&
        ++guard_pos_ > (unsigned long long)4096)
      Fail("metablock never opened");
    if (metablock::mk.Offen()) block_opened_ = true;
    if (metablock::mk.BZeileFertig()) {
      const size_t k = metablock::mk.BRang();
      const long long t_k = metablock::mk.BT();
      if (k > 0) {
        const long long fehler = t_k - letzte_prognose_;
        if (fehler == 0) ++st_.hits;
        st_.abs_error += fehler < 0 ? -fehler : fehler;
      }
      sch_.Beobachte(Rev(k), t_k);
      ++st_.b_zeilen;
      const size_t kn = k + 1;
      if (kn < metablock::mk.RangZahl()) Einspeisen(kn);
    }
    if (metablock::mk.NachlaufEndet()) {
      st_.a_zeilen = metablock::mk.GetStats().a_zeilen;
      st_.block_bytes = metablock::mk.GetStats().block_bytes;
      part_b_done_ = true;
      phase_ = kFertig;
      Bericht("Blockende");
    }
    return;
#endif
  }

  int TakeBlock(const unsigned char** p) {
    if (!have_block_) return 0;
    have_block_ = false;
    *p = block_.data();
    return (int)block_.size();
  }

  void Finish() {}

  void Report(const char* page_index) const {
    (void)page_index;
  }

  const Stats& GetStats() const { return st_; }

 private:
  enum Phase { kBlock = 0, kFertig = 1 };

  static void Fail(const char* what) {
    fprintf(stderr, "timestamps: %s\n", what);
    fflush(stderr);
    exit(1);
  }

  static bool Oben(size_t k) {
    return (long long)k >= (long long)87813;
  }

  long long Rev(size_t k) const {
    return metablock::mk.RevidNachRang(k);
  }

  // ---- Einspeisung -------------------------------------------------------
  void Einspeisen(size_t kn) {
    block_.clear();
    letzte_prognose_ = sch_.Prognose(Rev(kn));
    if (Oben(kn)) {
      Dezimal(letzte_prognose_ - (sch_.Leer() ? 0LL : sch_.LetztesT()));
      block_.push_back(kTab);
      have_block_ = !block_.empty();
      if (have_block_) { ++st_.stellen; st_.bytes += block_.size(); }
      return;
    }
    if (!Oben(kn)) {
      long long thk = letzte_prognose_;
      if (thk < 0) thk = 0;
      {
        int yk; long long mk, sk;
        SekZuTs(thk, &yk, &mk, &sk);
        if (yk < 0) yk = 0;
        if (yk > 99) yk = 99;
        if (mk < 0) mk = 0;
        if (mk > 9999) mk = 9999;
        if (sk < 0) sk = 0;
        if (sk > 999999) sk = 999999;
        TsFest((long long)yk, 1);
        TsFest(mk, 2);
        TsFest(sk, 3);
      }
      block_.push_back(kTab);
      have_block_ = !block_.empty();
      if (have_block_) { ++st_.stellen; st_.bytes += block_.size(); }
      return;
    }
    long long th = letzte_prognose_;
    int yy; long long mt, sek;
    if (th < 0) th = 0;
    SekZuTs(th, &yy, &mt, &sek);
    if (yy < 0) yy = 0;
    if (yy > 99) yy = 99;
    if (mt < 0) mt = 0;
    if (mt > 9999) mt = 9999;
    if (sek < 0) sek = 0;
    if (sek > 999999) sek = 999999;
    MpFest((long long)yy, 1);
    MpFest(mt, 2);
    MpFest(sek, 3);
    block_.push_back(kTab);
    have_block_ = !block_.empty();
    if (have_block_) { ++st_.stellen; st_.bytes += block_.size(); }
  }

  void TsFest(long long v, int n) {
    unsigned char z[8];
    for (int i = n - 1; i >= 0; --i) {
      z[i] = (unsigned char)((unsigned)0x80 + (unsigned)(v % 100));
      v /= 100;
    }
    for (int i = 0; i < n; ++i) block_.push_back(z[i]);
  }

  void MpFest(long long v, int n) {
    unsigned char z[8];
    for (int i = n - 1; i >= 0; --i) {
      z[i] = (unsigned char)((unsigned)0x80 + (unsigned)(v % 100));
      v /= 100;
    }
    for (int i = 0; i < n; ++i) block_.push_back(z[i]);
  }

  void Dezimal(long long v) {
    unsigned char z[24];
    int n = 0;
    if (v < 0) { block_.push_back(0x2d); v = -v; }
    do { z[n++] = (unsigned char)(0x30 + (int)(v % 10)); v /= 10; } while (v);
    while (n > 0) block_.push_back(z[--n]);
  }

  void Bericht(const char*) {}

  Phase phase_ = kBlock;
  bool block_opened_ = false, line_has_tab_ = false, part_b_done_ = false, have_block_ = false;
  unsigned long long guard_pos_ = 0;
  std::vector<unsigned char> block_;
  Schaetzer sch_;
  long long letzte_prognose_ = 0;
  Stats st_;
};

inline Aut aut;

static inline void KalenderWacht() {
  if (DaysFromCivil(2001, 1, 1) != kEpocheTage) {
    fprintf(stderr, "guard - calendar constant wrong\n");
    exit(1);
  }
}

}

#endif
