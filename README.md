# cmix-neif-beehive

A Hutter Prize entry: a lossless compressor for `enwik9` (the first 10^9 bytes of the English Wikipedia
dump used by the prize). It is a derivative of **cmix-neif-1.0** by Marcus Wölk
(`github.com/marwoelk/cmix-neif-1.0`, commit `1c9ea5ed57ec`), distributed under the GNU General Public
License version 3 (see `LICENSE`, `NOTICE.md`).

The entry keeps the whole cmix-neif-1.0 compressor (preprocessing, dictionary, context-mixing models, PPMd,
BITLSTM, arithmetic coder, self-extracting archive) and replaces its pre-trained transformer with a new model,
adds three compiled-in context-mixing settings, and retunes one memory-prefetch constant. Everything else is
unchanged.

The upstream README, referenced by section number from many source files, `CHANGES.md`, `NOTICE.md` and
`environment.txt` as "`README.md`", is preserved as **`README_cmix-neif-1.0.md`**. Read those
references as pointing to that file.

---

## 1. Result

| quantity | value |
|---|---|
| `comp9` (compressor, `S` counts it once) | 3,941,780 B, sha256 `db2ddfb9955f530f116b482491dea4370ebf455490de35ed1c8b5163dcd416d0` |
| `archive9` (self-extracting archive of `enwik9`) | 90,763,792 B, sha256 `bffe73800d53606a27152461fac1478bcf38cc80394cce4dbabc3471f6ea56b8` |
| `comp9.args` (the command line, counted as in cmix-neif-1.0) | 19 B |
| **`S` = comp9 + archive9 + comp9.args** | **94,705,591 B** |
| against cmix-neif-1.0 (`S` = 95,914,051 B, published, the newest entry) | **-1,208,460 B = -1.260 %** |
| against the last awarded record, fx2-cmix-transformer (L = 100,424,672 B, as cited in `README_cmix-neif-1.0.md`) | -5,719,081 B = -5.695 % |

The three files and their checksums are release assets of tag `cmix-neif-beehive`
(<https://github.com/galqiwi/cmix-neif-beehive/releases/tag/cmix-neif-beehive>):
[`comp9`](https://github.com/galqiwi/cmix-neif-beehive/releases/download/cmix-neif-beehive/comp9), [`archive9`](https://github.com/galqiwi/cmix-neif-beehive/releases/download/cmix-neif-beehive/archive9), [`comp9.args`](https://github.com/galqiwi/cmix-neif-beehive/releases/download/cmix-neif-beehive/comp9.args),
[`SHA256SUMS`](https://github.com/galqiwi/cmix-neif-beehive/releases/download/cmix-neif-beehive/SHA256SUMS) (`sha256sum -c SHA256SUMS` in the download directory).

`archive9` was produced twice, by two independent full encodes on two machines (an AMD EPYC 9755 and the
EPYC 7R13 host below), with byte-identical coded streams; the EPYC 7R13 encode wrote exactly the `archive9` above.

## 2. Hardware, time and resources (measured)

Both programs were run solo: one core, pinned with `taskset`, nothing else running on the host.

| | compression `./comp9 -e enwik9 archive9` | decompression `./archive9` |
|---|---|---|
| machine | AWS EC2, AMD EPYC 7R13, 8 vCPU (4 cores x 2 threads), 30 GiB RAM, 16 MB L3, Ubuntu 24.04.5 LTS, Linux 7.0.0-1014-aws | a second instance of the same type |
| Geekbench 5 single-core `T` | 1292 ([browser.geekbench.com/v5/cpu/24736264](https://browser.geekbench.com/v5/cpu/24736264)), multi-core 5520 | not measured separately; same instance type, so the same limit is applied |
| time limit 70,000 / T | 54.18 h | 54.18 h |
| **wall clock** | **52:08:41 = 52.14 h** (96.2 % of the limit) | **50:57:23 = 50.96 h** (94.0 % of the limit) |
| user / system CPU | 186,058 s / 1,477 s | 181,698 s / 1,547 s |
| peak resident set | 10,223,372 KiB = 9.75 GiB | 10,223,544 KiB = 9.75 GiB |
| result | exit 0, `archive9` as in section 1 | exit 0, output byte-identical to `enwik9` (`cmp`; sha256 `159b8535...44bc`) |

**Memory.** The peak resident set is 9.75 GiB, below 10 GiB and the same as cmix-neif-1.0's (9.75 GiB, see
`README_cmix-neif-1.0.md`). In decimal bytes it is 10.47 x 10^9 B, so under a decimal reading of "10 GB" it is
0.47 GB above, exactly as the published entry.

**Disk.** The working directory holds up to ~20.9 GB of files during compression: `ppm.page` 14.68 GB (a
sparse page file of the PPMd model; physically only ~0.1-0.2 GB were allocated in our runs), the
intermediate transforms of `enwik9` (~5.2 GB in total) and `enwik9` itself. Decompression holds ~15 GB of
files at 17 % (0.37 GB physically allocated) and writes the 1 GB output. Provide **at least 21 GB free** in
the working directory, as for cmix-neif-1.0.

**Round trip.** The archive above decodes to a file byte-identical to `enwik9`: on the EPYC 7R13 host (the timed
run above) and, independently, on an AMD EPYC 9755 (35:39:12). An `archive9` built with the decoder of the
pre-prefetch build (section 3.4; same coded stream and size, sha256 `c35c3dc8…`) also decodes exactly (35:50:44 on
the EPYC 9755).

## 3. What changed against cmix-neif-1.0

The complete change is the single commit on top of the upstream commit (`git diff 1c9ea5e cmix-neif-beehive`).
The training of the weights is in `training/README.md` section 12.

### 3.1 Transformer model
The cmix-neif transformer (16 layers, d_model 192, 208-symbol vocabulary, Kimi-delta-attention and windowed
attention layers, PPMd prior input) with:
- AttnRes in replace mode, block size 2: a learned depth-mixing of block outputs replaces the skip and
  token re-injection connections;
- a per-token PPMd-prior logit-mixing output head;
- one key/value head for the three query heads (grouped-query attention 3:1), with a fused shared-KV attention
  kernel;
- value embeddings on layers 3, 7, 11 and 15;
- MLP width 1024;
- attention window 1536 (was 2048);
- mixed-precision quantized weights: 99.52 % of rows 4-bit, 0.47 % 2-bit, 0.0025 % 3-bit, chosen by Fisher
  information under a size budget, with a differentiable container-size term during quantization-aware training.

Strict deployed cross-entropy: **1.27404 bits/token** (cmix-neif-1.0: 1.2915). Weight container
`weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4`: **3,649,198 B** (cmix-neif-1.0: 3,716,667 B), sha256
`141ee13c316811157bea167846b3f45472affff7adaa1b8b263e30621e490885` (container format v4, `config.ints[6]` = 1536).
The file keeps the upstream name because the build and the fingerprint stamp refer to it; its FNV-1a-64
fingerprint `0xf6bec91b6fe34355` is stamped in `src/transformer.h`. The weights are stored directly in Git.

### 3.2 Inference engine (`src/tf1/`)
Kernels for the new model parts, enabled by defines appended to `TF1_CXXFLAGS` in the makefile:
- AttnRes replace mode, block size 2 (`TF_AR=2`), and the per-token PPMd-prior logit-mixing head (`TF_PLM=1`);
- one key/value head (`TF_KVH=1`), value embeddings (`TF_VE=1`, sentinel clamp `TF_VE_SENT=1`), mixed-width weight
  kernels (`TF_W5`), MLP width 1024 (`TF_DMLP=1024`);
- shared-KV attention (`TF_ATTN_SHKV`, `attn.cpp`): the three query heads read one KV ring through a fused scan;
  outputs are bit-identical to the per-head kernel (exact integer dot products, identical float operations), about
  3 % faster for the whole compressor;
- attention window 1536 (`ATTN_WIN` in `attn.h`), matched by `config.ints[6]` of the container.

The container loader (`src/tf1/weights_io_compressed.cpp`) decodes the mixed-precision row format.

### 3.3 Context-mixing side
Three settings compiled in through the makefile (`-DCMX_FINAL -DCMX_F_NOLSTM=1 -DCMX_F_LR0=2000
-DCMX_F_TFMIX=1`, added after every `-DUPDATE_LIMIT=3000`): byte LSTM off (the PPMd distribution is passed through),
layer-0 mixer learning rate x2, and a transformer-confidence context for the last layer-0 mixer. Files:
`src/cmx_exp.h` (new), `src/predictor.cpp`, `src/mixer/byte-mixer.cpp`, `src/context-manager.h`,
`src/coder/encoder.cpp`, `src/runner.cpp`, `src/bit_head.cpp`, `src/tf1/weights_io_compressed.cpp`.

The settings are compile-time constants in `comp9` and in the decompressor it constructs, so compression and
decompression always use the same model; no environment variable selects them. Research builds without
`CMX_FINAL` read them from the environment and honour them only in `-e` mode, so a released binary must be a
`CMX_FINAL` build.

### 3.4 Prefetch distance
`src/tf1/opt/qmat_dense.cpp`: the software-prefetch distance of the dense int8 weight kernels (`run_g8`,
`run_g4_768`, `run_packed`) goes from 2 KB to 6 KB. The dense weights (~6 MB per token) stream from L3; on the
EPYC 7R13 the deeper prefetch measured -4.1 to -6.3 % engine time (`Tf1Engine::Step` over the real coder stream
with a stand-in for the CM's memory traffic, 3 interleaved runs per variant). 8 KB was similar, 12 KB slower; the
sparse and attention prefetch distances, a spill-free `pv_int_pairs3` and nibble-packed dense weights were
neutral or slower and are not included.

Prefetches only: outputs and archive bytes are unchanged. The pre-prefetch build (this tree without this change)
is `comp9` 3,941,780 B, sha256 `901471238d1e1c78520b2f83030da52aeae42cae694dcbc964e2411d2b6ac5fd`; on `in16`
(section 4) both builds give the same `out16`.

### 3.5 Inherited unchanged
The dictionary and its derivation, preprocessing, article order, PPMd, the packer and the build procedure.
- BITLSTM: weights `src/bit_head_weights.h` byte-identical to cmix-neif-1.0 (sha256 `c5977bd52281d05d…`), not
  retrained; `src/bit_head.h` unchanged. `src/bit_head.cpp` adds only a research trace writer (namespace
  `bit_head_trace`) that is dormant unless the environment variable `CMX_TRACE_DIR` is set (then it writes
  multi-GB trace files); it does not change outputs.
- Byte LSTM: present but disabled by `CMX_F_NOLSTM=1`.
- PGO profile (`pgo_data/default.profdata`): the upstream one. It no longer covers every changed function exactly,
  which affects speed only, not correctness.
- `dictionary/`, `prof_input/`, `src/models/fxcmv1.cpp` and the cmix-neif `training/` sources are unchanged.

## 4. Build, compress, decompress

Ubuntu 24.04 LTS, x86-64 with AVX2/FMA. The toolchain is pinned to an Ubuntu archive snapshot; the procedure is
cmix-neif-1.0's, unchanged.

```bash
bash install.sh                       # once; needs sudo and network; clang/lld/llvm 18.1.3 from snapshot 20260927T160000Z
mkdir ../build && cd ../build
bash ../cmix-neif-beehive/build.sh    # offline; writes ./comp9
sha256sum comp9                       # db2ddfb9955f530f116b482491dea4370ebf455490de35ed1c8b5163dcd416d0, 3,941,780 B
curl -LO https://github.com/galqiwi/cmix-neif-beehive/releases/download/cmix-neif-beehive/comp9 --output-dir /tmp && cmp comp9 /tmp/comp9   # the released binary of section 1
```

Two clean builds give the same `comp9`. The same environment rebuilds the published cmix-neif-1.0 source to its
release binary, which checks the toolchain:

```bash
git worktree add ../baseline upstream-cmix-neif-1.0
mkdir ../build-baseline && cd ../build-baseline && bash ../baseline/build.sh
sha256sum comp9                       # 8b07d85f0a374b55688fd08d1a464222f766efbb5bfefc2bc94d6abb201133d7 (cmix-neif-1.0)
```

Ubuntu clang 18.1.3 and nix clang 18.1.8 produce byte-identical archives for cmix-neif-1.0 and for this engine on a
16 MB test input.

Compress (one core, >= 21 GB free disk, ~10 GiB RAM):

```bash
mkdir enc && cp comp9 enc/ && cp /path/to/enwik9 enc/ && cd enc
./comp9 -e enwik9 archive9            # writes archive9 (90,763,792 B)
```

Decompress, in an otherwise empty directory (`archive9` unpacks its own decoder, dictionary and weights; the
`archive9` of section 1 is the release asset):

```bash
mkdir dec && cp archive9 dec/ && cd dec
./archive9                            # writes enwik9_uncompressed, identical to enwik9
cmp enwik9_uncompressed /path/to/enwik9
```

A bounded check without the full run: `./comp9 -c dictionary/english.dic in16 out16`, where `in16` is the first
16,000,000 B of `.ready4cmix` (the 888,096,401 B preprocessed `enwik9` that `./comp9 -e enwik9 archive9` writes into
its working directory once the console prints `preprocessing enwik9 ... s.`, about 5.5 minutes into the run on the
EPYC 7R13; the run can be stopped then: `head -c 16000000 .ready4cmix > in16`, sha256
`7b827308fcb2247d4d246cc5f7bd434947b11ce66e0e41f675b2ccdee3f64f26`), gives `out16` sha256
`d94eecc6db4398043de98bac65efc777070792d834ed7caf87f6f42732bb9bec`; `./comp9 -d dictionary/english.dic out16 dec16`
restores `in16` exactly.

## 5. Reproducing the weights

`training/beehive/` (described in `training/README.md` section 12) contains the training code, configurations, a
one-command reproduction script
(`scripts/reproduce.sh`), the exporter and the records of the shipped run: bf16 training from random
initialization (8 GPUs, ~40 h), Fisher bit allocation, 64-epoch quantization-aware finish with the size term,
one fp32 epoch, export at window 1536. From the fp32 checkpoint on, the export reproduces the shipped container
byte for byte; the GPU stages reproduce the recipe, not bit-identical bytes. Nothing in `training/` is compiled
into `comp9` or `archive9`.

## 6. Repository and release package

- Git history: the upstream commit `1c9ea5e` (Marcus Wölk, tag `upstream-cmix-neif-1.0`), then exactly one
  commit with all changes (tag `cmix-neif-beehive`).
- GitHub release `cmix-neif-beehive` (<https://github.com/galqiwi/cmix-neif-beehive/releases/tag/cmix-neif-beehive>):
  the tested `comp9`, the `archive9` of section 1, `comp9.args` and `SHA256SUMS` over the three. `build.sh` on
  tag `cmix-neif-beehive` gives the released `comp9` byte for byte.
- Release asset `cmix-neif-beehive-logs.tar.gz` (53,234 B, sha256
  `2288fd3f8632725354ed27c8a0763997a0b5e242fb449122f1fb8d1b4991cd49`): the benchmark and verification logs of
  sections 1 and 2 (Geekbench, build, compression, decompression, round trips); `logs/README.txt` is the index and
  says what is redacted.

## 7. Credits

cmix-neif-1.0 and everything it inherits (cmix, cmix-hp, fast-cmix, fx-cmix, fx2-cmix, cmix-lex,
fx2-cmix-transformer and the other components named in `NOTICE.md` and `training/ATTRIBUTION.md`) are the work of
their authors and keep their copyright and licence notices. The new transformer model, its training, the engine and
context-mixing changes, the prefetch tuning and the verification were done in October
2026 (`NOTICE.md`, section "Changes in cmix-neif-beehive").
