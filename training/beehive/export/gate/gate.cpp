// gate.cpp <container.c4> : decode with the engine's own loader and print "name sha-ish digest" per tensor (FNV-1a-64 over dtype, shape, bytes)
#include "weights_io.h"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
using namespace fx2;
int main(int argc, char** argv) {
  WeightsFile wf = WeightsFile::load_compressed(argv[1]);
  std::vector<std::string> names;
  for (auto& kv : wf.tensors) names.push_back(kv.first);
  std::sort(names.begin(), names.end());
  for (auto& n : names) {
    const WTensor& t = wf.tensors.at(n);
    unsigned long long h = 0xcbf29ce484222325ULL;
    auto mix = [&](unsigned char b) { h ^= b; h *= 0x100000001b3ULL; };
    mix(t.dtype); for (auto s : t.shape) for (int k = 0; k < 4; k++) mix((s >> (8 * k)) & 255);
    for (auto b : t.data) mix(b);
    printf("%s %016llx\n", n.c_str(), h);
  }
  return 0;
}
