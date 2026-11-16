#!/usr/bin/env bash
# Phase C — run Zeo++ on every CIF with probe settings matched to the Mercado
# (Chem. Mater. 2018) training-set descriptors.
#
# Zeo++ `network` is SINGLE-THREADED per structure, so we parallelize ACROSS
# structures with xargs -P (one process per CIF). Set JOBS to your core count.
#
# Usage:  ./scripts/run_zeopp.sh <cifs_dir> <network_binary> [out_dir] [jobs]
#   ./scripts/run_zeopp.sh ./curated_cifs /path/to/zeo++/network ./zeo_out 12
#
# Produces, per structure <name>:  <name>.res  <name>.sa  <name>.vol
#
# Probe settings (documented for the paper Methods):
#   - pore diameters Di/Df/Dif : geometric, no probe        -> network -ha -res
#   - accessible surface area  : methane probe r=1.86 Å      (TraPPE CH4 σ=3.73 Å /2)
#   - void fraction            : helium probe r=1.0 Å        (~He Widom void fraction)
set -uo pipefail

CIFS="${1:?cif dir}"
NETWORK="${2:?path to zeo++ network binary}"
OUT="${3:-zeo_out}"
JOBS="${4:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p "$OUT"

export NETWORK OUT
export CH4_PROBE=1.86      # methane probe radius [Å] (surface area)
export HE_PROBE=1.0        # helium  probe radius [Å] (void fraction)
export SAMPLES=2000        # Monte-Carlo samples per atom (SA)
export VOL_SAMPLES=50000   # samples for accessible volume

process_one() {
  local cif="$1"
  local name; name=$(basename "$cif" .cif)
  # skip if already done (idempotent / resumable)
  if [ -s "$OUT/$name.res" ] && [ -s "$OUT/$name.sa" ] && [ -s "$OUT/$name.vol" ]; then
    return
  fi
  "$NETWORK" -ha -res "$OUT/$name.res" "$cif" >/dev/null 2>&1 || { echo "res FAIL $name"; return; }
  "$NETWORK" -ha -sa  "$CH4_PROBE" "$CH4_PROBE" "$SAMPLES"     "$OUT/$name.sa"  "$cif" >/dev/null 2>&1 || echo "sa FAIL $name"
  "$NETWORK" -ha -vol "$HE_PROBE"  "$HE_PROBE"  "$VOL_SAMPLES" "$OUT/$name.vol" "$cif" >/dev/null 2>&1 || echo "vol FAIL $name"
}
export -f process_one

total=$(ls "$CIFS"/*.cif 2>/dev/null | wc -l)
echo "并行 Zeo++：$total 个结构，$JOBS 路并发 -> $OUT/"
ls "$CIFS"/*.cif | xargs -P "$JOBS" -I {} bash -c 'process_one "$@"' _ {}
done_n=$(ls "$OUT"/*.res 2>/dev/null | wc -l)
echo "完成: $done_n/$total 个结构有 .res 输出"
