# rocm_env.sh -- shared ROCm runtime environment for telemetry collection.
# Source (don't execute) this after activating the venv:
#     source "${REPO_ROOT}/scripts/rocm_env.sh"
#
# Why this is needed:
#   - RDC Tier-2 (rdc_bootstrap) loads librdc_bootstrap.so, whose dependencies
#     live in <rocm>/lib; without it on LD_LIBRARY_PATH the bindings fail to
#     load and Tier-2 metrics are silently skipped.
#   - The PyTorch ROCm wheels need libgomp.so, shipped under <rocm>/lib/llvm/lib
#     on cloud images that don't install a system libgomp.
#   - RDC_HOME helps the bindings locate their install root.
#
# Picks the newest /opt/rocm-* (falls back to the /opt/rocm symlink). Safe to
# source on consumer cards too — it only sets paths that exist.

_rocm_root="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1)"
if [[ -z "${_rocm_root}" && -e /opt/rocm ]]; then
    _rocm_root="$(readlink -f /opt/rocm)"
fi

if [[ -n "${_rocm_root}" && -d "${_rocm_root}" ]]; then
    export RDC_HOME="${_rocm_root}"
    export LD_LIBRARY_PATH="${_rocm_root}/lib:${_rocm_root}/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
else
    echo "WARNING: rocm_env.sh could not find a ROCm install under /opt; " \
         "RDC Tier-2 and/or PyTorch ROCm may fail to load." >&2
fi
unset _rocm_root
