#!/bin/bash
# Set up the B200-capable PRIME stack: clone upstream verl at the pinned commit, check
# out the recipe/prime submodule at its pinned commit, and apply the recipe port patch.
# We do NOT vendor verl into this repo — it stays a clean upstream checkout + one patch.
#
# Usage: bash together_prime/setup.sh [dest_dir]   (default: ./verl next to this script)
set -euxo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/VERSIONS"

DEST="${1:-$HERE/verl}"

if [ ! -d "$DEST/.git" ]; then
    git clone "$VERL_REPO" "$DEST"
fi
cd "$DEST"
git fetch --depth 1 origin "$VERL_COMMIT"
git checkout "$VERL_COMMIT"

# recipe/ is a submodule (verl-project/verl-recipe); init then pin to the exact commit
# the port patch was diffed against.
git submodule update --init --depth 1 recipe
git -C recipe fetch --depth 1 origin "$RECIPE_COMMIT"
git -C recipe checkout "$RECIPE_COMMIT"

# Apply the PRIME recipe port (bucket A — the only code change).
git -C recipe apply --verbose "$HERE/patches/recipe-prime-port.patch"

echo "SETUP_DONE: verl@$VERL_COMMIT + recipe@$RECIPE_COMMIT (+ recipe-prime-port.patch) at $DEST"
