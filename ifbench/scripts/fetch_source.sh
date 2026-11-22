#!/usr/bin/env bash
# Re-create the pinned IFBench checkout and the vendored NLTK data, verifying every hash.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=488918665d53f867fa908a293769ec6e71e74aec
mkdir -p third_party
if [ ! -d third_party/IFBench/.git ]; then
  git clone -q https://github.com/allenai/IFBench.git third_party/IFBench
fi
git -C third_party/IFBench checkout -q "$COMMIT"
test "$(git -C third_party/IFBench rev-parse HEAD)" = "$COMMIT"

mkdir -p third_party/nltk_zips
declare -A SHA=(
  [tokenizers/punkt_tab]=e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106
  [tokenizers/punkt]=51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec
  [corpora/stopwords]=48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b
  [taggers/averaged_perceptron_tagger_eng]=6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b
)
for pkg in "${!SHA[@]}"; do
  name=$(basename "$pkg"); dir=$(dirname "$pkg")
  zip=third_party/nltk_zips/$name.zip
  [ -f "$zip" ] || curl -sSf -o "$zip" "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/$pkg.zip"
  echo "${SHA[$pkg]}  $zip" | sha256sum -c --quiet
  mkdir -p "third_party/nltk_data/$dir"
  [ -d "third_party/nltk_data/$pkg" ] || unzip -q "$zip" -d "third_party/nltk_data/$dir"
done
# NLTK >= 3.10 refuses to read data that other accounts can write.
chmod -R go-w third_party/nltk_data
echo "IFBench $COMMIT and NLTK data ready."
