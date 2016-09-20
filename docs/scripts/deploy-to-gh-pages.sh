#!/bin/bash
# Based off instructions from:
# http://www.steveklabnik.com/automatically_update_github_pages_with_travis_example/

set -o errexit -o nounset

if [ "$TRAVIS_BRANCH" != "master" ]; then
    echo "This commit was made against the $TRAVIS_BRANCH and not the master! No deploy!"
    exit 0
fi
rev=$(git rev-parse --short HEAD)

cd $(dirname $(readlink -f $0))/../

rm -rf build/

make html
cd build/html/

git init
git config user.email "ceholden@gmail.com"
git config user.name "Chris Holden"
git remote add upstream "https://${GITHUB_TOKEN}@${GITHUB_REPO}"
git fetch upstream
git reset upstream/gh-pages
git add -A .
git commit -m "Rebuild gh-pages docs at ${rev}"
git push --quiet upstream HEAD:gh-pages
