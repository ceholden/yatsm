#!/bin/bash
# Based off instructions from:
# http://www.steveklabnik.com/automatically_update_github_pages_with_travis_example/

set -o errexit -o nounset

DST_BRANCH=gh-pages

git config user.email "ceholden@gmail.com"
git config user.name "Chris Holden"

# if [ "$TRAVIS_BRANCH" != "master" ]; then
#     echo "This commit was made against the $TRAVIS_BRANCH and not the master! No deploy!"
#     exit 0
# fi
rev=$(git rev-parse --short HEAD)
SRC_BRANCH=$(git rev-parse --abbrev-ref HEAD)

docs=$(dirname $(readlink -f $0))/../
cd $docs

rm -rf build

mkdir -p yatsm_docs/$SRC_BRANCH && cd yatsm_docs
git init
git remote add origin "https://${GITHUB_TOKEN}@${GITHUB_REPO}"
git fetch origin $DST_BRANCH
git checkout $DST_BRANCH
cd ..

make html
cp -R build/html/* yatsm_docs/$SRC_BRANCH/

cd yatsm_docs/
git add -A .
git commit -m "Rebuild $DST_BRANCH docs on ${SRC_BRANCH} at ${rev}"
git push --quiet origin HEAD:$DST_BRANCH
