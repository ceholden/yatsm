#!/bin/bash
# Based off instructions from:
# http://www.steveklabnik.com/automatically_update_github_pages_with_travis_example/

set -o errexit -o nounset

KEY_FILE=.deploy_key

REPO=$(git config remote.origin.url)
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
REV=$(git rev-parse --short HEAD)

SRC_BRANCH=$(git rev-parse --abbrev-ref HEAD)
DST_BRANCH=gh-pages

# CONFIGURE GIT
ENCRYPTED_KEY_VAR="encrypted_${ENCRYPTION_LABEL}_key"
ENCRYPTED_IV_VAR="encrypted_${ENCRYPTION_LABEL}_iv"
ENCRYPTED_KEY=${!ENCRYPTED_KEY_VAR}
ENCRYPTED_IV=${!ENCRYPTED_IV_VAR}
openssl aes-256-cbc \
    -K $ENCRYPTED_KEY \
    -iv $ENCRYPTED_IV \
    -in ${KEY_FILE}.enc \
    -out ${KEY_FILE} \
    -d
chmod 600 ${KEY_FILE}
eval `ssh-agent -s`
ssh-add ${KEY_FILE}

git config user.email $COMMIT_AUTHOR_EMAIL
git config user.name $COMMIT_AUTHOR_NAME

# Find and clean
docs=$(dirname $(readlink -f $0))/../
cd $docs

rm -rf build

# Create branch directory and grab Git repo
mkdir -p yatsm_docs/$SRC_BRANCH && cd yatsm_docs
git init
git remote add origin $SSH_REPO
git fetch origin $DST_BRANCH
git checkout $DST_BRANCH
cd ..

make html
cp -R build/html/* yatsm_docs/$SRC_BRANCH/

cd yatsm_docs/
git add -A .
git commit -m "Rebuild $DST_BRANCH docs on ${SRC_BRANCH} at ${REV}"
git push --quiet origin HEAD:$DST_BRANCH
