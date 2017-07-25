#!/bin/bash
# Based off instructions from:
# https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

set -o errexit -o nounset

PACKAGE="yatsm"

DOCS=$(dirname $(readlink -f $0))/../
APIDOC="${DOCS}/source/${PACKAGE}"

KEY_FILE=.deploy_key
SOURCE=_build

REPO=$(git config remote.origin.url)
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
REV=$(git rev-parse --short HEAD)

SRC_BRANCH=$(git rev-parse --abbrev-ref HEAD)
DST_BRANCH=gh-pages

set +u
if [ "$TRAVIS" == "true" ]; then
    set -u
    if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
        echo "Not building docs for PR"
        exit 0
    fi

    # Override what git says
    SRC_BRANCH=$TRAVIS_BRANCH

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

    git config --global user.email $COMMIT_AUTHOR_EMAIL
    git config --global user.name $COMMIT_AUTHOR_NAME
fi
set -u

echo "Building docs for branch: $SRC_BRANCH"

# START
cd $DOCS/

# Clean
rm -rf build
rm -rf $SOURCE

# Create branch directory and grab Git repo
git clone $SSH_REPO $SOURCE/
cd $SOURCE/
git checkout $DST_BRANCH || git checkout --orphan $DST_BRANCH
cd $DOCS/
rm -rf $SOURCE/**/* || exit 0

sphinx-apidoc -f -e -o $APIDOC ../${PACKAGE}/

make html
rm -rf $SOURCE/$SRC_BRANCH/
mkdir -p $SOURCE/$SRC_BRANCH/
cp -R build/html/* $SOURCE/$SRC_BRANCH/

cd $SOURCE/
git add -A .
git commit -m "Rebuild $DST_BRANCH docs on $SRC_BRANCH: ${REV}"
git push origin HEAD:$DST_BRANCH
