#!/bin/bash

for md in *.md; do
    pandoc -o $(basename $md .md).rst $md
done
