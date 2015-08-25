#!/bin/bash

set +e

pattern="L*stack"
istart=9
sstart=0
overwrite=0
relative=0
verbose=0

function usage() {
    cat << EOF

    usage: $0 [options] <root_directory> <output_file>

    This script will output a CSV list of the date and full filepath for
    all images within <root_directory> to <output_file>. Currently the date
    must be in YYYYDOY format.

    Options:
        -p         Filename pattern [default: $pattern]
        -d         Starting index of date within filename [default: $istart]
        -s         Starting index of sensor within filename [default: $sstart]
        -r         Use relative paths
        -o         Overwrite <output_file> if exists
        -v         Be verbose
        -h         Show help
EOF

}

function main() {
    # Header
    echo "date,sensor,filename" > $output

    images=$(find $root -follow -name "$pattern")
    nimages=$(echo $images | awk '{ print NF }')
    if [ "$nimages" == 0 ]; then
        echo "Error - found no images"
        rm $output
        exit 1
    fi

    if [ $verbose -eq 1 ]; then
        echo "Found $nimages images"
    fi

    for img in $images; do
        if [ $relative -eq 0 ]; then
            name=$(readlink -f $img)
        else
            name=$img
        fi
        bn=$(basename $img)
        id=$(basename $(dirname $img))
        ydoy=${id:$istart:7}
        sensor=${id:$sstart:3}

        echo "$ydoy,$sensor,$name"
    done | sort >> $output

}

while getopts "hp:d:s:orv" opt; do
    case $opt in
    h)
        usage
        exit 0
        ;;
    p)
        pattern=$OPTARG
        ;;
    d)
        istart=$OPTARG
        ;;
    s)
        sstart=$OPTARG
        ;;
    o)
        overwrite=1
        ;;
    r)
        relative=1
        ;;
    v)
        verbose=1
        ;;
    esac
done

shift $(($OPTIND - 1))

if [ -z $1 ] || [ -z $2 ]; then
    echo "Must specify <root_directory> and <output_file>"
    usage
    exit 1
else
    root=$1
    output=$2
fi

if [ $verbose -eq 1 ]; then
    echo "Searching in $root"
    echo "Searching for pattern: $pattern"
    echo "YYYYDOY starts at $istart"
    echo "Output file is $output"
fi

if [ -f $output ] && [ $overwrite -ne 1 ]; then
    echo "Error - $output already exists and we're not overwriting"
    exit 1
fi

set -e
echo -n "" > $output
if [ $? -ne 0 ]; then
    echo "Error writing to $output"
    exit 1
fi
set +e

main
