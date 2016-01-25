$ gen_date_file.sh -h

    usage: ./../scripts/gen_date_file.sh [options] <root_directory> <output_file>

    This script will output a CSV list of the date and full filepath for
    all images within <root_directory> to <output_file>. Currently the date
    must be in YYYYDOY format.

    Options:
        -p         Filename pattern [default: L*stack]
        -d         Starting index of date within filename [default: 9]
        -s         Starting index of sensor within filename [default: 0]
        -r         Use relative paths
        -o         Overwrite <output_file> if exists
        -v         Be verbose
        -h         Show help
