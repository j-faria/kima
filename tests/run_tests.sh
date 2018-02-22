#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

for D in examples/{BL2009,CoRoT7,default_priors}; do
    if [ -d "${D}" ]; then

        # use as seed the first 6 digits of the md5sum of the directory
        seed=`echo "${D}" | md5sum | sed 's/[a-z]*//g'`
        seed=${seed:0:6} # remove trailing - with %%-
        printf "   >  %-23s   (seed=${seed}) " "${D}"

        # go in the example directory, compile and run
        cd $D
        # echo -e "   compiling... \c"
        make --silent
        # echo "running the example (could take a while) ..."
        SEED=${seed} ./run > /dev/null 2> /dev/null
        
        # check (checks.md5 should already be in the directory)
        md5sum sample.txt levels.txt sample_info.txt > tests.md5
        if diff -q tests.md5 checks.md5; then
            printf "   >> ${GREEN}PASSED${NC}!"
        else 
            printf "   >> ${RED}FAILED${NC}!!"
        fi
        echo
        # rm tests.md5
        make -s cleanout
        cd ../..
    fi
done
