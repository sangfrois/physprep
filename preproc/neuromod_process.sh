#! bin/bash

SOURCE="/scratch/mepicard/physio_data/emotionvideos"
OUTDIR="/scratch/mepicard/physio_data/emotionvideos/processed"

for dir in $SOURCE/*/
do
    #Iterate over sub folders
    if [[ $dir == *"sub"* ]]
    then
        SUB=$(basename $dir)
        #Iterate over ses folders
        for i in $dir*/
        do
            SES=$(basename $i)
            python neuromod_process.py $SOURCE $SUB $SES $OUTDIR True
        done
    fi
done