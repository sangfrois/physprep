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
            # neuromod_process.py takes in argument :
            # 1. the path to source data
            # 2. the subject id (folder name)
            # 3. the ses id (folder name)
            # 4. the path to save the outputs
            # 5. Boolean to indicate wheter or not the outputs should be saved
            python process.py $SOURCE $SUB $SES $OUTDIR True
        done
    fi
done