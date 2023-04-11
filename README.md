# physprep pipeline
This standalone repository acts as use-case documentation for [phys2bids](https://github.com/physiopy/phys2bids). Thanks to the generous data donation from a subject of the [Courtois-Neuromod project](https://www.cneuromod.ca/), research communities like [PhysioPy](physiopy.github.io) will benefit from common data access to test and optimize their physio data preparation workflows, using [BIDS](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continous-recordings.html) format. 

The repo is separated in three main modules, and provides a setp-by-step tutorial for each of them:

`utils\` 

1. load BIOPAC sourcedata from the cloud;
2. match physio and neuro acquisition sessions;
3. log every file from one subject in a command line interface;
4. generate metadata from every sourcedata physio files of this one subject;
5. segment and convert files using phys2bids;
6. rename raw files that were converted in step #5, and make sure they follow the BIDS structure;

`preproc\`

to-do:
- [ ] link OSF storage and write a script to load the data locally
- [ ] Make sure Datalad integration works for metadata
