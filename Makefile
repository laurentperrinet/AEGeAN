default: run_CFD

# RUNNING
run_CFD:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.runs_path = "AEGEAN_long"; opt.n_epochs=1000; AG.learn(opt)'

run_simpsons:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.runs_path = "Simpsons_long"; opt.datapath="../database/Simpsons-Face_clear/cp/"; opt.n_epochs=1000; AG.learn(opt)'

# CODING
pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402

tb:
	tensorboard --logdir runs

# FILES
MARIA_URL = laurent@10.164.6.243:GAN-SDPC/AEGEAN
maria_pull:
	# rsync --progress -avhuz --exclude-from=.ExclusionRSync $(URL)/ .
	rsync --progress -avhuz $(MARIA_URL)/runs .

maria_push:
	rsync --progress -avhuz --exclude-from=.ExclusionRSync ../AEGEAN $(MARIA_URL)/..
	rsync --progress -avhuz --delete --exclude-from=.ExclusionRSync ../database $(MARIA_URL)/..


MESO_URL = lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/SDPC/2019_AEGeAN
meso_pull:
	# rsync --progress -avhuz --exclude-from=.ExclusionRSync $(URL)/ .
	rsync --progress -avhuz $(MESO_URL)/runs .

meso_push:
	rsync --progress -avhuz runs $(MESO_URL)
	# rsync --progress -avhuz --delete --exclude-from=.ExclusionRSync ../database $(MESO_URL)/..

load_modules:
	module purge; module load userspace/all; module load python3/3.6.3

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

## INSTALL
install:
	python3 -m pip install --user -r requirements.txt
	python3 -m pip install --user -e .
