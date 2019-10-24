default: meso_rsync

run:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.runs_path = "AEGEAN_long"; opt.n_epochs=1000; AG.learn(opt)' 

load_modules:
	module purge; module load userspace/all; module load python3/3.6.3

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402

maria_rsync:
	rsync --progress -avhuz --exclude-from=.ExclusionRSync  laurent@10.164.6.243:research/GAN-SDPC/GAN-SDPC/ .

URL = lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/SDPC/2019_AEGeAN
meso_pull:
	# rsync --progress -avhuz --exclude-from=.ExclusionRSync $(URL)/ .
	rsync --progress -avhuz $(URL)/runs .

meso_push:
	rsync --progress -avhuz runs $(URL)
	# rsync --progress -avhuz --exclude-from=.ExclusionRSync ../database $(URL)/..
