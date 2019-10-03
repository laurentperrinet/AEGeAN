default: meso_rsync

load_modules:
	module purge; module load userspace/all; module load python3/3.6.3

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402

maria_rsync:
	rsync --progress -avhuz --exclude-from=.ExclusionRSync  laurent@10.164.6.243:research/GAN-SDPC/GAN-SDPC/ .

URL = lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/SDPC/2019_AEGeAN/
meso_rsync:
	rsync --progress -avhuz --exclude-from=.ExclusionRSync $(URL) .

meso_sync:
	rsync --progress -avhuz runs $(URL)
	rsync --progress -avhuz $(URL)/runs .
	# rsync --progress -avhuz --exclude-from=.ExclusionRSync ../database $(URL)/..
