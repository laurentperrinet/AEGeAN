default: meso_rsync

load_modules:
	module purge
	module load userspace/all
	module load python3/3.6.3

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402

maria_rsync:
	rsync --progress -avhuz --exclude-from=.ExclusionRSync  laurent@10.164.6.243:science/GAN-SDPC/GAN-SDPC/ .

meso_rsync:
	rsync --progress -avhuz --exclude-from=.ExclusionRSync lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/SDPC/2019_AEGeAN/ .
