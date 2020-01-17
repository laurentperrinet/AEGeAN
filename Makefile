default: run

# RUNNING
run:
	sbatch launch.sh

run_CFD:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.runs_path = "AEGEAN_long"; opt.n_epochs=16384; opt.sample_interval=128; AG.learn(opt)'

run_128:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.img_size = 128; opt.runs_path = "AEGEAN_128_long"; opt.n_epochs=16384; opt.sample_interval=128; AG.learn(opt)'

run_simpsons:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.runs_path = "Simpsons_long"; opt.datapath="../database/Simpsons-Face_clear/cp/"; opt.n_epochs=16384; opt.sample_interval=128; AG.learn(opt)'

run_butterflies:
	python3 -c'import AEGEAN as AG; opt = AG.init(); opt.runs_path = "butterflies_long"; opt.datapath="../database/swapnesh_butterflies/";  opt.n_epochs=16384; opt.sample_interval=128; AG.learn(opt)'

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


MESO_URL = lperrinet@login.mesocentre.univ-amu.fr:/home/lperrinet/science/AEGeAN
meso_run:
	# https://mesocentre.univ-amu.fr/slurm/
	srun -p gpu -A h146 -t 4-2 --gres=gpu:1 --gres-flags=enforce-binding --pty bash -i

meso_pull:
	# rsync --progress -avhuz --exclude-from=.ExclusionRSync $(URL)/ .
	rsync --progress -avh  $(MESO_URL)/runs/ runs
	# rsync --progress -avh --delete $(MESO_URL)/runs/ runs

meso_push:
	#rsync --progress -avhuz runs $(MESO_URL)
	rsync --progress -avhuz --delete --exclude-from=.ExclusionRSync ../database $(MESO_URL)/..

load_modules:
	module purge; module load userspace/all; module load python3/3.6.3; module load cuda/10.1

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

clean:
	rm -fr runs

## INSTALL
install:
	python3 -m pip install --user -r requirements.txt
	python3 -m pip install --user -e .
