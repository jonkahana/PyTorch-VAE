

sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/bbvae__smallnorb.log --job-name=bbvae_norb bash_scripts/BBVAE/smallnorb.sh
sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/bbvae__cars3d.log --job-name=bbvae_cars3d bash_scripts/BBVAE/cars3d.sh
sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/bbvae__shapes3d.log --job-name=bbvae_shapes3d bash_scripts/BBVAE/shapes3d.sh


sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/btcvae__smallnorb.log --job-name=btcvae_norb bash_scripts/BTCVAE/smallnorb.sh
sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/btcvae__cars3d.log --job-name=btcvae_cars3d bash_scripts/BTCVAE/cars3d.sh
sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/btcvae__shapes3d.log --job-name=btcvae_shapes3d bash_scripts/BTCVAE/shapes3d.sh

sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/factorvae__smallnorb.log --job-name=fvae_norb bash_scripts/FactorVAE/smallnorb.sh

sbatch --mem=24g -c4 --time=7-0 --gres=gpu:1,vmem:10g --output=logfiles/dipvae__smallnorb.log --job-name=dipvae_norb bash_scripts/DIPVAE/smallnorb.sh
