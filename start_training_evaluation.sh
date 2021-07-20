
mkdir -p logs || exit 1

randomseed=0 # 0, 1, 2, ...
modelconfig=conf/training_mdl/se_mcg_res2net50_cesoftmax.json # configuration files in conf/training_mdl
feats=la_lfcc  # `la_spec`, `la_cqt` or `la_lfcc`
runid=MCG-Res2Net50-CESoftmax-LALFCC0
pretrained="none"
. ./parse_options.sh || exit 1

echo $randomseed
echo $modelconfig
echo $feats
echo $runid

echo "Start training."
if [ $pretrained == "none" ]
then
      python3 train.py --run-id $runid --random-seed $randomseed --data-feats $feats --configfile $modelconfig >logs/$runid || exit 1
else
      echo "using pretrained model: "$pretrained
      python3 train.py --run-id $runid --random-seed $randomseed --data-feats $feats --configfile $modelconfig --pretrained $pretrained >>logs/$runid || exit 1
fi

echo "Start evaluation on all checkpoints."
for model in model_snapshots/$runid/*_[0-9]*.pth.tar; do
      python3 eval.py --random-seed $randomseed --data-feats $feats --configfile $modelconfig --pretrained $model || exit 1
done

