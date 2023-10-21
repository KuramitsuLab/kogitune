# kogitune-examples

### scripts/train.py
メインの学習スクリプト。分散の有無に関わらず使えます。
モデルの指定は引数でできるようになっていないので、直で書き換えが必要です。

### scripts/train.sh
シングルノード・シングルGPUの学習に使用できます。

### scripts/train_distributed.sh
シングルノード・マルチGPUの学習に使用できます。
（deepspeedのconfigはzero2のみ、`ds_config/ds_config_zero2.json` にサンプルが存在します）

### config/training_setup.yaml
実行に関する設定（wandb、モデルのハイパーパラメータ、学習設定等）です。
sample_sizeを指定するとURLごとに指定した数の少量データのテストできて、指定しないと全データで学習となる点に注意してください。

#### wandbの設定について
- entity：　Teams
- project: Project
- name: Run


### datasets/urls.txt
データURLを置いておくtxtファイルです。
