CUDA_VISIBLE_DEVICES=0 python main.py \
--mode 'train' \
--model 'hqs-net' \
--acc 10 \
--batch_size 1 \
--lr 1e-3 \
--val_on_epochs 2 \
--num_epoch 101 \
--train_path "data/fs_train.npy" \
--val_path "data/fs_val.npy" \
--test_path "data/fs_test.npy"