CUDA_VISIBLE_DEVICES=0 python main.py \
--mode 'train' \
--model 'hqs-net-unet' \
--acc 5 \
--resume 0 \
--batch_size 1 \
--lr 1e-3 \
--val_on_epochs 2 \
--num_epoch 100 \
--train_path "data/fs_train.npy" \
--val_path "data/fs_val.npy" \
--test_path "data/fs_test.npy"