CUDA_VISIBLE_DEVICES=0 python main.py \
--mode 'test' \
--model 'hqs-net-unet' \
--acc 10 \
--batch_size 1 \
--lr 1e-3 \
--val_on_epochs 2 \
--num_epoch 300 \
--train_path "data/fs_train.npy" \
--val_path "data/fs_val.npy" \
--test_path "data/fs_test.npy"