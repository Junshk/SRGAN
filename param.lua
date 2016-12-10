require 'cutorch'
cutorch.setDevice(1)
torch.setnumthreads(2)
scale = 4
batch_size = 16
patch_size =24
unit_size =batch_size
-------fold------------------
local fold ='/SR_ILSVRC2015_val_4_'

rgb_fold = fold.. 'rgb/' 
Y_fold = fold.. 'Y/'
gt_fold = fold.. 'GT/'
ilr_fold = fold..'ILR/'
lr_fold = fold..'LR/'

-----iter----------------------

save_iter = 200
print_iter = 10
plot_iter = 100
test_iter = 10

-----test--------------------
test_flag = true
----criterion----------------
loss_type = {mse = 1 , adv =0, vgg=0}


-----------------------------
continue = false--true
img_type = 'rgb'--'yuv'

nettype = 'ResVGG'
netname = nettype .. '_1207'
