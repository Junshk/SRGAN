
scale = 4

-------fold-------------------
local fold ='data/SR_ILSVRC2015_val_4_'

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

savefold = 'result/'
setnumber =5

 setname ='Set' ..setnumber

