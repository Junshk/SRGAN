require 'cunn'
require 'cudnn'
require 'image'

local netname = 'ResMSE_1'..'.t7'

local net =torch.load(netname)

net:evaluate()

local testSet = { 'BSD100','Set5','Set14'}

for k,set in pairs(testSet) do
  
  local folder =  'data/test/'..set..'/image_SRF_4/'

  local img_num = 1

  while paths.filep(folder..string.format('img_%03d_SRF_4_LR.png',img_num)) == true do
  local lr = image.load(folder..string.format('img_%03d_SRF_4_LR.png',img_num))
  local hr = image.load(folder..string.format('img_%03d_SRF_4_HR.png',img_num))
 

  if lr:size(1) == 1 then 
    lr = torch.cat({lr,lr,lr},1)
  end  


  print(folder,img_num)
  -- test and save result
  lr = nn.Unsqueeze(1):forward(lr)
  local result = net:forward(lr:cuda()):squeeze():float()
  
  image.save(folder..string.format('img_%03d_SRF_4_SR.png',img_num),result)

  img_num = img_num + 1
  end

end  



