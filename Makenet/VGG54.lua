require 'cunn'
require 'loadcaffe'


function VGG54()

local VGGloss = {}
local VGG = loadcaffe.load('deplot.prototxt','VGG_ILSVRC_19_layer.caffemodel','cudnn')

for iter = 1,1 do
VGG:remove(iter)

end

VGG.accGradParameters = function() end
VGG:evaluate()

VGGloss.VGG = VGG
VGGloss.mse = nn.MSECrierion():cuda()

function VGGloss:forward(input,target)
  self.input_ = self.VGG:forward(input)
  self.target_ = self.VGG:forward(target)

  local loss = self.mse:forward(input_,target_)

  return loss
end


function VGGloss:backward(input,target)
  
  local grad_mse = self.mse:backward(self.input_,self.target_)
  local grad = VGG:backward(self.input_,grad_mse)
  
  return grad
end


return VGGloss
end 
