require 'cunn'
require 'loadcaffe'

local VGGloss , parent = torch.class('nn.VGGloss','nn.Criterion')


function VGGloss:__init()

local VGG = loadcaffe.load('Makenet/VGG_ILSVRC_19_layers_deploy.prototxt','Makenet/VGG_ILSVRC_19_layers.caffemodel','cudnn')
assert(VGG)

for iter = 36,46 do
VGG:remove(#VGG.modules)
end

VGG.accGradParameters = function() end
VGG:evaluate()

self.VGG = VGG
self.mse = nn.MSECriterion():cuda()

print(self.VGG)
end

function VGGloss:updateOutput(input,target)
  
--  self.VGG:clearState()
  --self.input_ = nil
  --self.target_ = nil
 self.target_ = self.VGG:forward(target):clone()
 self.input_ = self.VGG:forward(input):clone()

 local loss = self.mse:forward(self.input_,self.target_)
  --assert(torch.sum(torch.ne(self.input_,self.target_))~=0,torch.sum(torch.ne(self.input_,self.target_)))

  --assert(nil)
  return loss
end


function VGGloss:updateGradInput(input,target)
  
  local grad_mse = self.mse:backward(self.input_,self.target_):clone()
  local grad = self.VGG:backward(input,grad_mse):clone()
  
  return grad
end


