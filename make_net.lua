require 'nn'
require 'Makenet/SRResnet'
require 'Makenet/VGGloss'
require 'Makenet/SRGAN'

package.path = package.path ..';Makenet/?.lua'

function make_net(option)

local net, criterion

if type(option) ~= 'string' then assert(nil, 'wrong net option') end

  if option == 'ResMSE' then
    net = SRResnet()
    criterion = nn.MSECriterion()
  elseif option == 'ResVGG' then
    
    net = SRResnet()
    criterion = nn.VGGloss()
  elseif option == 'GANMSE' then
    net = SRGAN()
    criterion = nn.MultiCriterion()
    criterion:add(nn.MSECriterion(),1)
    criterion:add(nn.VGGloss(),1e-3)
  elseif option == 'GANVGG' then
    net = SRGAN()
    criterion = nn.VGGloss()
  else assert(nil, 'wrong net type') end
return net, criterion 
end

