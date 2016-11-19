require 'nn'
require 'Makenet/SRResnet'
require 'Makenet/VGG54'
require 'Makenet/SRGAN'


function make_net(option)

local net, criterion

if type(option) ~= 'string' then assert(nil, 'wrong net option') end

  if option == 'ResMSE' then
    net = SRResnet()
    criterion = nn.MSECriterion()
  elseif option == 'ResVGG' then
    net = SRResnet()
    criterion = VGG54()
  elseif option == 'GANMSE' then
    net = SRGAN()
    criterion = nn.MSECriterion()
  elseif option == 'GANVGG' then
    net = SRGAN()
    criterion = VGG54()
  else assert(nil, 'wrong net type') end
return net, criterion 
end

