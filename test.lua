require 'cunn'

require 'image'
require 'cudnn'
require 'TestParam'

--require 'divideForward'
require 'MethodofTest'

-----------------------------------------
cmd = torch.CmdLine()

cmd:option('-model','','string option')
cmd:option('-scale',2,'int option')
cmd:option('-set',5, 'set number')


local matio = require 'matio'
 local testtype = 'cuda'

torch.setdefaulttensortype('torch.FloatTensor')

-----flag


local cuManage = require 'cuManage'

local pDevice = cuManage.Device()

cutorch.setDevice(pDevice)
-------- model adjustment

  local boundofTestSize = 300

--local netname =srname ..scale ..'.t7'


function test(netname,scale,setnumber)

local model = torch.load(netname)

model:clearState()
print('netname: ',netname)
torch.save('TEST'..netname,model)
model =nil;
collectgarbage();


local Tnetname ='TEST'.. netname -- ..scale .. 'testModel.t7'
local re = string.sub(netname,1,-4)
print(re)
setname = 'Set'..setnumber
local resultName = savefold..re..'scale'..scale..'resultof'..setname

for i=1,setnumber do
local result 
local start= torch.Timer()  ;
  local imgname = Rnamet7..setnumber..'_scale'..scale..'_lr' ..i..'.t7'
  local reName = resultName..i..'.mat'
	print(imgname)
  local img = torch.load(imgname)
  result =torch.FloatTensor(img:size())
--print(result:size())
img = img:float()
--print('size', img:size())
local w, h = img:size(4),img:size(3)
result = result:float()
--print(img:type(),result:type())
--print(netname)

--print('dvdTest')
local fm,tm = cutorch.getMemoryUsage(pDevice)
  --  print('fm',fm)
result = dvdTest(Tnetname,img)

    matio.save(reName,result)--+img:float())
--print(result)
print(reName)
    result =nil;img = nil;
print('t:' , start:time().real)
collectgarbage();
img =nil;
end


os.execute('cd dataset/ ; matlab  -nosplash -r \'Cpsnr '.. re.. ' ' .. scale..' '..setnumber.. '\'')

end
------------------------------
params = cmd:parse(arg or {})
if params.model ~= '' then

test(params.model,params.scale,params.set)

end

