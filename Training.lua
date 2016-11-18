require  'optim'
require 'cunn'
require 'cudnn'
require 'extraction'
require 'gnuplot'

cudnn.fastest = true
cudnn.benchmark = true

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(2)

local optimState = {
learningRate = 1e-4
,beta1 = 0.9
  }
local patch_size = 96
local batch_size = 16

-------------------------------------------
function training(model,criterion,netname)
  model = cudnn.convert(model,cudnn):cuda()
  criterion:cuda()


  print(model)
  print(netname)

 
  local params, grads =model:getParameters()

--------------------------------------------------------------
  local feval = function(x)

    if x~=params then
      params:copy(x)
    end

    local batch_target,batch_input = extractData()
    
    if batch_input:size(1) ~= batch_size then
    print('size',batch_input:size(1)) end

------forward
    grads:zero()
    
    local f = 0
    
    local numUnit = batch_size/unit_size
    
    
    
    for batch_iter = 1,numUnit do
      
      --unit setting
      local unitInput = batch_input[{{1+(batch_iter-1)*unit_size,batch_iter*unit_size}}]
      local unitTarget = batch_target[{{1+(batch_iter-1)*unit_size,batch_iter*unit_size}}]
      
  
      local output = model:forward(unitInput:cuda()):float()
      local err =criterion:forward(output:cuda(),unitTarget:cuda())
      f = f+ err
      local df_do = criterion:backward(output:cuda(),unitTarget:cuda()):float()
model:backward(unitInput:cuda(),df_do:cuda())
      
      
    end
    
    grads:div(unit_size)
    f = f/unit_size
    
   

--- clip gradient element-wise

  if clipping ==true then
  clipvalue = clipMag  / optimState.learningRate
  grads:clamp(-clipvalue,clipvalue)
  end
  
  return f,grads
        
end


--------------------------------




  local losses ={}
  local testloss ={}
  
  local epoch_iter = math.floor(dataNmax/batch_size)
  print('epoch',epoch_iter)
 
 
-- local iter1 =  1e6

  
--  local iter2 = 20 *epoch_iter
  --local iter3 = 30 *epoch_iter
  local enditer = 1e6--50 *epoch_iter

for iteration =1,enditer do

    local  _ ,loss=optim.adam(feval,params,optimState)
    if iteration% print_iter ==0 then
      
      print(string.format("iteration %4d, loss = %6.6f",iteration,loss[1]))
   
    end
   
   local testEpoch =100
   
   
   if iteration %test_iter ==0 and test_flag ==true then
 
   
    model:evaluate()
    local p = model:forward(testData.input)[{{1}}]
    local vec = p:float() -testData.target[{{1}}]:float();
    local max_signal = 1
    
    vec= vec:view(-1)
    testloss[#testloss+1] = torch.norm(vec,2)/math.sqrt(vec:numel())

    print('psnr y',- math.log10(testloss[#testloss]/max_signal)*20)
    model:training() 
   end
  
  
  losses[#losses+1] = loss[1]
  

  if iteration % plot_iter == 0 then

      gnuplot.plot(
        {netname..'testerr',torch.range(1,#testloss)*testEpoch,torch.Tensor(testloss),'-'},
        {netname ..'trainerr',torch.range(1,#losses)*testEpoch,torch.Tensor(losses):clamp(0,plotbound),'-'}) 

	 
  end
  
  if iteration% save_iter == 0 then
  
  torch.save(netname..'trainerr.t7',losses)
  torch.save(netname..'testerr.t7',testloss)
 torch.save(netname..'.t7',model)

  end



end

  torch.save(netname..'loss.t7',losses)
  torch.save(netname..'testerr.t7',testerres)


end --function end
