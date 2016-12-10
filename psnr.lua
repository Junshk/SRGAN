require 'image'

function psnr(sr, hr)
  local batch = sr:size(1)
  local sz = sr:size(3)
  if sr:dim() == 3 then 
    batch = 1 ;sr= sr:view(1,sr:size(1),sz,sz); hr =hr:view(1,hr:size(1),sz,sz)
  end
    -- y channel

  local y_sr, y_hr = torch.Tensor(batch,1,sz,sz), torch.Tensor(batch,1,sz,sz)

  for iter = 1, batch do
    y_sr[{{iter}}] = image.rgb2y(sr[{{iter}}]:squeeze())
   -- print(y_sr)
    y_hr[{{iter}}] = image.rgb2y(hr[{{iter}}]:squeeze())
  end  

  -- quantize
--  y_sr = torch.floor(16 + y_sr*219)
--  y_hr = torch.floor(16 + y_hr*219)

  local mse = torch.norm(y_sr-y_hr)/math.sqrt(y_sr:numel())
  
print(mse)
  return -20*math.log10(mse)

end
