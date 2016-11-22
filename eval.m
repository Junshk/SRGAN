

%system('th test.lua')

folder = {'BSD100', 'Set14','Set5'};


for i=1:length(folder)

  
  sum_of_psnr = 0;
  img_num = 1;
  while exist(sprintf('data/test/%s/image_SRF_4/img_%03d_SRF_4_HR.png',folder{i},img_num)) == 2 

  hr = imread(sprintf('data/test/%s/image_SRF_4/img_%03d_SRF_4_HR.png',folder{i},img_num));
  sr = imread(sprintf('data/test/%s/image_SRF_4/img_%03d_SRF_4_SR.png',folder{i},img_num));
  
  if size(hr,3) == 1
      y_hr = hr ;
  else
  y_hr = rgb2ycbcr(hr);
  y_hr = y_hr(:,:,1);
  end
  
  y_sr = rgb2ycbcr(sr);
  y_sr = y_sr(:,:,1);
  PSNR = psnr(y_sr,y_hr);
  sum_of_psnr = sum_of_psnr + PSNR;
  
  img_num = img_num + 1;
  end
  
    folder{i} 
  ev = sum_of_psnr/(img_num-1)
  

  end



