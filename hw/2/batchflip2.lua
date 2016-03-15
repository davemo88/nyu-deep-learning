do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

 
 function BatchFlip:updateOutput(input)
    out = torch.Tensor(input:size()):copy(input)
    if self.train then
      local bs = input:size(1)
      --print(type(out))
      --print(out:size())
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(out[i], out[i]) end
        --Add Gaussian Noise
        uNoise = torch.normal(0,0.3)
        vNoise = torch.normal(0,0.3)
        out[i][2] = (out[i][2] + uNoise)/1.3
        out[i][3] = (out[i][3] + vNoise )/1.3
        --Translate
        xTrans = torch.random(-9,9)
        yTrans = torch.random(-9,9)
        out[i] = image.translate(out[i], xTrans, yTrans) 
        -- Rotate
        theta = torch.uniform(-0.2,0.2)
        out[i] = image.rotate(out[i], theta, 'bilinear')
      end
    end
    self.output = out
    return self.output
  end
end