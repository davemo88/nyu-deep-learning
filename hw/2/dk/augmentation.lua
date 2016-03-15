require 'image'

local Augmentation,parent = torch.class('nn.Augmentation', 'nn.Module')

function Augmentation:__init()
  parent.__init(self)
  self.train = true
end

function Augmentation:updateOutput(input)
  if self.train then
    local bs = input:size(1)
    local flip_mask = torch.randperm(bs):le(bs/2)
    for i=1,input:size(1) do
      if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      -- Add Gaussian Noise
      uNoise = torch.normal(0,0.2)
      vNoise = torch.normal(0,0.2)
      input[i][2] = (input[i][2] + uNoise)/1.2
      input[i][3] = (input[i][3] + vNoise )/1.2
      -- Rotate
      deg = torch.uniform(-0.2,0.2)
      input[i] = image.rotate(input[i], deg, 'bilinear') 

      -- Translate
      xTrans = torch.random(-6,6)
      yTrans = torch.random(-6,6)
      input[i] = image.translate(input[i], xTrans, yTrans)
    end
  end
  self.output:set(input)
  return self.output
end