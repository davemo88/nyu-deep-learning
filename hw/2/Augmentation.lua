do -- data augmentation module
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
      end
    end
    self.output:set(input)
    return self.output
  end
end