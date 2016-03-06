require 'nn'
require 'image'
require 'optim'
require 'cunn'

for i=1,9 do
	x = torch.load('swwae_' .. i .. '.t7b')
	x:float()
	torch.save('swwae_' .. i .. '.t7b', x)
end
