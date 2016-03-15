require 'image'
require 'nn'
require 'itorch'
dofile 'provider.lua'
dofile 'batchflip2.lua'

model =nn.Sequential()
model:add(nn.BatchFlip())
model:training()
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
images = provider.trainData.data[{{200,205}}]
print(images:size())
t = torch.Tensor(30,3,96,96)
t[1] = images[1]
t[11] = images[2]
t[21] = images[3]
for i=1,9 do
	aug = model:forward(images)
	for j=1,3 do
		t[(j-1)*10 + i + 1] = aug[j]
	end
end
image.save('Augmentations.png', itorch.image(t))




