

m = require 'manifold'

dofile('./Augmentation.lua')
dofile('./supervised_provider.lua')

model = torch.load('logs/model.net')
model:evaluate()



p = torch.load('provider.t7')
N = 1000

p.trainData.data  = p.trainData.data[{{1,N}}]
p.trainData.labels  = p.trainData.labels[{{1,N}}]



x = torch.DoubleTensor(p.trainData.data:size()):copy(p.trainData.data)
x:resize(x:size(1), x:size(2) * x:size(3) * x:size(4))

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)

im_size = 4096
map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, 28, 28), im_size, 0, true)

--return map_im