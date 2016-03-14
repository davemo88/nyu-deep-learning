require 'xlua'
require 'optim'
require 'unsup'
require 'cunn'
require 'image'

dofile './supervised_provider.lua'
dofile './augmentation.lua'

local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 32)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
]]

print(opt)


-- do -- data augmentation module
--   local Augmentation,parent = torch.class('nn.Augmentation', 'nn.Module')

--   function Augmentation:__init()
--     parent.__init(self)
--     self.train = true
--   end

--   function Augmentation:updateOutput(input)
--     if self.train then
--       local bs = input:size(1)
--       self.output = torch.FloatTensor(input:size()):copy(input)
--       local flip_mask = torch.randperm(bs):le(bs/2)
--       for i=1,bs do
--         -- Flip
--         if flip_mask[i] == 1 then image.hflip(self.output[i], self.output[i]) end
--         -- Add Gaussian Noise
--         uNoise = torch.normal(0,0.2)
--         vNoise = torch.normal(0,0.2)
--         self.output[i][2] = (self.output[i][2] + uNoise)/1.2
--         self.output[i][3] = (self.output[i][3] + vNoise )/1.2
--         -- Rotate
--         deg = torch.uniform(-0.2,0.2)
--         self.output[i] = image.rotate(self.output[i], deg, 'bilinear') 

--         -- Translate
--         xTrans = torch.random(-6,6)
--         yTrans = torch.random(-6,6)
--         self.output[i] = image.translate(self.output[i], xTrans, yTrans) 
--       end
--     end
--     return self.output
--   end
-- end

psd_conv = torch.load('out4.psd/psd,encoderType=tanh,kernelsize=3,lambda=1/model.bin')

vgg = dofile(opt.model..'.lua')
vgg:get(1).weight = psd_conv.encoder:get(1).weight:reshape(64,3,3,3)

--vgg = torch.load('dk-model.net')

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.Augmentation():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(vgg:cuda())
model:get(2).updateGradInput = function(input) return end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v):float()

    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()

      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1

  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
    print('==> saving model to '..filename)
    torch.save(filename, model)
  end

end


function val()
  -- disable flips, dropouts and batch normalization
  -- model:remove(2)
  -- model:remove(1)
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,provider.valData.data:size(1),bs do

    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.valData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
  
  if valLogger then
    paths.mkdir(opt.save)
    valLogger:add{train_acc, confusion.totalValid * 100}
    valLogger:style{'-','-'}
    valLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/val.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 5 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
    print('==> saving model')
    torch.save(filename, model)
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  val()
end


