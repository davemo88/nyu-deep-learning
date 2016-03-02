require 'xlua'
require 'optim'
require 'cunn'
dofile './unsupervised_provider.lua'
dofile './models/unsupervised_model.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 32)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default unsupervised_model)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
]]

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
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

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end
print(model)

-- need to get unlabeled data here
print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()

print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' setting criterion')
-- one MSECriterion for each nested autoencoder
criterion = nn.ParallelCriterion():cuda()
criterion:add(nn.MSECriterion():cuda())
criterion:add(nn.MSECriterion():cuda())
criterion:add(nn.MSECriterion():cuda())
criterion:add(nn.MSECriterion():cuda())

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

  -- local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

-- need to get unlabeled data
    local inputs = provider.trainData.data:index(1,v)
    local targets = inputs:clone():cuda()

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      local outputs = model:forward(inputs)
      -- print(type(targets:cuda()), type(maxpool2.output:cuda()), type(decode_2.output:cuda()))
      crit_inputs = {targets, 
                     maxpool1.output,
                     maxpool2.output,
                     maxpool3.output,
                     maxpool4.output}

      crit_targets = {targets,
                      decode_2.output,
                      decode_3.output,
                      decode_4.output,
                      decode_5.output}

      local f = criterion:forward(crit_inputs, crit_targets)
      local df_do = criterion:backward(crit_inputs, crit_targets)
      -- local f = criterion:forward(outputs, targets)
      -- local df_do = criterion:backward(outputs, targets)]
      model:backward(inputs, df_do[1])
      decode_2:backward(maxpool1.output, df_do[2])
      decode_3:backward(maxpool2.output, df_do[3])
      decode_4:backward(maxpool3.output, df_do[4])
      decode_5:backward(maxpool4.output, df_do[5])

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  if epoch % 10 == 0 then 
    torch.save('trained/swwae_'.. epoch .. '.t7b', model) 
  else
    torch.save('trained/swwae_'.. epoch .. '.t7b', model)
  end

  epoch = epoch + 1
  end

for i=1,opt.max_epoch do
  train()
end


