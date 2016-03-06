require 'xlua'
require 'optim'
require 'cunn'
-- dofile './unsupervised_provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default unsupervised_model_simple)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --clumps                   (default 26)            number of training clumps
]]

print(opt)

print(c.blue '==>' ..' configuring model')
local model = dofile('models/'..opt.model..'.lua'):cuda()

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model, cudnn)
end
print(model)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' setting criterion')
-- one MSECriterion for each nested autoencoder
criterion = nn.MSECriterion():cuda()
--criterion:add(nn.MSECriterion():cuda())
--criterion:add(nn.MSECriterion():cuda())
print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

last_rec_err = 1


function train()
  model:training()
  epoch = epoch or 1
  this_rec_err = 0
  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. 
    ' [batchSize = ' .. opt.batchSize .. 
    ', clumps = ' .. opt.clumps .. ']')

  for i=1, opt.clumps do
    print(c.blue '==>' ..' loading clump ' .. i .. '/' .. opt.clumps)
    clump = torch.load('clumps/' .. i .. '.t7b')
    clump.data = clump.data:float()
    local targets = torch.CudaTensor(opt.batchSize)
    local indices = torch.randperm(clump.data:size(1)):long():split(opt.batchSize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil

    local tic = torch.tic()
    for t,v in ipairs(indices) do
      xlua.progress(t, #indices)

  -- need to get unlabeled data
      local inputs = clump.data:index(1,v):cuda()
      local targets = inputs:clone():cuda()




      local feval = function(x)
        if x ~= parameters then parameters:copy(x) end
        gradParameters:zero()
        local outputs = model:forward(inputs)
        -- print(type(targets:cuda()), type(maxpool2.output:cuda()), type(decode_2.output:cuda()))
        --crit_inputs = {outputs, 
        --                decode1.output
        --               }

        --crit_targets = {targets,
        --                maxpool1.output
        --               }
        --local f = criterion:forward(crit_inputs, crit_targets)
        --local df_do = criterion:backward(crit_inputs, crit_targets)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        this_rec_err = this_rec_err + f
        model:backward(inputs:cuda(), df_do)
        --decode1:backward(maxpool1.output, df_do[2])
        return f,gradParameters
      end
      optim.sgd(feval, parameters, optimState)
    end
  end

  if epoch % 10 == 0 and epoch > 1 then 
    torch.save('trained/cae_simple'.. epoch .. '.t7b', model) 
  end
  if epoch >1 then
    print("Improvement %: " .. ((100*(last_rec_err-this_rec_err))/last_rec_err))
  end
  last_rec_err = this_rec_err
  epoch = epoch + 1
-- why not?
  collectgarbage()
end

for i=1,opt.max_epoch do
  train()
end


