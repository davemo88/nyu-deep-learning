-- here is our code to:
-- fetch test data
-- fetch model file also containing training data mean and std
-- generate predicitions csv for test data using model

require 'torch'
require 'xlua' -- progress bars
require 'nn' -- required to load model
require 'Dropconnect' -- required to load model

-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MNIST Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full')
   cmd:text()
   opt = cmd:parse(arg or {})
end

-- allow for full or small 
if opt.size == 'full' then
   print '==> using regular, full training data'
   tesize = 10000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   tesize = 1000
end

print '==> downloading dataset'

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = 'mnist.t7'
-- we only care about the test file this time around
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

print '==> downloading Dropconnect code'

dropconnect_code = 'Dropconnect.lua'

if not paths.filep(dropconnect_code) then
-- go grab the model file if we don't already have it
    os.execute('wget ' .. 'http://cs.nyu.edu/~dk2353/deeplearning/hw/1/' .. dropconnect_code)
end

print '==> downloading saved model'

model_file = '4_relu_maxpool_dropconnect.t7b'

if not paths.filep(model_file) then
-- go grab the model file if we don't already have it
    os.execute('wget ' .. 'http://cs.nyu.edu/~dk2353/deeplearning/hw/1/' .. model_file)
end

saved = torch.load(model_file)

print '==> normalize test data using training mean and std'

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

testData.data = testData.data:float()

-- Normalize test data, using the training mean/std
-- we saved the training mean/std with our model
testData.data[{ {},1,{},{} }]:add(-saved.mean)
testData.data[{ {},1,{},{} }]:div(saved.std)

--sanity check
testMean = testData.data[{ {},1 }]:mean()
testStd = testData.data[{ {},1 }]:std()
print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)

m = saved.model
m:evaluate()

-- function to get the index of maximum value
-- i.e. the class with the highest probability
-- adapted from http://www.lua.org/pil/5.1.html
function maximum (a)
    local mi = 1          -- maximum index
    local m = a[mi]       -- maximum value
    for i,val in ipairs(a) do
        if val > m then
            mi = i
            m = val
        end
    end
    return m, mi
end

f = io.open('predictions.csv', 'w')

f:write('Id,Prediction\n')

print('==> testing on test set:')
for t = 1,testData:size() do
    -- disp progress
    xlua.progress(t, testData:size())
    -- get new sample
    -- cpa only so don't need to worry about input:cuda()
    local input = testData.data[t]:double()
    local target = testData.labels[t]

    -- test sample
    local pred = m:forward(input)
-- get the id of the class with the highest probability
    prob, pred_class = maximum(torch.totable(pred))
    f:write(t .. ',' .. pred_class .. '\n')
end

f:close()