-- here is our code to:
-- fetch test data
-- fetch model file also containing training data mean and std
-- generate predicitions csv for test data using model

require 'torch'
require 'xlua'

print '==> downloading dataset'

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = 'mnist.t7'
-- we only care about the test file this time around
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

print '==> downloading saved model'

model_location = 'http://cs.nyu.edu/~dk2353/deeplearning/hw/1/model.t7b'

os.execute('wget ' .. model_location)

saved = torch.load(model_location)

print '==> normalize test data using training mean and std'

-- Normalize test data, using the training mean/std
-- we saved the training mean/std with our model
testData.data[{ {},1,{},{} }]:add(-saved.mean)
testData.data[{ {},1,{},{} }]:div(saved.std)

-- sanity check
testMean = testData.data[{ {},1 }]:mean()
testStd = testData.data[{ {},1 }]:std()
print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)

m = saved.model
m:evaluate()

f = io.open('predictions.csv', 'w')

print('==> testing on test set:')
for t = 1,testData:size() do
    -- disp progress
    xlua.progress(t, testData:size())

    -- get new sample
    -- cpa only so don't need to worry about input:cuda()
    local input = testData.data[t]:double()
    local target = testData.labels[t]

    -- test sample
    local pred = model:forward(input)
    f:write(t .. ',' .. pred .. '\n')
end

f:close()