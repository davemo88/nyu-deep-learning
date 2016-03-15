-- here is our code to:
-- fetch test data
-- fetch model file also containing training data mean and std
-- generate predicitions csv for test data using model

require 'torch'
--require 'xlua' -- progress bars
require 'cunn'

a2_url = 'http://s3.amazonaws.com/hodgetheaters/a2/'

print 'downloading provider code'
test_provider_code = 'test_provider.lua'
if not paths.filep(test_provider_code) then
    os.execute('wget ' .. a2_url .. test_provider_code)
end

dofile('./test_provider.lua')

print 'downloading augmentation code'
augmentation_code = 'Augmentation.lua'
if not paths.filep(augmentation_code) then
    os.execute('wget ' .. a2_url .. augmentation_code)
end

dofile('./Augmentation.lua')

print 'downloading test provider'
test_provider = 'test_provider.t7'
if not paths.filep(test_provider) then
    os.execute('wget ' .. a2_url .. test_provider)
end

p = torch.load('test_provider.t7')

print '==> downloading saved model'

model_file = 'model_a2.t7b'

if not paths.filep(model_file) then
-- go grab the model file if we don't already have it
    os.execute('wget ' .. a2_url .. model_file)
end

-- here's our model
m = torch.load(model_file)
-- remove our data augmentation modules since they are unnecessary for testing
m:remove(2)
m:remove(1)
m:evaluate()

-- function to get the index of maximum value
-- i.e. the class with the highest probability
-- from http://www.lua.org/pil/5.1.html
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
local batchsize = 10
print('==> testing on test set:')
for t = 1, p.testData.size(),batchsize do
    -- display progress
    xlua.progress(t, p.testData.size())
    end_index = math.min(t+batchsize-1, p.testData.size())
    local input = p.testData.data[{{t,end_index}}]:cuda()
    --print(input:size())

    -- test sample
    local pred = m:forward(input)
    -- get the id of the class with the highest probability
    for i=1,batchsize do

        prob, pred_class = maximum(torch.totable(pred[i]))
        f:write(t+i-1 .. ',' .. pred_class .. '\n')
    end
end

f:close()