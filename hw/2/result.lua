-- here is our code to:
-- fetch test data
-- fetch model file also containing training data mean and std
-- generate predicitions csv for test data using model

require 'torch'
require 'xlua' -- progress bars
require 'nn' -- required to load model
require 'cunn'

a2_url = 'http://s3.amazon.com/hodgetheaters/a2/'

print 'downloading provider'
test_provider = 'test_provider.t7'
if not paths.filep(test_provider) then
    os.execute('wget ' .. a2_url .. test_provider)
end

p = torch.load(test_provider)

print '==> downloading augmentation code'
augmentation_code = 'augmentation.lua'
if not paths.filep(provider_code) then
-- go grab the dropconnect code if we don't already have it
    os.execute('wget ' .. a2_url .. augmentation_code)
end

-- need to do this after downloading the code
dofile './test_provider.lua'
dofile './augmentation.lua'

-----------------

print '==> downloading saved model'

model_file = 'model_a2.t7b'

if not paths.filep(model_file) then
-- go grab the model file if we don't already have it
    os.execute('wget ' .. a2_url .. model_file)
end

-- here's our model
saved = torch.load(model_file)

-----------------

m = model
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

print('==> testing on test set:')
for t = 1, p.testData.data:size()[1] do
    -- display progress
    xlua.progress(t, testData.data:size()[1])
    
    local input = testData.data[t]:cuda()
    local target = testData.labels[t]

    -- test sample
    local pred = m:forward(input)
    -- get the id of the class with the highest probability
    prob, pred_class = maximum(torch.totable(pred))
    f:write(t .. ',' .. pred_class .. '\n')
end

f:close()