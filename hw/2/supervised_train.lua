require 'xlua'
require 'optim'
require 'cunn'
require 'image'

local c = require 'trepl.colorize'

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.Augmentation():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(torch.load('logs/model.net')):cuda()
model:get(2).updateGradInput = function(input) return end

model = torch.load('logs/model.net')

print(c.blue '==>' ..' loading data')
provider = torch.load 'test_provider.t7'
provider.trainData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)


function val()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,provider.valData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)

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

  -- save model every 50 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model)
  end

  confusion:zero()
end

val()


