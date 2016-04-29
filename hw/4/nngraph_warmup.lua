require 'nngraph'
require 'graph'

x = nn.Identity()()
y = nn.Identity()()
z = nn.Identity()()
xlin = nn.Linear(4,2)(x)
xlin.weight = torch.Tensor({{1,1},{1,1},{1,1},{1,1}})
xlin.bias = torch.Tensor({1,1,1,1})
ylin = nn.Linear(5,2)(y)
ylin.bias = torch.Tensor({1,1,1,1,1})
ylin.weight = torch.Tensor({{1,1},{1,1},{1,1},{1,1},{1,1}})
xsq = nn.Square()(nn.Tanh()(xlin))
ysq = nn.Square()(nn.Sigmoid()(ylin))
cxy = nn.CMulTable()({xsq,ysq})
a = nn.CAddTable()({cxy, z})
m = nn.gModule({x,y,z},{a})
graph.dot(m.fg,'M','mygraph')

function warmup(x,y,z)
    print(m:forward({x,y,z}))
    gradOutput = torch.Tensor({1,1})
    print(m:backward({x,y,z},gradOutput))
end
