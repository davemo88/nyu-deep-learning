require 'nngraph'
require 'graph'
nhid = 10
function grucell()-- (input, prevh)
    input = nn.Identity()()
    prevh = nn.Identity()()
    local i2h = nn.Linear(nhid, 3 * nhid)(input) 
    local h2h = nn.Linear(nhid, 3 * nhid)(prevh) 
    local gates = nn.CAddTable()({
        nn.Narrow(2, 1, 2 * nhid)(i2h), 
        nn.Narrow(2, 1, 2 * nhid)(h2h),
    }) 
    gates = nn.SplitTable(2)(nn.Reshape(2, nhid)(gates)) 
    local resetgate = nn.Sigmoid()(nn.SelectTable(1)(gates)) 
    local updategate = nn.Sigmoid()(nn.SelectTable(2)(gates)) 
    local output = nn.Tanh()(nn.CAddTable()({
        nn.Narrow(2, 2 * nhid+1, nhid)(i2h), 
        nn.CMulTable()({resetgate, 
                        nn.Narrow(2, 2 * nhid+1, nhid)(h2h),})})) 
    local nexth = nn.CAddTable()({prevh,
        nn.CMulTable()({updategate, 
        nn.CSubTable()({output, prevh,}),}), 
    })
    return nn.gModule({input, prevh}, {nexth})
    --return nexth 
end

g = grucell()
graph.dot(g.fg,'G','grucell')