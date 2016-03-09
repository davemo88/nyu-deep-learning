-- Generates matching arrays of targets and inputs for the parallel criterion
function gradient(model, input,  depth, criterion )
  crit_inputs = {}
  crit_targets = {}
  for i=1,depth do
    output = model.output
    crit_targets[i] = input
    crit_inputs[i] = model.output
    input = model:get(1).output
    model = model:get(2)
  end
  f = criterion:forward(crit_inputs, crit_targets)
  df_do = criterion:backward(crit_inputs, crit_targets)
  return f, df_do
end

-- Propagates the gradients to the appropriate modules
function backward(model, input, depth, df_do)
  for i=1,depth do 
    model:backward(input, df_do[i])
    input = model:get(1).output
    model = model:get(2)
  end
end
