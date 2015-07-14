require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'

require 'nn'
require 'optim'

local nonlinear = nn.Sequential()
nonlinear:add(nn.Linear(1,10))
nonlinear:add(nn.Sigmoid())
nonlinear:add(nn.Linear(10,10))
nonlinear:add(nn.Sigmoid())
nonlinear:add(nn.Linear(10,1))

criterion = nn.MSECriterion()
criterion.sizeAverage = true

--nonlinear.modules[1].weights = torch.rand(20,1):mul(0.0000001)
--nonlinear.modules[3].weights = torch.rand(20,1):mul(0.0000001)

standard = nn.ReLU()
standard:cuda()
nonlinear:cuda()
criterion:cuda()

input = torch.CudaTensor(100,1):copy((torch.rand(100,1) - 0.5)*10)

for i=1,2 do
	  local p,g = nonlinear:parameters()
	 print(p[5])
    print(g[5])
	input = torch.CudaTensor(1,1):copy((torch.rand(1,1) - 0.5)*10)
	target = standard:forward(input)
	output = nonlinear:forward(input)
    err = criterion:forward(output, target)	
	df_do = criterion:backward(output, target)
    nonlinear:backward(input, df_do)
    nonlinear:updateParameters(1e-5)
    cutorch.synchronize()
  
    print("------")
    print(err)
    print(output)
    print(input)
    print(df_do)
   
end


