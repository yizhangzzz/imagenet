require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'

require 'nn'
require 'optim'


model = torch.load('model_1.t7')

local p,g = model:parameters()
local b = torch.Tensor(25,75):copy(p[1]:view(25,75))
print(p[1][15][1])
print(b[15])

local a = p[1]:clone()
local img = torch.Tensor(125,15)

for i=1,25 do
for j=1,3 do
piece = a[i][j]
img[{{(i-1)*5+1,i*5},{(j-1)*5+1,j*5}}]:copy(piece:add(-piece:min()):div(piece:max()-piece:min()))
end
end

image.save('filter.jpg',image.scale(img,60,500))

--[[
   local model = nn.Sequential() -- branch 1
   --model:add(cudnn.SpatialConvolution(3,25,5,5,1,1,2,2))       -- 224 -> 224
   model:add(nn.SoftShrink(5))
model:cuda()

inputCPU = torch.Tensor(4,5)
i = 0

inputCPU:apply(function()
  i = i + 1
  return i
end)

inputCPU = inputCPU - 10

input = torch.CudaTensor():resize(inputCPU:size()):copy(inputCPU)
local output = model:forward(input)
local g = model:backward(input, output)

print(g)
--]]







