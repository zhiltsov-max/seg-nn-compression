local nn = require 'nn'
require 'cunn'

local Conv = nn.SpatialDilatedConvolution
local Deconv = nn.SpatialFullConvolution
local Avg = cudnn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU
local BatchNorm = nn.SpatialBatchNormalization

local function Kaiming(v)
    local n = v.kW*v.kH*v.nOutputPlane
    v.weight:normal(0, math.sqrt(2/n))
end
local function Constant(weights, c)
    weights:fill(c)
end
local function NoBias(v)
    if cudnn.version >= 4000 then
        v.bias = nil
        v.gradBias = nil
    else
        v.bias:zero()
    end
end

local function LinearInit(name)
    for k,v in pairs(model:findModules(name)) do
        NoBias(v)
    end
end
local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
        Kaiming(v)
        NoBias(v)
    end
end
local function DeconvInit(name)
    for k,v in pairs(model:findModules(name)) do
        Kaiming(v)
        -- NoBias(v)
    end
end
local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
        Constant(v.weight, 1)
        NoBias(v)
    end
end

local function create_model_camvid(options)
    local model = nn.Sequential()
    
    model:add(Conv(64, 7, 7, 3, 3, 2, 2))
    model:add(BatchNorm(64, 1e-3))
    model:add(ReLU(true))
    model:add(Max(3, 3, 2, 2, 1, 1))

    model:add(Conv(64, 5, 5, 4, 4, 1, 1, 2, 2))
    
    model:add(BatchNorm(64, 1e-3))
    model:add(ReLU(true))
    model:add(Conv(64, 5, 5, 4, 4, 1, 1, 2, 2))

    model:add(BatchNorm(64, 1e-3))
    model:add(ReLU(true))
    model:add(Conv(128, 5, 5, 4, 4, 2, 2, 2, 2))
    


    -- Classifier
    model:add(nn.BatchNorm(64,1e-3))
    model:add(nn.SpatialFullConvolution(64, 32, 1, 1))

    model = model:cuda()



    ConvInit('nn.SpatialConvolution')
    DeconvInit('nn.SpatialFullConvolution')
    LinearInit('nn.Linear')
    BNInit('nn.SpatialBatchNormalization')

    model:get(1).gradInput = nil

    local loss = cudnn.SpatialCrossEntropyCriterion()
    loss = loss:cuda()

    return model, loss
end

return create_model_camvid