local nn = require 'nn'
require 'cunn'

local Conv = nn.SpatialConvolution
local Upconv = nn.SpatialFullConvolution
local Avg = nn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local ReLU = nn.ReLU
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

local function LinearInit(model, name)
    for k,v in pairs(model:findModules(name)) do
        NoBias(v)
    end
end
local function ConvInit(model, name)
    for k,v in pairs(model:findModules(name)) do
        Kaiming(v)
        Constant(v.bias, 0)
        -- NoBias(v)
    end
end
local function DeconvInit(model, name)
    for k,v in pairs(model:findModules(name)) do
        Kaiming(v)
        Constant(v.bias, 0)
        -- NoBias(v)
    end
end
local function BNInit(model, name)
    for k,v in pairs(model:findModules(name)) do
        Constant(v.weight, 1)
        Constant(v.bias, 0)
        -- NoBias(v)
    end
end

local function create_model_camvid(options)
    local class_count = options.classCount
    local input_channels = options.inputChannelsCount

    -- Learning regime:
    -- 1-2k epochs
    -- base lr: 2, lr decay: 0.02, wdecay: 5e-4
    -- converges at ~300 epoch

    local model = nn.Sequential()
    
    model:add(Conv(input_channels, 32, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(32))
    model:add(Max(3, 3, 2, 2, 1, 1))

    model:add(Conv(32, 64, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(64))
    model:add(Max(3, 3, 2, 2, 1, 1))

    model:add(Conv(64, 128, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(128))
    model:add(Max(3, 3, 2, 2, 1, 1))
    

    model:add(Upconv(128, 64, 4, 4, 2, 2, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(64))

    model:add(Upconv(64, 32, 4, 4, 2, 2, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(32))

    model:add(Upconv(32, 32, 4, 4, 2, 2, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(32))
    
    model:add(Conv(32, class_count, 1, 1))

    -- ConvInit(model, 'nn.SpatialConvolution')
    -- DeconvInit(model, 'nn.SpatialFullConvolution')
    -- LinearInit(model, 'nn.Linear')
    -- BNInit(model, 'nn.SpatialBatchNormalization')

    model = model:cuda()

    local loss = cudnn.SpatialCrossEntropyCriterion()
    loss = loss:cuda()

    return model, loss
end

return create_model_camvid