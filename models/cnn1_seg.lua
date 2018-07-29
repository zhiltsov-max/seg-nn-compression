local nn = require 'nn'
require 'cunn'

local Conv = nn.SpatialConvolution
local Upconv = nn.SpatialFullConvolution
local Avg = nn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local ReLU = nn.ReLU
local BatchNorm = nn.SpatialBatchNormalization
local Dropout = nn.SpatialDropout

local function create_model_camvid(options)
    local class_count = options.classCount
    local input_channels = options.inputChannelsCount

    -- Learning regime:
    -- 2-3k epochs
    -- base lr: 2, lr decay: 0.02, wdecay: 5e-4
    -- converges at ~1400 epoch

    local model = nn.Sequential()
    
    model:add(Conv(input_channels, 32, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(32))
    model:add(Max(3, 3, 2, 2, 1, 1))

    model:add(Dropout(0.33))
    model:add(Conv(32, 64, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(64))
    model:add(Max(3, 3, 2, 2, 1, 1))


    model:add(Dropout(0.33))
    model:add(Upconv(64, 32, 4, 4, 2, 2, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(32))

    model:add(Dropout(0.33))
    model:add(Upconv(32, 32, 4, 4, 2, 2, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(32))
    
    model:add(Conv(32, class_count, 1, 1))

    local loss = cudnn.SpatialCrossEntropyCriterion()

    return model, loss
end

return create_model_camvid