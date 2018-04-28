-- Achieves 62.6% top1 on validation set at 35 epochs with this regime:
-- {  1,        9,    1e-1,    5e-4, },
-- { 10,      19,    1e-2,    5e-4  },
-- { 20,      25,    1e-3,    0 },
-- { 26,      30,    1e-4,    0 },
-- Trained model:
-- https://gist.github.com/szagoruyko/0f5b4c5e2d2b18472854


local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'

local function create_model_camvid()
    local nin = nn.Sequential()
    local function block(...)
        local arg = {...}
        local no = arg[2]
        nin:add(nn.SpatialConvolution(...))
        nin:add(nn.SpatialBatchNormalization(no,1e-3))
        nin:add(nn.ReLU(true))
        nin:add(nn.SpatialConvolution(no, no, 1, 1, 1, 1, 0, 0))
        nin:add(nn.SpatialBatchNormalization(no,1e-3))
        nin:add(nn.ReLU(true))
        nin:add(nn.SpatialConvolution(no, no, 1, 1, 1, 1, 0, 0))
        nin:add(nn.SpatialBatchNormalization(no,1e-3))
        nin:add(nn.ReLU(true))
    end

    local function mp(...)
        nin:add(nn.SpatialMaxPooling(...))
    end

    block(3, 96, 11, 11, 4, 4, 5, 5)
    mp(3, 3, 2, 2, 1, 1)
    block(96, 256, 5, 5, 1, 1, 2, 2)
    mp(3, 3, 2, 2, 1, 1)
    block(256, 384, 3, 3, 1, 1, 1, 1)
    mp(3, 3, 2, 2, 1, 1)
    
    local classifier = nn.Sequential()
    classifier:add(nn.SpatialConvolution(384, 32, 1, 1))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.SpatialBatchNormalization(32,1e-3))
    classifier:add(nn.SpatialConvolution(32, 32, 1, 1))
    classifier:add(nn.SpatialUpSamplingBilinear({oheight=512, owidth=512}))

    local model = nn.Sequential()
        :add(nin)
        :add(classifier)
        :cuda()

    local loss = cudnn.SpatialCrossEntropyCriterion()
    loss = loss:cuda()

    return model, loss
end

return create_model_camvid()