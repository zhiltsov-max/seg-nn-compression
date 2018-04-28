local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'

local function create_model_camvid()
    -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
    -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
    local features = nn.Sequential()
    features:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))         -- 224 -> 55
    features:add(nn.SpatialBatchNormalization(64,1e-3))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2))                     -- 55 ->  27
    features:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))         --  27 -> 27
    features:add(nn.SpatialBatchNormalization(192,1e-3))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2))                     --  27 ->  13
    features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))        --  13 ->  13
    features:add(nn.SpatialBatchNormalization(384,1e-3))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))        --  13 ->  13
    features:add(nn.SpatialBatchNormalization(256,1e-3))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))        --  13 ->  13
    features:add(nn.SpatialBatchNormalization(256,1e-3))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2))                     -- 13 -> 6

    local classifier = nn.Sequential()
    classifier:add(nn.SpatialConvolution(256, 32, 1, 1))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.SpatialBatchNormalization(32,1e-3))
    classifier:add(nn.SpatialConvolution(32, 32, 1, 1))
    classifier:add(nn.SpatialUpSamplingBilinear({oheight=512, owidth=512}))

    local model = nn.Sequential()
        :add(features)
        :add(classifier)
        :cuda()

	local loss = cudnn.SpatialCrossEntropyCriterion()
	loss = loss:cuda()

    return model, loss
end

return create_model_camvid