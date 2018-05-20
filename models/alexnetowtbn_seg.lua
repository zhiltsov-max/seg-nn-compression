local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'

local Convolution = nn.SpatialConvolution
local Upconvolution = nn.SpatialFullConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function create_model_camvid(options)
    local class_count = options.class_count

    -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
    -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
    local features = nn.Sequential()
    features:add(Convolution(3,64,11,11,4,4,2,2))         -- 224 -> 55
    features:add(ReLU(true))
    features:add(SBatchNorm(64,1e-3))
    features:add(Max(3,3,2,2))                     -- 55 ->  27
    features:add(Convolution(64,192,5,5,1,1,2,2))         --  27 -> 27
    features:add(ReLU(true))
    features:add(SBatchNorm(192,1e-3))
    features:add(Max(3,3,2,2))                     --  27 ->  13
    features:add(Convolution(192,384,3,3,1,1,1,1))        --  13 ->  13
    features:add(ReLU(true))
    features:add(SBatchNorm(384,1e-3))
    features:add(Convolution(384,256,3,3,1,1,1,1))        --  13 ->  13
    features:add(ReLU(true))
    features:add(SBatchNorm(256,1e-3))
    features:add(Convolution(256,256,3,3,1,1,1,1))        --  13 ->  13
    features:add(ReLU(true))
    features:add(SBatchNorm(256,1e-3))
    features:add(Max(3,3,2,2))                     -- 13 -> 6

    local upsampling = nn.Sequential()
    upsampling:add(SBatchNorm(256,1e-3))
    upsampling:add(Upconvolution(256, 128, 4, 4, 2, 2, 1, 1, 1, 1))
    upsampling:add(ReLU(true))
    upsampling:add(SBatchNorm(128,1e-3))
    upsampling:add(Upconvolution(128, 128, 4, 4, 2, 2, 1, 1, 1, 1))
    upsampling:add(ReLU(true))
    upsampling:add(SBatchNorm(128,1e-3))
    upsampling:add(Upconvolution(128, 64, 4, 4, 2, 2, 1, 1, 1, 1))
    upsampling:add(ReLU(true))
    upsampling:add(SBatchNorm(64,1e-3))
    upsampling:add(Upconvolution(64, 32, 4, 4, 2, 2, 1, 1, 1, 1))
    upsampling:add(ReLU(true))
    upsampling:add(SBatchNorm(32,1e-3))
    upsampling:add(Upconvolution(32, 32, 5, 5, 2, 2, 1, 1, 1, 1))
    upsampling:add(ReLU(true))

    local classifier = nn.Sequential()
    classifier:add(SBatchNorm(32,1e-3))
    classifier:add(Convolution(32, class_count, 1, 1))

    local model = nn.Sequential()
        :add(features)
        :add(upsampling)
        :add(classifier)
        :cuda()

	local loss = cudnn.SpatialCrossEntropyCriterion()
	loss = loss:cuda()

    return model, loss
end

return create_model_camvid