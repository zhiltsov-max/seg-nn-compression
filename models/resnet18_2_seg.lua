local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'

local Convolution = nn.SpatialConvolution
local Upconvolution = nn.SpatialFullConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local Dropout = nn.SpatialDropout

local function create_model_camvid(options)
    local class_count = options.class_count

    -- Learning regime:
    -- 2-3k epochs
    -- base lr: 1, lr decay: 0.01, wdecay: 5e-4

    local depth = 18
    local shortcutType = 'B' -- 'C' or 'B'
    local iChannels

    -- The shortcut layer is either identity or 1x1 convolution
    local function shortcut(nInputPlane, nOutputPlane, stride)
        local useConv = shortcutType == 'C' or
            (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
        if useConv then
            -- 1x1 convolution
            return nn.Sequential()
                :add(SBatchNorm(nInputPlane))
                :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
        elseif nInputPlane ~= nOutputPlane then
            -- Strided, zero-padded identity shortcut
            return nn.Sequential()
                :add(nn.SpatialAveragePooling(1, 1, stride, stride))
                :add(nn.Concat(2)
                    :add(nn.Identity())
                    :add(nn.MulConstant(0)))
        else
            return nn.Identity()
        end
    end

    -- The basic residual layer block for 18 and 34 layer network, and the
    -- CIFAR networks
    local function basicblock(n, stride)
        local nInputPlane = iChannels
        iChannels = n

        local s = nn.Sequential()
        s:add(SBatchNorm(nInputPlane))
        s:add(ReLU(true))
        -- s:add(Dropout(0.25))
        s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
        s:add(SBatchNorm(n))
        s:add(ReLU(true))
        -- s:add(Dropout(0.25))
        s:add(Convolution(n,n,3,3,1,1,1,1))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, n, stride)))
            :add(nn.CAddTable(true))
    end

    -- The bottleneck residual layer for 50, 101, and 152 layer networks
    local function bottleneck(n, stride)
        local nInputPlane = iChannels
        iChannels = n * 4

        local s = nn.Sequential()
        s:add(SBatchNorm(nInputPlane))
        s:add(ReLU(true))
        s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
        s:add(SBatchNorm(n))
        s:add(ReLU(true))
        s:add(Convolution(n,n,3,3,stride,stride,1,1))
        s:add(SBatchNorm(n))
        s:add(ReLU(true))
        s:add(Convolution(n,n*4,1,1,1,1,0,0))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, n * 4, stride)))
            :add(nn.CAddTable(true))
    end

    -- Creates count residual blocks with specified number of features
    local function layer(block, features, count, stride)
        local s = nn.Sequential()
        for i=1,count do
            s:add(block(features, i == 1 and stride or 1))
        end
        return s
    end

    local function upsamplingblock1(bottomdim, outputdim)
        local internaldim = outputdim

        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(Upconvolution(bottomdim, internaldim, 1, 1, 2, 2, 0, 0, 1, 1))
        upsampling:add(ReLU(true))

        upsampling:add(SBatchNorm(internaldim))
        upsampling:add(Upconvolution(internaldim, outputdim, 5, 5, 1, 1, 2, 2))
        upsampling:add(ReLU(true))

        return upsampling
    end

    local function upsamplingblock2(bottomdim, outputdim)
        local internaldim = outputdim

        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(Upconvolution(bottomdim, internaldim, 1, 1, 2, 2, 0, 0, 1, 1))
        upsampling:add(ReLU(true))

        upsampling:add(SBatchNorm(internaldim))
        upsampling:add(Upconvolution(internaldim, internaldim, 3, 3, 1, 1, 1, 1))
        upsampling:add(ReLU(true))

        upsampling:add(SBatchNorm(internaldim))
        upsampling:add(Upconvolution(internaldim, outputdim, 3, 3, 1, 1, 1, 1))
        upsampling:add(ReLU(true))

        return upsampling
    end

    local function upsamplingblock3(bottomdim, outputdim)
        local internaldim = outputdim

        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(Upconvolution(bottomdim, internaldim, 3, 3, 2, 2, 1, 1, 1, 1))
        upsampling:add(ReLU(true))

        upsampling:add(SBatchNorm(internaldim))
        upsampling:add(Upconvolution(internaldim, outputdim, 3, 3, 1, 1, 1, 1))
        upsampling:add(ReLU(true))

        return upsampling
    end

    local function upsamplingblock4(bottomdim, outputdim)

        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(Upconvolution(bottomdim, outputdim, 4, 4, 2, 2, 1, 1))
        upsampling:add(ReLU(true))

        return upsampling
    end

    local function upsamplingblock5(bottomdim, outputdim)
        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(ReLU(true))
        -- upsampling:add(Dropout(0.25))
        upsampling:add(Upconvolution(bottomdim, outputdim, 4, 4, 2, 2, 1, 1))

        local skip = nn.Sequential()
        skip:add(Upconvolution(bottomdim, outputdim, 1, 1, 2, 2, 0, 0, 1, 1))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(skip)
                :add(upsampling)
            )
            :add(nn.CAddTable(true))
    end

    local function upsamplingblock6(bottomdim, outputdim)
        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(ReLU(true))
        upsampling:add(Upconvolution(bottomdim, outputdim, 4, 4, 2, 2, 1, 1))

        local skip = nn.Sequential()
        skip:add(Upconvolution(bottomdim, outputdim, 1, 1, 2, 2, 0, 0, 1, 1))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(skip)
                :add(upsampling)
            )
            :add(nn.JoinTable(2))
    end

    local function upsamplingblock7(bottomdim, outputdim)
        local internaldim = bottomdim
        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(ReLU(true))
        upsampling:add(Upconvolution(bottomdim, internaldim, 4, 4, 2, 2, 1, 1))
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(ReLU(true))
        upsampling:add(Upconvolution(internaldim, outputdim, 3, 3, 1, 1, 1, 1))

        local skip = nn.Sequential()
        skip:add(Upconvolution(bottomdim, outputdim, 1, 1, 2, 2, 0, 0, 1, 1))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(skip)
                :add(upsampling)
            )
            :add(nn.JoinTable(2))
    end

    local function upsamplingblock8(bottomdim, outputdim)
        local internaldim = bottomdim
        local upsampling = nn.Sequential()
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(ReLU(true))
        upsampling:add(Upconvolution(bottomdim, internaldim, 4, 4, 2, 2, 1, 1))
        upsampling:add(SBatchNorm(bottomdim))
        upsampling:add(ReLU(true))
        upsampling:add(Upconvolution(internaldim, outputdim, 5, 5, 1, 1, 2, 2))

        local skip = nn.Sequential()
        skip:add(Upconvolution(bottomdim, outputdim, 1, 1, 2, 2, 0, 0, 1, 1))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(skip)
                :add(upsampling)
            )
            :add(nn.CAddTable(true))
    end

    -- Creates block with the followning structure:
    -- input -> residual block -> bottom block        -> eltwise sum -> upsampling block -> output
    --   /                     -> residual connection ->
    -- input dim = output dim
    -- bottom block input dim = bottom block output dim
    local sblock1 = function(bottomblock, downsamplingblock, upsamplingblock)
        local block = nn.Sequential()
        if (bottomblock ~= nil) and (downsamplingblock ~= nil) then
            block:add(downsamplingblock)
            block:add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(bottomblock)
                    )
            block:add(nn.CAddTable(true))
        else 
            if downsamplingblock ~= nil then
                block:add(downsamplingblock)
            end
            
            if bottomblock ~= nil then
                block:add(bottomblock)
            end
        end

        if upsamplingblock ~= nil then
            block:add(upsamplingblock)
        end

        return block
    end

    local sblock2 = function(bottomblock, downsamplingblock, upsamplingblock)
        local block = nn.Sequential()
        if (bottomblock ~= nil) and (downsamplingblock ~= nil) then
            block:add(downsamplingblock)
            block:add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(bottomblock)
                    )
            block:add(nn.JoinTable(2))
        else 
            if downsamplingblock ~= nil then
                block:add(downsamplingblock)
            end
            
            if bottomblock ~= nil then
                block:add(bottomblock)
            end
        end

        if upsamplingblock ~= nil then
            block:add(upsamplingblock)
        end

        return block
    end

    local model = nn.Sequential()
    -- Configurations for ResNet:
    --  num. residual blocks, num features, residual block function
    local cfg = {
        [18]  = {{2, 2, 2, 2}, 512, basicblock},
        [34]  = {{3, 4, 6, 3}, 512, basicblock},
        [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
        [101] = {{3, 4, 23, 3}, 2048, bottleneck},
        [152] = {{3, 8, 36, 3}, 2048, bottleneck},
    }

    assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
    local def, nFeatures, block = table.unpack(cfg[depth])
    iChannels = 64
    print(' | ResNet-' .. depth .. ' ImageNet')

    -- The ResNet ImageNet model
    model:add(Convolution(3,64,7,7,2,2,3,3)) -- original 3,64,7,7,2,2,3,3
    model:add(SBatchNorm(64))
    model:add(ReLU(true))
    model:add(Max(3,3,2,2,1,1))
    -- creation order matters
    local block1residual = layer(block, 64, def[1])
    local block2residual = layer(block, 128, def[2], 2)
    local block3residual = layer(block, 256, def[3], 2)
    local block4residual = layer(block, 512, def[4], 2)
    local block5 = layer(block, 512, 2)
    iChannels = 256
    local block4_us = nn.Sequential()
        :add(upsamplingblock5(512, 256))
        :add(layer(block, 256, 2))
    local block4 = sblock1(block5, block4residual, block4_us)
    iChannels = 128
    local block3_us = nn.Sequential()
        :add(upsamplingblock5(256, 128))
        :add(layer(block, 128, 2))
    local block3 = sblock1(block4, block3residual, block3_us)
    iChannels = 64
    local block2_us = nn.Sequential()
        :add(upsamplingblock5(128, 64))
        :add(layer(block, 64, 2))
    local block2 = sblock1(block3, block2residual, block2_us)
    iChannels = 32
    local block1_us = nn.Sequential()
        :add(upsamplingblock5(64, 32))
        :add(layer(block, 32, 2))
    local block1 = sblock1(block2, block1residual, block1_us)
        :add(SBatchNorm(32))
        :add(ReLU(true))
        -- :add(Dropout(0.25))
        :add(Upconvolution(32, 32, 4, 4, 2, 2, 1, 1))
    model:add(block1)

    -- Classifier
    model:add(SBatchNorm(32))
    model:add(ReLU(true))
    model:add(Upconvolution(32, class_count, 1, 1))

    model = model:cuda()

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
            -- NoBias(v)
        end
    end

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