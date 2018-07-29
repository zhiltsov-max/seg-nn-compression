-- http://ccvl.stat.ucla.edu/ccvl/DeepLab-MSc-COCO-LargeFOV/train.prototxt
-- Model from above link. Above has caffe model
-- Implementation from https://github.com/Aliases/torch-models

local nn = require 'nn'
require 'nngraph'
require 'cunn'
local cudnn = require 'cudnn'

local MaxPooling2D = nn.SpatialMaxPooling
local Convolution = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local UpConvolution = nn.SpatialFullConvolution
local Identity = nn.Identity
local ReLU = nn.ReLU
local Dropout = nn.Dropout

  -- net:add(Convolution(nIn, nOut, kernelX, kernelY, strideX, strideY, padX, padY))

-- In caffe layers, default stride = 1 , pad = 0
-- Same in torch

local function DirectPath(nIn, receptiveField, noClasses)
  -- Input is data
  local nInp = nIn
  local nOut = 128
  local stride = receptiveField or 1 -- stride
  -- 8 for path0
  -- 4 for path1
  -- 2 for path2
  -- 1 for path3
  -- 1 for path4
  local net = nn.Sequential()
  net:add(Convolution(nInp, nOut, 3, 3, stride, stride, 1, 1 ))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, nOut, 1, 1, 1, 1)) -- padding not specified
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(nOut, noClasses, 1, 1)) -- padding and stride not specified
  return net
end

local function Intermediate(nIn, poolStride, numConv, nOut_)
  local noChannels = 1
  local nInp = nIn or noChannels
  local nOut
  local stride = poolStride or 2
  if nInp == noChannels or nInp < 4 then -- < 4 because possible to have rgb channel.
    -- Now noChannels argument not neeeded in this function.
    -- Good to remove it to make life generic
    nOut = 64
  else
    nOut = 2*nInp
  end
  nOut = nOut_ or nOut

  local net = nn.Sequential()
  net:add(Convolution(nInp, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))
  net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
  net:add(ReLU(true))

  if numConv == 3 then -- for intermediate2 and intermediate3
    net:add(Convolution(nOut, nOut, 3, 3, 1, 1, 1, 1)) -- stride not specified
    net:add(ReLU(true))
  end

  net:add(MaxPooling2D(3, 3, stride, stride, 1, 1)) -- pool1
  -- stride = 2 for intermediate0
  -- stride = 2 for intermediate1
  -- stride = 2 for intermediate2
  -- stride = 1 for intermediate3
  return net
end

local function kerEff(ker, hole)
  return ker + (ker -1) * (hole - 1)
end

local function lastPart(noClasses)
  -- input is pool4
  local net = nn.Sequential()
  net:add(Convolution(512, 512, 5, 5, 1, 1, 2, 2)) -- stride not specified -- hole = 2, ker =3, ker_eff = ker_h + (ker_h -1 )*(hole-1),
  net:add(ReLU(true))
  net:add(Convolution(512, 512, 5, 5, 1, 1, 2, 2)) -- stride not specified -- hole = 2
  net:add(ReLU(true))
  net:add(Convolution(512, 512, 5, 5, 1, 1, 2, 2)) -- stride not specified -- hole = 2
  net:add(ReLU(true))
  net:add(MaxPooling2D(3, 3, 1, 1, 1, 1)) -- pool4
  net:add(nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
  net:add(Convolution(512, 1024, 25, 25, 1, 1, 12, 12)) -- stride not specified - hole =12
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(1024, 1024, 1, 1))
  net:add(ReLU(true))
  net:add(Dropout(0.5))
  net:add(Convolution(1024, noClasses, 1, 1)) -- fc8_$EXP
  -- Fuse layers after this
  return net
end


function create_model_camvid(options)
  local noClasses = options.classCount
  local nInChannels = options.inputChannelsCount

  local input = nn.Identity()()


  local data_ms = DirectPath(nInChannels, 8, noClasses)(input)

  local pool1 = Intermediate(nInChannels, 2)(input)
  local pool1_ms = DirectPath(64, 4, noClasses)(pool1)

  local pool2 = Intermediate(64, 2)(pool1)
  local pool2_ms = DirectPath(128, 2, noClasses)(pool2)

  local pool3 = Intermediate(128, 2, 3)(pool2)
  local pool3_ms = DirectPath(256, 1, noClasses)(pool3)

  local pool4 = Intermediate(256, 1, 3)(pool3)
  local pool4_ms = DirectPath(512, 1, noClasses)(pool4)

  local fc8_ = lastPart(noClasses)(pool4)

  -- fuse layers
  -- fuse data_ms , pool1_ms , pool2_ms , pool3_ms, pool4_ms, fc8_$Exp
  local output = nn.CAddTable()({data_ms, pool1_ms, pool2_ms, pool3_ms, pool4_ms, fc8_})

  -- Either the below 3 layers can be used to scale to 8 times back to original size
  -- or use SpatialUpSamplingNearest as done here
  -- output = UpConvolution(noClasses, noClasses, 2, 2, 2, 2)(output)
  -- output = UpConvolution(noClasses, noClasses, 2, 2, 2, 2)(output)
  -- output = UpConvolution(noClasses, noClasses, 2, 2, 2, 2)(output)

  output = nn.SpatialUpSamplingNearest(8)(output)

  -- local model = nn.gModule({input}, {data_ms, pool1_ms, pool2_ms, pool3_ms, pool4_ms, fc8_, output})
  local model = nn.gModule({input}, {output})
  model =  model:cuda()

  local loss = cudnn.SpatialCrossEntropyCriterion()
  loss = loss:cuda()

  return model, loss
end

return create_model_camvid