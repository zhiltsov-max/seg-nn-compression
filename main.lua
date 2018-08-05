require 'pl'
local lapp = require 'pl.lapp'
require 'paths'
require 'cutorch'
require 'cunn'
local nn = require 'nn'
local cudnn = require 'cudnn'
local models = require 'models.init'
local datasets = require 'datasets.common_loader'
local checkpoints = require 'checkpoints'
require 'paths'


function parse_args(arg)
    local opt = lapp [[
        Command line options:

        General:
        --train                                       Do model training
        --test                                        Do model testing
        
        Training Related:
        --learningRate          (default 5e-4)        learning rate
        --lrDecay               (default 1e-1)        Learning rate decay
        --weightDecay           (default 5e-4)        L2 penalty on the weights
        --momentum              (default 0.9)         Momentum
        -b,--batchSize          (default 10)          Batch size
        -e,--epochs             (default 30)          Maximum number of training epochs
        --testStep              (default 100)         Do model testing every X epoch on train data; 0 - disable
        --snapshotStep          (default 100)         Do model snapshot every X epoch; 0 - disable
        --save                  (default ./)          Save model snapshots and resulting model here
        --snapshot              (default none)        Model snapshot to start from

        Testing Related:
        --inferencePath         (default ./)          Path for the inference
        
        Device Related:
        -i,--devid              (default 1)           Device ID (if using CUDA)
        --nGPU                  (default 1)           Number of GPUs you want to train on
        --tensorType            (default torch.CudaTensor)
        --cudnnMode             (default fastest)     CuDNN mode: fastest | default. Fastest increases speed and memory consumption.
        --cudnnDebug                                  Print debug information for CuDNN
        --manualSeed            (default 0)           Random seed
        --deterministic                               Use deterministic computations. Slower, so use it for testing only!
        
        Dataset Related:
        --datapath              (default none)        Dataset location
        --dataset               (default voc)         Dataset type: voc, camvid, camvid12
        --imHeight              (default 512)         Image height
        --imWidth               (default 512)         Image width
        
        Model Related:
        --model                 (default none)        Model description file name
        --transferFrom          (default none)        Try to transfer weights from model; path
        --optnet                                      Use optnet for memory optimizations
    ]]

    -- Set default values

    if (opt.datapath == 'none') then
        opt.datapath = 'datasets/' .. opt.dataset
    end

    return opt
end

function main()
    local opt = parse_args(arg)
    print("Current execution options: ")
    print(opt)

    torch.setdefaulttensortype('torch.FloatTensor')
    cutorch.setDevice(opt.devid)
    cudnn.benchmark = (opt.cudnnMode == 'fastest')
    cudnn.fastest = (opt.cudnnMode == 'fastest')
    cudnn.verbose = (opt.cudnnDebug == true)
    if opt.manualSeed ~= 0 then
        torch.manualSeed(opt.manualSeed)
        cutorch.manualSeedAll(opt.manualSeed)
        math.randomseed(opt.manualSeed)
    end

    local dataset = datasets.loadDataset(opt.dataset, opt.datapath, opt)
    opt.classCount = dataset.class_count
    opt.inputChannelsCount = dataset.input_channel_count

    local model, cost = models.init(opt)

    local solver = nil
    if (opt.train == true) then
        print("Loading solver")
        solver = require 'train'

        if (opt.snapshot ~= 'none') then
            print('Loading checkpoint from \'' .. opt.snapshot .. '\'')
            local model_state, solver_state = checkpoints.load(opt.snapshot)
            models.restore_model_state(model_state, model)
            solver:init(model, cost, dataset, opt, solver_state)
        else
            solver:init(model, cost, dataset, opt)
        end

        print("Model save directory is '" .. opt.save .. "'")
        os.execute('mkdir -p ' .. opt.save)
    end

    local test = nil
    if (opt.test == true) then
        print("Loading tester")
        test = require 'test'
        
        print("Inference directory is '" .. opt.inferencePath .. "'")
        os.execute('mkdir -p ' .. opt.inferencePath)
    end

    if (opt.train == true) then
        print("Starting training")
        training_time = sys.clock()
        local epoch = solver:get_epoch()
        while (epoch <= opt.epochs) do
            solver:run_training_epoch()
            epoch = solver:get_epoch()

            if (((opt.snapshotStep ~= 0) and (epoch % opt.snapshotStep == 0)) or
            	(epoch == opt.epochs)) then
            
                checkpoints.save(model, solver, opt.save)
            end

            if ((opt.test == true) and (opt.testStep ~= 0) and (epoch % opt.testStep == 0)) then
                local subset = 'val'
                test(model, dataset, subset, true, paths.concat(opt.inferencePath, "epoch_" .. epoch, subset), opt)

                local subset = 'train'
                test(model, dataset, subset, true, paths.concat(opt.inferencePath, "epoch_" .. epoch, subset), opt)

                local subset = 'test'
                test(model, dataset, subset, true, paths.concat(opt.inferencePath, "epoch_" .. epoch, subset), opt)
            end
        end
        training_time = sys.clock() - training_time
        print(string.format("Training time: %.3fs", training_time))
    end

    if (opt.test == true) then
        print("Starting testing")
        local subset = 'test'
        test(model, dataset, subset, true, paths.concat(opt.inferencePath, "testing", subset), opt)
    end
end

main()