require 'pl'
local lapp = require 'pl.lapp'

local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local models = require 'models.init'
local datasets = require 'datasets.common_loader'
local checkpoints = require 'checkpoints'


function parse_args(arg)
    local opt = lapp [[
        Command line options:

        General:
        --train                                       Do model training
        --test                                        Do model testing
        
        Training Related:
        --learningRate          (default 5e-4)        learning rate
        --lrDecay               (default 1e-1)        Learning rate decay
        --lrDecayStep           (default 100)         Learning rate decay step (in # epochs)
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
        
        Dataset Related:
        --channels              (default 3)           Number of input channels
        --datapath              (default ./datasets/VOC) Dataset location
        --dataset               (default voc)         Dataset type: voc (PASCAL VOC 2012), camvid
        --imHeight              (default 512)         Image height (voc: 512)
        --imWidth               (default 512)         Image width  (voc: 512)
        
        Model Related:
        --model                 (default none)        Model description file name
        --transferFrom          (default none)        Try to transfer weights from model; path
        --optnet                                      Use optnet for memory optimizations
    ]]

    return opt
end

function main()
    local opt = parse_args(arg)
    print("Current execution options: ")
    print(opt)

    torch.setdefaulttensortype('torch.FloatTensor')
    
    cutorch.setDevice(opt.devid)
    print("Running on device " .. opt.devid)

    local model, cost = models.init(opt)
    print(model)
    print("Output shape is: " ..
        table.concat(
            torch.totable(
                model:forward(torch.Tensor(1, 3, opt.imHeight, opt.imWidth):cuda()):size()
            ),
            'x'
        )
    )

    local dataset = datasets.loadDataset(opt.dataset, opt.datapath, opt)

    local solver = nil
    if (opt.train == true) then
        print("Loading solver")
        solver = require 'train'

        if (opt.snapshot ~= 'none') then
            print('Loading checkpoint from \'' .. opt.snapshot .. '\'')
            local old_model_state, old_solver_state = checkpoints.load(opt.snapshot)
            models.restore_model_state(old_model_state, model)
            solver:init(model, cost, dataset, opt, old_solver_state)
            solver:set_epoch(solver:get_epoch() + 1)
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
        
        os.execute('mkdir -p ' .. opt.inferencePath)
    end


    if (opt.train == true) then
        print("Starting training")
        local epoch = solver:get_epoch()
        while (epoch <= opt.epochs) do
            solver:run_training_epoch()
            if ((opt.snapshotStep ~= 0) and (epoch % opt.snapshotStep == 0)) then
                checkpoints.save(model, solver, opt.save)
            end

            if ((opt.testStep ~= 0) and (epoch % opt.testStep == 0)) then
                test(model, dataset, epoch, true, opt)
            end

            epoch = solver:get_epoch()
        end
    end

    if (opt.test == true) then
        test(model, dataset, 0, true, opt)
    end
end

main()