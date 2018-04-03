function parse_args(arg)
    require 'pl'
    local lapp = require 'pl.lapp'
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
        --testStep              (default 100)         Do model testing every X epoch on train data
        --snapshotStep          (default 100)         Do model snapshot every X epoch
        --save                  (default ./)          Save trained model here

        Testing Related:
        --inferencePath         (default ./)          Path for the inference
        
        Device Related:
        -i,--devid              (default 1)           Device ID (if using CUDA)
        --nGPU                  (default 1)           Number of GPUs you want to train on
        
        Dataset Related:
        --channels              (default 3)
        --datapath              (default ./datasets/VOC) Dataset location
        --dataset               (default voc)         Dataset type: voc(PASCAL VOC 2012)
        --imHeight              (default 512)         Image height  (voc: 512)
        --imWidth               (default 512)         Image width   (voc: 512)
        --labelHeight           (default 512)         Label height  (voc: 512)
        --labelWidth            (default 512)         Label width   (voc: 512)
        
        Model Related:
        --model                 (default none)        Model path
    ]]

    return opt
end

function save_model(model, epoch, options)
    -- this have to be after :cuda and after :getParameters
    local lightModel = model:getParameters()
    local filepath = paths.concat(options.save, 'model_iter_' .. epoch .. '.t7')
    torch.save(filepath, lightModel)
end

function main()
    local opt = parse_args(arg)
    print("Current execution options: ")
    print(opt)

    torch.setdefaulttensortype('torch.FloatTensor')
    require 'nn'
    require 'cudnn'
    require 'cunn'
    print("Running on device " .. opt.devid .. ":")
    -- print(cutorch.getDeviceProperties(opt.devid))
    cutorch.setDevice(opt.devid)

    require 'datasets/common_loader'
    local dataset = loadDataset(opt.dataset, opt.datapath, opt)

    print("Model folder is '" .. opt.save .. "'")
    os.execute('mkdir -p ' .. opt.save)

    if opt.test == true then
        os.execute('mkdir -p ' .. opt.inferencePath)
    end

    print("Loading selected model")
    local model = paths.dofile(opt.model)(opt)
    print(model.model)
    print("Output shape is:", model.model:forward(torch.Tensor(1, 3, opt.imHeight, opt.imWidth):cuda()):size())

    local solver = nil
    if opt.train == true then
        print("Loading solver")
        solver = require 'train'
        solver:init(model, dataset, opt)
    end

    local test = nil
    if opt.test == true then
        print("Loading tester")
        test = require 'test'
    end

    if opt.train == true then
        print("Starting training")
        for epoch = 1, opt.epochs do
            solver:train(model, epoch)
            if opt.snapshotStep ~= nil then
                if epoch % opt.snapshotStep == 0 then
                    save_model(model.model, epoch, opt)
                end
            end

            if opt.testStep ~= nil then
                if epoch % opt.testStep == 0 then
                    test(model.model, dataset, epoch, opt)
                end
            end
        end

        save_model(model.model, epoch, opt)
    end

    if opt.test == true then
        test(model.model, dataset, 0, opt)
    end
end

main()