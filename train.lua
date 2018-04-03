require 'optim'

local solver = {}

function solver.init(solver, model, dataset, options)
    solver.sgd = {}
    solver.sgd.learningRate = options.learningRate
    solver.sgd.momentum = options.momentum
    solver.sgd.learningRateDecay = options.lrDecay
    solver.sgd.learningRateDecayStep = options.lrDecayStep

    solver.dataset = dataset
    solver.train_data, _ = dataset:get_iterators()
    solver.image_size = {height = options.imHeight, width = options.imWidth}
    solver.minibatchCount = math.floor(#solver.dataset.train_data / options.batchSize)

    solver.batchSize = options.batchSize
    solver.weightDecay = options.weightDecay

    solver.inputs = torch.CudaTensor(options.batchSize, 
        options.channels, options.imHeight, options.imWidth)
    solver.targets = torch.CudaTensor(options.batchSize, 
        options.labelHeight, options.labelWidth)

    solver.weights, solver.dE_dw = model.model:getParameters() -- should be called after model:cuda()
    
    -- solver.confusion = optim.ConfusionMatrix(options.dataClasses)

    return solver
end

function solver.update_learning_rate(solver, epoch)
    local sgd = solver.sgd
    if (epoch ~= 0) and (epoch % sgd.learningRateDecayStep == 0) then 
        sgd.learningRate = sgd.learningRate * sgd.learningRateDecay
    end
end

function solver.load_batch(self, minibatch)
    local shuffle = self.shuffle
    local train_data = self.train_data

    self.inputs:zero()
    self.targets:zero()

    for i = 1, self.batchSize do
        local dataset_index = shuffle[(minibatch - 1) * self.batchSize + i]
        local dataset_entry = train_data[dataset_index]
        self.inputs[i]:copy(dataset_entry[1]:clone())
        self.targets[i]:copy(dataset_entry[2]:clone())
    end

    return self.inputs:cuda(), self.targets:cuda()
end

function solver.shuffle_data(self)
    self.shuffle = torch.randperm(#self.train_data)
end

function solver.train(solver, model, epoch)
    -- Perform one training epoch

    local sgd = solver.sgd
    local criterion = model.loss
    local net = model.model
    local minibatchCount = solver.minibatchCount

    net:training()
    solver:update_learning_rate(epoch)
    solver:shuffle_data()

    local epoch_time = 0

    for minibatch = 1, minibatchCount do
        local minibatch_time = sys.clock()

        local batch_loading_time = minibatch_time
        local x, y = solver:load_batch(minibatch)
        batch_loading_time = sys.clock() - batch_loading_time

        -- create closure to evaluate E(W) and dE/dW
        local eval_E = function(weights)
            -- reset gradients
            solver.dE_dw:zero()

            -- evaluate function for complete mini batch
            local f = net:forward(x)
            local loss = criterion:forward(f, y)
            -- estimate gradients dE_dw (stored in net)
            local dE_df = criterion:backward(f, y)
            net:backward(solver.inputs, dE_df)

            if solver.weightDecay ~= 0 then
                local norm = torch.norm

                -- Loss:
                loss = loss + solver.weightDecay * 0.5 * (norm(weights, 2) ^ 2)

                -- Gradients:
                solver.dE_dw:add(solver.weightDecay, weights)
            end

            -- print("w:", weights:min(), weights:max())
            -- print("f:", f:min(), f:max())

            return loss, solver.dE_dw
        end

        local optim_time = sys.clock()
        -- Perform SGD step:
        local sgd_state = {
            learningRate = sgd.learningRate,
            momentum = sgd.momentum,
            learningRateDecay = sgd.learningRateDecay
        }
        local _, loss = optim.sgd(eval_E, solver.weights, sgd)
        optim_time = sys.clock() - optim_time

        minibatch_time = sys.clock() - minibatch_time
        epoch_time = epoch_time + minibatch_time

        io.write(
            "\r",
            string.format(
                "Epoch #%d (%d/%d)" ..
                    ", %.3fs" ..
                    ", batch time %.3fs" ..
                " (" ..
                    "loading %.3fs" ..
                    ", optimization %.3fs" ..
                ")" ..
                " | " ..
                    "lr %.2e" ..
                    ", loss %.4f",
                epoch, minibatch, minibatchCount,
                epoch_time, minibatch_time, 
                batch_loading_time, optim_time, 
                sgd.learningRate, 
                loss[1]
            ),
            "\r") 
    end

    collectgarbage()

    return net, loss
end

return solver