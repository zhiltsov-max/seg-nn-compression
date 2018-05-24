require 'optim'

local solver = {}

function solver.init(self, model, cost, dataset, options, previous_state)
    self.state = previous_state or {
        sgd = {
            learningRate = options.learningRate,
            momentum = options.momentum,
	        weightDecay = options.weightDecay
        },
        basicLrDecay = options.lrDecay,
        epoch = 1
    }

    self.model = model:cuda()
    self.cost = cost:cuda()

    self.dataset = dataset
    self.train_data, _ = dataset:get_iterators()
    self.image_size = {height = options.imHeight, width = options.imWidth}

    self.minibatch_count = math.floor(#self.dataset.train_data / options.batchSize)
    self.batch_size = options.batchSize
    self.inputs = torch.CudaTensor(options.batchSize,
        options.channels, options.imHeight, options.imWidth)
    self.targets = torch.CudaTensor(options.batchSize,
        options.imHeight, options.imWidth)
    self.state.sgd.learningRateDecay = self.state.basicLrDecay / self.minibatch_count

    self.weights, self.dE_dw = model:getParameters() -- should be called after model:cuda()

    return self
end

function solver.get_state(self)
    return self.state
end

function solver.get_epoch(self)
    return self.state.epoch
end

function solver.set_epoch(self, epoch)
    self.state.epoch = epoch
end

function solver.load_batch(self, minibatch)
    local shuffle = self.shuffle
    local train_data = self.train_data

    self.inputs:zero()
    self.targets:zero()

    for i = 1, self.batch_size do
        local dataset_index = shuffle[(minibatch - 1) * self.batch_size + i]
        local dataset_entry = train_data[dataset_index]
        self.inputs[i]:copy(dataset_entry[1])
        self.targets[i]:copy(dataset_entry[2])
    end

    return self.inputs:cuda(), self.targets:cuda()
end

function solver.shuffle_data(self)
    self.shuffle = torch.randperm(#self.train_data)
end

function solver.run_training_epoch(self)
    local epoch = self.state.epoch
    local model = self.model
    local cost = self.cost
    local weights = self.weights
    local sgd = self.state.sgd
    local dE_dw = self.dE_dw

    self:shuffle_data()

    model:training()

    local epoch_time = 0

    local minibatch_count = self.minibatch_count
    for minibatch = 1, minibatch_count do
        local minibatch_time = sys.clock()

        local x, y = self:load_batch(minibatch)

        -- create closure to evaluate E(W) and dE/dW
        local eval_E = function(weights)
            -- evaluate function for complete minibatch
            local f = model:forward(x)
            local loss = cost:forward(f, y)

            -- reset gradients
            dE_dw:zero()
            -- estimate gradients dE_dw (stored in model)
            local dE_df = cost:backward(f, y)
            model:backward(x, dE_df)

            return loss, dE_dw
        end

        local _, loss = optim.sgd(eval_E, weights, sgd)

        minibatch_time = sys.clock() - minibatch_time
        epoch_time = epoch_time + minibatch_time

        io.write(
            "\r",
            string.format(
                "Epoch #%d (%d/%d)" ..
                    ", time %.3fs" ..
                    ", batch time %.3fs" ..
                " | " ..
                    "lr %.2e" ..
                    ", loss %.4f",
                epoch, minibatch, minibatch_count,
                epoch_time, minibatch_time,
                sgd.learningRate / (1 + sgd.evalCounter * sgd.learningRateDecay), -- from optim/sgd
                loss[1]
            ),
            "\r")
    end

    self:set_epoch(epoch + 1)

    collectgarbage()

    return model, loss
end

return solver