require 'xlua'
require 'optim'
require 'image'

local function test(model, dataset, epoch, options)
    -- local confusion_matrix = optim.ConfusionMatrix(options.dataClasses)

    model:evaluate()

    _, test_data = dataset:get_iterators()
    test_data_size = #test_data
    print("Running for " .. test_data_size .. " images...")

    local time = sys.clock()
    for i = 1, test_data_size do
        if i % math.max(1, math.floor(test_data_size / 10)) then
            io.write("\r", "Inference: ", i, "/", test_data_size, "\r")
        end
        
        local input_image = test_data[i][1]:clone()
        input_image = input_image:cuda()

        local prediction = model:forward(input_image)
        prediction = prediction[1]
        -- TODO: add confusion matrix computations

        local _, max_indices = torch.max(prediction, 1)
        local output_image = max_indices:float() / (256.0)
        local filepath = paths.concat(options.inferencePath, i .. "-" .. epoch .. ".png")
        image.save(filepath, output_image)
    end
    time = sys.clock() - time
    
    print("Inference time: " .. (time * 1000) .. "ms")
    print("Average time per image: " .. (time / test_data_size * 1000) .. "ms")

    collectgarbage()

    return total_error
end

return test