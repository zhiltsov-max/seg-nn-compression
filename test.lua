require 'xlua'
require 'optim'
require 'image'
require 'paths'


local function rfind(str, pattern)
    local pos = str:find(pattern)
    local cur_pos = pos
    while (cur_pos ~= nil) do
        pos = cur_pos
        cur_pos = str:find(pattern, cur_pos + 1)
    end
    return pos
end

local function get_file_name(path)
    local begin_idx = (rfind(path, "/") or 0) + 1
    local end_idx = (path:find(".", begin_idx, true) or 0) - 1
    return path:sub(begin_idx, end_idx)
end

local function test(model, dataset, epoch, validation, options)
    local confusion_matrix = nil
    if (validation == true) then
        confusion_matrix = optim.ConfusionMatrix(dataset.class_count, dataset.classes)
        confusion_matrix:zero()
    end

    model:evaluate()

    local _, test_data = dataset:get_iterators()
    local test_data_size = #test_data
    print("Running for " .. test_data_size .. " images...")

    local save_dir = paths.concat(options.inferencePath, "epoch_" .. epoch)
	os.execute('mkdir -p ' .. save_dir)
	local save_dir_raw = paths.concat(save_dir, 'raw')
	os.execute('mkdir -p ' .. save_dir_raw)
	local save_dir_painted = paths.concat(save_dir, 'painted')
	os.execute('mkdir -p ' .. save_dir_painted)

    local test_list = dataset.val_list

    -- Preallocation for inputs
    local input = torch.CudaTensor(1, 
        options.channels, options.imHeight, options.imWidth)

    local time = sys.clock()
    for i = 1, test_data_size do
        if i % math.max(1, math.floor(test_data_size / 10)) then
            io.write("\r", "Inference: ", i, "/", test_data_size, "\r")
        end
        
        input:zero()
        
        local input_image = test_data[i][1]
        input:copy(input_image):cuda()

        local output = model:forward(input)
        local prediction = output[1]
        -- TODO: add confusion matrix computations

        local max_values, max_indices = torch.max(prediction, 1)

        if (validation == true) then
            local target = test_data[i][2]
            local mask = target:lt(255)
            local target_size = target:size()
            confusion_matrix:batchAdd(max_indices[1][{{1, target_size[1]}, {1, target_size[2]}}][mask]:view(-1), target[mask]:view(-1))
        end

        local image_name = get_file_name(test_list[i][1])

        local output_image_raw = max_indices:float():mul(1.0 / 255.0)
        local filepath_raw = paths.concat(save_dir_raw, image_name .. ".png")
        image.save(filepath_raw, output_image_raw)
        
        local input_size = input_image:size()
        local output_image_painted = dataset:paint_segmentation(max_indices:float()):mul(1.0 / 255.0)
        local filepath_painted = paths.concat(save_dir_painted, image_name .. ".png")
        image.save(filepath_painted, output_image_painted)
    end
    time = sys.clock() - time
    
    print(string.format("Inference time: %.3fs", time))
    print(string.format("Average time per image: %.3fs", time / test_data_size))

    if (validation == true) then
        print(confusion_matrix)
    end

    collectgarbage()
end

return test