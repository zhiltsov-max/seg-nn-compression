-- Adapter for SegNet modified CamVid dataset
-- Data can be obtained here: https://github.com/alexgkendall/SegNet-Tutorial

require 'paths'
require 'image'

local CamVid = {}
CamVid.__index = CamVid
CamVid.input_channel_count = 3
CamVid.classes = {
    'Sky',
    'Building',
    'Pole',
    'Road',
    'Pavement',
    'Tree',
    'SignSymbol',
    'Fence',
    'Car',
    'Pedestrian',
    'Bicyclist',
    'Unlabelled'
}
CamVid.class_count = #CamVid.classes
CamVid.train_data = nil
CamVid.test_data = nil
CamVid.image_size = { width = 0, height = 0 }

function CamVid.get_train_entry(self, index)
    return self.train_data[index]
end

function CamVid.get_test_entry(self, index)
    return self.test_data[index]
end

function CamVid.get_paths(list_file_path, dataset_path)
    local img_paths = {}
    local gt_paths = {}

    local file = assert(io.open(list_file_path, 'r'))
    local line = file:read()
    while line ~= nil do
        local col1, col2 = line:match("(.+) (.+)")
        col1 = dataset_path .. col1
        col2 = dataset_path .. col2
        table.insert(img_paths, col1)
        table.insert(gt_paths, col2)

        line = file:read()
    end

    file:close()

    return img_paths, gt_paths
end

local function crop_or_pad(img, crop_size, filler)
    local pad = ((img:size()[2] < crop_size.height) or (img:size()[3] < crop_size.width))
    local real_crop_size = {}
    real_crop_size.height = math.min(img:size()[2], crop_size.height)
    real_crop_size.width = math.min(img:size()[3], crop_size.width)
    if (pad == true) then
        local output = torch.Tensor(img:size()[1], crop_size.height, crop_size.width)
        if (filler ~= nil) then
            if (type(filler) == 'number') then
                output:fill(filler)
            elseif (filler:size()[1] == output:size()[1]) then
                for c = 1, output:size()[1] do
                    output[c]:fill(filler[c])
                end
            else
                error('Wrong padding filler type.')
            end
        end
        output[{ {}, {1, real_crop_size.height}, {1, real_crop_size.width} }] = image.crop(img, 'c', real_crop_size.width, real_crop_size.height)
        return output
    else
        return image.crop(img, 'c', real_crop_size.width, real_crop_size.height)
    end
end

function CamVid.load_sample(self, image_path, gt_path)
    local img = image.load(image_path, 3, 'float')
    local gt_img = nil
    if (gt_path ~= nil) then
        gt_img = image.load(gt_path, 1, 'byte')
    end

    return img, gt_img
end

function CamVid.get_statistics(self, list)
    local dataset_mean = torch.Tensor(self.input_channel_count):zero()
    local dataset_std = torch.Tensor(self.input_channel_count):zero()
    for i = 1, #list do
        local img = image.load(list[i][1], self.input_channel_count, 'float')
        img = img:view(self.input_channel_count, -1)
        dataset_mean:add(img:mean(2))
        dataset_std:add(torch.pow(img:std(2), 2))
    end

    dataset_mean:mul(1.0 / #list)
    dataset_std:sqrt():mul(1.0 / (#list - 1.0))

    return {mean = dataset_mean, std = dataset_std}
end

function CamVid.preprocess_input(self, input, target, stats, crop_size)
    if (crop_size ~= nil) then
        input = crop_or_pad(input, crop_size, stats.mean or nil)

        if (target ~= nil) then
            local void_class = self.class_count - 1
            target = crop_or_pad(target, crop_size, void_class)
            target = target:squeeze():float():add(1)
        end
    end

    if (stats ~= nil) then
        -- Subtract mean, divide on std
        for c = 1, input:size(1) do
            if stats.mean ~= nil then
                input[c]:add(-stats.mean[c])
            end
            if stats.std ~= nil then
                input[c]:div(stats.std[c])
            end
        end
    end

    return input, target
end

function CamVid.load_data(self, list, stats)
    local data = {}
    for i = 1, #list do
        local paths = list[i]
        local input, target = self:load_sample(paths[1], paths[2])
        input, target = self:preprocess_input(input, target, stats, self.image_size)
        table.insert(data, { input, target })
    end
    return data
end

function CamVid.get_iterators(self)
    return self.train_data, self.val_data
end

local colormap_rgb = {}
colormap_rgb[1 + 0] =  {128,128,128}
colormap_rgb[1 + 1] =  {128,0,0}
colormap_rgb[1 + 2] =  {192,192,128}
colormap_rgb[1 + 3] =  {128,64,128}
colormap_rgb[1 + 4] =  {60,40,222}
colormap_rgb[1 + 5] =  {128,128,0}
colormap_rgb[1 + 6] =  {192,128,128}
colormap_rgb[1 + 7] =  {64,64,128}
colormap_rgb[1 + 8] =  {64,0,128}
colormap_rgb[1 + 9] = {64,64,0}
colormap_rgb[1 + 10] = {0,128,192}
colormap_rgb[1 + 11] = {0,0,0}
colormap_rgb[1 + 255] = {0,0,0}


local function apply_colormap_r(c)
    return colormap_rgb[c][1]
end
local function apply_colormap_g(c)
    return colormap_rgb[c][2]
end
local function apply_colormap_b(c)
    return colormap_rgb[c][3]
end

function CamVid.paint_segmentation(self, class_index_map)
    local rgb_maps = class_index_map:clone():repeatTensor(3, 1, 1)
    rgb_maps[1]:apply(apply_colormap_r)
    rgb_maps[2]:apply(apply_colormap_g)
    rgb_maps[3]:apply(apply_colormap_b)
    return rgb_maps
end

function CamVid.create(t)
    t = t or {}
    setmetatable(t, CamVid)
    return t
end

local function load(dataset_path, options)
    local function create_list(dataset_path, list_path)
        print("Extracting file names from: " .. list_path)
        local img_paths, gt_paths = CamVid.get_paths(list_path, dataset_path)

        local result = {}
        for i = 1, #img_paths do
            f = assert(io.open(img_paths[i], 'r'))
            f:close()

            f = assert(io.open(gt_paths[i], 'r'))
            f:close()

            table.insert(result, {img_paths[i], gt_paths[i]})
        end
        
        return result
    end

    local train_list = create_list(dataset_path, paths.concat(dataset_path, "list", "train.txt"))
    local val_list = create_list(dataset_path, paths.concat(dataset_path, "list", "val.txt"))
    local test_list = create_list(dataset_path, paths.concat(dataset_path, "list", "test.txt"))

    camvid = CamVid.create()
    camvid.image_size = {width = options.imWidth, height = options.imHeight}

    camvid.train_list = train_list
    camvid.train_data_stats = camvid:get_statistics(train_list)
    camvid.train_data = camvid:load_data(train_list, camvid.train_data_stats)

    camvid.val_list = val_list
    camvid.val_data_stats = camvid:get_statistics(val_list)
    camvid.val_data = camvid:load_data(val_list, camvid.val_data_stats)

    camvid.test_list = test_list
    camvid.test_data_stats = camvid:get_statistics(test_list)
    camvid.test_data = camvid:load_data(test_list, camvid.val_data_stats)

    return camvid
end

return load