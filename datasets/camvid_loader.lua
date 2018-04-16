require 'paths'
require 'image'

local CamVid = {}
CamVid.__index = CamVid
CamVid.input_channel_count = 3
CamVid.class_count = 32
CamVid.classes = {
    'Void',
    'Archway',
    'Bicyclist',
    'Bridge',
    'Building',
    'Car',
    'CartLuggagePram',
    'Child',
    'Column_Pole',
    'Fence',
    'LaneMkgsDriv',
    'LaneMkgsNonDriv',
    'Misc_Text',
    'MotorcycleScooter',
    'OtherMoving',
    'ParkingBlock',
    'Pedestrian',
    'Road',
    'RoadShoulder',
    'Sidewalk',
    'SignSymbol',
    'Sky',
    'SUVPickupTruck',
    'TrafficCone',
    'TrafficLight',
    'Train',
    'Tree',
    'Truck_Bus',
    'Tunnel',
    'VegetationMisc',
    'Animal',
    'Wall'
}
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

    local file = io.open(list_file_path, 'r')
    if file == nil then
        error("Failed to open list file '" .. list_file_path .. "'.")
    end
    local line = file:read()
    while line ~= nil do
        local col1, col2 = line:match("(.+) (.+)")
        col1 = dataset_path .. col1
        col2 = dataset_path .. col2
        table.insert(img_paths, col1)
        table.insert(gt_paths, col2)

        line = file:read()
    end

    return img_paths, gt_paths
end

local function crop(img, crop_size)
    return image.crop(img, 'c', crop_size.width, crop_size.height)
end

function CamVid.load_sample(self, image_path, gt_path, crop_size)
    local img = image.load(image_path, 3, 'float')
    local gt_img = nil
    if gt_path ~= nil then
        gt_img = image.load(gt_path, 1, 'byte')
    end
    if crop_size ~= nil then
        img = crop(img, crop_size)

        if gt_img ~= nil then
            gt_img = crop(gt_img, crop_size)
        end
    end

    if gt_img ~= nil then
        gt_img = gt_img:squeeze():float():add(1)
    end

    return img, gt_img
end

function CamVid.get_statistics(self, list)
    local dataset_mean = torch.DoubleTensor(self.input_channel_count):zero()
    local dataset_std = torch.DoubleTensor(self.input_channel_count):zero()
    for i = 1, #list do
        local img = image.load(list[i][1], self.input_channel_count, 'float')
        for c = 1, self.input_channel_count do
            dataset_mean[c] = dataset_mean[c] + img:select(1, c):mean()
            dataset_std[c] = dataset_std[c] + img:select(1, c):std()
        end
    end

    dataset_mean:mul(1.0 / #list)
    dataset_std:mul(1.0 / (#list - 1.0))

    return {mean = dataset_mean, std = dataset_std}
end

function CamVid.preprocess_input(self, input, stats)
    -- Subtract mean, divide on std
    for c = 1, input:size(1) do
        if stats[mean] ~= nil then 
            input[c]:add(-stats.mean[c])
        end
        if stats[std] ~= nil then
            input[c]:div(stats.std[c])
        end
    end
end

function CamVid.load_data(self, list, stats)
    local data = {}
    for i = 1, #list do
        local paths = list[i]
        local input, target = self:load_sample(paths[1], paths[2], self.image_size)

        if stats ~= nil then
            self:preprocess_input(input, stats)
        end
        table.insert(data, { input, target })
    end
    return data
end

function CamVid.get_iterators(self)
    return self.train_data, self.val_data
end

local colormap_rgb = {}
colormap_rgb[1 + 0] = {0, 0, 0}
colormap_rgb[1 + 1] = {192, 0, 128}
colormap_rgb[1 + 2] = {0, 128, 192}
colormap_rgb[1 + 3] = {0, 128, 64}
colormap_rgb[1 + 4] = {128, 0, 0}
colormap_rgb[1 + 5] = {64, 0, 128}
colormap_rgb[1 + 6] = {64, 0, 192}
colormap_rgb[1 + 7] = {192, 128, 64}
colormap_rgb[1 + 8] = {192, 192, 128}
colormap_rgb[1 + 9] = {64, 64, 128}
colormap_rgb[1 + 10] = {128, 0, 192}
colormap_rgb[1 + 11] = {192, 0, 64}
colormap_rgb[1 + 12] = {128, 128, 64}
colormap_rgb[1 + 13] = {192, 0, 192}
colormap_rgb[1 + 14] = {128, 64, 64}
colormap_rgb[1 + 15] = {64, 192, 128}
colormap_rgb[1 + 16] = {64, 64, 0}
colormap_rgb[1 + 17] = {128, 64, 128}
colormap_rgb[1 + 18] = {128, 128, 192}
colormap_rgb[1 + 19] = {0, 0, 192}
colormap_rgb[1 + 20] = {192, 128, 128}
colormap_rgb[1 + 21] = {128, 128, 128}
colormap_rgb[1 + 22] = {64, 128, 192}
colormap_rgb[1 + 23] = {0, 0, 64}
colormap_rgb[1 + 24] = {0, 64, 64}
colormap_rgb[1 + 25] = {192, 64, 128}
colormap_rgb[1 + 26] = {128, 128, 0}
colormap_rgb[1 + 27] = {192, 128, 192}
colormap_rgb[1 + 28] = {64, 0, 64}
colormap_rgb[1 + 29] = {192, 192, 0}
colormap_rgb[1 + 30] = {64, 128, 64}
colormap_rgb[1 + 31] = {6, 192, 0}
colormap_rgb[1 + 255] = {0, 0, 0}

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
            io.open(img_paths[i], 'r')
            io.open(gt_paths[i], 'r')
            table.insert(result, {img_paths[i], gt_paths[i]})
        end
        
        return result
    end

    local train_list = create_list(dataset_path, paths.concat(dataset_path, "list", "train.txt"))
    local val_list = create_list(dataset_path, paths.concat(dataset_path, "list", "val.txt"))

    camvid = CamVid.create()
    camvid.image_size = {width = options.imWidth, height = options.imHeight}

    camvid.train_list = train_list
    camvid.train_data_stats = camvid:get_statistics(train_list)
    camvid.train_data = camvid:load_data(train_list, camvid.train_data_stats)

    camvid.val_list = val_list
    camvid.val_data_stats = camvid:get_statistics(val_list)
    camvid.val_data = camvid:load_data(val_list, camvid.val_data_stats)

    return camvid
end

return load