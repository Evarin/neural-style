require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'lfs'

local loadcaffe_wrap = require 'loadcaffe_wrapper'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_dir', 'examples/vangogh/',
           'Style target images')
cmd:option('-image_size', 256, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-seed', -1)

cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

function nn.SpatialConvolutionMM:accGradParameters()
  -- nop.  not needed by our net
end

local function main(params)
  if params.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  else
    params.backend = 'nn-cpu'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  print('loadcaffe')
  local cnn = loadcaffe_wrap.load(params.proto_file, params.model_file, params.backend):float()
  print('loadcuda')
  if params.gpu >= 0 then
    cnn:cuda()
  end
  
  -- Lists all the files from the style directory
  print('get images')
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_images = {}
  for file in io.popen('find "'..params.style_dir..'" -maxdepth 1 -type f'):lines() do
    print("found file "..file)
    table.insert(style_images, file)
  end
  
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style descriptor modules
  local style_descrs = {}
  local next_style_idx = 1
  local net = nn.Sequential()
  for i = 1, #cnn do
    if next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if params.gpu >= 0 then avg_pool_layer:cuda() end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local style_module = nn.StyleDescr(params.style_weight):float()
        if params.gpu >= 0 then
          style_module:cuda()
        end
        net:add(style_module)
        table.insert(style_descrs, style_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remote these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  
  
  -- Run each image to the network and save their style features
  for i=1, #style_images do
    local img = image.load(style_image[i], 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    if params.gpu >= 0 then
      img = img:cuda()
    end
    net:forward(img)
    for _, mod in ipairs(style_descrs) do
      -- TODO SAVE mod.G
    end
  end
  
  -- Unsupervised learning of the features
  -- TODO PCA on the style features
  
  -- Save final features
end


function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  return string.format('%s_%d.%s', basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style description (Gram Matrix) in-place
local StyleDescr, parent = torch.class('nn.StyleDescr', 'nn.Module')

function StyleDescr:__init(strength)
  parent.__init(self)
  self.strength = strength
  
  self.gram = GramMatrix()
  self.G = nil
end

function StyleDescr:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.output = input
  return self.output
end

-- Useless ?
function StyleDescr:updateGradInput(input, gradOutput)
  return gradOutput
end


local params = cmd:parse(arg)
main(params)
