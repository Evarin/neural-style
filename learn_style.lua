require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'lfs'
require 'io'
require 'loadcaffe'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_dir', 'examples/vangogh/',
           'Style target images')
cmd:option('-style_max_number', 100000)
cmd:option('-image_size', 256, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Output options
cmd:option('-output_file', 'style.data')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)

cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

function nn.SpatialConvolutionMM:accGradParameters()
   -- nop.  not needed by our net
end

local function main(params)
   if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
   
  print('loadcaffe')
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
   
   
   -- Lists all the files from the style directory
   print('get images')
   local style_size = math.ceil(params.style_scale * params.image_size)
   local style_images = {}
   for file in io.popen('find "'..params.style_dir..'" -maxdepth 1 -type f'):lines() do
      print("found file "..file)
      if #style_images < params.style_max_number then
         table.insert(style_images, file)
      end
   end
   
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
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  avg_pool_layer:cuda()
	       else
		  avg_pool_layer:cl()
	       end
	    end
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
	       if params.backend ~= 'clnn' then
		  style_module:cuda()
	       else
		  style_module:cl()
	       end
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
   
   local outputs = {}
   for i=1, #style_descrs do
      table.insert(outputs, {})
   end
   
   -- Run each image to the network and save their style features
   for i=1, #style_images do
      print('processing image '..i..': '..style_images[i])
      local img = image.load(style_images[i], 3)
      if img then
	 img = image.scale(img, style_size, 'bilinear')
	 local img_caffe = preprocess(img):float()
	 if params.gpu >= 0 then	    
	    if params.backend ~= 'clnn' then
	       img_caffe = img_caffe:cuda()
	    else
	       img_caffe = img_caffe:cl()
	    end
	 end
	 net:forward(img_caffe)
	 for j, mod in ipairs(style_descrs) do
	    table.insert(outputs[j], mod.G:clone())
	 end
      end
   end
   

   net = nil
   collectgarbage()

   -- Unsupervised learning of the features
   local style_outputs = {}
   for i=1, #style_descrs do
      local st = stack_gram(outputs[i])
      if params.gpu >= 0 then	    
	 if params.backend ~= 'clnn' then
	    st = st:cuda()
	 else
	    st = st:cl()
	 end
      end
      print('Variance '..i..': layer '..style_layers[i])
      local mean, var = variance(st, params.gpu, params.backend)
      table.insert(style_outputs, {layer = style_layers[i], mean = mean, var = var})
   end
   
   -- Save final features
   torch.save(params.output_file, style_outputs)
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

-- Converts table of gram matrices into Matrix
function stack_gram(gs)
   local sz = gs[1]:nElement()
   local st = gs[1]:reshape(gs[1], 1, sz)
   for j=2, #gs do
      st = st:cat(torch.reshape(gs[j], 1, sz))
   end
   return st
end

-- Converts table of gram matrices into Matrix, keeping only upper-triangular matrix
function linearize_gram(gs)
   local sz = gs[1]:size()[1]
   local indices = torch.Tensor((sz*(sz+1))/2)
   local ik = 1
   for j=1, sz do
      for k=1, j do
	 indices[ik] = (j-1)*sz + k
	 ik = ik+1
      end
   end
   local st = torch.Tensor(#gs, indices:nElement())
   for j=1, #gs do
      local o = gs[j]:storage()
      for k=1, indices:nElement() do
	 st[j][k] = o[indices[k]]
      end
   end
   return st, sz
end

-- Undo the previous process with the eigenvector matrix
-- from S(S-1)/2 square matrix, outputs S x S(S-1)/2 matrix (i.e. same number of eigenvectors)
function delinearize_gram(sz, ev)
   local gsz = sz*(sz+1)/2
   local output = torch.Tensor(sz*sz, gsz)
   local ik = 1
   for j=1, sz do
      for k=1, j-1 do
	 local i1 = (j-1)*sz + k
	 local i2 = (k-1)*sz + j
	 for l=1, gsz do
	    output[i1][l] = ev[ik][l]/2
	    output[i2][l] = ev[ik][l]/2
	 end
	 ik = ik+1
      end
      local i = (j-1)*sz + j
      for l=1, gsz do
	 output[i][l] = ev[ik][l]
      end
      ik = ik+1
   end
   return output
end

-- from package unsup by koraykv
-- PCA using covariance matrix
-- x is supposed to be MxN matrix, where M samples(trials) and each sample(trial) is N dim
-- returns the eigen values and vectors of the covariance matrix in increasing order
function pcacov(x, gpu, backend)
   local mean = torch.mean(x,1)
   local xm = x - torch.ger(torch.ones(x:size(1)),mean:squeeze())
   local c = torch.mm(xm:t(),xm)
   c:div(x:size(1)-1)
   if gpu >=0 then
      if backend ~= 'clnn' then
	 c = c:cuda()
      else
	 -- not implemented in cltorch
	 c = c
      end
   end
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end

-- variance
-- x is supposed to be MxN matrix, where M samples(trials) and each sample(trial) is N dim
-- returns the eigen values and vectors of the covariance matrix in increasing order
function variance(x, gpu, backend)
   local mean = torch.mean(x, 1)
   local var = torch.mean(torch.pow(x, 2), 1) - torch.pow(mean, 2)
   return mean, var
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
