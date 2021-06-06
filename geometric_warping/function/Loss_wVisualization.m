classdef Loss_wVisualization < dagnn.Loss
  methods
    function outputs = forward(obj, inputs, params)
        [~,~,~,nbatch] = size(inputs{1});
        vis_index = randi(nbatch);
        avim = gpuArray(single(cat(3,123.680,116.779,103.939)));
        inputs_3 = bsxfun(@plus,inputs{3}(:,:,:,vis_index),avim);
        inputs_4 = bsxfun(@plus,inputs{4}(:,:,:,vis_index),avim);
        inputs_5 = bsxfun(@plus,inputs{5}(:,:,:,vis_index),avim);
        figure(2); imshow(uint8(cat(2,inputs_3,inputs_4,inputs_5)));     
        
%         mask = vl_nnsoftmax(inputs{1}*10);
%         mask = repmat(mask(:,:,41,:)>0.013,1,1,size(inputs{2},3),1);
%         outputs{1} = vl_nnloss(inputs{1}, mask.*inputs{2}, [], 'loss', obj.loss, obj.opts{:});
        outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:});
        obj.accumulateAverage(inputs, outputs);
    end

    function accumulateAverage(obj, inputs, outputs)
      if obj.ignoreAverage, return; end
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 1) *  size(inputs{1}, 2) * size(inputs{1}, 4);
      obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%       mask = vl_nnsoftmax(inputs{1});
%       mask = repmat(mask(:,:,41,:)>0.013,1,1,size(inputs{2},3),1);
%       derInputs{1} = vl_nnloss(inputs{1}, mask.*inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:});
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:});
      derInputs{2} = [];
      derInputs{3} = [];
      derInputs{4} = gpuArray(zeros(size(inputs{4}),'single'));
      derInputs{5} = [];
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss_wVisualization(varargin)
      obj.load(varargin) ;
    end
  end
end
