classdef Transform_to_Flow_Layer < dagnn.ElementWise
    
    properties
        Ho = 0;
        Wo = 0;
    end
    
	properties (Transient)
        xx;
        yy;
        zero;
	end

  methods
    function outputs = forward(obj, inputs, params)
        outputs{1} = cat(3,inputs{1}(:,:,3,:),inputs{1}(:,:,6,:));
    end
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)  
        derInputs{1} = gpuArray(zeros(size(inputs{1}),'single'));
        derInputs{1}(:,:,3,:) = derOutputs{1}(:,:,1,:);
        derInputs{1}(:,:,6,:) = derOutputs{1}(:,:,2,:);
        derParams = {} ;
    end

    function obj = Transform_to_Flow_Layer(varargin)
      obj.load(varargin);
    end
   
  end
  
end
