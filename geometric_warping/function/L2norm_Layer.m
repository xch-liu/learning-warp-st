classdef L2norm_Layer < dagnn.ElementWise

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_l2norm(inputs{1}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_l2norm(inputs{1}, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = L2norm_Layer(varargin)
      obj.load(varargin) ;
    end
  end
  
end
