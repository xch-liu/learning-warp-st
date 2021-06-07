%% Script for Geometric Style Transfer
clc; clear; close all;
run('vlfeat-0.9.21/toolbox/vl_setup');
run('matconvnet-1.0-beta25/matlab/vl_setupnn.m');
addpath('model');
addpath('function');
addpath('flow-code-matlab');

%% Load model
load('model/warpnet.mat');
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
net.vars(net.getVarIndex('comp_iter4_flow_transform_x1')).precious = 1;

%% Read content image and style image
CONTENT_IMAGE = imread('inputs/clock.png');
STYLE_IMAGE = imread('inputs/dali_melting_clock.png');

%% Original size of content image and style image
[heightS_Ori, widthS_Ori, dimS_Ori] = size(STYLE_IMAGE);
[heightC_Ori, widthC_Ori, dimC_Ori] = size(CONTENT_IMAGE);

%% Resize images
content_image = imresize(CONTENT_IMAGE, [256, 256]);
style_image = imresize(STYLE_IMAGE, [256, 256]);

%% Init
init = zeros(16,16,6,'single');
init(:,:,1) = 1; init(:,:,5) = 1;
imgs = single(style_image);
imgc = single(content_image);
avim = single(cat(3,123.680,116.779,103.939));
imgs = bsxfun(@minus,imgs,avim);
imgc = bsxfun(@minus,imgc,avim);        
inputs = {'init', init, 'f1_input', imgs, 'f2_input', imgc};   

%% Get warp field
net.eval(inputs);
results = gather(net.vars(net.getVarIndex('comp_iter4_flow_transform_x1')).value);
vx = imresize(results(:,:,2,:),[256,256],'bilinear');
vy = imresize(results(:,:,1,:),[256,256],'bilinear');
vx = vx*112;
vy = vy*112;
flow = cat(3,vx,vy);

%% Warped content image
imgWarping = warpImage(im2double(content_image),vx,vy);

%% Resize warped content image to style size
output = imresize(uint8(imgWarping*256), [heightS_Ori, widthS_Ori]);

%% Save results
save_folder = 'demo';
style_dir = fullfile(save_folder,'style.png');
content_dir = fullfile(save_folder,'content.png');
flow_dir = fullfile(save_folder,'flow.png');
flowmat_dir = fullfile(save_folder,'flow.mat');
output_dir = fullfile(save_folder,'output.png');

imwrite(STYLE_IMAGE, style_dir);
imwrite(CONTENT_IMAGE, content_dir);
imwrite(uint8(flowToColor(flow)), flow_dir);
save(flowmat_dir,'flow');
imwrite(output, output_dir);

%% Visualize
figure(1);
subplot(2,2,1), imshow(STYLE_IMAGE)
subplot(2,2,2), imshow(CONTENT_IMAGE)
subplot(2,2,3), imshow(output)
subplot(2,2,4), imshow(flowToColor(flow))
