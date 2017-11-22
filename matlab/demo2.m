close all; clear all;

addpath('/home/duncanlocal/source/caffe/matlab')

param.model_file = '../net/deploy.prototxt';
param.maxDisp = 242; % maximum displacement for both x and y direction
param.ratio   = 3;   % downsample scale
param.P1      = 7;   % SGM param
param.P2      = 485; % SGM param
param.outOfRange    = 0.251; % Default cost for out-of-range displacements
param.occ_threshold = 0.8;   % threshold for fwd+bwd consisntency check

param.P2 = 600;
param.weight_file = '../net/sintel.caffemodel';
im1 = imread('../data/im1.png');
im2 = imread('../data/im2.png');

caffe.set_mode_gpu();
net = caffe.Net(param.model_file, param.weight_file, 'test');

[feat1, feat2] = getfeatures(im1, im2, param, net);

s = prod(size(feat1));
nChannels = size(feat1,3); 
width = size(feat1,2);

display(nChannels);
display(nChannels);

total = 0;

fileID = fopen('test.bin', 'w');
for y_m = 1:size(feat1,1)
    for x_m = 1:size(feat1,2)
        A = zeros(nChannels,1);
        for c_m = 1:size(feat1,3)
            A(c_m) = feat1(y_m,x_m,c_m);
        end
        fwrite(fileID,A,'float');
    end
end
fclose(fileID);

feat = permute(feat1, [3 2 1]);

fileID = fopen('test2.bin', 'w');
fwrite(fileID,feat,'float');
fclose(fileID);




