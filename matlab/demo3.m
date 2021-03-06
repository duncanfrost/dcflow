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
% param.weight_file = '../net/sintel.caffemodel';

param.weight_file = '../net/kitti.caffemodel';
im2 = imread('../data/im2.png');

caffe.set_mode_gpu();
net = caffe.Net(param.model_file, param.weight_file, 'test');

directory = '/home/duncanlocal/Data/TUM/rgbd_dataset_freiburg2_desk/rgb/';
outDir = '/home/duncanlocal/Data/TUM/rgbd_dataset_freiburg2_desk/feature2/';
list = dir(directory);





for i = 1:size(list)
    l = list(i);
    if (~l.isdir)
        imgname = l.name;
        name = imgname(1:length(imgname)-4);
        binname = [name '.bin'];
        binpath = [outDir binname];
        impath = [directory imgname];
        display(binpath);

        im1 = imread(impath);
        [feat1, feat2] = getfeatures(im1, im2, param, net);
        feat = permute(feat1, [3 2 1]);
        fileID = fopen(binpath, 'w');
        fwrite(fileID,feat,'float');
        fclose(fileID);


        display(binname);
    end
end



