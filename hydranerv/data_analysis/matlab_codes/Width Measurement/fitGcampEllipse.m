function [bw,a,b,cent,theta] = fitGcampEllipse(movieParam,ifvisualize)
% Segment hydra region from video and generate a mask matrix
% SYNOPSIS:
%     [bw,bg] = gnMask(movieParam,ifInvert)
% INPUT:
%     movieParam: a struct returned by paramAll
%     ifInvert: 0 if white object, black background, 1 otherwise
% OUTPUT:
%     bw: a binary matrix with the same size as input video
%     bg: a binary matrix with the same size as one frame of the video,
%       with the estimated background region
% 
% Shuting Han, 2017

% initialization
bw = false(movieParam.imageSize(1),movieParam.imageSize(2),movieParam.numImages);
a = zeros(movieParam.numImages,1);
b = zeros(movieParam.numImages,1);
theta = zeros(movieParam.numImages,1);
cent = zeros(movieParam.numImages,2);

% area threshold
P = round(movieParam.imageSize(1)*movieParam.imageSize(2)/400);

writerobj = VideoWriter('out.avi');
open(writerobj);
hf = figure;

sig = 3;
fgauss = fspecial('gaussian',2*ceil(2*sig)+1,sig);
for n = 1:movieParam.numImages
    
    % read image
    im = double(imread([movieParam.filePath movieParam.fileName '.tif'],n));
    
    if n==1
        dims = size(im);
%         figure; imagesc(im)
    end
    
    % smoothe image
    im = imfilter(im,fgauss);
    
    % segmentation
%     im_seg = reshape(kmeans(im(:),4),dims(1),dims(2));
%     f_bw = im_seg~=mode(im_seg(:));
    f_bw = im>multithresh(im);
    
    % threshold area
    f_bw = bwareaopen(f_bw,P);
    
    % fit ellipse
    [a(n),b(n),cent(n,:),theta(n)] = findEllipseInBw(f_bw);
    
    bw(:,:,n) = f_bw;
    
    % visualize
    if ifvisualize
        rs = struct();
        rs.Centroid = cent(n,:); rs.Orientation = theta(n);
        rs.MajorAxisLength = a(n); rs.MinorAxisLength = b(n);
        hold off;
        imagesc(bw(:,:,n)); hold on
        plot_ellipse2(rs);
        quiver(rs.Centroid(1),rs.Centroid(2),cos(degtorad(rs.Orientation))*...
        rs.MajorAxisLength,-sin(degtorad(rs.Orientation))*rs.MajorAxisLength);
        xlim([0 dims(1)]);ylim([0 dims(2)]);
        hold off
        axis equal tight
        title(num2str(n));pause(0.01);
        F = getframe(hf);
        writeVideo(writerobj,F);
    end    
    
end

close(writerobj);

end