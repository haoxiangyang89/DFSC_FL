for i = 0:29
    astr = sprintf('~/Dropbox/NU Documents/Hurricane/Data/GEFS_Simu/Path14_%d.png',i);
    a = imread(astr);
    F(i+1) = im2frame(a);
end

vidObj = VideoWriter('~/Dropbox/NU Documents/Hurricane/Data/GEFS_Simu/Path14.avi');
vidObj.FrameRate = 1;
open(vidObj);
writeVideo(vidObj,F);
close(vidObj);