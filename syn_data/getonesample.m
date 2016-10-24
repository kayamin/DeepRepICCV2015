function [cFrames] = getonesample(noiselevel,nframes,motion_type,subtype,subtlety,scalefactor,current_label)
%subtype 1 rotates individual patches

if motion_type==6
    [cFrames] = getonesampletype6(noiselevel,nframes,motion_type,subtype,subtlety,current_label);
    return
end

if ~exist('subtype','var')
    subtype = 0;
end

if ~exist('subtlety','var')
    subtlety = 0;
end

if ~exist('scalefactor','var')
    scalefactor = 1;
end

bkg = rand(100,100)/3;
G = fspecial('gaussian',[3 3],0.6);

nobjects = 2 + randi(6-2, 1);
objsizes = (15 + randi(40-15, [1 nobjects]))/scalefactor;
% lencycle is random from 3-10
if (current_label == 8) % no repetition mode.
    lencycle = 20;
    subtlerot = 1;
elseif (current_label == 9) % no movement mode
    lencycle = 20;
    subtlety = 10;
    subtlerot = 10;
else
    lencycle = current_label + 3;
    subtlerot = 1;
end

for i = 1:nobjects
    cI{i} = imresize(rand(6),[objsizes(i),objsizes(i)]);
end

% random patches location
x = 10 + randi(80,nobjects,1);
y = 10 + randi(80,nobjects,1);
%x = randn(nobjects,1)*14+50;
%y = randn(nobjects,1)*14+50;
loc = (1:nframes)/lencycle;
loc = loc-floor(loc);

[locx,locy,curr_scale_x,curr_scale_y,vmotion_type] = getmotionpath(motion_type,nobjects,subtlety,x,y,loc);

% jittering
locx = locx + randn(size(locx,1),size(locx,2))*noiselevel/3;
locy = locy + randn(size(locy,1),size(locy,2))*noiselevel/3;


if subtype==1
    degrot0 = rand(nobjects,1)*360;
    if ~subtlety
        degrotmax = min(360,degrot0 + rand(nobjects,1)*60);
    else
        degrotmax = min(360,degrot0 + rand(nobjects,1)*(60/subtlety));
    end
    degrot = bsxfun(@plus,degrotmax*loc,degrot0);
end

for j = 1:nframes,
    cFrames{j} = bkg;
    for i = 1:nobjects,
        tmp = zeros(300);
        tmp(round(locy(i,j)+100),round(locx(i,j)+100)) = 1;
        if (vmotion_type(i) == 3)
            cPatch = imresize(cI{i}, [size(cI{i},1)*curr_scale_x(i,j)  size(cI{i},2)*curr_scale_y(i,j)]);
            thispatch = cPatch;
        else
            thispatch = cI{i};
        end 
        if subtype==1 
            thispatch = imrotate(thispatch,degrot(i,j));
        end
        tmp = conv2(tmp,thispatch,'same');
        tmp = tmp(101:200,101:200);
        cFrames{j} = (tmp==0).*cFrames{j} + (tmp>0).*tmp;
    end
    cFrames{j} = imresize(cFrames{j},[50 50]);
    % rotate image by [-15:15] deg.
    randDeg = (-16 + randi(31, 1))/subtlerot;
    cFrames{j} = imrotate(cFrames{j},randDeg,'bilinear','crop');
    % fill black corners with noise
    tmpbackgrnd = rand(50)/3;
    tmpbackgrnd = imfilter(tmpbackgrnd,G,'same');
    cFrames{j}(cFrames{j}==0) = tmpbackgrnd(cFrames{j}==0);
end
