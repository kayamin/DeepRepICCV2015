function [cFrames] = getonesampletype6(noiselevel,nframes,motion_type,subtype,subtlety,current_label)
%subtype 1 rotates individual patches

if ~exist('subtype','var')
    subtype = 0;
end

if ~exist('subtlety','var')
    subtlety = 0;
end

G = fspecial('gaussian',[3 3],0.6);

if current_label>7
    error('unsupported at the moment');
end

lencycle = current_label + 3;
subtlerot = 1;

cbkg = cell(lencycle,1);
for ii = 1:lencycle,
     bkg = rand(20,20);
     bkg = imfilter(bkg,G,'same');
     cbkg{ii} = bkg;
end

locii = repmat((1:lencycle),[1 ceil(nframes/lencycle)]);
locii = locii(1:nframes);


for j = 1:nframes,
    dx = round(randn*noiselevel/5);    
    dy = round(randn*noiselevel/5);
    
    cFrames{j} = imtranslate(cbkg{locii(j)},[dx,dy],-1);
    cFrames{j} = imresize(cFrames{j},[50 50]);
    tmpbackgrnd = rand(50);
    tmpbackgrnd = imfilter(tmpbackgrnd,G,'same');
    cFrames{j}(cFrames{j}<0) = tmpbackgrnd(cFrames{j}<0);    
    % rotate image by [-15:15] deg.
    randDeg = (-11 + randi(21, 1))/subtlerot;
    cFrames{j} = imrotate(cFrames{j},randDeg,'bilinear','crop');
    % fill black corners with noise
    tmpbackgrnd = rand(50);
    tmpbackgrnd = imfilter(tmpbackgrnd,G,'same');
    cFrames{j}(cFrames{j}==0) = tmpbackgrnd(cFrames{j}==0);
end
