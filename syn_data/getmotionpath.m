function [locx,locy,curr_scale_x,curr_scale_y,vmotion_type] = getmotionpath(motion_type,nobjects,subtlety,x,y,loc)

nframes = length(loc);
curr_scale_x = ones(nobjects,nframes);
curr_scale_y = ones(nobjects,nframes);
vmotion_type = ones(nobjects,1)*motion_type;

% patches moving along random line.
if (motion_type == 1)
    velocity_x = -50 + randi(100, [1 nobjects]);    % random between -50:50
    velocity_y = -50 + randi(100, [1 nobjects]);
    if subtlety
        velocity_x = velocity_x./subtlety;
        velocity_y = velocity_y./subtlety;
    end
    locx = bsxfun(@plus,x, bsxfun(@times,loc,velocity_x'));
    locy = bsxfun(@plus,y, bsxfun(@times,loc,velocity_y'));
end


% patches moving along line and back to original position
if (motion_type == 2)
    loc(loc>0.5) = 1 - loc(loc>0.5);  % change second half to 'moving back'
    velocity_x = -50 + randi(100, [1 nobjects]);    % random between -50:50
    velocity_y = -50 + randi(100, [1 nobjects]);
    % can move twice the velocity since it is going half way and moving back
    velocity_x = velocity_x *2;
    velocity_y = velocity_y *2;
    if subtlety
        velocity_x = velocity_x./subtlety;
        velocity_y = velocity_y./subtlety;
    end
    locx = bsxfun(@plus,x, bsxfun(@times,loc,velocity_x'));
    locy = bsxfun(@plus,y, bsxfun(@times,loc,velocity_y'));
end


% shapes are in place, jitter a bit and only inflate and shrink in a cyclic manner
if (motion_type == 3)    
    scale_x = (4 + randi(20-5, [1 nobjects]))/10;   % scale between 0.5 to 2
    scale_y = (4 + randi(20-5, [1 nobjects]))/10;
    if subtlety
        scale_x = scale_x./subtlety;
        scale_y = scale_y./subtlety;
    end
    phase = rand(1, nobjects)*pi;
    curr_scale_x = bsxfun(@times,sin(bsxfun(@plus,pi*loc, phase'))+0.2,scale_x');
    curr_scale_y = bsxfun(@times,sin(bsxfun(@plus,pi*loc, phase'))+0.2,scale_y');
    curr_scale_x(curr_scale_x<0.1) = 0.1;
    curr_scale_y(curr_scale_y<0.1) = 0.1;
    %curr_scale_y = bsxfun(@times,sin(pi*loc)+0.2,scale_y');
    bla = ones(1,nobjects);
    locx = bsxfun(@plus,x, bsxfun(@times,loc,bla'));
    locy = bsxfun(@plus,y, bsxfun(@times,loc,bla'));
end

% patches moving in a cyclic path
if (motion_type == 4)
    theta0 = rand(nobjects,1)*2*pi;
    theta = bsxfun(@plus,(loc)*2*pi,theta0);
    
    radius = 5 + randi(40-5, [1 nobjects]);
    if subtlety
        radius = radius./subtlety;
    end
    
    locx = bsxfun(@plus,x,bsxfun(@times,cos(theta),radius'));
    locy = bsxfun(@plus,y,bsxfun(@times,sin(theta),radius'));
end

if (motion_type == 5)
    %mechanism to ensure a fair distribution where all motion types appear
    vmotion_type = [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4];
    vmotion_type = vmotion_type(1:nobjects);
    tmp = randperm(4);
    vmotion_type = tmp(vmotion_type);
    for ii = 1:nobjects,
        [locx(ii,:),locy(ii,:),curr_scale_x(ii,:),curr_scale_y(ii,:)] = getmotionpath(vmotion_type(ii),1,subtlety,x(ii),y(ii),loc);
    end
end
